//
// Created by mhaidl on 30/05/16.
//

#include "pacxx/detail/cuda/CUDARuntime.h"
#include "pacxx/detail/common/Exceptions.h"
#include "pacxx/detail/common/LLVMHelper.h"
#include "pacxx/detail/common/Timing.h"
#include <llvm/IR/Constants.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/Transforms/Utils/Cloning.h>

#include <llvm/Transforms/PACXXTransforms.h>
#include <llvm/Transforms/Scalar.h>
#include <llvm/Transforms/Vectorize.h>

namespace llvm {
FunctionPass *createNVPTXInferAddressSpacesPass();
}

using namespace llvm;

namespace pacxx {
namespace v2 {
CUDARuntime::CUDARuntime(unsigned dev_id)
    : _context(nullptr), _compiler(std::make_unique<CompilerT>()),
      _delayed_compilation(false) {
  SEC_CUDA_CALL(cuInit(0));
  CUcontext old;
  SEC_CUDA_CALL(cuCtxGetCurrent(&old)); // check if there is already a context
  if (old) {
    _context = old; // we use the existing context
  }
  if (!_context) { // create a new context for the device
    CUdevice device;
    SEC_CUDA_CALL(cuDeviceGet(&device, dev_id));
    SEC_CUDA_CALL(cuCtxCreate(&_context, CU_CTX_SCHED_AUTO, device));
    __verbose("Creating cudaCtx for device: ", dev_id, " ", device, " ",
              _context);
  }
  cudaDeviceProp prop;
  SEC_CUDA_CALL(cudaGetDeviceProperties(&prop, dev_id));

  unsigned CC = prop.major * 10 + prop.minor;

  __verbose("Initializing PTXBackend for ", prop.name, " (dev: ", dev_id,
            ") with compute capability ", prop.major, ".", prop.minor);
  _compiler->initialize(CC);
}

CUDARuntime::~CUDARuntime() {}

void CUDARuntime::link(std::unique_ptr<llvm::Module> M) {

  _rawM = std::move(M);
  llvm::legacy::PassManager PM;

  PM.add(createPACXXSpirPass());
  PM.add(createPACXXClassifyPass());
  PM.add(createPACXXNvvmPass());

  PM.add(createPACXXNvvmRegPass(false));
  PM.add(createPACXXInlinerPass());
  PM.add(createPACXXDeadCodeElimPass());
  PM.add(createCFGSimplificationPass());
  PM.add(createNVPTXInferAddressSpacesPass());
  PM.add(createSROAPass());
  PM.add(createPromoteMemoryToRegisterPass());
  PM.add(createDeadStoreEliminationPass());
  PM.add(createInstructionCombiningPass());
  PM.add(createCFGSimplificationPass());
  PM.add(createSROAPass());
  PM.add(createPromoteMemoryToRegisterPass());
  PM.add(createInstructionCombiningPass());
  PM.add(createNVPTXInferAddressSpacesPass());

  PM.run(*_rawM);

  _M = CloneModule(_rawM.get());
  _M->setDataLayout(_rawM->getDataLayoutStr());

  auto reflect = _M->getFunction("__pacxx_reflect");
  if (!reflect || reflect->getNumUses() == 0) {
    compileAndLink();
  } else {
    __verbose("Module contains unresolved calls to __pacxx_reflect. Linking "
              "delayed!");
    _delayed_compilation = true;
  }
}

void CUDARuntime::compileAndLink() {
  std::string MC;
  SCOPED_TIMING { MC = _compiler->compile(*_M); };
  float walltime;
  char error_log[81920];
  char info_log[81920];
  size_t logSize = 81920;

  // Setup linker options

  CUjit_option lioptions[] = {
      CU_JIT_WALL_TIME,                  // Return walltime from JIT compilation
      CU_JIT_INFO_LOG_BUFFER,            // Pass a buffer for info messages
      CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES, // Pass the size of the info buffer
      CU_JIT_ERROR_LOG_BUFFER,           // Pass a buffer for error message
      CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES, // Pass the size of the error buffer
      CU_JIT_LOG_VERBOSE                  // Make the linker verbose
  };

  void *opt_values[] = {
      reinterpret_cast<void *>(&walltime), reinterpret_cast<void *>(info_log),
      reinterpret_cast<void *>(logSize),   reinterpret_cast<void *>(error_log),
      reinterpret_cast<void *>(logSize),   reinterpret_cast<void *>(1)};

  __verbose(MC);
  SEC_CUDA_CALL(
      cuModuleLoadDataEx(&_mod, MC.c_str(), 6, lioptions, opt_values));
  if (info_log[0] != '\0')
    __verbose("Linker Output: \n", info_log);
  _delayed_compilation = false;
}

Kernel &CUDARuntime::getKernel(const std::string &name) {
  auto It = std::find_if(_kernels.begin(), _kernels.end(),
                         [&](const auto &p) { return name == p.first; });
  if (It == _kernels.end()) {
    CUfunction ptr = nullptr;
    if (!_delayed_compilation) {
      SEC_CUDA_CALL(cuModuleGetFunction(&ptr, _mod, name.c_str()));
      if (!ptr)
        throw common::generic_exception("Kernel function not found in module!");
    }
    auto kernel = new CUDAKernel(*this, ptr);
    kernel->setName(name);
    _kernels[name].reset(kernel);
    auto buffer_size =
        _msp_engine.getArgBufferSize(*_rawM->getFunction(name), *kernel);
    kernel->setHostArgumentsSize(buffer_size);
    return *kernel;
  } else {
    return *It->second;
  }
}

size_t CUDARuntime::getPreferedMemoryAlignment() {
  return 256; // on CUDA devices memory is best aligned at 256 bytes
}

RawDeviceBuffer *CUDARuntime::allocateRawMemory(size_t bytes) {
  CUDARawDeviceBuffer raw;
  raw.allocate(bytes);
  auto wrapped = new CUDADeviceBuffer<char>(std::move(raw));
  _memory.push_back(std::unique_ptr<DeviceBufferBase>(
      static_cast<DeviceBufferBase *>(wrapped)));
  return wrapped->getRawBuffer();
}

void CUDARuntime::deleteRawMemory(RawDeviceBuffer *ptr) {
  auto It = std::find_if(_memory.begin(), _memory.end(), [&](const auto &uptr) {
    return static_cast<CUDADeviceBuffer<char> *>(uptr.get())->getRawBuffer() ==
           ptr;
  });
  if (It != _memory.end())
    _memory.erase(It);
  else
    __error("ptr to delete not found");
}

void CUDARuntime::initializeMSP(std::unique_ptr<llvm::Module> M) {
  if (!_msp_engine.isDisabled())
    return;
  _msp_engine.initialize(std::move(M));
}

void CUDARuntime::evaluateStagedFunctions(Kernel &K) {
  if (K.requireStaging()) {
    if (_msp_engine.isDisabled())
      return;
    _msp_engine.evaluate(*_rawM->getFunction(K.getName()), K);
  }
}

void CUDARuntime::requestIRTransformation(Kernel &K) {
  if (_msp_engine.isDisabled())
    return;
  _M = CloneModule(_rawM.get());
  _M->setDataLayout(_rawM->getDataLayoutStr());

  _msp_engine.transformModule(*_M, K);

  compileAndLink();

  CUfunction ptr = nullptr;
  SEC_CUDA_CALL(cuModuleGetFunction(&ptr, _mod, K.getName().c_str()));
  static_cast<CUDAKernel &>(K).overrideFptr(ptr);
}

const llvm::Module &CUDARuntime::getModule() { return *_M; }

void CUDARuntime::synchronize() { SEC_CUDA_CALL(cudaDeviceSynchronize()); }

llvm::legacy::PassManager &CUDARuntime::getPassManager() {
  return _compiler->getPassManager();
}
}
}
