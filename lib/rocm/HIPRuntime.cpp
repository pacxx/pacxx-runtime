//
// Created by mhaidl on 30/05/16.
//

#include "pacxx/detail/rocm/HIPErrorDetection.h"
#include "pacxx/detail/rocm/HIPRuntime.h"
#include "pacxx/detail/common/Exceptions.h"
#include "pacxx/detail/common/LLVMHelper.h"
#include "pacxx/detail/common/Timing.h"
#include <llvm/IR/Constants.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/Transforms/Utils/Cloning.h>

#include <llvm/Transforms/Scalar.h>
#include <llvm/Transforms/Vectorize.h>

#include <hip/hip_runtime.h>

using namespace llvm;

namespace pacxx {
namespace v2 {
HIPRuntime::HIPRuntime(unsigned dev_id)
    : IRRuntime(RuntimeKind::RK_HIP), _context(nullptr), _compiler(std::make_unique<CompilerT>()),
      _dev_props(16), _delayed_compilation(false) {
  SEC_HIP_CALL(hipInit(0));
  hipCtx_t old;
  SEC_HIP_CALL(hipCtxGetCurrent(&old)); // check if there is already a context
  if (old) {
    _context = old; // we use the existing context
  }
  if (!_context) { // create a new context for the device
    hipDevice_t device;
    SEC_HIP_CALL(hipDeviceGet(&device, dev_id));
    SEC_HIP_CALL(hipCtxCreate(&_context, 0, device));
    __verbose("Creating hipCtx for device: ", dev_id, " ", device, " ",
              _context);
  }

  auto &prop = _dev_props[dev_id];
  SEC_HIP_CALL(hipGetDeviceProperties(&prop, dev_id));

  unsigned CC = prop.major * 10 + prop.minor;

  __verbose("Initializing PTXBackend for ", prop.name, " (dev: ", dev_id,
            ") with compute capability ", prop.major, ".", prop.minor);
  _compiler->initialize(CC);
}

HIPRuntime::~HIPRuntime() {}

void HIPRuntime::link(std::unique_ptr<llvm::Module> M) {

  _rawM = std::move(M);

  auto RM = _compiler->prepareModule(*_rawM);
  initializeMSP(std::move(RM));

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

void HIPRuntime::compileAndLink() {
  std::string MC;
  SCOPED_TIMING { MC = _compiler->compile(*_M); };

  SEC_HIP_CALL(hipModuleLoadData(&_mod, MC.c_str()));

  _delayed_compilation = false;
}

Kernel &HIPRuntime::getKernel(const std::string &name) {
  auto It = std::find_if(_kernels.begin(), _kernels.end(),
                         [&](const auto &p) { return name == p.first; });
  if (It == _kernels.end()) {
    hipFunction_t ptr = nullptr;
    if (!_delayed_compilation) {
      SEC_HIP_CALL(hipModuleGetFunction(&ptr, _mod, name.c_str()));
      if (!ptr)
        throw common::generic_exception("Kernel function not found in module!");
    }

    _kernels[name].reset(new HIPKernel(*this, ptr, name));

    return *_kernels[name];
  } else {
    return *It->second;
  }
}

size_t HIPRuntime::getPreferedMemoryAlignment() {
  return 256; // on HIP devices memory is best aligned at 256 bytes
}

RawDeviceBuffer *HIPRuntime::allocateRawMemory(size_t bytes, MemAllocMode mode) {
  HIPRawDeviceBuffer raw([this](HIPRawDeviceBuffer& buffer){ deleteRawMemory(&buffer); }, mode);
  raw.allocate(bytes);
  auto wrapped = new HIPDeviceBuffer<char>(std::move(raw));
  _memory.push_back(std::unique_ptr<DeviceBufferBase>(
      static_cast<DeviceBufferBase *>(wrapped)));
  return wrapped->getRawBuffer();
}

void HIPRuntime::deleteRawMemory(RawDeviceBuffer *ptr) {
  auto It = std::find_if(_memory.begin(), _memory.end(), [&](const auto &uptr) {
    return static_cast<HIPDeviceBuffer<char> *>(uptr.get())->getRawBuffer() ==
           ptr;
  });
  if (It != _memory.end())
    _memory.erase(It);
  else
    __error("ptr to delete not found");
}



void HIPRuntime::requestIRTransformation(Kernel &K) {
  if (_msp_engine.isDisabled())
    return;
  _M = CloneModule(_rawM.get());
  _M->setDataLayout(_rawM->getDataLayoutStr());

  _msp_engine.transformModule(*_M, K);

  compileAndLink();

  hipFunction_t ptr = nullptr;
  SEC_HIP_CALL(hipModuleGetFunction(&ptr, _mod, K.getName().c_str()));
  static_cast<HIPKernel &>(K).overrideFptr(ptr);
}

void HIPRuntime::synchronize() { SEC_HIP_CALL(hipDeviceSynchronize()); }

size_t HIPRuntime::getPreferedVectorSize(size_t dtype_size) {
  return 1;
}

size_t HIPRuntime::getPreferedVectorSizeInBytes(){
  return 8;
}

size_t HIPRuntime::getConcurrentCores() {
  int dev = -1;
  SEC_HIP_CALL(hipGetDevice(&dev));
  return _dev_props[dev].multiProcessorCount;
}

bool HIPRuntime::supportsUnifiedAddressing(){
  return false;
}

bool HIPRuntime::checkSupportedHardware() {
  int count = -1;
  SEC_HIP_CALL(hipGetDeviceCount(&count));
  __verbose("HIPRuntime has found ", count, " HIP devices");
  return count != 0;
}

const llvm::Module &HIPRuntime::getModule() { return *_M; }

}
}
