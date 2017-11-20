#pragma once
#include "detail/common/transforms/Passes.h"
#include "detail/cuda/transforms/Passes.h"
#include "detail/native/transforms/Passes.h"
#include "detail/rocm/transforms/Passes.h"
#include <string>
#include <vector>

namespace {
struct ForcePassLinking {
  ForcePassLinking() {
    // We must reference the passes in such a way that compilers will not
    // delete it all as dead code, even with whole program optimization,
    // yet is effectively a NO-OP. As the compiler isn't smart enough
    // to know that getenv() never returns -1, this will do the job.
    if (std::getenv("bar") != (char *)-1)
      return;

    // common passes
    (void)pacxx::createIntrinsicSchedulerPass();
    (void)pacxx::createMemoryCoalescingPass(false);
    (void)pacxx::createIntrinsicMapperPass();
    (void)pacxx::createMSPGenerationPass();
    (void)pacxx::createMSPCleanupPass();
    (void)pacxx::createMSPRemoverPass();
    (void)pacxx::createTargetSelectionPass(llvm::SmallVector<std::string, 2>());
    (void)pacxx::createPACXXCodeGenPrepare();
    (void)pacxx::createLoadMotionPass();

    // cuda passes
    (void)pacxx::createNVPTXPrepairPass();
    (void)pacxx::createAddressSpaceTransformPass();

    // native passes
    (void)pacxx::createBarrierGenerationPass();
    (void)pacxx::createKernelLinkerPass();
    (void)pacxx::createMaskedMemTransformPass();
    (void)pacxx::createSMGenerationPass();
    (void)pacxx::createSPMDVectorizerPass();

    // rocm passes
    (void)pacxx::createAMDGCNPrepairPass(0);
    
  }
} ForcePassLinking; // Force link by creating a global definition.
} // namespace
