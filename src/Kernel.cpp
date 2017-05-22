//
// Created by m_haid02 on 28.04.17.
//

#include "pacxx/detail/common/Log.h"
#include "pacxx/detail/Kernel.h"
#include "pacxx/detail/KernelConfiguration.h"
#include "pacxx/detail/IRRuntime.h"

namespace pacxx {
namespace v2 {

Kernel::Kernel(IRRuntime &runtime, std::string name)
    : _runtime_ref(runtime), _staged_values_changed(false),
      _disable_staging(false), _name(name) {
  auto &M = _runtime_ref.getModule();
  auto F = M.getFunction(_name);
  size_t offset = 0;
  std::for_each(F->arg_begin(), F->arg_end(),
                [&](const auto &arg) {
                  auto arg_size =
                      M.getDataLayout().getTypeAllocSize(arg.getType());
                  auto arg_alignment =
                      M.getDataLayout().getPrefTypeAlignment(arg.getType());

                  auto arg_offset =
                      (offset + arg_alignment - 1) & ~(arg_alignment - 1);

                  offset = arg_offset + arg_size;
                  _argBufferSize = offset;
                  return arg_offset;
                });

}

KernelConfiguration Kernel::getConfiguration() const { return _config; }

void Kernel::setCallback(std::function<void()> callback) {
  _callback = callback;
};

void Kernel::setStagedValue(int ref, long long value, bool inScope) {
  auto old = _staged_values[ref];
  if (old != value) {
    _staged_values[ref] = value;
    if (inScope)
      _staged_values_changed = true;
  }
}

const std::map<int, long long> &Kernel::getStagedValues() const {
  return _staged_values;
}

void Kernel::setName(std::string name) { _name = name; }

const std::string &Kernel::getName() const { return _name; }

void Kernel::disableStaging() { _disable_staging = true; }

bool Kernel::requireStaging() { return !_disable_staging; }

}
}