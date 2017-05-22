//
// Created by m_haid02 on 28.04.17.
//

#include "pacxx/detail/common/Log.h"
#include "pacxx/detail/Kernel.h"
#include "pacxx/detail/KernelConfiguration.h"
#include "pacxx/detail/IRRuntime.h"

namespace pacxx {
namespace v2 {

Kernel::Kernel(IRRuntime &runtime)
    : _runtime_ref(runtime), _staged_values_changed(false),
      _disable_staging(false) {}

KernelConfiguration Kernel::getConfiguration() const { return _config; }

void Kernel::setCallback(std::function<void()> callback) {
  _callback = callback;
};

void Kernel::setArguments(const std::vector<char> &arg_buffer) {
  __verbose("Set kernel args");
  _args = arg_buffer;
}

const std::vector<char> &Kernel::getArguments() const { return _args; }

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

const std::vector<size_t> &Kernel::getArugmentBufferOffsets() {
  if (_arg_offsets.size() == 0) {
    auto &M = _runtime_ref.getModule();
    auto F = M.getFunction(_name);
    size_t offset = 0;
    size_t old = 0;
    _arg_offsets.resize(F->arg_size());
    std::transform(F->arg_begin(), F->arg_end(), _arg_offsets.begin(),
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

  return _arg_offsets;

}

size_t Kernel::getArgBufferSize() {
  return _argBufferSize;
}

}
}