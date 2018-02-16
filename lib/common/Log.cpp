//===-----------------------------------------------------------*- C++ -*-===//
//
//                       The LLVM-based PACXX Project
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "pacxx/detail/common/Log.h"
#include <sstream>
#include <string>

#include <llvm/IR/Value.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/raw_ostream.h>

using namespace llvm;

namespace pacxx {
namespace common {

void dumpToLog(const llvm::Value &V, std::string prefix, const char *file,
               int line) {
  std::string str;
  llvm::raw_string_ostream ss(str);
  V.print(ss);
  pacxx_log_print<LOG_LEVEL::verbose>(file, line, "[", prefix, "] ", ss.str());
}

Log &Log::get() {
  static Log the_log;
  return the_log;
}

Log::Log() : _silent(false), _no_warnings(false), output(std::cout) {
  _old_buffer = output.rdbuf();
  auto str = GetEnv("PACXX_LOG_LEVEL");
  log_level = 0;
  if (str.length() > 0) {
    log_level = std::stoi(str);
  }
}

Log::~Log() { resetStream(); }

void Log::printDiagnositc(const std::string &program,
                          const std::stringstream &file_line,
                          const std::string &msg_str,
                          LOG_LEVEL::LEVEL debug_level) {
  SourceMgr::DiagKind kind = SourceMgr::DiagKind::DK_Note;

  switch (debug_level) {
  case LOG_LEVEL::exception:
  case LOG_LEVEL::fatal:
  case LOG_LEVEL::error:
    kind = SourceMgr::DiagKind::DK_Error;
    break;
  case LOG_LEVEL::warning:
    if (_no_warnings)
      return;
    kind = SourceMgr::DiagKind::DK_Warning;
    break;
  default:
    break;
  }

  SMDiagnostic diag(file_line.str(), kind, msg_str);
  std::string message;
  raw_string_ostream os(message);
  diag.print(program.c_str(), os);

  if (kind == SourceMgr::DiagKind::DK_Error)
    std::cerr << os.str(); 
  else 
    std::cout << os.str(); 
}
} // namespace common
} // namespace pacxx
