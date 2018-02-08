#!/usr/bin/env python

import sys
import os
from subprocess import call, check_output
import tempfile
import shutil

def execute(command):
    if "PACXX_FE_VERBOSE" in os.environ:
        print(" ".join(command))
    retcode = call(command)
    if not retcode: 
        return
    else:
        sys.exit(retcode) 

def xxd(file_path, name):
    with open(file_path, 'r') as f:
        output = "unsigned char %s[] = {" % name
        length = 0
        while True:
            buf = f.read(12)

            if not buf:
                output = output[:-2]
                break
            else:
                output += "\n  "

            for i in buf:
                output += "0x%02x, " % ord(i)
                length += 1
        output += "\n};\n"
        output += "unsigned int %s_len = %d;" % (name, length)
        return output


def remove_opt_level(args):
    new_args = []
    for s in args:
        if not s.startswith("-O"):
            new_args.append(s)
    return new_args



class Context:
    includes = []
    libs = []
    workingdir = ""
    pacxx_dir = ""
    llvm_dir = ""
    llvm_libs= []
    sys_libs = []
    clang = ""
    opt = ""
    nm = ""
    input_files = []
    mode = 0
    flags = []

    def __init__(self):
        self.llvm_config()
        self.prepare_dependencies()

    def __del__(self):
        shutil.rmtree(self.workingdir)

    def llvm_config(self):
        self.workingdir = tempfile.mkdtemp()
        self.pacxx_dir = os.path.dirname(os.path.realpath(__file__))
        self.llvm_dir = check_output([self.pacxx_dir + "/llvm-config", "--prefix"]).rstrip()
        self.llvm_libs= check_output([self.pacxx_dir + "/llvm-config", "--libs"]).rstrip().split(" ")
        self.sys_libs = check_output([self.pacxx_dir + "/llvm-config", "--system-libs"]).rstrip().split(" ")
        self.clang = self.llvm_dir + "/bin/clang++"
        self.opt = self.llvm_dir + "/bin/opt"
        self.nm = self.llvm_dir + "/bin/llvm-nm"

    def prepare_dependencies(self):
        self.lookup_dependency(self.llvm_dir)
        self.lookup_dependency("/usr/local/cuda", "64")
        self.lookup_dependency("/usr/local/cuda/extras/CUPTI", "64")
        self.lookup_dependency("/opt/rocm")
    
    def lookup_dependency(self, path, suffix=None):
        if os.path.exists(path):
            self.insert_include(path)
            self.insert_libdir(path, suffix)

    def insert_include(self, path):
        if path:
            self.includes.append("-I")
            self.includes.append(path+"/include")

    def insert_libdir(self, path, suffix):
        if path:
            self.libs.append("-L")
            self.libs.append(path+"/lib"+(suffix or ''))
    
    def parse_args(self, args):
        for s in args:
            if s.endswith(".cpp"):
                self.input_files.append(s)
            if s.endswith(".cxx"):
                self.input_files.append(s)
            if s.endswith(".cc"):
                self.input_files.append(s)
        self.flags = [x for x in args[1:] if x not in self.input_files]
        if not "-c" in self.flags:
            self.libs = self.libs + ["-Wl,--start-group", "-lpacxxrt2", "-lPACXXTransforms", "-lRV"] + self.llvm_libs + self.sys_libs
            if os.path.exists(self.llvm_dir + "/lib/libPACXXBeROCm.so"):
                self.libs = self.libs + ["-lPACXXBeROCm", "-lhsa-runtime64"]
            if os.path.exists(self.llvm_dir + "/lib/libPACXXBeCUDA.so"):
                self.libs = self.libs + ["-lPACXXBeCUDA","-lcudart", "-lcuda", "-lcupti"]
            self.libs = self.libs + ["-Wl,--end-group"]
        else:
            self.mode = 1 # compile only

    def compile(self):
        if len(self.input_files) > 0:
            args = self.flags
            object_files = []
            original_output = []
            if "-o" in args:
                index = args.index("-o")
                original_output = args[index + 1]
                del args[index + 1]
                args.remove("-o")
            else: 
                original_output = "a.out"
            for file in self.input_files:
                filename_only = os.path.basename(file)
                header_name = self.workingdir+ "/" + filename_only + "_integration.h"
                kernel_name = self.workingdir+ "/" + filename_only + "_kernel.bc"
                object_name = self.workingdir+ "/" + filename_only + ".o"
                dev_args = ["-pacxx", "-emit-llvm", "-c"]

                #compile the device code to llvm bitcode
                execute([ self.clang ] + dev_args + self.includes + remove_opt_level(args) + [file] + ["-o", kernel_name])
                execute([ self.opt ] + ["-load=libPACXXTransforms.so", "-pacxx-codegen-prepare", "-inline", kernel_name, "-o", kernel_name])
                output = check_output([self.nm, kernel_name])
                num_kernels = len(output.split('\n')) - 1
                integration_header = []
                if num_kernels:
                    #encode the kernel.bc file to a char array and include it into the integration header
                    encoded = xxd(kernel_name, "kernel")
                    with open(self.llvm_dir + "/include/pacxx/detail/ModuleIntegration.h", 'r') as include_header:
                        data = include_header.read().replace('/*FILECONTENT*/', encoded)
                        with open(header_name, "w") as integration_header_file:
                            integration_header_file.write(data)
                    integration_header = ["--include", header_name]

                #compile the host code with the integration header to an object file 
                object_files.append(object_name)   
                execute([ self.clang ] + integration_header + self.includes + self.flags + [file, "-c", "-o", object_name])
    
            #compile objects to the desired output
            if self.mode == 0: 
                execute([ self.clang ] + object_files  + self.libs + ["-o", original_output])
            elif self.mode == 1: 
                shutil.copyfile(object_files[0], original_output)
        else:
            execute([ self.clang ] + self.includes + self.flags + self.libs)

def main(argv):
    ctx = Context()
    try:
        ctx.parse_args(argv)
        ctx.compile()
    finally:
        del ctx

if __name__ == "__main__":
    main(sys.argv)