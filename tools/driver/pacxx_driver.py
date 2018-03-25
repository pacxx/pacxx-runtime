#!/usr/bin/env python

import sys
import os
from subprocess import call, check_output
import tempfile
import shutil
import time; 

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
	output += "static __module_registrator __reger(%s, %s_len);" % (name, name)
        return output


def remove_opt_level(args):
    new_args = []
    for s in args:
        if not s.startswith("-O"):
            new_args.append(s)
    return new_args

def remove_target(args):
    new_args = []
    for s in args:
        if not s.startswith("-target") and not s.startswith("nvptx"):
            new_args.append(s)
    return new_args



def handle_pacxx_args(args):
    return

class Context:
    includes = []
    libs = []
    workingdir = ""
    pacxx_dir = ""
    llvm_dir = ""
    llvm_libs= []
    sys_libs = []
    clang = ""
    llc = ""
    opt = ""
    nm = ""
    input_files = []
    object_files = []
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
        self.llvm_dir = check_output([self.pacxx_dir + "/llvm-config", "--prefix"]).decode("utf-8").rstrip()
        self.llvm_libs= check_output([self.pacxx_dir + "/llvm-config", "--libs"]).decode("utf-8").rstrip().split(" ")
        self.sys_libs = check_output([self.pacxx_dir + "/llvm-config", "--system-libs"]).decode("utf-8").rstrip().split(" ")
        self.clang = self.llvm_dir + "/bin/clang++"
        self.llc = self.llvm_dir + "/bin/llc"
        self.link = self.llvm_dir + "/bin/llvm-link"
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
            self.includes.append(path + "/include")

    def insert_libdir(self, path, suffix):
        if path:
            self.libs.append("-L")
            self.libs.append(path + "/lib" + (suffix or ''))
            self.libs.append("-Wl,-rpath")
            self.libs.append("-Wl," + path + "/lib" + (suffix or ''))
    
    def parse_args(self, args):
        for s in args:
            if s.endswith(".cpp"):
                self.input_files.append(s)
            if s.endswith(".cxx"):
                self.input_files.append(s)
            if s.endswith(".cc"):
                self.input_files.append(s)
            if s.endswith(".o"): 
		self.object_files.append(s)
        self.flags = [x for x in args[1:] if x not in self.input_files and x not in self.object_files]
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
                integration_file = self.workingdir+ "/" + filename_only + "_integration.cpp"
                ir_module = self.workingdir+ "/" + filename_only + ".bc"
                kernel_module = self.workingdir+ "/" + filename_only + "_kernel.bc"
                host_module = self.workingdir+ "/" + filename_only + "_host.bc"
                integration_module = self.workingdir+ "/" + filename_only + "_kernel_integration.bc"
                host_object = self.workingdir+ "/" + filename_only + "_host.o"

                dev_args = ["-pacxx", "-emit-llvm", "-c", "-Wno-unused-command-line-argument"]

                #compile the device code to llvm bitcode
                execute([ self.clang ] + dev_args + self.includes + args + [file] + ["-o", ir_module])
                execute([ self.opt ] + ["-load=libPACXXTransforms.so", "-strip-debug", "-pacxx-codegen-prepare", "-simplifycfg", "-inline", ir_module, "-o", kernel_module])
                execute([ self.opt ] + ["-load=libPACXXTransforms.so", "-pacxx-kernel-eraser", ir_module, "-o", host_module])
                output = check_output([self.nm, kernel_module])
                num_kernels = len(output.split('\n')) - 1
                if num_kernels:
                    #encode the kernel.bc file to a char array and include it into the integration header
                    encoded = xxd(kernel_module, "kernel" + str(time.time()).replace(".", ""))
                    with open(self.llvm_dir + "/include/pacxx/detail/ModuleIntegration.inc", 'r') as include_header:
                        data = include_header.read().replace('/*FILECONTENT*/', encoded)
                        with open(integration_file, "w") as integration_header_file:
                            integration_header_file.write(data)
                    execute([ self.clang ] + self.includes + remove_target(self.flags) + [integration_file, "-c", "-emit-llvm", "-o", integration_module])
                    execute([ self.link, integration_module, host_module, "-o", host_module])
                execute([ self.llc ] + [host_module, "-filetype=obj", "-o", host_object])
    
            #compile objects to the desired output
            if self.mode == 0: 
                execute([ self.clang ] + [host_object] + self.object_files  + self.libs + ["-o", original_output])
            elif self.mode == 1: 
                shutil.copyfile(host_object, original_output)
        else:
            if len(self.flags) > 0:
                execute([ self.clang ] + self.includes + self.flags + self.libs)
            else:
                execute([ self.clang ])

def main(argv):
    clang_args = []
    pacxx_args = []
    try:
        index = argv.index("--")
        clang_args = argv[0:index]
        pacxx_args = argv[index+1:]
        handle_pacxx_args(pacxx_args)
    except:
        clang_args = argv

    ctx = Context()
    try:
        ctx.parse_args(clang_args)
        ctx.compile()
    except KeyboardInterrupt:
        pass
    finally:
        del ctx

if __name__ == "__main__":
    main(sys.argv)
