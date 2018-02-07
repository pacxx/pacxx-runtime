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

def insert_libdir(list, path, suffix):
    if not path:
        return list
    else:
        list.append("-L")
        list.append(path+"/lib"+(suffix or ''))
        return list

def insert_include(list, path):
    if not path:
        return list
    else:
        list.append("-I")
        list.append(path+"/include")
        return list

def lookup_dependency(path, include_dirs, lib_dirs, suffix=None):
    if os.path.exists(path):
        include_dirs = insert_include(include_dirs, path)
        lib_dirs = insert_libdir(lib_dirs, path, suffix)
    return include_dirs, lib_dirs



def main(argv):
    try:
        includes = []
        libs = []
        workingdir = tempfile.mkdtemp()
        pacxx_dir = os.path.dirname(os.path.realpath(__file__))
        llvm_dir = check_output([pacxx_dir + "/llvm-config", "--prefix"]).rstrip();
        llvm_libs= check_output([pacxx_dir + "/llvm-config", "--libs"]).rstrip().split(" ");
        sys_libs = check_output([pacxx_dir + "/llvm-config", "--system-libs"]).rstrip().split(" ");
        clang = llvm_dir + "/bin/clang++"
        opt = llvm_dir + "/bin/opt"
        nm = llvm_dir + "/bin/llvm-nm"

        includes, libs = lookup_dependency(llvm_dir, includes, libs)
        includes, libs = lookup_dependency("/usr/local/cuda", includes, libs, "64")
        includes, libs = lookup_dependency("/usr/local/cuda/extras/CUPTI", includes, libs, "64")
        includes, libs = lookup_dependency("/opt/rocm", includes, libs)
        
        input_files = [];
        for s in argv:
            if s.endswith(".cpp"):
                input_files.append(s);
            if s.endswith(".cxx"):
                input_files.append(s);
            if s.endswith(".cc"):
                input_files.append(s);

        flags = [x for x in argv[1:] if x not in input_files]

        mode = 0 # 0 = compile and link | 1 = compile only
        
        if not "-c" in flags:
            libs = libs + ["-Wl,--start-group", "-lpacxxrt2", "-lPACXXTransforms", "-lRV"] + llvm_libs + sys_libs; 
            if os.path.exists(llvm_dir + "/lib/libPACXXBeROCm.so"):
                libs = libs + ["-lPACXXBeROCm", "-lhsa-runtime64"]
            if os.path.exists(llvm_dir + "/lib/libPACXXBeCUDA.so"):
                libs = libs + ["-lPACXXBeCUDA","-lcudart", "-lcuda", "-lcupti"]
            libs = libs + ["-Wl,--end-group"]
        else:
            mode = 1 # compile only

        if len(input_files) > 0:
            args = flags; 
            object_files = []
            original_output = []
            if "-o" in args:
                index = args.index("-o")
                original_output = args[index + 1]
                del args[index + 1]
                args.remove("-o")
            else: 
                original_output = "a.out"
            for file in input_files:
                filename_only = os.path.basename(file);
                header_name = workingdir+ "/" + filename_only + "_integration.h"
                kernel_name = workingdir+ "/" + filename_only + "_kernel.bc"
                object_name = workingdir+ "/" + filename_only + ".o"
                dev_args = ["-pacxx", "-emit-llvm", "-c"]

                #compile the device code to llvm bitcode
                execute([ clang ] + dev_args + includes + args + [file] + ["-o", kernel_name])
                execute([ opt ] + ["-load=libPACXXTransforms.so", "-pacxx-codegen-prepare", "-inline", kernel_name, "-o", kernel_name])
                output = check_output([nm, kernel_name])
                num_kernels = len(output.split('\n')) - 1
                integration_header = []
                if num_kernels:
                #encode the kernel.bc file to a char array and include it into the integration header
                    encoded = xxd(kernel_name, "kernel");
                    with open(llvm_dir + "/include/pacxx/detail/ModuleIntegration.h", 'r') as include_header:
                        data = include_header.read().replace('/*FILECONTENT*/', encoded)
                        with open(header_name, "w") as integration_header_file:
                            integration_header_file.write(data)
                    integration_header = ["--include", header_name]

                #compile the host code with the integration header to an object file 
                object_files.append(object_name)   
                execute([ clang ] + integration_header + includes + flags + [file, "-c", "-o", object_name])
    
            #compile objects to the desired output
            if mode == 0: 
                execute([ clang ] + object_files  + libs + ["-o", original_output])
            elif mode == 1: 
                shutil.copyfile(object_files[0], original_output)
        else:
            execute([ clang ] + includes + argv[1:] + libs)
    finally:
        #cleanup 
        shutil.rmtree(workingdir)

if __name__ == "__main__":
    main(sys.argv)