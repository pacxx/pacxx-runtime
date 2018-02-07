#!/usr/bin/env python

import sys
import os
from subprocess import call, check_output
import tempfile
import shutil

def execute(command):
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


def lookup_include(path):
    if os.path.exists(path):
        return path+"/include"
    else:
        return ""

def insert_include(list, path):
    if not path:
        return list
    else:
        list.append("-I")
        list.append(path)
        return list

def main(argv):
    includes = []
    workingdir = tempfile.mkdtemp()
    llvm_dir = check_output(["llvm-config", "--prefix"]).rstrip();

    clang = llvm_dir + "/bin/clang++"
    opt = llvm_dir + "/bin/opt"

    insert_include(includes, lookup_include(llvm_dir))
    insert_include(includes, lookup_include("/usr/local/cuda"))
    insert_include(includes, lookup_include("/opt/rocm"))
    input_files = [];
    for s in argv:
        if s.endswith(".cpp"):
            input_files.append(s);
        if s.endswith(".cxx"):
            input_files.append(s);
        if s.endswith(".cc"):
            input_files.append(s);

    flags = [x for x in argv[1:] if x not in input_files]

    libs = []
    mode = 0 # 0 = compile and link | 1 = compile only
    
    if not "-c" in flags:
        libs = libs + ["-lpacxxrt2", "-lPACXXBeCUDA", "-lcuda"]
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
            dev_args = ["-std=c++17", "-pacxx", "-emit-llvm", "-c"]

            #compile the device code to llvm bitcode
            execute([ clang ] + dev_args + includes + args + [file] + ["-o", kernel_name])
            execute([ opt ] + ["-load=libPACXXTransforms.so", "-pacxx-codegen-prepare", "-inline", kernel_name, "-o", kernel_name])

            #encode the kernel.bc file to a char array and include it into the integration header
            encoded = xxd(kernel_name, "kernel");
            with open(llvm_dir + "/include/pacxx/detail/ModuleIntegration.h", 'r') as include_header:
                data = include_header.read().replace('##FILECONTENT##', encoded)
                with open(header_name, "w") as integration_header:
                    integration_header.write(data)

            #compile the host code with the integration header to an object file 
            object_files.append(object_name)   
            execute([ clang ] + ["--include", header_name] + includes + flags + [file, "-c", "-o", object_name])
 
        #compile objects to the desired output
        if mode == 0: 
            execute([ clang ] + libs + object_files + ["-o", original_output])
        elif mode == 1: 
            shutil.copyfile(object_files[0], original_output)
    else:
        execute([ clang ] + libs + includes + argv[1:] )
    #cleanup 
    shutil.rmtree(workingdir)

if __name__ == "__main__":
    print(" ".join(sys.argv))
    main(sys.argv)