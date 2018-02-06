#!/usr/bin/env python

import sys
import os
from subprocess import call, check_output
import tempfile
import shutil


# ... do stuff with dirpath


def lookup_include(path):
    if os.path.exists(path):
        return path+"/include"
    else:
        return ""

def insert_include(list, path):
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
    input_files = [s for s in argv if ".cpp" in s];
    dev_args = ["-std=c++17", "-pacxx", "-emit-llvm", "-c", "-o", workingdir+"/kernel.bc"]
    host_args = ["--include", workingdir+"/pacxx_integration.h"]

    if not "-c" in argv:
        host_args = host_args + ["-lpacxxrt2", "-lPACXXBeCUDA", "-lcuda"]

    command = [ clang ] + dev_args + includes + argv[1:]

    #compile the device code to llvm bitcode
    print(" ".join(command))
    call(command)
    command = [ opt ] + ["-load=libPACXXTransforms.so", "-pacxx-codegen-prepare", "-inline", workingdir+"/kernel.bc", "-o", workingdir+"/kernel.bc"]
    call(command)
    #encode the kernel.bc file to a char array and include it into the integration header
    current_dir = os.getcwd(); 
    os.chdir(workingdir)
    encoded = check_output(["xxd", "-i", "kernel.bc"]).rstrip();
    with open(llvm_dir + "/include/pacxx/detail/ModuleIntegration.h", 'r') as include_header:
        data = include_header.read().replace('##FILECONTENT##', encoded)
        with open(workingdir+"/pacxx_integration.h", "w") as integration_header:
            integration_header.write(data)
    os.chdir(current_dir)

    #compile the host code with the integration header 
    command = [ clang ] + host_args + includes + argv[1:]
    call(command)
    print(" ".join(command))   

    #cleanup 
    shutil.rmtree(workingdir)

if __name__ == "__main__":
    main(sys.argv)