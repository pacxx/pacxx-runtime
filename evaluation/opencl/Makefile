CC=clang++
CXX_FLAGS= -O3 -std=c++11 -L /opt/intel/opencl-1.2-5.2.0.10002/lib64/ -lOpenCL
CXX_INCLUDES=

all : vadd matmul dot mandel

vadd : vectorAdd.cpp
	$(CC) $(CXX_FLAGS) $(CXX_INCLUDES) $? -o $@

matmul : matMul.cpp
	$(CC) $(CXX_FLAGS) $(CXX_INCLUDES) $? -o $@

dot: dot.cpp
	$(CC) $(CXX_FLAGS) $(CXX_INCLUDES) $? -o $@

mandel: mandel.cpp
	$(CC) $(CXX_FLAGS) $(CXX_INCLUDES) $? -o $@

clean : cleanup
	./cleanup
