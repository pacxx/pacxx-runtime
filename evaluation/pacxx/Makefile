CC=../../pacxx
PACXXV2_HOME = $(PACXX_PATH)/include/pacxx

CXX_FLAGS=-nvvm -D__PACXX_V2 -Wno-invalid-noreturn -Wno-zero-length-array 
CXX_INCLUDES=-I$(PACXXV2_HOME)

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
