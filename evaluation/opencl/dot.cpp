#include <CL/cl.h>
#include <string.h>
#include <vector>
#include <random>
#include <type_traits>
#include <typeinfo>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <limits>


std::vector<int> M;
std::vector<int> N;
std::vector<int> P_opencl;
size_t Width;
int Num_Threads;

// fill f width size many random int values
void fill(int* f, int size) {
  for(unsigned i = 0; i < size; ++i)
    f[i] = std::rand();
}

void printPlatform(cl_platform_id* platform, cl_uint platformCount) {
  int i, j;
  char* info;
  size_t infoSize;
  const char* attributeNames[5] = { "Name", "Vendor", "Version", "Profile", "Extensions" };
  const cl_platform_info attributeTypes[5] = { CL_PLATFORM_NAME, CL_PLATFORM_VENDOR,
    CL_PLATFORM_VERSION, CL_PLATFORM_PROFILE, CL_PLATFORM_EXTENSIONS };
  const int attributeCount = sizeof(attributeNames) / sizeof(char*);
  for(int i = 0; i < platformCount; ++i) {
    for (j = 0; j < attributeCount; j++) {
      clGetPlatformInfo(platform[i], attributeTypes[j], 0, NULL, &infoSize);
      info = (char*) malloc(infoSize);
      clGetPlatformInfo(platform[i], attributeTypes[j], infoSize, info, NULL);
      printf("  %d.%d %-11s: %sn", i+1, j+1, attributeNames[j], info);
      free(info);
    }
    printf("\n");
  }
}

// ######################################################
// Start OpenCL section
cl_platform_id*   platform;
cl_uint           platformCount;
cl_device_id      device;
cl_context        context;
cl_command_queue  commandQueue;
cl_kernel         kernel;

// check err for an OpenCL error code
void checkError(cl_int err) {
  if (err != CL_SUCCESS)
    printf("Error with errorcode: %d\n", err);
}

void initOpenCL() {
  cl_int err;
  size_t valueSize;
  char* value;

  // Speichere 1 Plattform in platform
  err = clGetPlatformIDs(5, NULL, &platformCount);
  checkError(err);

  platform = (cl_platform_id*) malloc(sizeof(cl_platform_id) * platformCount);
  err = clGetPlatformIDs(platformCount, platform, NULL);
  checkError(err);

  printPlatform(platform, platformCount);

  printf("platform selected\n");

  // Speichere 1 Device beliebigen Typs in device
  err = clGetDeviceIDs(platform[2], CL_DEVICE_TYPE_CPU, 1, &device, NULL);
  checkError(err);
  printf("device selected\n");

  clGetDeviceInfo(device, CL_DEVICE_NAME, 0, NULL, &valueSize);
  value = (char*) malloc(valueSize);
  clGetDeviceInfo(device, CL_DEVICE_NAME, valueSize, value, NULL);
  printf("Device: %s \n", value);
  free(value);

  // erzeuge Context fuer das Device device
  context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
  checkError(err);
  printf("context created\n");

  // erzeuge Command Queue zur Verwaltung von device
  commandQueue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
  checkError(err);
  printf("commandQueue created\n");
}

void printBuildLog(cl_program program, cl_device_id device) {
  cl_int err;
  char* build_log;
  size_t build_log_size;
  // Speichere den Build Log fuer program und device in build_log
  err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &build_log_size);
  checkError(err);

  build_log = (char*) malloc(build_log_size);

  err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, build_log_size, build_log, NULL);
  checkError(err);

  printf("Log:\n%s\n", build_log);

  free(build_log);
}

void makeKernel() {
  cl_int err;
  // Kernel Quellcode
  const char* kernelSource = "__kernel \
void dotKernel(__global int* Md, \
                      __global int* Nd, \
                      __global int* Pd, \
                      __local int* sm, \
                      int width) { \
  int tmp = 0; \
  size_t local_x = get_local_id(0); \
  size_t global_size = get_num_groups(0) * get_local_size(0); \
  for(size_t global_x = get_global_id(0); global_x < width; global_x += global_size) { \
    tmp += Md[global_x] * Nd[global_x]; \
  }\
  sm[local_x] = tmp;\
  barrier(CLK_LOCAL_MEM_FENCE);\
  for(unsigned i = get_local_size(0) / 2; i > 0; i /= 2) {\
    if(local_x < i) \
      sm[local_x] += sm[local_x + i];\
    barrier(CLK_LOCAL_MEM_FENCE);\
  }\
  if(local_x == 0)\
    Pd[get_group_id(0)] = sm[0];\
}";
  // Laenge des Kernel Quellcodes
  size_t sourceLength = strlen(kernelSource);
  cl_program program;
  // Ein Programm aus dem Kernel Quellcode wird erzeugt
  program = clCreateProgramWithSource(context, 1, &kernelSource, &sourceLength, &err);
  checkError(err);
  printf("program created\n");
  // Das Programm wird fuer alle Devices des Contextes gebaut
  err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
  if (err != CL_SUCCESS)
    printBuildLog(program, device);
  else
    printf("program build successfully\n");
  kernel = clCreateKernel(program, "dotKernel", &err);
  checkError(err);
  printf("kernel created\n");
}

void dotOpenCL(int* M, int* N, int* P, size_t width) {
  cl_int err;

  int size = width * sizeof(int);

  size_t globalSize[] = {width};
  size_t localSize[] = {512};

  // Buffer Md erzeugen und direkt auf das Device kopieren
  cl_mem Md = clCreateBuffer(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, size, M, &err);
  checkError(err);
  printf("buffer md created and copied\n");

  // Buffer ND erzeugen ohne zu kopieren
  cl_mem Nd = clCreateBuffer(context, CL_MEM_READ_ONLY, size, NULL, &err);
  checkError(err);
  printf("buffer nd created\n");
  // Daten explizit auf das Device kopieren
  // Dieser Aufruf ist nicht blockierend (CL_FALSE)
  err = clEnqueueWriteBuffer(commandQueue, Nd, CL_FALSE, 0, size, N, 0, NULL, NULL);
  checkError(err);
  printf("enqueued write buffer nd\n");

  // Speicher fuer Ergebnis Matrix reservieren
  cl_mem Pd = clCreateBuffer(context, CL_MEM_READ_WRITE, size, NULL, &err);
  checkError(err);
  printf("buffer pd created and memory allocated\n");

  // Setze Argument fuer den Kernel
  err  = clSetKernelArg( kernel, 0, sizeof(cl_mem), &Md );
  err |= clSetKernelArg( kernel, 1, sizeof(cl_mem), &Nd );
  err |= clSetKernelArg( kernel, 2, sizeof(cl_mem), &Pd );
  err |= clSetKernelArg( kernel, 3, sizeof(int) * localSize[0] , NULL);
  err |= clSetKernelArg( kernel, 4, sizeof(int), &width );
  checkError(err);
  printf("kernel arguments set\n");

  unsigned runs = 1000;

  double total_time;

  for(unsigned i = 0; i < runs; ++i) {
    cl_event event;
    cl_ulong time_start, time_end;
    err = clEnqueueNDRangeKernel( commandQueue, kernel, 1, NULL, globalSize, localSize, 0, NULL, &event);
    checkError(err);
    clWaitForEvents(1, &event);
    clFinish(commandQueue);

    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);

    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);

    total_time += time_end - time_start;

    clReleaseEvent(event);
  }

  printf("OpenCl Execution time is: %0.3f us \n",(total_time/1000.0) / runs);

  // Daten vom Device kopieren
  // Dieser Aufruf ist blockierend (CL_TRUE)
  err = clEnqueueReadBuffer( commandQueue, Pd,  CL_TRUE, 0, size, P, 0, NULL, NULL );
  checkError(err);
  printf("enqueued read buffer pd\n");
}

// end OpenCL section
// ######################################################

void init() {
  Width = 1048576;
  M = std::vector<int>(Width);
  N = std::vector<int>(Width);
  P_opencl  = std::vector<int>(Width);

  fill(M.data(), Width);
  fill(N.data(), Width);
  initOpenCL();
  makeKernel();
};

int main(void) {
  init();
  dotOpenCL(M.data(), N.data(), P_opencl.data(), Width);
  int openCL = std::accumulate(P_opencl.begin(), P_opencl.end(), 0);
  int seq = std::inner_product(M.begin(), M.end(), N.begin(), 0, std::plus<int>(), std::multiplies<int>());
  std::cout << "Equal: " << (seq == openCL) << std::endl;
  return 0;
}

