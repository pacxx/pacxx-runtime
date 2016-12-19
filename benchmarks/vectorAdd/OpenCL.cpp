#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <CL/cl.h>
#include <string.h>

float* M;
float* N;
float* P_opencl;
float* P_seq;
int Width;
int Num_Threads;

// fill f width size many random float values
void fill(float* f, int size) {
  srand( time(NULL) );
  int i;
  for (i = 0; i < size; i+=1)
    f[i] = ((float)rand()) / RAND_MAX;
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

// compares every pair lhs[i] and rhs[i] for i < width
void compare(float* lhs, float* rhs, int width) {
  int errors = 0;
  int i;
  for (i = 0; i < width; i+=1) {
    if ((lhs[i] - rhs[i]) != 0) {
      printf("%f : %f\n", lhs[i], rhs[i]);
      errors += 1;
    }
  }
  if (errors > 0)
    printf("%d errors occured.\n", errors);
  else
    printf("no errors occured.\n");
}

// sequentiell matrix multiplication
void VectorAddSeq() {
  int Col, Row, k;
    for (k = 0; k < Width; k+=1) {
      P_seq[k] = M[k] + N[k];
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
void MatrixMultKernel(__global float* Md, \
                      __global float* Nd, \
                      __global float* Pd, int width) { \
  int idx = get_global_id(0); \
  if(idx < width) \
    Pd[idx] = Md[idx] + Nd[idx]; \
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
  kernel = clCreateKernel(program, "MatrixMultKernel", &err);
  checkError(err);
  printf("kernel created\n");
}

void VectorAddOpenCL(float* M, float* N, float* P, int width) {
  cl_int err;

  int size = width * sizeof(float);

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
  err |= clSetKernelArg( kernel, 3, sizeof(int), &width );
  checkError(err);
  printf("kernel arguments set\n");

  unsigned runs = 1000;
  size_t globalSize[] = {width};
  size_t localSize[] = {512};
  double total_time;

  err = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, globalSize, localSize, 0, NULL, NULL);
  checkError(err);

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
  Width = 40960000;
  M = (float*)malloc(Width*sizeof(float));
  N = (float*)malloc(Width*sizeof(float));
  P_opencl  = (float*)malloc(Width*sizeof(float));
  P_seq     = (float*)malloc(Width*sizeof(float));

  fill(M, Width);
  fill(N, Width);
  initOpenCL();
  makeKernel();
};

int main(void) {
  init();
  VectorAddOpenCL(M, N, P_opencl, Width);
  VectorAddSeq();
  compare(P_seq, P_opencl, Width);
  return 0;
}

