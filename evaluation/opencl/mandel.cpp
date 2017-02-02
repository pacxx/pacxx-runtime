#include <stdio.h>
#include <iostream>
#include <stdlib.h>
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


int Width;
int Height;
const int bailout = 1000;
float cr1, cr2, ci1, ci2;

std::vector<int> buffer;

typedef unsigned int u32;
typedef unsigned char u8;

// Translate (center + zoom) to (upper left + lower right)
void GetTranslatedCoordinates(float* cr1, float* cr2, float* ci1, float* ci2, float center_r, float center_i, float zoom) {
  *cr1 = center_r - zoom;
  *cr2 = center_r + zoom;
  float aspect_ratio = (float)width / (float)height;
  *ci1 = center_i - (zoom / aspect_ratio);
  *ci2 = center_i + (zoom / aspect_ratio);
}


void writePPM(int* mandel_bailouts) {
  ofstream ofs("mandelbrot.ppm", ios::binary);
  ofs << "P6" << "\n" << width << " " << height << " " << 255 << "\n";
  for (u32 y = 0; y < height; ++y) {
    for (u32 x = 0; x < width; ++x) {
      int v = mandel_bailouts[x + (y * width)];
      if (v > 255) {
        v = 255;
      }
      u8 vb = static_cast<u8>(v);
      ofs.write((char*)&vb, sizeof(u8));
      ofs.write((char*)&vb, sizeof(u8));
      ofs.write((char*)&vb, sizeof(u8));
    }
  }
}

unsigned div_up(unsigned a, unsigned b) {
  return ((a % b) != 0) ? (a / b + 1) : (a / b);
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
void dot(__global int* Md, \
                      float cr1, \
                      float cr2, \
                      float ci1, \
                      float ci2) { \
  u32 x(get_group_id(0) * get_local_size(0) + get_local_id(0)); \
  u32 y(get_group_id(1) * get_local_size(1) + get_local_id(1)); \
  if (x >= width || y >= height) { \
    return; \
  }\
  float cr = (x / (float)width) * (cr2 - cr1) + cr1; \
  float ci = (y / (float)height) * (ci2 - ci1) + ci1; \
  float zi = 0.0f, zr = 0.0f, zr2 = 0.0f, zi2 = 0.0f, zit; \
  u32 iter = bailout; \
  while(--iter && zr2 + zi2 < 4.0f) { \
    zit = zr * zi; \
    zi = zit + zit + ci; \
    zr = (zr2 - zi2) + cr; \
    zr2 = zr * zr; \
    zi2 = zi * zi; \
  }\
  if (iter) { \
    iter = bailout - iter; \
  } \
  out[x + y * width] = iter * 5.0f; \
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

void mandelOpenCL(int* buffer, float cr1, float cr2, float ci1, float ci2) {
  cl_int err;

  int size = Width * Height * 3 * sizeof(int);

  size_t globalSize[] = {div_up(Width, 512), div_up(Height, 1024)};
  size_t localSize[] = {512, 1024};

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
  err  = clSetKernelArg( kernel, 0, sizeof(cl_mem), &buffer );
  err |= clSetKernelArg( kernel, 1, sizeof(float), &cr1 );
  err |= clSetKernelArg( kernel, 2, sizeof(float), &cr2 );
  err |= clSetKernelArg( kernel, 3, sizeof(float), &ci1);
  err |= clSetKernelArg( kernel, 4, sizeof(float), &ci2);
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

  Width = 1024;
  Height = 1024;
  size_t buffer_size = sizeof(int) * Width * Height * 3;

  buffer = std::vector<int>(buffer_size);

  float center_r = -0.5f, center_i = 0.0f;
  float zoom = 1.5f;
  GetTranslatedCoordinates(&cr1, &cr2, &ci1, &ci2, center_r, center_i, zoom);

  initOpenCL();
  makeKernel();
};

int main(void) {
  init();
  mandelOpenCL(buffer, cr1, cr2, ci1, ci2);
  writePPM(buffer.data());
  return 0;
}

