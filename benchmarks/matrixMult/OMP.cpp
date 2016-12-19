#include <vector>
#include <random>
#include <chrono> 
#include <iostream>
#include <type_traits>
#include <typeinfo>
#include <cstdio>
#include <omp.h>

#define WIDTH 1024
#define RUNS 2

float getRandom() {
  static std::default_random_engine e;
  static std::uniform_real_distribution<> dis(0, 5000); // rage 0 - 1
  return dis(e);
}

void initMatrix(float* matrix) {
  for(int i = 0; i < WIDTH; ++i)
    for(int j=0; j < WIDTH; ++j)
      matrix[i * WIDTH + j] = getRandom();
}

void clearMatrix(float* matrix) {
  for(int i = 0; i < WIDTH; ++i)
    for(int j=0; j < WIDTH; ++j)
      matrix[i * WIDTH + j] = 0;
}

void calcSeq(const std::vector<float>& a, const std::vector<float>& b, std::vector<float>& c) {
  for(int i = 0; i < WIDTH; ++i)
    for(int j=0; j < WIDTH; ++j)
      for(int k = 0; k < WIDTH; ++k)
        c[i * WIDTH + j] += a[i * WIDTH + k] * b[k * WIDTH + j];
}

bool compareMatrices(const std::vector<float> &first, const std::vector<float> &second) {
  bool equal = true;
  for(int i = 0; i < WIDTH; ++i)
    for(int j = 0; j < WIDTH; ++j)
      if(first[i * WIDTH +j] != second[i * WIDTH +j])
          equal = false;
  return equal;
}

void prinMatrix(const std::vector<float> matrix) {
  for(int i = 0; i < WIDTH; ++i) {
    for(int j = 0; j < WIDTH; ++j)
      std::cout << matrix[i * WIDTH +j] << '\t';
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

void calcOmp(float* a, float* b, float* c) {

  std::chrono::high_resolution_clock::time_point start, end;
  omp_set_num_threads(16);

  int width = WIDTH;

  start = std::chrono::high_resolution_clock::now();

  #pragma omp parallel for simd default(none) shared(a, b, c, width)
    for (int i = 0; i < width; ++i)
      for(int j = 0; j < width; ++j)
        for (int k = 0; k < width; ++k)
          c[i * width + j] += a[i * width + k] * b[k * width + j];

  end = std::chrono::high_resolution_clock::now();

  auto time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

  std::cout << "timed (openmp): " << time << "us" << std::endl;
}

int main(int argc, char **argv) {

  size_t matrix_size = WIDTH * WIDTH;

  std::vector<float> a(matrix_size), b(matrix_size), c(matrix_size), c_host(matrix_size);

  initMatrix(a.data());

  initMatrix(b.data());

  for(int i = 0; i < RUNS; ++i) {
    clearMatrix(c.data());
    calcOmp(a.data(), b.data(), c.data());
  }

  calcSeq(a, b, c_host);

  std::cout << compareMatrices(c, c_host) << std::endl;

  return 0;
}


