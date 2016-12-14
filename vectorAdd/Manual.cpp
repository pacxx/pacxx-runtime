#include <vector>
#include <iostream>
#include <chrono>
#include <random>
#include <type_traits>
#include <typeinfo>
#include <cstdio>
#include <thread>
#include <immintrin.h>
#include "tbb/tbb.h"

using namespace std;

#define OPT_N (40960000)
#define THREAD_N 512 
#define BLOCKS (OPT_N / THREAD_N)
#define RUNS 1000

#define DIV_UP(a, b) (((a) + (b)-1) / (b))

float getRandom() {
  static std::default_random_engine e;
  static std::uniform_real_distribution<> dis(0, 5000); // rage 0 - 1
  return dis(e);
}

void initVector(std::vector<float>& vector) {
  for(unsigned i = 0; i < vector.size(); ++i)
      vector[i] = getRandom();
}

void calcSeq(const std::vector<float>& a, const std::vector<float>& b, std::vector<float>& c) {
  for(unsigned i = 0; i < a.size(); ++i)
    c[i] = a[i] + b[i];
}

void calcAVX(const float* a, const float* b, float* c, unsigned thread) {
  for(unsigned i = 0; i < THREAD_N; i+=8) {
    __m256 v1, v2, vadd;
    v1 = _mm256_loadu_ps(&a[thread + i]);
    v2 = _mm256_loadu_ps(&b[thread + i]);
    vadd = _mm256_add_ps(v1, v2);
    _mm256_storeu_ps(&c[thread + i], vadd);
  }
}


void calcSSE(const float*  a, const float* b, float*  c, unsigned thread) {
  for(unsigned i = 0; i < THREAD_N; i+=4) {
    __m128 v1, v2, vadd;
    v1 = _mm_loadu_ps(&a[thread + i]);
    v2 = _mm_loadu_ps(&b[thread + i]);
    vadd = _mm_add_ps(v1, v2);
    _mm_storeu_ps(&c[thread + i], vadd);
  }
}

void calcThread(const std::vector<float>& a, const std::vector<float>& b, std::vector<float>& c_sse, 
    std::vector<float>& c_avx) {

  std::vector<std::thread> threads;
  std::chrono::high_resolution_clock::time_point start_sse, end_sse, start_avx, end_avx;

  start_sse = std::chrono::high_resolution_clock::now();

  for(unsigned i = 0; i < RUNS; ++i) {

    tbb::parallel_for( int(0), BLOCKS, [&](int thread) {
        calcSSE(a.data(), b.data(), c_sse.data(), thread * THREAD_N);
    });
  }

  end_sse = std::chrono::high_resolution_clock::now();

  auto time_sse = std::chrono::duration_cast<std::chrono::microseconds>(end_sse - start_sse).count();
  std::cout << "timed (SSE): " << time_sse / RUNS << "us" << std::endl;

  start_avx = std::chrono::high_resolution_clock::now();

  for(unsigned i = 0; i < RUNS; ++i) {

    tbb::parallel_for( int(0), BLOCKS, [&](int thread) {
        calcAVX(a.data(), b.data(), c_avx.data(), thread * THREAD_N);
    });
  }

  end_avx = std::chrono::high_resolution_clock::now();

  auto time_avx = std::chrono::duration_cast<std::chrono::microseconds>(end_avx - start_avx).count();
  std::cout << "timed (AVX): " << time_avx / RUNS << "us" << std::endl;
}

bool compare(const std::vector<float> &first, const std::vector<float> &second) {
  bool equal = true;
  for(unsigned i = 0; i < first.size(); ++i)
    if(first[i] != second[i])
      equal = false;
  return equal;
}

void printVector(std::vector<float>& vec) {
  for(size_t i = 0; i < vec.size(); ++i)
    std::cout << vec[i] << std::endl;
}

int main(int argc, char **argv) {

  std::vector<float> a(OPT_N), b(OPT_N), c_sse(OPT_N), c_avx(OPT_N), c_seq(OPT_N);

  initVector(a);
  initVector(b);

  size_t count = OPT_N;

  calcThread(a, b, c_sse, c_avx);

  calcSeq(a, b, c_seq);

  std::cout << "Compare (SSE, SEQ): " << compare(c_sse, c_seq) << std::endl;
  std::cout << "Compare (AVX, SEQ): " << compare(c_avx, c_seq) << std::endl;

  return 0;
}
