#include <vector>
#include <random>
#include <PACXX.h>
#include <type_traits>
#include <typeinfo>
#include <cstdio>
#include <iostream>
#include <fstream>

using namespace pacxx::v2;
using namespace std;

#define WIDTH 1024
#define THREADS (256)

float getRandom() {
  static std::default_random_engine e;
  static std::uniform_real_distribution<> dis(0, 5000); // rage 0 - 1
  return dis(e);
}

void initMatrix(float* matrix) {
  srand( time(NULL) );
  for(int i = 0; i < WIDTH; ++i)
    for(int j=0; j < WIDTH; ++j)
      matrix[i * WIDTH + j] = ((float)rand()) / RAND_MAX;
 }

void clearMatrix(float* matrix) {
  for(int i = 0; i < WIDTH; ++i)
    for(int j=0; j < WIDTH; ++j)
      matrix[i * WIDTH + j] = 0;
}

void calcSeq(const std::vector<float>& a, const std::vector<float>& b, std::vector<float>& c) {
  for (int Col = 0; Col < WIDTH; ++Col)
    for (int Row = 0; Row < WIDTH; ++Row) {
      float sum = 0;
      for (int k = 0; k < WIDTH; ++k) {
        sum += a[Row * WIDTH + k] * b[k * WIDTH + Col];
      }
      c[Row * WIDTH + Col] = sum;
    }
}

bool compareMatrices(const std::vector<float> &first, const std::vector<float> &second) {
  bool equal = true;
  for(int i = 0; i < WIDTH; ++i)
    for(int j = 0; j < WIDTH; ++j)
      if(first[i * WIDTH +j] - second[i * WIDTH +j] != 0)
          equal = false;
  return equal;
}

void printMatrix(const std::vector<float> matrix, std::string filename) {
  ofstream outFile(filename);
  for(int i = 0; i < WIDTH; ++i) {
    for(int j = 0; j < WIDTH; ++j)
      outFile << matrix[i * WIDTH +j] << '\t';
    outFile << std::endl;
  }
  outFile << std::endl;
}

void calcPACXX(float* a, float* b, float* c, size_t matrix_size) {

  auto test = [](const float* a, const float* b, float* c, unsigned width) {
    auto column = get_global_id(0);
    auto row = get_global_id(1);
    float val = 0;
    for(unsigned i = 0; i < width; ++i)
      val += a[row * width + i] * b[i * width + column];
    c[row * width + column] = val;
  };

  auto& exec = get_executor<NativeRuntime>();

  auto& dev_a = exec.allocate<float>(matrix_size, a);
  auto& dev_b = exec.allocate<float>(matrix_size, b);
  auto& dev_c = exec.allocate<float>(matrix_size, c);

  auto matMul =
      kernel<NativeRuntime>(test, {{2, 1024}, {512, 1}});

  matMul(dev_a.get(), dev_b.get(), dev_c.get(), WIDTH);

  exec.synchronize();
}

int main(int argc, char **argv) {

  size_t matrix_size = WIDTH * WIDTH;

  std::vector<float> a(matrix_size), b(matrix_size), c(matrix_size), c_host(matrix_size);

  initMatrix(a.data());

  initMatrix(b.data());

  clearMatrix(c.data());

  calcPACXX(a.data(), b.data(), c.data(), matrix_size);

  calcSeq(a, b, c_host);

  std::cout << compareMatrices(c, c_host) << std::endl;

  return 0;
}

