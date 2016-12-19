#include <vector>
#include <random>
#include <PACXX.h>
#include <type_traits>
#include <typeinfo>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <limits>

using namespace pacxx::v2;
using namespace std;

//40960000
#define OPT_N 40960000
#define THREAD_N 512

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

bool compare(const std::vector<float> &first, const std::vector<float> &second) {
  bool equal = true;
  for(unsigned i = 0; i < first.size(); ++i)
    if(first[i] != second[i])
      equal = false;
  return equal;
}

void printVector(std::vector<float>& vec, std::string filename) {
  ofstream outFile(filename);
  for(size_t i = 0; i < vec.size(); ++i)
    outFile << vec[i] << std::endl;
}

int main(int argc, char **argv) {

  std::chrono::high_resolution_clock::time_point start, end;
  std::vector<float> a(OPT_N), b(OPT_N), c(OPT_N), c_seq(OPT_N), c_sse(OPT_N);

  initVector(a); 
  initVector(b);

  size_t count = OPT_N;

  auto test = [](const float* a, const float* b, float* c, unsigned size) {
    auto idx = Thread::get().global.x;
    if(idx < size)
      c[idx] = a[idx] + b[idx];
  };

  auto& exec = get_executor<NativeRuntime>();

  auto& dev_a = exec.allocate<float>(count);
  auto& dev_b = exec.allocate<float>(count);
  auto& dev_c = exec.allocate<float>(count);

  dev_a.upload(a.data(), count);
  dev_b.upload(b.data(), count);

  auto vaddKernel =
      kernel<NativeRuntime>(test, {{(OPT_N + THREAD_N - 1)/THREAD_N}, {THREAD_N}});

  start = std::chrono::high_resolution_clock::now();

  vaddKernel(dev_a.get(), dev_b.get(), dev_c.get(), OPT_N);

  end = std::chrono::high_resolution_clock::now();

  auto time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

  std::cout << "timed (pacxx): " << time /1000 << "us" << std::endl;

  dev_c.download(c.data(), count);

  calcSeq(a, b, c_seq);

  std::cout << "Equal: " << compare(c, c_seq) << std::endl;

  return 0;
}
