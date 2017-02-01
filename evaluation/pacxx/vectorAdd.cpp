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

#define OPT_N 1048576
#define THREAD_N 512

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

  size_t count = OPT_N;

  std::vector<float> a(count), b(count), c(count), c_seq(count), c_sse(count);

  initVector(a);
  initVector(b);

  auto test = [](const float* a, const float* b, float* c, unsigned size) {
    auto idx = get_global_id(0);
    if(idx < size)
      c[idx] = a[idx] + b[idx];
  };

  auto& exec = get_executor<NativeRuntime>();

  auto& dev_a = exec.allocate<float>(count, a.data());
  auto& dev_b = exec.allocate<float>(count, b.data());
  auto& dev_c = exec.allocate<float>(count, c.data());

  auto vaddKernel =
      kernel<NativeRuntime>(test, {{OPT_N / THREAD_N}, {THREAD_N}});

  vaddKernel(dev_a.get(), dev_b.get(), dev_c.get(), count);

  exec.synchronize();

  calcSeq(a, b, c_seq);

  std::cout << "Equal: " << compare(c, c_seq) << std::endl;

  return 0;
}
