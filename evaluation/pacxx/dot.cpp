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

void initVector(std::vector<int>& vector) {
  for(unsigned i = 0; i < vector.size(); ++i)
      vector[i] = std::rand();
}

bool compare(int first, int second) {
  std::cout << "Expected: " << second << ". Got: " << first << std::endl;
  return first == second;
}

int main(int argc, char **argv) {

  size_t count = OPT_N;

  std::vector<int> a(count), b(count), c(count);

  initVector(a);
  initVector(b);

  auto test = [](const int* a, const int* b, int* c, unsigned size) {

    shared_memory<int> sm;
    int tmp = 0;

    auto local_x = get_local_id(0);
    auto global_size = get_num_groups(0) * get_local_size(0);

    for(auto global_x = get_global_id(0); global_x < size; global_x += global_size) {
      tmp += a[global_x] * b[global_x];
    }

    sm[local_x] = tmp;
    barrier(0);

    for(unsigned i = get_local_size(0) / 2; i > 0; i /= 2) {
      if(local_x < i)
        sm[local_x] += sm[local_x + i];
      barrier(0);
    }

    if(local_x == 0)
      c[get_group_id(0)] = sm[0];
  };

  auto& exec = get_executor<NativeRuntime>();

  auto& dev_a = exec.allocate<int>(count, a.data());
  auto& dev_b = exec.allocate<int>(count, b.data());
  auto& dev_c = exec.allocate<int>(count, c.data());

  auto dot =
      kernel<NativeRuntime>(test, {{OPT_N / THREAD_N}, {THREAD_N}, sizeof(int) * THREAD_N});

  dot(dev_a.get(), dev_b.get(), dev_c.get(), count);

  exec.synchronize();

  int pacxx_result = std::accumulate(c.begin(), c.end(), 0);

  int seq_result = std::inner_product(a.begin(), a.end(), b.begin(), 0, std::plus<>(), std::multiplies<>());

  std::cout << "Equal: " << compare(pacxx_result, seq_result) << std::endl;

  return 0;
}
