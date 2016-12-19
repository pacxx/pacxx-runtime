#include <vector>
#include <random>
#include <PACXX.h>
#include <type_traits>
#include <typeinfo>
#include <cstdio>

using namespace pacxx::v2;
using namespace std;

int main(int argc, char **argv) {


  std::vector<int> dst(8), src(8);

  auto test = [](int* dst, int* src, int alpha) {
    auto idx = Thread::get().global.x;
    dst[idx] = src[idx] + (alpha - 1);
  };

  auto& exec = get_executor<NativeRuntime>();

  auto& dev_dst = exec.allocate<int>(8);
  auto& dev_src = exec.allocate<int>(8);

  dev_dst.upload(dst.data(), 8);
  dev_src.upload(src.data(), 8);

  auto testKernel =
      kernel<NativeRuntime>(test, {{1}, {8}});

  testKernel(dev_dst.get(), dev_src.get(), 4);

  return 0;
}
