#include <vector>
#include <PACXX.h>
#include <type_traits>
#include <typeinfo>

using namespace pacxx::v2;
using namespace std;

#define OPT_N (1)
#define THREAD_N 1

#define DIV_UP(a, b) (((a) + (b)-1) / (b))

int main(int argc, char **argv) {

  std::vector<int> a(OPT_N), b(OPT_N), c(OPT_N);

  size_t count = OPT_N;

  auto test = [](int* a) {
      a[0] = 20;
  };

  auto& exec = get_executor<CUDARuntime>();

  auto& dev_a = exec.allocate<int>(count);

  auto vaddKernel =
      kernel<CUDARuntime>(test, {{(OPT_N + THREAD_N - 1) / THREAD_N}, {THREAD_N}});

  vaddKernel(dev_a.get());

  exec.synchronize();

  dev_a.download(a.data(), count);

  std::cout << a[0] << std::endl;
  return 0;
}
