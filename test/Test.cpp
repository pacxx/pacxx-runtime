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

  auto test = [](int* a) {
      a[idx] = 20;
  };

  Executor& exec = get_executor<CUDARuntime>();

  DeviceBuffer<int>& dev_a = exec.allocate(1);

  Kernel vaddKernel =
      kernel<CUDARuntime>(test, {{(OPT_N + THREAD_N - 1) / THREAD_N}, {THREAD_N}});

  vaddKernel(dev_a.get());


  dev_a.download(&a.begin(), sizeof(int));

  exec.synchronize();


  std::cout << a[0] << std::endl;
  return 0;
}
