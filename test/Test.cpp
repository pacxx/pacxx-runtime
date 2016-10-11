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

  auto test = [](const vector<int> &a) {
    auto idx = Thread::get().global.x;
    if (idx < a.size())
      a[idx] = 20;
  };

  DeviceBuffer devA = MemoryManager::translateVector(a);

  auto vaddKernel =
      kernel<NativeRuntime>(vectorAdd, {{(OPT_N + THREAD_N - 1) / THREAD_N}, {THREAD_N}});

  vaddKernel(a);

  auto& exec = get_executor<NativeRuntime>();
  exec.synchronize();

  devA.download(a, OPT_N * sizeof(int));

  std::cout << a[0] << std::endl;
  return 0;
}
