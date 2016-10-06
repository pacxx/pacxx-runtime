#include <vector>
#include <PACXX.h>

using namespace pacxx::v2;
using namespace std;

#define OPT_N (1)
#define THREAD_N 1

#define DIV_UP(a, b) (((a) + (b)-1) / (b))

int main(int argc, char **argv) {
  std::vector<int> a(OPT_N), b(OPT_N), c(OPT_N);

  auto vectorAdd = [](const vector<int> &a, const vector<int> &b,
                      vector<int> &c) {
    auto idx = Thread::get().global.x;
    if (idx < a.size())
      c[idx] = a[idx] + b[idx];
  };

  auto vaddKernel =
      kernel<CUDARuntime>(vectorAdd, {{(OPT_N + THREAD_N - 1) / THREAD_N}, {THREAD_N}});

  vaddKernel(a, b, c);

  auto& exec = get_executor();
  std::cout << typeid(exec.rt()).name() << '\n';
  exec.synchronize();
  return 0;
}
