#include "PACXX.h"
#include "Executor.h"
#include "lodepng/lodepng.h"

#include <vector>
#include "algorithm.h"
#include "views.h"
#include <string>
#include <type_traits>

using namespace std;
using namespace pacxx;

using byte_t = unsigned char;

template <typename Rng, typename UnaryOp>
void for_each(const Rng &rng, UnaryOp func) {
  auto E = rng.end();
  auto I = rng.begin();

  for (; I != E; ++I) {
    func(*I);
  }
}

template <typename T> struct matrix_access {
  using value_type = T;

  matrix_access() : invalid(0), width(0), height(0){};

  template <typename X>
  matrix_access(X *ptr, const unsigned &width, const unsigned &height,
                int padding = 0)
      : ptr(ptr), invalid(0xFF00FFFF), width(width), height(height),
        padding(padding) {}
  T *ptr;
  T invalid;
  const unsigned width;
  const unsigned height;
  int padding;

  T &operator()(int x, int y) const {
    return ptr[(y + padding) * (2 * padding + width) + (x + padding)];
  }

  T at(int x, int y) const {

    // x += padding;
    // y += padding;

    if (is_inbounds(x, y)) {
      y += padding;
      x += padding;
      return ptr[y * (2 * padding + width) + x];
    } else
      return invalid;
  }
  T& at(int x, int y){

    // x += padding;
    // y += padding;

    if (is_inbounds(x, y)) {
      y += padding;
      x += padding;
      return ptr[y * (2 * padding + width) + x];
    } else
      return invalid;
  }

  bool is_inbounds(int x, int y) const {
    return x >= 0 && x < static_cast<int>(width + 2 * padding) && y >= 0 &&
           y < static_cast<int>(height + 2 * padding);
  }
};

template <typename T, class VECT> auto vector_cast(VECT &v) {
  auto data = &v[0];
  using type = typename std::conditional<
      std::is_const<typename std::remove_pointer<decltype(data)>::type>::value,
      const T, T>::type;
  return reinterpret_cast<type *>(data);
}

template <typename T>
auto to_matrix(T *ptr, unsigned width, unsigned height, int offset = 0,
               unsigned padding = 0) {
  return matrix_access<T>(ptr, width, height, padding);
}

template <typename T, typename VECT>
auto to_matrix(VECT &v, unsigned width, unsigned height) {
  using type = typename std::conditional<
      std::is_const<typename std::remove_reference<VECT>::type>::value, const T,
      T>::type;

  return matrix_access<type>(vector_cast<T>(v), width, height);
}

template <typename T> struct regular_stencil {

  using value_type = std::tuple<T, int, int>;
  using reference = const value_type &;
  using data_type = T *;

  regular_stencil() : range(0), x(0), y(0){};

  regular_stencil(int range, T *data, unsigned width, unsigned height, int x,
                  int y)
      : data(data), range(range), x(x), y(y), width(width), height(height) {}

  struct Sentinel;
  struct Iterator {

    friend struct Sentinel;

    Iterator(const regular_stencil &base, const int range, const int x,
             const int y)
        : i(-range), j(-range), range(range), x(x), y(y), _base(base) {}

    value_type operator*() {
      return std::make_tuple(_base.at(x + i, y + j), i, j);
    }

    Iterator &operator++() {
      ++i;
      if (i > range) {
        ++j;
        i = -range;
      }
      return *this;
    }

    bool done() const {
      bool done = j > range;
      return done;
    }

    int i, j;
    const int range, x, y;
    const regular_stencil &_base;
  };

  T at(const int i, const int j) const {
    return to_matrix<T>(data, width, height).at(i, j);
  }

  bool is_inbounds(const int i, const int j) const {
    return to_matrix<T>(data, width, height).is_inbounds(i, j);
  }

  Iterator begin() const { return Iterator(*this, range, x, y); }

  struct Sentinel {
    friend bool operator==(const Iterator &lhs, const Sentinel &rhs) {
      return lhs.done();
    }
    friend bool operator==(const Sentinel &lhs, const Iterator &rhs) {
      return rhs.done();
    }

    friend bool operator!=(const Iterator &lhs, const Sentinel &rhs) {
      return !lhs.done();
    }
    friend bool operator!=(const Sentinel &lhs, const Iterator &rhs) {
      return !rhs.done();
    }
  };

  Sentinel end() const { return Sentinel(); }

  T *data;
  int range;
  int x, y;
  unsigned width, height;
};

template <typename T> struct sm_regular_stencil {

  using value_type = std::tuple<T, int, int>;
  using reference = const value_type &;
  using data_type = T *;

  sm_regular_stencil()
      : data(nullptr), range(0), x(0), y(0), width(0), height(0) {}

  sm_regular_stencil(int range, T *data, unsigned width, unsigned height, int x, int y)
      : data(data), range(range), x(x), y(y), width(width), height(height) {
  }

  struct Sentinel;
  struct Iterator {

    friend struct Sentinel;

    Iterator(const sm_regular_stencil &base, const int range, const int x,
             const int y)
        : i(-range), j(-range), range(range), x(x), y(y), _base(base) {}

    value_type operator*() {
      auto tid = Thread::get().index;
      auto block = Block::get();
      auto bwidth = block.range.x;
      auto bheight = block.range.y;
      auto sm = to_matrix(&_base.sm_data[0], bwidth, bheight, 0, range);

      return std::make_tuple(sm.at(tid.x + i, tid.y + j), i, j);
    }

    Iterator &operator++() {
      ++i;
      if (i > range) {
        ++j;
        i = -range;
      }
      return *this;
    }

    bool done() const {
      bool done = j > range;
      return done;
    }

    int i, j;
    const int range, x, y;
    const sm_regular_stencil &_base;
  };

  T at(const int i, const int j) const {
    return to_matrix<T>(data, width, height).at(i, j);
  }

  bool is_inbounds(const int i, const int j) const {
    return to_matrix<T>(data, width, height).is_inbounds(i, j);
  }

  Iterator begin() const { 
    fetch();
    return Iterator(*this, range, x, y); 
  }

  struct Sentinel {
    friend bool operator==(const Iterator &lhs, const Sentinel &rhs) {
      return lhs.done();
    }
    friend bool operator==(const Sentinel &lhs, const Iterator &rhs) {
      return rhs.done();
    }

    friend bool operator!=(const Iterator &lhs, const Sentinel &rhs) {
      return !lhs.done();
    }
    friend bool operator!=(const Sentinel &lhs, const Iterator &rhs) {
      return !rhs.done();
    }
  };

  Sentinel end() const { return Sentinel(); }

  T *data;
  int range;
  int x, y;
  unsigned width, height;
  mutable v2::shared_memory<std::remove_cv_t<T>> sm_data;

  void fetch() const {
    auto id = Thread::get().global;
    auto tid = Thread::get().index;
    auto block = Block::get();
    auto bwidth = block.range.x;
    auto bheight = block.range.y;

    //   v2::shared_memory<T> sm_data;
    auto mdata = to_matrix<T>(data, width, height);
    auto sm = to_matrix(&sm_data[0], bwidth, bheight, 0, range);

    sm(tid.x, tid.y) = mdata.at(id.x, id.y); // read centers

    if (tid.y == 0) {
      for (int i = range; i > 0; --i)
        sm(tid.x, tid.y - i) = mdata.at(id.x, id.y - i); // read upper rows
    }

    if (tid.y == bheight - 1) {
      for (int i = 1; i <= range; ++i)
        sm(tid.x, tid.y + i) = mdata.at(id.x, id.y + i); // read lower rows
    }

    if (tid.x == 0) {
      for (int i = range; i > 0; --i)
        sm(tid.x - i, tid.y) = mdata.at(id.x - i, id.y); // read left cols
    }

    if (tid.x == bwidth - 1) {
      for (int i = 1; i <= range; ++i)
        sm(tid.x + i, tid.y) = mdata.at(id.x + i, id.y); // read right cols
    }

    if (tid.x == 0 && tid.y == 0) {
      for (int j = range; j > 0; --j) {
        for (int i = range; i > 0; --i) {
          sm(tid.x - i, tid.y - j) =
              mdata.at(id.x - i, id.y - j); // read corners
        }
      }
    }
    if (tid.x == bwidth - 1 && tid.y == 0) {
      for (int j = range; j > 0; --j) {
        for (int i = range; i > 0; --i) {
          sm(tid.x + i, tid.y - j) =
              mdata.at(id.x + i, id.y - j); // read corners
        }
      }
    }

    if (tid.x == 0 && tid.y == bheight - 1) {
      for (int j = range; j > 0; --j) {
        for (int i = range; i > 0; --i) {
          sm(tid.x - i, tid.y + j) =
              mdata.at(id.x - i, id.y + j); // read corners
        }
      }
    }

    if (tid.x == bwidth - 1 && tid.y == bheight - 1) {
      for (int j = range; j > 0; --j) {
        for (int i = range; i > 0; --i) {
          sm(tid.x + i, tid.y + j) =
              mdata.at(id.x + i, id.y + j); // read corners
        }
      }
    }

    block.synchronize();

    /*    for (int j = -range; j <= range; ++j) {
          for (int i = -range; i <= range; ++i) {
            // if (data.is_inbounds(id.x + i, id.y + j))
            func(sm(tid.x + i, tid.y + j), i, j);
          }
        }

        block.synchronize();*/
  }
};

template <typename T, typename I> struct pixel {
  I data;

  pixel(I &data) : data(data) {}

  pixel(T r, T g, T b, T a) {

    auto ptr = reinterpret_cast<T *>(&data);
    ptr[0] = r;
    ptr[1] = g;
    ptr[2] = b;
    ptr[3] = a;
  }

  auto &r() { return reinterpret_cast<T *>(&data)[0]; }
  auto &g() { return reinterpret_cast<T *>(&data)[1]; }
  auto &b() { return reinterpret_cast<T *>(&data)[2]; }
  auto &a() { return reinterpret_cast<T *>(&data)[3]; }
  I convert() { return data; }
  operator I() { return data; }
};

template <typename T> auto to_pixel(T &input) {
  using type = conditional_t<is_const<T>::value, const byte_t, byte_t>;
  return pixel<type, T>(input);
}

int main(int argc, char *argv[]) {
  if (argc < 3) {
    __error("missing input PNG and output file names");
    return -1;
  }
  const char *infile = argv[1];
  const char *outfile = argv[2];

  int range = 1;
  if (argc >= 4)
    range = stoi(argv[3]);

  vector<byte_t> image, output1, output2;
  unsigned width = 0, height = 0;

  if (auto error = lodepng::decode(image, width, height, infile)) {
    __error("image decoding failed: ", lodepng_error_text(error));
    return -2;
  }

  output1.resize(image.size());
  output2.resize(image.size());
  __message("input size: ", image.size(), " bytes");

  int fwhm = 5;
  int offset = (2 * range + 1) / 2;

  float a = (fwhm / 2.354);

  vector<float> blur(2 * range + 1);

  for (auto i = -offset; i <= ((2 * range + 1) - offset - 1); ++i)
    blur[i + offset] = std::exp(-i * i / (2 * a * a));

  /*  vector<float> g(9);
    g[0] = -1.0f;
    g[2] = 1.0f;
    g[3] = -2.0f;
    g[5] = 2.0f;
    g[6] = -1.0f;
    g[8] = 1.0f;*/

  auto gaussian = [](const auto &in, const auto &blur, auto range) {

    float alpha = 255;

    float sx = 0.0f;
    float sy = 0.0f;
    float sz = 0.0f;
    float sw = 0.0f;
    // unsigned int range = 1;

    for_each(in, [&](const auto &tuple) {
      pacxx::exp::apply([&](auto value, int i, int j) {

        auto p = to_pixel(value);
        sx += p.r() * blur[(i + j) / 2 + range];
        sy += p.g() * blur[(i + j) / 2 + range];
        sz += p.b() * blur[(i + j) / 2 + range];

        if (i == 0 && j == 0) {
          alpha = p.a();
        }
        sw += blur[(i + j) / 2 + range];
      }, tuple);
    });

    sx /= sw;
    sy /= sw;
    sz /= sw;

    auto to_char = [](float v, float high = 255.f, float low = 0.0f) -> byte_t {
      return (v > high) ? 255 : (v < low) ? 0 : static_cast<byte_t>(v);
    };
    pixel<byte_t, int> out{to_char(sx), to_char(sy), to_char(sz),
                           to_char(alpha)};
    return out.convert();

  };

  dim3 threads(32, 4, 1);
  dim3 blocks((width + threads.x - 1) / threads.x,
              (height + threads.y - 1) / threads.y, 1);
  size_t sm_size =
      (threads.x + 2 * range) * (threads.y + 2 * range) * sizeof(byte_t) * 4;

  pacxx::exp::pacxx_execution_policy pacxx{{blocks, threads, sm_size}};

  pacxx::exp::scalar_view<const vector<float> &> s_blur(blur);
  pacxx::exp::value_view<unsigned int> v_range(range);
  auto stencil = pacxx::exp::stencil<regular_stencil, unsigned int>(
        image, width, height, range);

  //auto stencil = pacxx::exp::stencil<sm_regular_stencil, unsigned int>(
  //    image, width, height, range);

  auto zrange = ranges::view::zip(stencil, s_blur, v_range);

  std::vector<unsigned int> out4(image.size() / 4);
  auto orange = pacxx::exp::matrix<1920>(out4);
  pacxx::exp::transform(pacxx, zrange, orange, [&](auto &&tuple) {
    return exp::apply(gaussian, tuple);
  });

  auto &exec = v2::Executor<v2::RuntimeT>::Create();

  auto &outM = exec.mm().translateVector(out4);
  outM.download(out4.data(), out4.size() * sizeof(unsigned int), 256);

  unsigned int *ptr = (unsigned int *)&output1[0];
  int i = 0;
  for (auto v : out4)
    ptr[i++] = v;

  printf("host 0x%08x\n", *(out4.data() + 50 * width + 50));

  if (auto error = lodepng::encode(outfile, output1, width, height)) {
    __error("image encoding failed: ", lodepng_error_text(error));
    return -2;
  }

  return 0;
}
