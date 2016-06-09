#include <vector>
#include "algorithm.h"
#include "iterator.h"
#include "execution_policy.h"
#include "views.h"
#include <iomanip>
#include <sstream>
#include "range/v3/all.hpp"
#include "PACXX.h"

using namespace std;
using namespace pacxx;
using namespace pacxx::exp;

#define THREADS 128
#define __sq(x) ((x) * (x))
#define __cu(x) ((x) * (x) * (x))

struct Point4 {
  using data_t = typename v2::vec<float, 4>::type;
  Point4() : data(data_t{0.0f, 0.0f, 0.0f, 0.0f}) {}

  Point4(const Point4 &rhs) : data(rhs.data) {}

  auto &operator=(const Point4 &rhs) {
    data.x = rhs.data.x;
    data.y = rhs.data.y;
    data.z = rhs.data.z;
    data.w = rhs.data.w;
    return *this;
  }

  auto &operator+(const Point4 &rhs) {
    data.x += rhs.data.x;
    data.y += rhs.data.y;
    data.z += rhs.data.z;
    return *this;
  }

  auto &operator-(const Point4 &rhs) {
    data.x -= rhs.data.x;
    data.y -= rhs.data.y;
    data.z -= rhs.data.z;
    return *this;
  }

  auto &operator*(float rhs) {
    data.x *= rhs;
    data.y *= rhs;
    data.z *= rhs;
    return *this;
  }

  data_t data;
};

using data_t = Point4;

int main(int argc, char *argv[]) {
  int runs = 1;
  int devid = 0;
  int particleCount = 1000;
  if (argc >= 2)
    particleCount = stoi(argv[1]);
  if (argc >= 3)
    devid = stoi(argv[2]);
  if (argc >= 4)
    runs = stoi(argv[3]);

  vector<data_t> position(particleCount), pos2(particleCount),
      velocity(particleCount);

  auto init = [](auto &pos) {
    mt19937 rng;

    uniform_real_distribution<float> rnd_pos(-1e11, 1e11);
    uniform_real_distribution<float> rnd_mass(1e22, 1e24);

    rng.seed(13122012);

    for (size_t i = 0; i != pos.size(); ++i) {
      pos[i].data.x = rnd_pos(rng);
      pos[i].data.y = rnd_pos(rng);
      pos[i].data.z = rnd_pos(rng);
      pos[i].data.w = rnd_mass(rng);
    }
  };

  init(position);

  pacxx_execution_policy pacxx{{{(position.size() + THREADS - 1) / THREADS},
                                {THREADS},
                                THREADS * sizeof(data_t)}};

  constexpr auto G = -6.673e-11f;
  constexpr auto dt = 3600.f;
  constexpr auto eps2 = 0.00125f;

  auto nbody = [=](auto p, auto &v, auto &np, const auto &particles) {
    data_t a, r;

    exp::for_each(particles, [&](auto particle) {
      r = p - particle;
      r.data.w = native::nvvm::rsqrt(__sq(r.data.x) + __sq(r.data.y) +
                                     __sq(r.data.z) + eps2);

      a.data.w = G * particle.data.w * __cu(r.data.w);

      a = a + r * a.data.w;
    });
    np = p + v * dt + a * 0.5f * __sq(dt);
    v = v + a * dt;
  };

  scalar_view<const vector<data_t> &> particle_view(position);
  auto range = ranges::view::zip(position, velocity, pos2, particle_view);

  for (int i = 0; i < runs; ++i)
    for_each(pacxx, range, [&](auto &&tuple) { exp::apply(nbody, tuple); });

  //  cira::sync();

  stringstream ss;

  for (auto f : pos2)
    ss << std::fixed << std::setw(11) << std::setprecision(6) << f.data.x << " "
       << f.data.y << " " << f.data.z << " " << f.data.w << "\n";

  common::write_string_to_file("pacxx.out", ss.str());

  return 0;
}
