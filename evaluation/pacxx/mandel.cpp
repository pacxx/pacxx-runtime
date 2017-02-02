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

const int bailout = 1000;
const int width = 1024;
const int height = 1024;

typedef unsigned int u32;
typedef unsigned char u8;

// Translate (center + zoom) to (upper left + lower right)
void GetTranslatedCoordinates(float* cr1, float* cr2, float* ci1, float* ci2, float center_r, float center_i, float zoom) {
  *cr1 = center_r - zoom;
  *cr2 = center_r + zoom;
  float aspect_ratio = (float)width / (float)height;
  *ci1 = center_i - (zoom / aspect_ratio);
  *ci2 = center_i + (zoom / aspect_ratio);
}


void writePPM(int* mandel_bailouts) {
  ofstream ofs("mandelbrot.ppm", ios::binary);
  ofs << "P6" << "\n" << width << " " << height << " " << 255 << "\n";
  for (u32 y = 0; y < height; ++y) {
    for (u32 x = 0; x < width; ++x) {
      int v = mandel_bailouts[x + (y * width)];
      if (v > 255) {
        v = 255;
      }
      u8 vb = static_cast<u8>(v);
      ofs.write((char*)&vb, sizeof(u8));
      ofs.write((char*)&vb, sizeof(u8));
      ofs.write((char*)&vb, sizeof(u8));
    }
  }
}

unsigned div_up(unsigned a, unsigned b) {
  return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

int main(int argc, char **argv) {

  size_t buffer_size = sizeof(int) * width * height * 3;

  std::vector<int> c(buffer_size);

  float cr1, cr2, ci1, ci2;
  float center_r = -0.5f, center_i = 0.0f;
  float zoom = 1.5f;
  GetTranslatedCoordinates(&cr1, &cr2, &ci1, &ci2, center_r, center_i, zoom);


  auto test = [](int *out, float cr1, float cr2, float ci1, float ci2) {
    u32 x(get_group_id(0) * get_local_size(0) + get_local_id(0));
    u32 y(get_group_id(1) * get_local_size(1) + get_local_id(1));

    if (x >= width || y >= height) {
      return;
    }

    float cr = (x / (float)width) * (cr2 - cr1) + cr1;
    float ci = (y / (float)height) * (ci2 - ci1) + ci1;

    float zi = 0.0f, zr = 0.0f, zr2 = 0.0f, zi2 = 0.0f, zit;
    u32 iter = bailout;
    while(--iter && zr2 + zi2 < 4.0f) {
      zit = zr * zi;
      zi = zit + zit + ci;
      zr = (zr2 - zi2) + cr;
      zr2 = zr * zr;
      zi2 = zi * zi;
    }

    if (iter) {
      iter = bailout - iter;
    }
    out[x + y * width] = iter * 5.0f;
  };

  auto& exec = get_executor<NativeRuntime>();

  auto& dev_c = exec.allocate<int>(buffer_size, c.data());

  auto mandel =
      kernel<NativeRuntime>(test, {{div_up(width, 512), div_up(height, 1024)}, {512, 1024}});

  mandel(dev_c.get(), cr1, cr2, ci1, ci2);

  exec.synchronize();

  writePPM(c.data());

  return 0;
}
