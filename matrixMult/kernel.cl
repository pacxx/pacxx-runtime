__kernel void MatrixMultKernel(__global int* Md, __global int* Nd, __global int* Pd, int width) {
  int col = get_global_id(0);
  int row = get_global_id(1);
  int sum = 0;
  for (int k = 0; k < width; k+=1)
    sum += Md[row * width + k] * Nd[k * width + col];
  Pd[row * width + col] = sum;
}

