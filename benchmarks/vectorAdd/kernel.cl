__kernel void MatrixMultKernel(__global float* Md, __global float* Nd, __global float* Pd, int width) { 
  int idx = get_global_id(0);
  if(idx < width)
    Pd[idx] = Md[idx] + Nd[idx];
}

