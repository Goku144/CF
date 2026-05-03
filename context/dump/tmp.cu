#include <iostream>

__global__ void vecAdd(float *a, float *b, float *c)
{
  int index = blockIdx.x * blockDim.x + blockIdx.x;

  c[index] = a[index] + b[index];
}


int main(void)
{
  float *a,*b,*c;
  vecAdd<<<4, 256>>>(a,b,c);
  return 0;
}