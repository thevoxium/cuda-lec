
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void hello(){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < 10) { 
        printf("Block Id: %d, Thread Id: %d, Global Idx: %d\n",
               blockIdx.x, threadIdx.x, idx);
    }
}

int main(){
  int N = 1 << 10;
  dim3 threadsPerBlock(256, 1, 1);
  dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x);
  hello<<<blocksPerGrid, threadsPerBlock>>>();
  cudaDeviceSynchronize();
  return 0;
}
