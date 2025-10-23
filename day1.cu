#include <cuda_runtime.h>

__global__ void hello(){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  printf("Block Id: %d, Thread Id: %d", blockIdx, threadIdx);
}

int main(){
  int N = 1 << 10;
  dim3 threadsPerBlock(256, 1, 1);
  dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x);
  hello<<<blocksPerGrid, threadsPerBlock>>>();
  return 0;
}
