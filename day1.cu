
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void hello() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < 10) {  
        printf("Block Id: %d, Thread Id: %d, Global Idx: %d\n",
               blockIdx.x, threadIdx.x, idx);
    }
}

int main() {
    int N = 1 << 10;
    dim3 threadsPerBlock(256, 1, 1);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x);
  
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
  
    cudaEventRecord(start);
    hello<<<blocksPerGrid, threadsPerBlock>>>();
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
  
    float elapsed;
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("Kernel execution time: %f ms\n", elapsed);
  
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
