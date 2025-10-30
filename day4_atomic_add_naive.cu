#include <cuda_runtime.h>
#include <stdio.h> 

#define TIME_KERNEL(...) do { \
    cudaEvent_t start, stop; \
    cudaEventCreate(&start); \
    cudaEventCreate(&stop); \
    cudaEventRecord(start); \
    __VA_ARGS__; \
    cudaEventRecord(stop); \
    cudaEventSynchronize(stop); \
    float ms = 0.0f; \
    cudaEventElapsedTime(&ms, start, stop); \
    printf("Time taken by %s: %.3f ms\n", #__VA_ARGS__, ms); \
    cudaEventDestroy(start); \
    cudaEventDestroy(stop); \
} while(0)



__global__ void atomicAddNaive(const double* d_a, double* d_out, int N){
  int idx = blockDim.x * blockIdx.x  + threadIdx.x;
  if (idx < N){
    atomicAdd(d_out, d_a[idx]);
  }
}


int main(){
  int N = 1 << 20;
  int size = N * sizeof(double);
  double* a = (double*) malloc(size);
  double* out = (double*) malloc(sizeof(double));

  *out = 0.0;
  for(int i=0; i < N; i++) a[i] = i;

  double* d_out;
  double* d_a;
  cudaMalloc((void**)&d_a, size);
  cudaMalloc((void**)&d_out, sizeof(double));
  cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_out, out, sizeof(double), cudaMemcpyHostToDevice);

  dim3 threadsPerBlock(256);
  dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x);
  TIME_KERNEL(atomicAddNaive<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_out, N));
  cudaDeviceSynchronize();

  cudaMemcpy(out, d_out, sizeof(double), cudaMemcpyDeviceToHost);

  printf("Result: %lf\n", *out);

  free(a);
  free(out);
  cudaFree(d_a);
  cudaFree(d_out);

  return 0;
}
