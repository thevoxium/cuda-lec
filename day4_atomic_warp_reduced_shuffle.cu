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



#define threads_per_block 256


__global__
void atomicAddReduced(const double* d_a, double* d_out, int N){
  __shared__ double S[threads_per_block];
  int i = blockDim.x * blockIdx.x * 2 + threadIdx.x;
  int tid = threadIdx.x;

  double sum = 0.0;
  if (i < N) sum += d_a[i];
  if (i + blockDim.x < N) sum += d_a[i + blockDim.x];
  S[tid] = sum;
  __syncthreads();


  for (int s = blockDim.x/2; s > 32; s>>=1){
    if (tid < s) S[tid] += S[tid + s];
    __syncthreads();
  }

  if (tid < 32){
    double val = S[tid] + S[tid + 32];
    for (int offset = 16; offset > 0; offset>>=1){
      val += __shfl_down_sync(0xffffffff, val, offset);
    }
    S[tid] = val;
  }

  if (tid == 0){
    atomicAdd(d_out, S[tid]);
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

  dim3 blocksPerGrid((N + 2*threads_per_block - 1) / (2*threads_per_block));
  TIME_KERNEL(atomicAddReduced<<<blocksPerGrid, threads_per_block>>>(d_a, d_out, N));
  cudaDeviceSynchronize();

  cudaMemcpy(out, d_out, sizeof(double), cudaMemcpyDeviceToHost);

  printf("Result: %lf\n", *out);

  free(a);
  free(out);
  cudaFree(d_a);
  cudaFree(d_out);

  return 0;
}
