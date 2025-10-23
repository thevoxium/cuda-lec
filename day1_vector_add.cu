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



__global__ void vectorAdd(float* a, float* b, float* c, int N){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N){
    c[idx] = b[idx] + a[idx];
  }
}

int main(){
  int N = 1 << 20;
  int size = N * sizeof(float);

  float* a = (float*) malloc(size);
  float* b = (float*) malloc(size);
  float* c = (float*) malloc(size);

  for(int i=0; i < N; i++){
    a[i] = i;
  }

  for(int i=0; i < N; i++){
    b[i] = 2*i;
  }
 
  float* da;
  float* db;
  float* dc;

  cudaMalloc((void**)&da, size);
  cudaMalloc((void**)&db, size);
  cudaMalloc((void**)&dc, size);

  cudaMemcpy(da, a, size, cudaMemcpyHostToDevice);
  cudaMemcpy(db, b, size, cudaMemcpyHostToDevice);


  dim3 threadsPerBlock(256, 1, 1);
  dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x);

  TIME_KERNEL(vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(da, db, dc, N));
  cudaDeviceSynchronize();

  cudaMemcpy(c, dc, size, cudaMemcpyDeviceToHost);

  for(int i=0; i < 5; i++){
    printf("%f, ", c[i]);
  }

  cudaFree(da);
  cudaFree(db);
  cudaFree(dc);

  free(a);
  free(b);
  free(c);

  return 0;
}

