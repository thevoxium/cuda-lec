#include <cuda_runtime.h>
#include <stdio.h>

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

  vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(da, db, dc, N);
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

