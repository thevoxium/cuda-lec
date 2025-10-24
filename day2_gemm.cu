#include <cuda_runtime.h>
#include <stdio.h>
#include <cublas_v2.h>
#include <stdlib.h>

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

#define N (1 << 13)
#define __threadCount 16

__global__ void gemmNaive(float* da, float* db, float* dc){
  int j = blockDim.x * blockIdx.x + threadIdx.x;
  int i = blockDim.y * blockIdx.y + threadIdx.y;
  if(i < N && j < N){
    float sum = 0.0f;
    for (int k=0; k < N; ++k){
      sum += (da[i * N + k] * db[k * N + j]);
    }
    dc[i * N + j] = sum;
  }
}

int main(){
    size_t size = N * N * sizeof(float);
    float* a = (float*)malloc(size);
    float* b = (float*)malloc(size);
    float* c = (float*)malloc(size);
    float* c_cublas = (float*)malloc(size);

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            a[i * N + j] = i + j;
            b[i * N + j] = 2 * (i + j);
        }
    }

    float* da;
    float* db;
    float* dc;
    float* dc_cublas;

    cudaMalloc((void**)&da, size);
    cudaMalloc((void**)&db, size);
    cudaMalloc((void**)&dc, size);
    cudaMalloc((void**)&dc_cublas, size);

    cudaMemcpy(da, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(db, b, size, cudaMemcpyHostToDevice);

    dim3 tpb(__threadCount, __threadCount, 1);
    dim3 bpg((N+__threadCount-1)/__threadCount, (N+__threadCount-1)/__threadCount, 1);

    TIME_KERNEL(gemmNaive<<<bpg, tpb>>>(da, db, dc));
    cudaDeviceSynchronize();

    cudaMemcpy(c, dc, size, cudaMemcpyDeviceToHost);

    //cublas implementation

    cublasHandle_t handle;
    cublasCreate(&handle);

    float alpha = 1.0f;
    float beta = 0.0f;

    TIME_KERNEL(cublasSgemm(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N,
        N,
        N,
        &alpha,
        db, N,
        da, N,
        &beta,
        dc_cublas, N));

    cudaMemcpy(c_cublas, dc_cublas, size, cudaMemcpyDeviceToHost);

    cublasDestroy(handle);

    free(a);
    free(b);
    free(c);
    free(c_cublas);
    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);
    cudaFree(dc_cublas);

    return 0;
}
