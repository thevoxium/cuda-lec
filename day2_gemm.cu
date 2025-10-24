#include <cuda_runtime.h>
#include <stdio.h>
#include <cublas_v2.h>
#include <stdlib.h>

#define N (1 << 8)
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

    gemmNaive<<<bpg, tpb>>>(da, db, dc);
    cudaDeviceSynchronize();

    cudaMemcpy(c, dc, size, cudaMemcpyDeviceToHost);

    for (int j = 0; j < N; j++) printf("%f ", c[j]); printf("\n");


    //cublas implementation

    cublasHandle_t handle;
    cublasCreate(&handle);

    float alpha = 1.0f;
    float beta = 0.0f;

    cublasSgemm(handle,
        CUBLAS_OP_T, CUBLAS_OP_T,  // Transpose inputs
        N,
        N,
        N,
        &alpha,
        db, N,
        da, N,
        &beta,
        dc_cublas, N);

    cudaMemcpy(c_cublas, dc_cublas, size, cudaMemcpyDeviceToHost);

    printf("First row (cuBLAS GEMM):\n");
    for (int j = 0; j < N; j++) printf("%f ", c_cublas[j]);
    printf("\n");

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
