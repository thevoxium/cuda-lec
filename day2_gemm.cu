
#include <cuda_runtime.h>
#include <stdio.h>
#include <cublas_v2.h>
#include <stdlib.h>

#define __threadCount 16

__global__ void gemmNaive(float* da, float* db, float* dc, int N){
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    if(i < N && j < N){
        float sum = 0.0f;
        for (int k=0; k < N; ++k){
            sum += da[i * N + k] * db[k * N + j];
        }
        dc[i * N + j] = sum;
    }
}

float timeKernel(void(*kernel)(float*, float*, float*, int), float* da, float* db, float* dc, int N, dim3 bpg, dim3 tpb){
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    kernel<<<bpg, tpb>>>(da, db, dc, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms;
}

int main(){
    printf(" N      NaiveTime(ms)  NaiveGFLOPs  cuBLASTime(ms)  cuBLASGFLOPs\n");
    for(int N = 512; N <= 8192; N += 512){
        size_t size = N * N * sizeof(float);
        float* a = (float*)malloc(size);
        float* b = (float*)malloc(size);
        float* c = (float*)malloc(size);
        float* c_cublas = (float*)malloc(size);

        for(int i=0;i<N;i++)
            for(int j=0;j<N;j++){
                a[i*N+j] = i+j;
                b[i*N+j] = 2*(i+j);
            }

        float *da, *db, *dc, *dc_cublas;
        cudaMalloc((void**)&da, size);
        cudaMalloc((void**)&db, size);
        cudaMalloc((void**)&dc, size);
        cudaMalloc((void**)&dc_cublas, size);

        cudaMemcpy(da, a, size, cudaMemcpyHostToDevice);
        cudaMemcpy(db, b, size, cudaMemcpyHostToDevice);

        dim3 tpb(__threadCount, __threadCount, 1);
        dim3 bpg((N+__threadCount-1)/__threadCount, (N+__threadCount-1)/__threadCount, 1);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        gemmNaive<<<bpg, tpb>>>(da, db, dc, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float naiveTime = 0.0f;
        cudaEventElapsedTime(&naiveTime, start, stop);

        float gflops_naive = 2.0f * N * N * N / (naiveTime * 1e6f);

        cublasHandle_t handle;
        cublasCreate(&handle);
        float alpha = 1.0f;
        float beta = 0.0f;

        cudaEventRecord(start);
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, db, N, da, N, &beta, dc_cublas, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float cublasTime = 0.0f;
        cudaEventElapsedTime(&cublasTime, start, stop);

        float gflops_cublas = 2.0f * N * N * N / (cublasTime * 1e6f);

        printf("%4d     %10.3f     %10.2f     %10.3f     %10.2f\n", N, naiveTime, gflops_naive, cublasTime, gflops_cublas);

        cublasDestroy(handle);
        free(a); free(b); free(c); free(c_cublas);
        cudaFree(da); cudaFree(db); cudaFree(dc); cudaFree(dc_cublas);
        cudaEventDestroy(start); cudaEventDestroy(stop);
    }
    return 0;
}
