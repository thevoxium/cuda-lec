
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

int main(){
    printf(" N      NaiveTime(ms)  NaiveGFLOPs   NaiveGFLOPs/s   cuBLASTime(ms)  cuBLASGFLOPs   cuBLASGFLOPs/s\n");
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
        float naiveTimeMs = 0.0f;
        cudaEventElapsedTime(&naiveTimeMs, start, stop);

        float naiveGFLOPs = 2.0f * N * N * N / 1e9f;
        float naiveGFLOPsPerSec = naiveGFLOPs / (naiveTimeMs / 1000.0f);

        cublasHandle_t handle;
        cublasCreate(&handle);
        float alpha = 1.0f;
        float beta = 0.0f;

        cudaEventRecord(start);
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, db, N, da, N, &beta, dc_cublas, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float cublasTimeMs = 0.0f;
        cudaEventElapsedTime(&cublasTimeMs, start, stop);

        float cublasGFLOPs = 2.0f * N * N * N / 1e9f;
        float cublasGFLOPsPerSec = cublasGFLOPs / (cublasTimeMs / 1000.0f);

        printf("%4d     %10.3f     %10.2f     %13.2f     %13.3f     %12.2f     %15.2f\n",
               N, naiveTimeMs, naiveGFLOPs, naiveGFLOPsPerSec,
               cublasTimeMs, cublasGFLOPs, cublasGFLOPsPerSec);

        cublasDestroy(handle);
        free(a); free(b); free(c); free(c_cublas);
        cudaFree(da); cudaFree(db); cudaFree(dc); cudaFree(dc_cublas);
        cudaEventDestroy(start); cudaEventDestroy(stop);
    }
    return 0;
}
