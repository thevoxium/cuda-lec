#include <cuda_runtime.h>
#define TILE 8 

__global__ void matrix_transpose_kernel(const float* input, float* output, int N, int M) {
    __shared__ float sm[TILE][TILE+1];
    int i = blockIdx.y * TILE + threadIdx.y;
    int j = blockIdx.x * TILE + threadIdx.x;

    int ty = threadIdx.y;
    int tx = threadIdx.x;

    if (i < N && j < M){
        sm[ty][tx] = input[i * M + j];
    }

    __syncthreads();

    int out_i = blockIdx.x * TILE + threadIdx.y;
    int out_j = blockIdx.y * TILE + threadIdx.x;

    if (out_i< M && out_j < N){
        output[out_i * N + out_j] = sm[tx][ty];
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int rows, int cols) {
    dim3 threadsPerBlock(TILE, TILE);
    dim3 blocksPerGrid((cols + TILE - 1) / TILE,
                       (rows + TILE - 1) / TILE);

    matrix_transpose_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
}
