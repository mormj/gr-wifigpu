#include <cuComplex.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void threshold_kernel(float *cor, float *plateau, float threshold,
                                 int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    plateau = cor[i] > threshold;
  }
}

__global__ void plateau_kernel(float *plateau, int *accum, int min_plateau,
                               int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    int a = 0;
    for (j = 0; j < min_plateau; j++) {
      if (plateau[i + j]) {
        a++;
      } else {
        a = 0;
      }
    }
  }
}

void exec_threshold(float *cor, float *plateau, float threshold, int n,
                    int grid_size, int block_size, cudaStream_t stream) {
  threshold_kernel<<<grid_size, block_size, 0, stream>>>(cor, plateau,
                                                         threshold, n);
}

void exec_plateau(float *plateau, int *accum, int min_plateau, int n,
                  int grid_size, int block_size, cudaStream_t stream) {
  plateau_kernel<<<grid_size, block_size, 0, stream>>>(plateau, accum, min_plateau, n);
}

void get_block_and_grid_plateau(int *minGrid, int *minBlock) {
  cudaOccupancyMaxPotentialBlockSize(minGrid, minBlock, plateau_kernel, 0, 0);
}