#include <cuComplex.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void multiply_const_kernel(cuFloatComplex *in, cuFloatComplex *out,
                                      cuFloatComplex k, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    // e ix = cos x + i sin x
    out[i] = cuCmulf(in[i], k);
  }
}

__global__ void multiply_phase_kernel(cuFloatComplex *in, cuFloatComplex *out,
                                      float beta, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    // e ix = cos x + i sin x
    out[i] = cuCmulf( in[i], make_cuFloatComplex(cos(beta), sin(beta)));
  }
}

void exec_multiply_const(cuFloatComplex *in, cuFloatComplex *out,
                         cuFloatComplex k, int n, int grid_size, int block_size,
                         cudaStream_t stream) {
  multiply_const_kernel<<<grid_size, block_size, 0, stream>>>(in, out, k, n);
}

void exec_multiply_phase(cuFloatComplex *in, cuFloatComplex *out, float beta,
                         int n, int grid_size, int block_size,
                         cudaStream_t stream) {
  multiply_phase_kernel<<<grid_size, block_size, 0, stream>>>(in, out, beta, n);
}

void get_block_and_grid_multiply_const(int *minGrid, int *minBlock) {
  cudaOccupancyMaxPotentialBlockSize(minGrid, minBlock, multiply_const_kernel,
                                     0, 0);
}