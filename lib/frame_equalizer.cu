#include <cuComplex.h>
#include <cuda.h>
#include <cuda_runtime.h>

__host__ __device__ double carg(const cuFloatComplex &z) {
  return atan2(cuCimagf(z), cuCrealf(z));
}
__host__ __device__ cuFloatComplex conj(const cuFloatComplex &z) {
  return make_cuFloatComplex(z.x, -z.y);
}

__global__ void calc_beta_err_kernel(cuFloatComplex *in, float *polarity,
                                     int current_symbol_index,
                                     cuFloatComplex *last_symbol, float bw,
                                     float freq, float *beta, float *err,
                                     int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < n) {
    cuFloatComplex *current_symbol = &in[i*64];
    cuFloatComplex pp0, pp1, pp2, pp3;

   
    if (i > 0) {
      pp0 = in[(i - 1)*64 + 11];
      pp1 = in[(i - 1)*64 + 25];
      pp2 = in[(i - 1)*64 + 39];
      pp3 = in[(i - 1)*64 + 53];
    }
    else {
      pp0 = last_symbol[11];
      pp1 = last_symbol[25];
      pp2 = last_symbol[39];
      pp3 = last_symbol[53];
    }

    float p = polarity[(current_symbol_index + i - 2) % 127];

		if(current_symbol_index+i <= 2) {
      pp1 = cuCmulf(pp1, make_cuFloatComplex(-1.0,0.0));
		} else {
      float last_p = polarity[(current_symbol_index + i - 2 - 1) % 127];
			pp0 = cuCmulf(pp0, make_cuFloatComplex(last_p,0.0));
			pp1 = cuCmulf(pp1, make_cuFloatComplex(last_p,0.0));
			pp2 = cuCmulf(pp2, make_cuFloatComplex(last_p,0.0));
			pp3 = cuCmulf(pp3, make_cuFloatComplex(-last_p,0.0));
		}

    if ((current_symbol_index + i) < 2) {
      beta[i] = carg(
          make_cuFloatComplex(current_symbol[11].x - current_symbol[25].x +
                                  current_symbol[39].x + current_symbol[53].x,
                              current_symbol[11].y - current_symbol[25].y +
                                  current_symbol[39].y + current_symbol[53].y));
    } else {
      beta[i] = carg(make_cuFloatComplex(
          (current_symbol[11].x * p) + (current_symbol[39].x * p) +
              (current_symbol[25].x * p) + (current_symbol[53].x * -p),
          (current_symbol[11].y * p) + (current_symbol[39].y * p) +
              (current_symbol[25].y * p) + (current_symbol[53].y * -p)));

    }

    err[i] = carg(cuCaddf(
        cuCaddf((make_cuFloatComplex(
                    cuCmulf(pp0, current_symbol[11]).x * p,
                    -cuCmulf(pp0, current_symbol[11]).y * p)),
                (make_cuFloatComplex(
                    cuCmulf(pp1, current_symbol[25]).x * p,
                    -cuCmulf(pp1, current_symbol[25]).y * p))),
        cuCaddf((make_cuFloatComplex(
                    cuCmulf(pp2, current_symbol[39]).x * p,
                    -cuCmulf(pp2, current_symbol[39]).y * p)),
                (make_cuFloatComplex(
                    cuCmulf(pp3, current_symbol[53]).x * -p,
                    -cuCmulf(pp3, current_symbol[53]).y * -p)))));

    err[i] *= (bw / (2 * M_PI * freq * 80));
  }
}

void exec_calc_beta_err(cuFloatComplex *in, float *polarity,
                        int current_symbol_index, cuFloatComplex *last_symbol,
                        float bw, float freq, float *beta, float *err, int n,
                        int grid_size, int block_size, cudaStream_t stream) {
  calc_beta_err_kernel<<<grid_size, block_size, 0, stream>>>(
      in, polarity, current_symbol_index, last_symbol, bw, freq, beta, err, n);
}

void get_block_and_grid_calc_beta_err(int *minGrid, int *minBlock) {
  cudaOccupancyMaxPotentialBlockSize(minGrid, minBlock, calc_beta_err_kernel, 0,
                                     0);
}