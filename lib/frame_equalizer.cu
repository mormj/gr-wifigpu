#include <cuComplex.h>
#include <cuda.h>
#include <cuda_runtime.h>

__host__ __device__ double carg(const cuFloatComplex& z) {return atan2(cuCimagf(z),   cuCrealf(z));}
__host__ __device__ cuFloatComplex conj(const cuFloatComplex& z) {return make_cuFloatComplex(z.x, -z.y);}

__global__ void calc_beta_err_kernel(cuFloatComplex *current_symbol, int8_t polarity,
                                     int current_symbol_index,
                                     cuFloatComplex *prev_pilots, float *beta,
                                     float *err, float bw, float freq,
                                     float *d_err, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  float p = (float)polarity;
  if (i < n) {

    if (current_symbol_index < 2) {
    //   *beta = carg(current_symbol[11] - current_symbol[25] + current_symbol[39] +
    //              current_symbol[53]);

    *beta = carg( make_cuFloatComplex( current_symbol[11].x - current_symbol[25].x + current_symbol[39].x +
        current_symbol[53].x,
        current_symbol[11].y - current_symbol[25].y + current_symbol[39].y +
        current_symbol[53].y    
    ));



    } else {
    //   *beta = carg((current_symbol[11] * p) + (current_symbol[39] * p) +
    //              (current_symbol[25] * p) + (current_symbol[53] * -p));
      *beta = carg(make_cuFloatComplex((current_symbol[11].x * p) + (current_symbol[39].x * p) +
                 (current_symbol[25].x * p) + (current_symbol[53].x * -p),
                 (current_symbol[11].y * p) + (current_symbol[39].y * p) +
                 (current_symbol[25].y * p) + (current_symbol[53].y * -p)
                ));
    }

    // *err = carg(    (conj(prev_pilots[0]) * current_symbol[11] * p) +
    //                 (conj(prev_pilots[1]) * current_symbol[25] * p) +
    //                 (conj(prev_pilots[2]) * current_symbol[39] * p) +
    //                 (conj(prev_pilots[3]) * current_symbol[53] * -p));

    *err = carg(    cuCaddf(cuCaddf((make_cuFloatComplex( cuCmulf(prev_pilots[0],current_symbol[11]).x * p, -cuCmulf(prev_pilots[0],current_symbol[11]).y * p    )), 
                    (make_cuFloatComplex( cuCmulf(prev_pilots[1],current_symbol[25]).x * p, -cuCmulf(prev_pilots[1],current_symbol[25]).y * p    ))),
                    cuCaddf((make_cuFloatComplex( cuCmulf(prev_pilots[2],current_symbol[39]).x * p, -cuCmulf(prev_pilots[2],current_symbol[39]).y * p    )),
                    (make_cuFloatComplex( cuCmulf(prev_pilots[3],current_symbol[53]).x * -p, -cuCmulf(prev_pilots[3],current_symbol[53]).y * -p   )))  ) );

    *err *= bw / (2 * M_PI * freq * 80);

    if (current_symbol_index < 2) {
        prev_pilots[0] = current_symbol[11];
        prev_pilots[1] = make_cuFloatComplex(-current_symbol[25].x, -current_symbol[25].y);
        prev_pilots[2] = current_symbol[39];
        prev_pilots[3] = current_symbol[53];
    } else {
        prev_pilots[0] = make_cuFloatComplex(current_symbol[11].x * p, current_symbol[11].y * p);
        prev_pilots[1] = make_cuFloatComplex(current_symbol[25].x * p, current_symbol[25].y * p);
        prev_pilots[2] = make_cuFloatComplex(current_symbol[39].x * p, current_symbol[39].y * p);
        prev_pilots[3] = make_cuFloatComplex(current_symbol[53].x * -p,current_symbol[53].y * -p);
    }
  }
}

void exec_calc_beta_err(cuFloatComplex *in, int8_t polarity,
                        int current_symbol_index, cuFloatComplex *prev_pilots,
                        float *beta, float *err, float bw, float freq,
                        float *d_err, int n, int grid_size, int block_size,
                        cudaStream_t stream) {
  calc_beta_err_kernel<<<grid_size, block_size, 0, stream>>>(
      in, polarity, current_symbol_index, prev_pilots, beta, err, bw, freq,
      d_err, n);
}

void get_block_and_grid_calc_beta_err(int *minGrid, int *minBlock) {
  cudaOccupancyMaxPotentialBlockSize(minGrid, minBlock, calc_beta_err_kernel, 0,
                                     0);
}