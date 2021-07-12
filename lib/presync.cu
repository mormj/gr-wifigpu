#include <cuComplex.h>
#include <cuda.h>
#include <cuda_runtime.h>


__global__ void
sync_kernel(cuFloatComplex* in, cuFloatComplex* out, float* cor, int n)
{
    int d = 16;
    int w = 48;
    int w2 = 64;

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        cuFloatComplex m = cuCmulf(cuConjf(in[i + d]),in[i]);

        float cplx_mag = in[i].x*in[i].x + in[i].y*in[i].y;
        cor[i] = cplx_mag;

        out[i] = m;
        __syncthreads();

        cuFloatComplex sum = make_cuFloatComplex(0,0);
        for (int j=0; j<w; j++)
        {   
            sum.x += out[i+j].x;
            sum.y += out[i+j].y;
        }
        
        float fsum = 0;
        for (int j=0; j<w2; j++)
        {
            fsum += cor[i+j];
        }

        __syncthreads();
        out[i] = sum;
        cor[i] = sqrt(sum.x*sum.x + sum.y*sum.y) / fsum;

    }
}

void exec_sync(cuFloatComplex* in,
                cuFloatComplex* out,
                float* cor,
                int n,
                int grid_size,
                int block_size,
                cudaStream_t stream)
{
    sync_kernel<<<grid_size, block_size, 0, stream>>>(in, out, cor, n);
}

void get_block_and_grid(int* minGrid, int* minBlock)
{
    cudaOccupancyMaxPotentialBlockSize(minGrid, minBlock, sync_kernel, 0, 0);
}