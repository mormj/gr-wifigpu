/* -*- c++ -*- */
/*
 * Copyright 2021 gr-wifigpu author.
 *
 * This is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3, or (at your option)
 * any later version.
 *
 * This software is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this software; see the file COPYING.  If not, write to
 * the Free Software Foundation, Inc., 51 Franklin Street,
 * Boston, MA 02110-1301, USA.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "presync_impl.h"
#include <gnuradio/io_signature.h>

extern void exec_sync(cuFloatComplex *in, cuFloatComplex *out, float *cor,
                      int n, int grid_size, int block_size,
                      cudaStream_t stream);
extern void get_block_and_grid(int *minGrid, int *minBlock);

namespace gr {
namespace wifigpu {

presync::sptr presync::make() {
  return gnuradio::get_initial_sptr(new presync_impl());
}

/*
 * The private constructor
 */
presync_impl::presync_impl()
    : gr::block("presync", gr::io_signature::make(1, 1, sizeof(gr_complex)),
                gr::io_signature::make(3, 3, sizeof(gr_complex))) {
  set_history(64);

  get_block_and_grid(&d_min_grid_size, &d_block_size);
  cudaDeviceSynchronize();
  std::cerr << "minGrid: " << d_min_grid_size << ", blockSize: " << d_block_size
            << std::endl;

  cudaStreamCreate(&d_stream);

  // Temporary Buffers
  checkCudaErrors(cudaMalloc((void **)&d_dev_in, d_max_out_buffer));
  checkCudaErrors(cudaMalloc((void **)&d_dev_out, d_max_out_buffer));
  checkCudaErrors(cudaMalloc((void **)&d_dev_abs, d_max_out_buffer));
  checkCudaErrors(cudaMalloc((void **)&d_dev_cor, d_max_out_buffer));
}

/*
 * Our virtual destructor.
 */
presync_impl::~presync_impl() {}

void presync_impl::forecast(int noutput_items,
                            gr_vector_int &ninput_items_required) {
  ninput_items_required[0] = noutput_items;
}

int presync_impl::general_work(int noutput_items, gr_vector_int &ninput_items,
                               gr_vector_const_void_star &input_items,
                               gr_vector_void_star &output_items) {
  const gr_complex *in = (const gr_complex *)input_items[0];
  gr_complex *out = (gr_complex *)output_items[0];
  gr_complex *abs = (gr_complex *)output_items[1];
  float *cor = (float *)output_items[2];

  checkCudaErrors(cudaMemcpyAsync(d_dev_in, in + 48,
                                  sizeof(gr_complex) * noutput_items,
                                  cudaMemcpyHostToDevice, d_stream));                            

  exec_sync((cuFloatComplex *)d_dev_in, (cuFloatComplex *)d_dev_abs, d_dev_cor,
            noutput_items, d_min_grid_size, d_block_size, d_stream);


  checkCudaErrors(cudaMemcpyAsync(abs, d_dev_abs,
                                  sizeof(gr_complex) * noutput_items,
                                  cudaMemcpyDeviceToHost, d_stream));

  checkCudaErrors(cudaMemcpyAsync(cor, d_dev_cor,
                                  sizeof(float) * noutput_items,
                                  cudaMemcpyDeviceToHost, d_stream));

  memcpy(out, in+63, noutput_items*sizeof(gr_complex));

  // Tell runtime system how many input items we consumed on
  // each input stream.
  consume_each(noutput_items);

  // Tell runtime system how many output items we produced.
  return noutput_items;
}

} /* namespace wifigpu */
} /* namespace gr */
