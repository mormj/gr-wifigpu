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

#include <cuda_buffer/cuda_buffer.h>

#include <pmt/pmt.h>

void exec_mov_avg(cuFloatComplex *in, float *mag, cuFloatComplex *out,
                  float *cor, int n, int grid_size, int block_size,
                  cudaStream_t stream);

void exec_corr_abs(cuFloatComplex *in, cuFloatComplex *out, float *mag, int n,
                   int grid_size, int block_size, cudaStream_t stream);

extern void get_block_and_grid(int *minGrid, int *minBlock);

namespace gr {
namespace wifigpu {

presync::sptr presync::make() {
  return gnuradio::get_initial_sptr(new presync_impl());
}

void presync_impl::forecast(int noutput_items,
                            gr_vector_int &ninput_items_required) {
  ninput_items_required[0] = noutput_items + 63;
}

#if 1 // CUDA Custom Buffers
presync_impl::presync_impl()
    : gr::block(
          "presync",
          gr::io_signature::make(1, 1, sizeof(gr_complex), cuda_buffer::type),
          gr::io_signature::make3(3, 3, sizeof(gr_complex), sizeof(gr_complex),
                                  sizeof(float), cuda_buffer::type,
                                  cuda_buffer::type, cuda_buffer::type)) {
  set_history(64);

  get_block_and_grid(&d_min_grid_size, &d_block_size);
  cudaDeviceSynchronize();
  std::cerr << "minGrid: " << d_min_grid_size << ", blockSize: " << d_block_size
            << std::endl;

  cudaStreamCreate(&d_stream);

  // Temporary Buffers
  checkCudaErrors(cudaMalloc(
      (void **)&d_dev_in, d_max_out_buffer + (64 + 16) * sizeof(gr_complex)));
  checkCudaErrors(cudaMalloc(
      (void **)&d_dev_out, d_max_out_buffer + (64 + 16) * sizeof(gr_complex)));
  checkCudaErrors(cudaMalloc(
      (void **)&d_dev_abs, d_max_out_buffer + (64 + 16) * sizeof(gr_complex)));
  checkCudaErrors(cudaMalloc((void **)&d_dev_cor,
                             d_max_out_buffer + (64 + 16) * sizeof(float)));
  checkCudaErrors(cudaMalloc(
      (void **)&d_dev_abs2, d_max_out_buffer + (64 + 16) * sizeof(gr_complex)));
  checkCudaErrors(cudaMalloc((void **)&d_dev_cor2,
                             d_max_out_buffer + (64 + 16) * sizeof(float)));

  checkCudaErrors(cudaMemset(
      d_dev_in, 0, d_max_out_buffer + (64 + 16) * sizeof(gr_complex)));
  checkCudaErrors(cudaMemset(
      d_dev_out, 0, d_max_out_buffer + (64 + 16) * sizeof(gr_complex)));
  checkCudaErrors(cudaMemset(
      d_dev_abs, 0, d_max_out_buffer + (64 + 16) * sizeof(gr_complex)));
  checkCudaErrors(
      cudaMemset(d_dev_cor, 0, d_max_out_buffer + (64 + 16) * sizeof(float)));
  checkCudaErrors(cudaMemset(
      d_dev_abs2, 0, d_max_out_buffer + (64 + 16) * sizeof(gr_complex)));
  checkCudaErrors(
      cudaMemset(d_dev_cor2, 0, d_max_out_buffer + (64 + 16) * sizeof(float)));

  set_output_multiple(1024 * 1024 / 8);
}

int presync_impl::general_work(int noutput_items, gr_vector_int &ninput_items,
                               gr_vector_const_void_star &input_items,
                               gr_vector_void_star &output_items) {
  const gr_complex *in = (const gr_complex *)input_items[0];
  gr_complex *out = (gr_complex *)output_items[0];
  gr_complex *abs = (gr_complex *)output_items[1];
  float *cor = (float *)output_items[2];

  //   std::cout << noutput_items << std::endl;

  noutput_items = std::min(ninput_items[0], noutput_items);

  //   checkCudaErrors(cudaMemcpyAsync(d_dev_in, in,
  //                                   sizeof(gr_complex) * (noutput_items +
  //                                   63), cudaMemcpyHostToDevice, d_stream));

  auto gridSize = (noutput_items + d_block_size - 1) / d_block_size;
  exec_corr_abs((cuFloatComplex *)in, (cuFloatComplex *)d_dev_abs, d_dev_cor,
                noutput_items, gridSize, d_block_size, d_stream);
  checkCudaErrors(cudaPeekAtLastError());
  exec_mov_avg((cuFloatComplex *)d_dev_abs, d_dev_cor, (cuFloatComplex *)abs,
               cor, noutput_items, gridSize, d_block_size, d_stream);
  checkCudaErrors(cudaPeekAtLastError());

  // std::cout << "presync: " << noutput_items << " / " << ninput_items[0] <<
  // std::endl;
  checkCudaErrors(cudaMemcpyAsync(out, in+47,
                                  sizeof(gr_complex) * noutput_items,
                                  cudaMemcpyDeviceToDevice, d_stream));

  cudaStreamSynchronize(d_stream);

  // add_item_tag(1, nitems_written(0), pmt::mp("frame"), pmt::from_long(0));

  // Tell runtime system how many input items we consumed on
  // each input stream.
  consume_each(noutput_items);

  // Tell runtime system how many output items we produced.
  return noutput_items;
}
#else

presync_impl::presync_impl()
    : gr::block("presync", gr::io_signature::make(1, 1, sizeof(gr_complex)),
                gr::io_signature::make3(3, 3, sizeof(gr_complex),
                                        sizeof(gr_complex), sizeof(float))) {
  set_history(64);

  get_block_and_grid(&d_min_grid_size, &d_block_size);
  cudaDeviceSynchronize();
  std::cerr << "minGrid: " << d_min_grid_size << ", blockSize: " << d_block_size
            << std::endl;

  cudaStreamCreate(&d_stream);

  // Temporary Buffers
  checkCudaErrors(cudaMalloc(
      (void **)&d_dev_in, d_max_out_buffer + (64 + 16) * sizeof(gr_complex)));
  checkCudaErrors(cudaMalloc(
      (void **)&d_dev_out, d_max_out_buffer + (64 + 16) * sizeof(gr_complex)));
  checkCudaErrors(cudaMalloc(
      (void **)&d_dev_abs, d_max_out_buffer + (64 + 16) * sizeof(gr_complex)));
  checkCudaErrors(cudaMalloc((void **)&d_dev_cor,
                             d_max_out_buffer + (64 + 16) * sizeof(float)));
  checkCudaErrors(cudaMalloc(
      (void **)&d_dev_abs2, d_max_out_buffer + (64 + 16) * sizeof(gr_complex)));
  checkCudaErrors(cudaMalloc((void **)&d_dev_cor2,
                             d_max_out_buffer + (64 + 16) * sizeof(float)));

  checkCudaErrors(cudaMemset(
      d_dev_in, 0, d_max_out_buffer + (64 + 16) * sizeof(gr_complex)));
  checkCudaErrors(cudaMemset(
      d_dev_out, 0, d_max_out_buffer + (64 + 16) * sizeof(gr_complex)));
  checkCudaErrors(cudaMemset(
      d_dev_abs, 0, d_max_out_buffer + (64 + 16) * sizeof(gr_complex)));
  checkCudaErrors(
      cudaMemset(d_dev_cor, 0, d_max_out_buffer + (64 + 16) * sizeof(float)));
  checkCudaErrors(cudaMemset(
      d_dev_abs2, 0, d_max_out_buffer + (64 + 16) * sizeof(gr_complex)));
  checkCudaErrors(
      cudaMemset(d_dev_cor2, 0, d_max_out_buffer + (64 + 16) * sizeof(float)));

  set_output_multiple(1024 * 1024 / 8);
}

int presync_impl::general_work(int noutput_items, gr_vector_int &ninput_items,
                               gr_vector_const_void_star &input_items,
                               gr_vector_void_star &output_items) {
  const gr_complex *in = (const gr_complex *)input_items[0];
  gr_complex *out = (gr_complex *)output_items[0];
  gr_complex *abs = (gr_complex *)output_items[1];
  float *cor = (float *)output_items[2];

  //   std::cout << noutput_items << std::endl;

  noutput_items = std::min(ninput_items[0], noutput_items);

  checkCudaErrors(cudaMemcpyAsync(d_dev_in, in,
                                  sizeof(gr_complex) * (noutput_items + 63),
                                  cudaMemcpyHostToDevice, d_stream));

  auto gridSize = (noutput_items + d_block_size - 1) / d_block_size;
  exec_corr_abs((cuFloatComplex *)d_dev_in, (cuFloatComplex *)d_dev_abs,
                d_dev_cor, noutput_items, gridSize, d_block_size, d_stream);

  exec_mov_avg((cuFloatComplex *)d_dev_abs, d_dev_cor,
               (cuFloatComplex *)d_dev_abs2, d_dev_cor2, noutput_items,
               gridSize, d_block_size, d_stream);

  checkCudaErrors(cudaMemcpyAsync(abs, d_dev_abs2,
                                  sizeof(gr_complex) * noutput_items,
                                  cudaMemcpyDeviceToHost, d_stream));

  checkCudaErrors(cudaMemcpyAsync(cor, d_dev_cor2,
                                  sizeof(float) * noutput_items,
                                  cudaMemcpyDeviceToHost, d_stream));

  memcpy(out, in + 47, noutput_items * sizeof(gr_complex));
  //   checkCudaErrors(cudaMemcpyAsync(out,  in + 47,
  //                                   sizeof(gr_complex) * noutput_items,
  //                                   cudaMemcpyDeviceToDevice, d_stream));

  cudaStreamSynchronize(d_stream);

  // add_item_tag(1, nitems_written(0), pmt::mp("frame"), pmt::from_long(0));

  // Tell runtime system how many input items we consumed on
  // each input stream.
  consume_each(noutput_items);

  // Tell runtime system how many output items we produced.
  return noutput_items;
}
#endif
} /* namespace wifigpu */
} /* namespace gr */
