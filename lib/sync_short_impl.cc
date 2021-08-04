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

#include "sync_short_impl.h"
#include <gnuradio/io_signature.h>
#include <cuda_buffer/cuda_buffer.h>

extern void exec_freq_correction(cuFloatComplex *in, cuFloatComplex *out, float freq_offset,
                          float start_idx, int n, int grid_size, int block_size,
                          cudaStream_t stream);
extern void get_block_and_grid_freq_correction(int *minGrid, int *minBlock);

namespace gr {
namespace wifigpu {

sync_short::sptr sync_short::make(float threshold, int min_plateau) {
  return gnuradio::get_initial_sptr(
      new sync_short_impl(threshold, min_plateau));
}

void sync_short_impl::forecast(int noutput_items,
                               gr_vector_int &ninput_items_required) {
  ninput_items_required[0] = noutput_items + d_min_plateau;
}

#if 1
/*
 * Custom Buffers
 */
sync_short_impl::sync_short_impl(float threshold, int min_plateau)
    : gr::block("sync_short",
                gr::io_signature::make3(3, 3, sizeof(gr_complex),
                                        sizeof(gr_complex), sizeof(float), cuda_buffer::type, cuda_buffer::type, cuda_buffer::type),
                gr::io_signature::make3(1, 3, sizeof(gr_complex),
                                        sizeof(uint8_t), sizeof(uint8_t), cuda_buffer::type)),
      d_threshold(threshold), d_min_plateau(min_plateau) {
  set_history(d_min_plateau);

  above_threshold.resize(8192);
  accum.resize(8192);
  d_host_cor.resize(8192);
  d_host_abs.resize(8192);

  // Temporary Buffers
  checkCudaErrors(cudaMalloc(
      (void **)&d_dev_in, d_max_out_buffer + (64 + 16) * sizeof(gr_complex)));
  checkCudaErrors(cudaMalloc(
      (void **)&d_dev_out, d_max_out_buffer + (64 + 16) * sizeof(gr_complex)));

  get_block_and_grid_freq_correction(&d_min_grid_size, &d_block_size);
  cudaDeviceSynchronize();
  std::cerr << "minGrid: " << d_min_grid_size << ", blockSize: " << d_block_size
            << std::endl;

  cudaStreamCreate(&d_stream);    
}

int sync_short_impl::general_work(int noutput_items,
                                  gr_vector_int &ninput_items,
                                  gr_vector_const_void_star &input_items,
                                  gr_vector_void_star &output_items) {
  const gr_complex *in = (const gr_complex *)input_items[0];
  const gr_complex *in_abs = (const gr_complex *)input_items[1];
  const float *in_cor = (const float *)input_items[2];
  gr_complex *out = (gr_complex *)output_items[0];
  // uint8_t *out_plateau = (uint8_t *)output_items[1];
  // uint8_t *out_accum = (uint8_t *)output_items[2];

  int h = history() - 1;
  if (noutput_items+h > above_threshold.size()) {
    above_threshold.resize(noutput_items + h);
    accum.resize(noutput_items);
       std::cout << "resizing to " << noutput_items+h << std::endl;
        d_host_cor.resize(noutput_items+h);
        d_host_abs.resize(noutput_items+h);
  }

    // Copy D2H in_cor to host
    checkCudaErrors(cudaMemcpyAsync(d_host_cor.data(),
                                    in_cor,
                                    sizeof(float) * (noutput_items+h),
                                    cudaMemcpyDeviceToHost,
                                    d_stream));
    checkCudaErrors(cudaMemcpyAsync(d_host_abs.data(),
                                    in_abs,
                                    sizeof(gr_complex) * (noutput_items+h),
                                    cudaMemcpyDeviceToHost,
                                    d_stream));
    cudaStreamSynchronize(d_stream);


  for (int i = 0; i < noutput_items + h; i++) {
    above_threshold[i] = d_host_cor[i] > d_threshold;
  }

  accum[0] = 0;
  for (int j = 0; j < h + 1; j++) {
    if (above_threshold[j]) {
      accum[0]++;
    } else {
      accum[0] = 0;
    }
  }

  auto nread = nitems_read(0);
  auto nwritten = nitems_written(0);

  for (int i = 1; i < noutput_items; i++) {
    if (above_threshold[i]) {
      accum[i] = accum[i - 1] + 1;

      if (accum[i] >= d_min_plateau &&
          nread + i - d_last_tag_location > MIN_GAP) {
        d_last_tag_location = nread + i;
        d_freq_offset = arg(d_host_abs[i]) / 16;
        insert_tag(nwritten + i, d_freq_offset, nread + i);
      }

    } else {
      accum[i] = 0;
    }
  }

  // std::cout << noutput_items << std::endl;

  // checkCudaErrors(cudaMemcpyAsync(d_dev_in, in + h,
  //                                 sizeof(gr_complex) * (noutput_items),
  //                                 cudaMemcpyHostToDevice, d_stream));

  // for (int o = 0; o < noutput_items; o++) {
  //   out[o] = in[o + h] * exp(gr_complex(0, -d_freq_offset * (nwritten + o)));
  // }

  auto gridSize = (noutput_items + d_block_size - 1) / d_block_size;
  exec_freq_correction((cuFloatComplex *)(in+h), (cuFloatComplex *)out, d_freq_offset,
                          nwritten, noutput_items, gridSize, d_block_size,
                          d_stream);

  // checkCudaErrors(cudaMemcpyAsync(out, d_dev_out,
  //                                 sizeof(gr_complex) * (noutput_items),
  //                                 cudaMemcpyDeviceToHost, d_stream));

  cudaStreamSynchronize(d_stream);


  // memcpy(out_plateau, above_threshold.data(), noutput_items);
  // memcpy(out_accum, accum.data(), noutput_items);

  // Tell runtime system how many input items we consumed on
  // each input stream.
  consume_each(noutput_items);

  // Tell runtime system how many output items we produced.
  return noutput_items;
}
#else
/*
 * Double Copy
 */
sync_short_impl::sync_short_impl(float threshold, int min_plateau)
    : gr::block("sync_short",
                gr::io_signature::make3(3, 3, sizeof(gr_complex),
                                        sizeof(gr_complex), sizeof(float)),
                gr::io_signature::make3(1, 3, sizeof(gr_complex),
                                        sizeof(uint8_t), sizeof(uint8_t))),
      d_threshold(threshold), d_min_plateau(min_plateau) {
  set_history(d_min_plateau);

  above_threshold.resize(8192);
  accum.resize(8192);

  // Temporary Buffers
  checkCudaErrors(cudaMalloc(
      (void **)&d_dev_in, d_max_out_buffer + (64 + 16) * sizeof(gr_complex)));
  checkCudaErrors(cudaMalloc(
      (void **)&d_dev_out, d_max_out_buffer + (64 + 16) * sizeof(gr_complex)));

  get_block_and_grid_freq_correction(&d_min_grid_size, &d_block_size);
  cudaDeviceSynchronize();
  std::cerr << "minGrid: " << d_min_grid_size << ", blockSize: " << d_block_size
            << std::endl;

  cudaStreamCreate(&d_stream);    
}

int sync_short_impl::general_work(int noutput_items,
                                  gr_vector_int &ninput_items,
                                  gr_vector_const_void_star &input_items,
                                  gr_vector_void_star &output_items) {
  const gr_complex *in = (const gr_complex *)input_items[0];
  const gr_complex *in_abs = (const gr_complex *)input_items[1];
  const float *in_cor = (const float *)input_items[2];
  gr_complex *out = (gr_complex *)output_items[0];
  // uint8_t *out_plateau = (uint8_t *)output_items[1];
  // uint8_t *out_accum = (uint8_t *)output_items[2];

  int h = history() - 1;
  if (noutput_items > above_threshold.size()) {
    above_threshold.resize(noutput_items + h);
    accum.resize(noutput_items);
  }

  for (int i = 0; i < noutput_items + h; i++) {
    above_threshold[i] = in_cor[i] > d_threshold;
  }

  accum[0] = 0;
  for (int j = 0; j < h + 1; j++) {
    if (above_threshold[j]) {
      accum[0]++;
    } else {
      accum[0] = 0;
    }
  }

  auto nread = nitems_read(0);
  auto nwritten = nitems_written(0);

  for (int i = 1; i < noutput_items; i++) {
    if (above_threshold[i]) {
      accum[i] = accum[i - 1] + 1;

      if (accum[i] >= d_min_plateau &&
          nread + i - d_last_tag_location > MIN_GAP) {
        d_last_tag_location = nread + i;
        d_freq_offset = arg(in_abs[i]) / 16;
        insert_tag(nwritten + i, d_freq_offset, nread + i);
      }

    } else {
      accum[i] = 0;
    }
  }

  // std::cout << noutput_items << std::endl;

  checkCudaErrors(cudaMemcpyAsync(d_dev_in, in + h,
                                  sizeof(gr_complex) * (noutput_items),
                                  cudaMemcpyHostToDevice, d_stream));

  // for (int o = 0; o < noutput_items; o++) {
  //   out[o] = in[o + h] * exp(gr_complex(0, -d_freq_offset * (nwritten + o)));
  // }

  auto gridSize = (noutput_items + d_block_size - 1) / d_block_size;
  exec_freq_correction(d_dev_in, d_dev_out, d_freq_offset,
                          nwritten, noutput_items, gridSize, d_block_size,
                          d_stream);

  checkCudaErrors(cudaMemcpyAsync(out, d_dev_out,
                                  sizeof(gr_complex) * (noutput_items),
                                  cudaMemcpyDeviceToHost, d_stream));

  cudaStreamSynchronize(d_stream);


  // memcpy(out_plateau, above_threshold.data(), noutput_items);
  // memcpy(out_accum, accum.data(), noutput_items);

  // Tell runtime system how many input items we consumed on
  // each input stream.
  consume_each(noutput_items);

  // Tell runtime system how many output items we produced.
  return noutput_items;
}
#endif

void sync_short_impl::insert_tag(uint64_t item, double freq_offset,
                                 uint64_t input_item) {
  // mylog(boost::format("frame start at in: %2% out: %1%") % item %
  // input_item);

  const pmt::pmt_t key = pmt::string_to_symbol("wifi_start");
  const pmt::pmt_t value = pmt::from_double(freq_offset);
  const pmt::pmt_t srcid = pmt::string_to_symbol(name());
  add_item_tag(0, item, key, value, srcid);
}

} /* namespace wifigpu */
} // namespace gr
