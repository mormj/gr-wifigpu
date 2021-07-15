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

#ifndef INCLUDED_WIFIGPU_SYNC_LONG_IMPL_H
#define INCLUDED_WIFIGPU_SYNC_LONG_IMPL_H

#include <wifigpu/sync_long.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cufft.h>

namespace gr {
namespace wifigpu {

class sync_long_impl : public sync_long {
private:
  enum { SYNC, COPY, RESET } d_state = SYNC;
  unsigned int d_sync_length;
  static const std::vector<gr_complex> LONG;
  int d_fftsize = 512;

  cufftHandle d_plan;
  cufftComplex *d_dev_training_freq;
  cufftComplex *d_dev_in;

  cudaStream_t d_stream;
  int d_min_grid_size;
  int d_block_size;

  std::vector<gr::tag_t> d_tags;

  int d_ncopied = 0;
  float d_freq_offset = 0;
  float d_freq_offset_short = 0;

  int d_num_syms = 0;

public:
  sync_long_impl(unsigned int sync_length);
  ~sync_long_impl();

  // Where all the action really happens
  void forecast(int noutput_items, gr_vector_int &ninput_items_required);

  int general_work(int noutput_items, gr_vector_int &ninput_items,
                   gr_vector_const_void_star &input_items,
                   gr_vector_void_star &output_items);
};

} // namespace wifigpu
} // namespace gr

#endif /* INCLUDED_WIFIGPU_SYNC_LONG_IMPL_H */
