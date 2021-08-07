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

#ifndef INCLUDED_WIFIGPU_SYNC_SHORT_IMPL_H
#define INCLUDED_WIFIGPU_SYNC_SHORT_IMPL_H

#include <wifigpu/sync_short.h>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuComplex.h>
#include <helper_cuda.h>

namespace gr {
namespace wifigpu {

class sync_short_impl : public sync_short {
private:
  float d_threshold;
  int d_min_plateau;
  uint64_t d_last_tag_location = 0;
  float d_freq_offset;

  cuFloatComplex* d_dev_in;
  cuFloatComplex* d_dev_out;
  cudaStream_t d_stream;
  int d_min_grid_size;
  int d_block_size;

  std::vector<float> d_host_cor;
	std::vector<gr_complex> d_host_abs;

  std::vector<uint8_t> above_threshold;
  std::vector<uint8_t> accum;

  static const int MIN_GAP = 480;
  static const int MAX_SAMPLES = 540 * 80;

  static const int d_max_out_buffer = 8*1024*1024;  // max bytes for output buffer

  int packet_cnt = 0;

public:
  sync_short_impl(float threshold, int min_plateau);
  ~sync_short_impl() { 
    std::cout << "sync_short: " << packet_cnt << std::endl;
  }

  void insert_tag(uint64_t item, double freq_offset, uint64_t input_item);

  // Where all the action really happens
  void forecast(int noutput_items, gr_vector_int &ninput_items_required);

  int general_work(int noutput_items, gr_vector_int &ninput_items,
                   gr_vector_const_void_star &input_items,
                   gr_vector_void_star &output_items);
};

} // namespace wifigpu
} // namespace gr

#endif /* INCLUDED_WIFIGPU_SYNC_SHORT_IMPL_H */
