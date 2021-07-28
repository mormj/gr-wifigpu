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

#ifndef INCLUDED_WIFIGPU_PRESYNC_IMPL_H
#define INCLUDED_WIFIGPU_PRESYNC_IMPL_H

#include <wifigpu/presync.h>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuComplex.h>
#include <helper_cuda.h>

namespace gr {
namespace wifigpu {

class presync_impl : public presync {
private:
  cudaStream_t d_stream;
  int d_min_grid_size;
  int d_block_size;

  static const int d_max_out_buffer = 8*1024*1024;  // max bytes for output buffer
  cuFloatComplex* d_dev_in;
  cuFloatComplex* d_dev_out;
  cuFloatComplex* d_dev_abs;
  float* d_dev_cor;
  cuFloatComplex* d_dev_abs2;
  float* d_dev_cor2;

public:
  presync_impl();

  // Where all the action really happens
  void forecast(int noutput_items, gr_vector_int &ninput_items_required);

  int general_work(int noutput_items, gr_vector_int &ninput_items,
                   gr_vector_const_void_star &input_items,
                   gr_vector_void_star &output_items);
};

} // namespace wifigpu
} // namespace gr

#endif /* INCLUDED_WIFIGPU_PRESYNC_IMPL_H */
