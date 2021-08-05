/* -*- c++ -*- */
/*
 * Copyright 2021 Josh Morman.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_WIFIGPU_GPU_CHANNELIZER_IMPL_H
#define INCLUDED_WIFIGPU_GPU_CHANNELIZER_IMPL_H

#include <wifigpu/gpu_channelizer.h>
#include <cusp/channelizer.cuh>
#include <cusp/deinterleave.cuh>

namespace gr {
namespace wifigpu {

class gpu_channelizer_impl : public gpu_channelizer {
private:
  size_t d_nchans;
  size_t d_taps;
  size_t d_overlap;

  void *d_dev_buf;
  void *d_dev_tail;
  cudaStream_t d_stream;

  std::shared_ptr<cusp::channelizer<gr_complex>> p_channelizer;
  std::shared_ptr<cusp::deinterleave> p_deinterleaver;

public:
  gpu_channelizer_impl(size_t nchans, const std::vector<float>& taps);
  ~gpu_channelizer_impl();

  // Where all the action really happens
  void forecast(int noutput_items, gr_vector_int &ninput_items_required);

  int general_work(int noutput_items, gr_vector_int &ninput_items,
                   gr_vector_const_void_star &input_items,
                   gr_vector_void_star &output_items);
};

} // namespace wifigpu
} // namespace gr

#endif /* INCLUDED_WIFIGPU_GPU_CHANNELIZER_IMPL_H */
