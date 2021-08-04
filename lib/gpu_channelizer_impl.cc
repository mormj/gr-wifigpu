/* -*- c++ -*- */
/*
 * Copyright 2021 Josh Morman.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "gpu_channelizer_impl.h"
#include <gnuradio/io_signature.h>
#include <cuda_buffer/cuda_buffer.h>
#include <helper_cuda.h>

namespace gr {
namespace wifigpu {

using input_type = gr_complex;
using output_type = gr_complex;

gpu_channelizer::sptr gpu_channelizer::make(size_t nchans,
                                            const std::vector<float> &taps) {
  return gnuradio::make_block_sptr<gpu_channelizer_impl>(nchans, taps);
}

/*
 * The private constructor
 */
gpu_channelizer_impl::gpu_channelizer_impl(size_t nchans,
                                           const std::vector<float> &taps)
    : gr::block("gpu_channelizer",
                gr::io_signature::make(1 /* min inputs */, 1 /* max inputs */,
                                       sizeof(input_type),  cuda_buffer::type),
                gr::io_signature::make(1 /* min outputs */, 1 /*max outputs */,
                                       sizeof(output_type),  cuda_buffer::type)),
      d_nchans(nchans)

{
  auto new_taps = std::vector<gr_complex>(taps.size());
  for (size_t i = 0; i < taps.size(); i++) {
    new_taps[i] = gr_complex(taps[i], 0);
  }

  p_channelizer =
      std::make_shared<cusp::channelizer<gr_complex>>(new_taps, nchans);

  set_output_multiple(nchans);
}

/*
 * Our virtual destructor.
 */
gpu_channelizer_impl::~gpu_channelizer_impl() {}

void gpu_channelizer_impl::forecast(int noutput_items,
                                    gr_vector_int &ninput_items_required) {

  ninput_items_required[0] = noutput_items;
}

int gpu_channelizer_impl::general_work(int noutput_items,
                                       gr_vector_int &ninput_items,
                                       gr_vector_const_void_star &input_items,
                                       gr_vector_void_star &output_items) {
  const input_type *in = reinterpret_cast<const input_type *>(input_items[0]);
  output_type *out = reinterpret_cast<output_type *>(output_items[0]);

  checkCudaErrors(p_channelizer->launch_default_occupancy(input_items, output_items,
                                          noutput_items / d_nchans));

  consume_each(noutput_items);

  // Tell runtime system how many output items we produced.
  return noutput_items;
}

} /* namespace wifigpu */
} /* namespace gr */
