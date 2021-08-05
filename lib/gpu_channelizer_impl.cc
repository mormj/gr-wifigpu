/* -*- c++ -*- */
/*
 * Copyright 2021 Josh Morman.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "gpu_channelizer_impl.h"
#include <cuda_buffer/cuda_buffer.h>
#include <gnuradio/io_signature.h>
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
                                       sizeof(input_type), cuda_buffer::type),
                gr::io_signature::make(nchans /* min outputs */,
                                       nchans /*max outputs */,
                                       sizeof(output_type), cuda_buffer::type)),
      d_nchans(nchans)

{
  auto new_taps = std::vector<gr_complex>(taps.size());
  for (size_t i = 0; i < taps.size(); i++) {
    new_taps[i] = gr_complex(taps[i], 0);
  }

  // quantize the overlap to the nchans
  d_overlap = d_nchans * ((taps.size() + d_nchans - 1) / d_nchans);
  checkCudaErrors(cudaMalloc(&d_dev_tail, d_overlap * sizeof(gr_complex)));
  checkCudaErrors(cudaMemset(d_dev_tail, 0, d_overlap * sizeof(gr_complex)));

  checkCudaErrors(cudaMalloc(
      &d_dev_buf, 16 * 1024 * 1024 * sizeof(gr_complex))); // 4M items max ??

  p_channelizer =
      std::make_shared<cusp::channelizer<gr_complex>>(new_taps, nchans);
  cudaStreamCreate(&d_stream);
  p_channelizer->set_stream(d_stream);

  p_deinterleaver =
      std::make_shared<cusp::deinterleave>(nchans, 1, sizeof(gr_complex));

  p_deinterleaver->set_stream(d_stream);
  // set_output_multiple(nchans);
  // set_min_noutput_items(d_overlap+1024);
}

/*
 * Our virtual destructor.
 */
gpu_channelizer_impl::~gpu_channelizer_impl() {}

void gpu_channelizer_impl::forecast(int noutput_items,
                                    gr_vector_int &ninput_items_required) {

  ninput_items_required[0] = noutput_items * d_nchans + d_overlap;
}

int gpu_channelizer_impl::general_work(int noutput_items,
                                       gr_vector_int &ninput_items,
                                       gr_vector_const_void_star &input_items,
                                       gr_vector_void_star &output_items) {
  const input_type *in = reinterpret_cast<const input_type *>(input_items[0]);
  output_type *out = reinterpret_cast<output_type *>(output_items[0]);

  // checkCudaErrors(p_channelizer->launch_default_occupancy(input_items,
  // output_items,
  //                                         noutput_items / d_nchans));

  // gr_complex host_in[ninput_items[0]];
  // checkCudaErrors(cudaMemcpy(host_in, in, ninput_items[0] * sizeof(gr_complex),
  //                            cudaMemcpyDeviceToHost));

  checkCudaErrors(p_channelizer->launch_default_occupancy(
      input_items, {d_dev_buf}, (noutput_items + d_overlap / d_nchans)));

  // gr_complex
  //     host_channelized[d_nchans * (noutput_items + d_overlap / d_nchans)];
  // checkCudaErrors(cudaMemcpy(host_channelized, d_dev_buf,
  //                            d_nchans * (noutput_items + d_overlap / d_nchans) *
  //                                sizeof(gr_complex),
  //                            cudaMemcpyDeviceToHost));

  // // copy the tail from the previous work()
  // checkCudaErrors(cudaMemcpyAsync(out, d_dev_tail,
  // d_overlap*sizeof(gr_complex), cudaMemcpyDeviceToDevice, d_stream));

  // // Save the tail
  // checkCudaErrors(cudaMemcpyAsync(d_dev_tail, out+noutput_items-d_overlap,
  // d_overlap*sizeof(gr_complex), cudaMemcpyDeviceToDevice, d_stream));

  // checkCudaErrors(cudaMemcpyAsync(out, (gr_complex *)d_dev_buf + d_overlap,
  //                                 noutput_items * sizeof(gr_complex),
  //                                 cudaMemcpyDeviceToDevice, d_stream));

  checkCudaErrors(p_deinterleaver->launch_default_occupancy(
      {(gr_complex *)d_dev_buf + d_overlap}, output_items,
      noutput_items * d_nchans));

  // gr_complex host_stream0[noutput_items];
  // checkCudaErrors(cudaMemcpy(host_stream0, out,
  //                            noutput_items * sizeof(gr_complex),
  //                            cudaMemcpyDeviceToHost));

  cudaStreamSynchronize(d_stream);

  consume_each(noutput_items * d_nchans);

  // Tell runtime system how many output items we produced.
  return noutput_items;
}

} /* namespace wifigpu */
} /* namespace gr */
