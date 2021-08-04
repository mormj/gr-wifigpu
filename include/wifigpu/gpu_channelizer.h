/* -*- c++ -*- */
/*
 * Copyright 2021 Josh Morman.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_WIFIGPU_GPU_CHANNELIZER_H
#define INCLUDED_WIFIGPU_GPU_CHANNELIZER_H

#include <gnuradio/block.h>
#include <wifigpu/api.h>

namespace gr {
namespace wifigpu {

/*!
 * \brief <+description of block+>
 * \ingroup wifigpu
 *
 */
class WIFIGPU_API gpu_channelizer : virtual public gr::block {
public:
  typedef std::shared_ptr<gpu_channelizer> sptr;

  /*!
   * \brief Return a shared_ptr to a new instance of wifigpu::gpu_channelizer.
   *
   * To avoid accidental use of raw pointers, wifigpu::gpu_channelizer's
   * constructor is in a private implementation
   * class. wifigpu::gpu_channelizer::make is the public interface for
   * creating new instances.
   */
  static sptr make(size_t nchans, const std::vector<float>& taps);
};

} // namespace wifigpu
} // namespace gr

#endif /* INCLUDED_WIFIGPU_GPU_CHANNELIZER_H */
