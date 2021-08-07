/* -*- c++ -*- */
/*
 * Copyright 2021 Josh Morman.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_WIFIGPU_PACKETIZE_FRAME_H
#define INCLUDED_WIFIGPU_PACKETIZE_FRAME_H

#include <gnuradio/block.h>
#include <wifigpu/api.h>

namespace gr {
namespace wifigpu {

/*!
 * \brief <+description of block+>
 * \ingroup wifigpu
 *
 */
class WIFIGPU_API packetize_frame : virtual public gr::block {
public:
  typedef std::shared_ptr<packetize_frame> sptr;

  /*!
   * \brief Return a shared_ptr to a new instance of wifigpu::packetize_frame.
   *
   * To avoid accidental use of raw pointers, wifigpu::packetize_frame's
   * constructor is in a private implementation
   * class. wifigpu::packetize_frame::make is the public interface for
   * creating new instances.
   */
  static sptr make(int algo, double freq, double bw);
};

} // namespace wifigpu
} // namespace gr

#endif /* INCLUDED_WIFIGPU_PACKETIZE_FRAME_H */
