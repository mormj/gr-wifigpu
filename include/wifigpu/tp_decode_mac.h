/* -*- c++ -*- */
/*
 * Copyright 2021 Josh Morman.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_WIFIGPU_TP_DECODE_MAC_H
#define INCLUDED_WIFIGPU_TP_DECODE_MAC_H

#include <gnuradio/block.h>
#include <wifigpu/api.h>

namespace gr {
namespace wifigpu {

/*!
 * \brief <+description of block+>
 * \ingroup wifigpu
 *
 */
class WIFIGPU_API tp_decode_mac : virtual public gr::block {
public:
  typedef std::shared_ptr<tp_decode_mac> sptr;

  /*!
   * \brief Return a shared_ptr to a new instance of wifigpu::tp_decode_mac.
   *
   * To avoid accidental use of raw pointers, wifigpu::tp_decode_mac's
   * constructor is in a private implementation
   * class. wifigpu::tp_decode_mac::make is the public interface for
   * creating new instances.
   */
  static sptr make(int num_threads, int queue_depth, bool log = false, bool debug = false);
};

} // namespace wifigpu
} // namespace gr

#endif /* INCLUDED_WIFIGPU_TP_DECODE_MAC_H */
