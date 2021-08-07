/* -*- c++ -*- */
/*
 * Copyright 2021 Josh Morman.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_WIFIGPU_TP_DECODE_MAC_IMPL_H
#define INCLUDED_WIFIGPU_TP_DECODE_MAC_IMPL_H

#include <wifigpu/tp_decode_mac.h>

namespace gr {
namespace wifigpu {

class tp_decode_mac_impl : public tp_decode_mac {
private:
  // Nothing to declare in this block.

public:
  tp_decode_mac_impl(int num_threads, int queue_depth, bool log = false, bool debug = false);
  ~tp_decode_mac_impl();

};

} // namespace wifigpu
} // namespace gr

#endif /* INCLUDED_WIFIGPU_TP_DECODE_MAC_IMPL_H */
