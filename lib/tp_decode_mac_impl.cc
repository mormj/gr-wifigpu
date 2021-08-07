/* -*- c++ -*- */
/*
 * Copyright 2021 Josh Morman.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "tp_decode_mac_impl.h"
#include <gnuradio/io_signature.h>

#include "threadpool/threadpool.h"

namespace gr {
namespace wifigpu {

tp_decode_mac::sptr tp_decode_mac::make(int num_threads, int queue_depth, bool log, bool debug) {
  return gnuradio::make_block_sptr<tp_decode_mac_impl>(num_threads, queue_depth, log, debug);
}

/*
 * The private constructor
 */
tp_decode_mac_impl::tp_decode_mac_impl(int num_threads, int queue_depth,bool log, bool debug)
    : gr::block(
          "tp_decode_mac",
          gr::io_signature::make(0,0,0),
          gr::io_signature::make(0,0,0)) {}

/*
 * Our virtual destructor.
 */
tp_decode_mac_impl::~tp_decode_mac_impl() {}


} /* namespace wifigpu */
} /* namespace gr */
