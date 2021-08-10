/* -*- c++ -*- */
/*
 * Copyright 2021 Josh Morman.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_WIFIGPU_TP_DECODE_MAC_IMPL_H
#define INCLUDED_WIFIGPU_TP_DECODE_MAC_IMPL_H

#include <wifigpu/tp_decode_mac.h>

#include <atomic>
#include <future>
#include <iostream>
#include <mutex>
#include <queue>
#include <thread>

#include "threadpool/threadpool.h"
#include "utils.h"
#include "viterbi_decoder/viterbi_decoder.h"
#include <wifigpu/mapper.h>

namespace gr {
namespace wifigpu {

class tp_decode_mac_impl : public tp_decode_mac {
private:
  void handle_pdu(pmt::pmt_t msg);
  static void monitor_queue(tp_decode_mac_impl *);
  std::thread *monitor_thread;
  threadpool *tp;
  bool exit_monitor_thread;
  int packet_cnt = 0;

  bool d_debug;
  bool d_log;

  pmt::pmt_t d_meta;



public:
  tp_decode_mac_impl(int num_threads, int queue_depth, bool log = false,
                     bool debug = false);
  ~tp_decode_mac_impl();

  bool stop();
};

} // namespace wifigpu
} // namespace gr

#endif /* INCLUDED_WIFIGPU_TP_DECODE_MAC_IMPL_H */
