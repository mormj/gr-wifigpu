/* -*- c++ -*- */
/*
 * Copyright 2021 Josh Morman.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "tp_decode_mac_impl.h"
#include <gnuradio/io_signature.h>

namespace gr {
namespace wifigpu {

tp_decode_mac::sptr tp_decode_mac::make(int num_threads, int queue_depth,
                                        bool log, bool debug) {
  return gnuradio::make_block_sptr<tp_decode_mac_impl>(num_threads, queue_depth,
                                                       log, debug);
}

/*
 * The private constructor
 */
tp_decode_mac_impl::tp_decode_mac_impl(int num_threads, int queue_depth,
                                       bool log, bool debug)
    : gr::block("tp_decode_mac", gr::io_signature::make(0, 0, 0),
                gr::io_signature::make(0, 0, 0)) {
#if 1
  exit_monitor_thread = false;
  tp = new threadpool(num_threads, queue_depth);
  monitor_thread = new std::thread([this]() {
    while (!exit_monitor_thread) {
      pmt::pmt_t pdu = this->tp->dequeue();
      while (pdu != pmt::PMT_NIL) {
        // std::cout << "pub ... " << std::endl;
        this->message_port_pub(pmt::mp("pdus"), pdu);
        pdu = this->tp->dequeue();
      }
    }
  });
#endif

  message_port_register_in(pmt::mp("pdus"));
  set_msg_handler(pmt::mp("pdus"),
                  [this](const pmt::pmt_t &msg) { this->handle_pdu(msg); });

  message_port_register_out(pmt::mp("pdus"));
}

void tp_decode_mac_impl::monitor_queue(tp_decode_mac_impl *top) {
  pmt::pmt_t pdu = top->tp->dequeue();
  while (pdu != pmt::PMT_NIL) {
    // std::cout << "pub ... " << std::endl;
    top->message_port_pub(pmt::mp("pdus"), pdu);
    pdu = top->tp->dequeue();
  }
}

void tp_decode_mac_impl::handle_pdu(pmt::pmt_t msg) {
  packet_cnt++;

  tp->enqueue(msg);

  // check flow control
}

/*
 * Our virtual destructor.
 */
tp_decode_mac_impl::~tp_decode_mac_impl() {
  std::cout << "got " << packet_cnt << " packets" << std::endl;
}

} /* namespace wifigpu */
} /* namespace gr */
