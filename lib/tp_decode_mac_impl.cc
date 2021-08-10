/* -*- c++ -*- */
/*
 * Copyright 2021 Josh Morman.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "tp_decode_mac_impl.h"
#include <gnuradio/io_signature.h>
#include <chrono>
using namespace std::chrono_literals;
// #include <boost/crc.hpp>

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
                gr::io_signature::make(0, 0, 0)),
      d_log(log), d_debug(debug) 
      // ,d_ofdm(num_threads), d_frame(num_threads),
      // d_rx_symbols(num_threads), d_rx_bits(num_threads),
      // d_deinterleaved_bits(num_threads), d_out_bytes(num_threads),
      // d_frame_complete(true) 
      {

  // uint8_t d_rx_symbols[48 * MAX_SYM];
  // uint8_t d_rx_bits[MAX_ENCODED_BITS];
  // uint8_t d_deinterleaved_bits[MAX_ENCODED_BITS];
  // uint8_t out_bytes[MAX_PSDU_SIZE + 2]; // 2 for signal field


  // for (int i = 0; i < num_threads; i++) {
  //   d_ofdm.push_back(std::make_shared<ofdm_param>(BPSK_1_2));
  //   d_frame.push_back(std::make_shared<frame_param>(d_ofdm[i].get(), 0));
  //   d_rx_symbols[i].resize(48 * MAX_SYM);
  //   d_rx_bits[i].resize(MAX_ENCODED_BITS);
  //   d_deinterleaved_bits[i].resize(MAX_ENCODED_BITS);
  //   d_out_bytes[i].resize(MAX_PSDU_SIZE + 2);
  // }

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


  
}

bool tp_decode_mac_impl::stop()
{
  while(tp->qsize() > 0)
  {
    std::cout << "got " << packet_cnt << " packets" << " qsize: " << tp->qsize() << std::endl;
    std::this_thread::sleep_for(100ms);
  }
  return true;
}

// void tp_decode_mac_impl::decode(uint8_t *rx_bits, uint8_t *rx_symbols,
//                                 uint8_t *deinterleaved_bits, uint8_t *out_bytes,
//                                 viterbi_decoder &decoder,
//                                 frame_param &frame_info,
//                                 ofdm_param &ofdm_info) {

//   for (int i = 0; i < frame_info.n_sym * 48; i++) {
//     for (int k = 0; k < ofdm_info.n_bpsc; k++) {
//       rx_bits[i * ofdm_info.n_bpsc + k] = !!(rx_symbols[i] & (1 << k));
//     }
//   }

//   deinterleave(rx_bits, deinterleaved_bits, frame_info.n_sym, ofdm_info);
//   uint8_t *decoded =
//       decoder.decode(&ofdm_info, &frame_info, deinterleaved_bits);

//   descramble(frame_info.psdu_size, decoded, out_bytes);
//   print_output(out_bytes, frame_info.psdu_size);

//   // skip service field
//   boost::crc_32_type result;
//   result.process_bytes(out_bytes + 2, frame_info.psdu_size);
//   if (result.checksum() != 558161692) {
//     dout << "checksum wrong -- dropping" << std::endl;
//     return;
//   }

//   mylog(boost::format("encoding: %1% - length: %2% - symbols: %3%") %
//         ofdm_info.encoding % frame_info.psdu_size % frame_info.n_sym);

//   // // create PDU
//   // pmt::pmt_t blob = pmt::make_blob(out_bytes + 2, frame_info.psdu_size - 4);
//   // d_meta = pmt::dict_add(d_meta, pmt::mp("dlt"),
//   //                        pmt::from_long(LINKTYPE_IEEE802_11));

//   // message_port_pub(pmt::mp("out"), pmt::cons(d_meta, blob));
// }

// void tp_decode_mac_impl::deinterleave(uint8_t *rx_bits,
//                                       uint8_t *deinterleaved_bits, size_t n_sym,
//                                       ofdm_param &ofdm_info) {

//   int n_cbps = ofdm_info.n_cbps;
//   int first[n_cbps];
//   int second[n_cbps];
//   int s = std::max(ofdm_info.n_bpsc / 2, 1);

//   for (int j = 0; j < n_cbps; j++) {
//     first[j] = s * (j / s) + ((j + int(floor(16.0 * j / n_cbps))) % s);
//   }

//   for (int i = 0; i < n_cbps; i++) {
//     second[i] = 16 * i - (n_cbps - 1) * int(floor(16.0 * i / n_cbps));
//   }

//   int count = 0;
//   for (int i = 0; i < n_sym; i++) {
//     for (int k = 0; k < n_cbps; k++) {
//       deinterleaved_bits[i * n_cbps + second[first[k]]] =
//           rx_bits[i * n_cbps + k];
//     }
//   }
// }

// void tp_decode_mac_impl::descramble(size_t psdu_size, uint8_t *decoded_bits,
//                                     uint8_t *out_bytes) {

//   int state = 0;
//   std::memset(out_bytes, 0, psdu_size + 2);

//   for (int i = 0; i < 7; i++) {
//     if (decoded_bits[i]) {
//       state |= 1 << (6 - i);
//     }
//   }
//   out_bytes[0] = state;

//   int feedback;
//   int bit;

//   for (int i = 7; i < psdu_size * 8 + 16; i++) {
//     feedback = ((!!(state & 64))) ^ (!!(state & 8));
//     bit = feedback ^ (decoded_bits[i] & 0x1);
//     out_bytes[i / 8] |= bit << (i % 8);
//     state = ((state << 1) & 0x7e) | feedback;
//   }
// }

// void tp_decode_mac_impl::print_output(uint8_t *out_bytes, size_t psdu_size) {

//   dout << std::endl;
//   dout << "psdu size" << psdu_size << std::endl;
//   for (int i = 2; i < psdu_size + 2; i++) {
//     dout << std::setfill('0') << std::setw(2) << std::hex
//          << ((unsigned int)out_bytes[i] & 0xFF) << std::dec << " ";
//     if (i % 16 == 15) {
//       dout << std::endl;
//     }
//   }
//   dout << std::endl;
//   for (int i = 2; i < psdu_size + 2; i++) {
//     if ((out_bytes[i] > 31) && (out_bytes[i] < 127)) {
//       dout << ((char)out_bytes[i]);
//     } else {
//       dout << ".";
//     }
//   }
//   dout << std::endl;
// }

} /* namespace wifigpu */
} /* namespace gr */
