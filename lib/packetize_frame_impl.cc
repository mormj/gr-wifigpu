/* -*- c++ -*- */
/*
 * Copyright 2021 Josh Morman.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "packetize_frame_impl.h"
#include <gnuradio/io_signature.h>

#include "equalizer/base.h"
#include "equalizer/comb.h"
#include "equalizer/lms.h"
#include "equalizer/ls.h"
#include "equalizer/sta.h"
#include "utils.h"

namespace gr {
namespace wifigpu {

packetize_frame::sptr packetize_frame::make(int algo, double freq, double bw) {
  return gnuradio::make_block_sptr<packetize_frame_impl>(algo, freq, bw);
}

/*
 * The private constructor
 */
packetize_frame_impl::packetize_frame_impl(int algo, double freq, double bw)
    : gr::block("packetize_frame",
                gr::io_signature::make(1 /* min inputs */, 1 /* max inputs */,
                                       64 * sizeof(gr_complex)),
                gr::io_signature::make(0, 0, 0)),
      d_current_symbol(0), d_equalizer(nullptr), d_freq(freq), d_bw(bw),
      d_frame_bytes(0), d_frame_symbols(0), d_freq_offset_from_synclong(0.0) {

  message_port_register_out(pmt::mp("pdus"));

  d_bpsk = constellation_bpsk::make();
  d_qpsk = constellation_qpsk::make();
  d_16qam = constellation_16qam::make();
  d_64qam = constellation_64qam::make();

  d_frame_mod = d_bpsk;

  set_tag_propagation_policy(block::TPP_DONT);
  set_algorithm((Equalizer)algo);
}

void packetize_frame_impl::set_algorithm(Equalizer algo) {
  // gr::thread::scoped_lock lock(d_mutex);
  delete d_equalizer;

  switch (algo) {

  case COMB:
    // dout << "Comb" << std::endl;
    d_equalizer = new equalizer::comb();
    break;
  case LS:
    // dout << "LS" << std::endl;
    d_equalizer = new equalizer::ls();
    break;
  case LMS:
    // dout << "LMS" << std::endl;
    d_equalizer = new equalizer::lms();
    break;
  case STA:
    // dout << "STA" << std::endl;
    d_equalizer = new equalizer::sta();
    break;
  default:
    throw std::runtime_error("Algorithm not implemented");
  }
}

packetize_frame_impl::~packetize_frame_impl() {
  std::cout << "packetized " << packet_cnt << std::endl;
}

void packetize_frame_impl::forecast(int noutput_items,
                                    gr_vector_int &ninput_items_required) {
  ninput_items_required[0] = noutput_items + 3;
}

int packetize_frame_impl::general_work(int noutput_items,
                                       gr_vector_int &ninput_items,
                                       gr_vector_const_void_star &input_items,
                                       gr_vector_void_star &output_items) {
  const gr_complex *in = (const gr_complex *)input_items[0];
  // uint8_t *out = (uint8_t *)output_items[0];

  int i = 0;
  int o = 0;
  gr_complex symbols[48];
  gr_complex current_symbol[64];

  auto nread = nitems_read(0);

  // dout << "FRAME EQUALIZER: input " << noutput_items << "  output " <<
  // noutput_items << std::endl;

  int symbols_to_consume = ninput_items[0];

  get_tags_in_window(tags, 0, 0, noutput_items,
                     pmt::string_to_symbol("wifi_start"));

  if (d_state == FINISH_LAST_FRAME) {
    if (noutput_items < symbols_to_consume) {
      symbols_to_consume = noutput_items;
    }
    if (tags.size()) {
      // only consume up to the next tag
      symbols_to_consume = tags[0].offset - nread;
      if (symbols_to_consume == 0) {
        d_state = WAITING_FOR_TAG;
        return 0;
      }
    }

    if (d_frame_symbols + 2 > d_current_symbol &&
        d_frame_symbols + 2 - d_current_symbol < symbols_to_consume) {
      symbols_to_consume = d_frame_symbols + 2 - d_current_symbol;
    }

    if (symbols_to_consume) {
      auto symbols_to_process = symbols_to_consume;
      if (symbols_to_process > (d_frame_symbols + 2)) {
        symbols_to_process = d_frame_symbols + 2;
      }

      size_t vlen;
      gr_complex *pdu_vec = pmt::c32vector_writable_elements(pmt::cdr(d_pdu), vlen);
      for (int o = 0; o < symbols_to_process; o++) {
        if (d_current_symbol > d_frame_symbols + 2) {
          d_state = WAITING_FOR_TAG;
          consume_each(symbols_to_consume);
          return 0;
        }
        memcpy(pdu_vec + 64 * (d_current_symbol - 3), in + o * 64,
               64 * sizeof(gr_complex));

        d_current_symbol++;
      }

      consume_each(symbols_to_consume);
      return 0;
    }
  } else {

    if (tags.size()) {
      // new frame -- send out the old frame
      if (d_pdu) {
        // std::cout << "publish frame" << std::endl;
        message_port_pub(pmt::mp("pdus"), d_pdu);
        d_pdu = nullptr;
      }

      d_current_symbol = 0;
      d_frame_symbols = 0;
      d_frame_mod = d_bpsk;

      d_freq_offset_from_synclong =
          pmt::to_double(tags.front().value) * d_bw / (2 * M_PI);
      d_epsilon0 =
          pmt::to_double(tags.front().value) * d_bw / (2 * M_PI * d_freq);
      d_er = 0;

      auto frame_start = tags[0].offset - nread;

      if (frame_start + 3 >= ninput_items[0]) {
        consume_each(frame_start);
        return 0;
      }

      static uint8_t signal_field[48];
      static gr_complex symbols[64];

      for (int i = 0; i < 3; i++) {
        process_symbol((gr_complex *)(in + frame_start + i * 64), symbols,
                       signal_field);
        d_current_symbol++;
      }

      if (decode_signal_field(signal_field)) {

        // check for maximum frame size
        if (d_frame_symbols > MAX_SYM || d_frame_bytes > MAX_PSDU_SIZE) {
          consume_each(frame_start + 3);
          d_state = WAITING_FOR_TAG;
          return 0;
        }

        // std::cout << d_frame_bytes << " / " << d_frame_encoding << std::endl;
        pmt::pmt_t d = pmt::make_dict();
        d = pmt::dict_add(d, pmt::mp("frame_bytes"),
                          pmt::from_uint64(d_frame_bytes));
        d = pmt::dict_add(d, pmt::mp("frame_symbols"),
                          pmt::from_uint64(d_frame_symbols));
        d = pmt::dict_add(d, pmt::mp("encoding"),
                          pmt::from_uint64(d_frame_encoding));
        d = pmt::dict_add(d, pmt::mp("snr"),
                          pmt::from_double(d_equalizer->get_snr()));
        d = pmt::dict_add(d, pmt::mp("freq"), pmt::from_double(d_freq));
        d = pmt::dict_add(d, pmt::mp("bw"), pmt::from_double(d_bw));
        d = pmt::dict_add(d, pmt::mp("freq_offset"),
                          pmt::from_double(d_freq_offset_from_synclong));
        d = pmt::dict_add(d, pmt::mp("H"),
                          pmt::init_c32vector(64, d_equalizer->get_H()));
        d = pmt::dict_add(d, pmt::mp("prev_pilots"),
                          pmt::init_c32vector(4, d_prev_pilots));

        auto samples = pmt::make_c32vector(64 * d_frame_symbols, gr_complex(0, 0));
        d_pdu = pmt::cons(d, samples);

        packet_cnt++;

                        
                        if (packet_cnt % 1000 == 0)
                        {
                        std::cout << "packetize_frame: " << packet_cnt << std::endl;
                        }

        d_state = FINISH_LAST_FRAME;

        consume_each(frame_start + 3);
        return 0;
      } else {
        consume_each(frame_start + 3);
        d_state = WAITING_FOR_TAG;
        return 0;
      }

    } else {
      consume_each(ninput_items[0]);
      return 0;
    }
  }

  throw std::runtime_error("should not get here");
  return 0;
}

void packetize_frame_impl::process_symbol(gr_complex *in, gr_complex *symbols,
                                          uint8_t *out) {
  static gr_complex current_symbol[64];

  std::memcpy(current_symbol, in, 64 * sizeof(gr_complex));

  // compensate sampling offset
  for (int i = 0; i < 64; i++) {
    current_symbol[i] *=
        exp(gr_complex(0, 2 * M_PI * d_current_symbol * 80 *
                              (d_epsilon0 + d_er) * (i - 32) / 64));
  }

  gr_complex p = equalizer::base::POLARITY[(d_current_symbol - 2) % 127];

  double beta;
  if (d_current_symbol < 2) {
    beta = arg(current_symbol[11] - current_symbol[25] + current_symbol[39] +
               current_symbol[53]);

  } else {
    beta = arg((current_symbol[11] * p) + (current_symbol[39] * p) +
               (current_symbol[25] * p) + (current_symbol[53] * -p));
  }

  double er = arg((conj(d_prev_pilots[0]) * current_symbol[11] * p) +
                  (conj(d_prev_pilots[1]) * current_symbol[25] * p) +
                  (conj(d_prev_pilots[2]) * current_symbol[39] * p) +
                  (conj(d_prev_pilots[3]) * current_symbol[53] * -p));

  er *= d_bw / (2 * M_PI * d_freq * 80);

  if (d_current_symbol < 2) {
    d_prev_pilots[0] = current_symbol[11];
    d_prev_pilots[1] = -current_symbol[25];
    d_prev_pilots[2] = current_symbol[39];
    d_prev_pilots[3] = current_symbol[53];
  } else {
    d_prev_pilots[0] = current_symbol[11] * p;
    d_prev_pilots[1] = current_symbol[25] * p;
    d_prev_pilots[2] = current_symbol[39] * p;
    d_prev_pilots[3] = current_symbol[53] * -p;
  }

  // compensate residual frequency offset
  for (int i = 0; i < 64; i++) {
    current_symbol[i] *= exp(gr_complex(0, -beta));
  }

  // update estimate of residual frequency offset
  if (d_current_symbol >= 2) {

    double alpha = 0.1;
    d_er = (1 - alpha) * d_er + alpha * er;
  }

  // do equalization
  d_equalizer->equalize(current_symbol, d_current_symbol, symbols, out,
                        d_frame_mod);
}

bool packetize_frame_impl::decode_signal_field(uint8_t *rx_bits) {

  static ofdm_param ofdm(BPSK_1_2);
  static frame_param frame(&ofdm, 0);

  deinterleave(rx_bits);
  uint8_t *decoded_bits = d_decoder.decode(&ofdm, &frame, d_deinterleaved);

  return parse_signal(decoded_bits);
}

void packetize_frame_impl::deinterleave(uint8_t *rx_bits) {
  for (int i = 0; i < 48; i++) {
    d_deinterleaved[i] = rx_bits[interleaver_pattern[i]];
  }
}

bool packetize_frame_impl::parse_signal(uint8_t *decoded_bits) {

  int r = 0;
  d_frame_bytes = 0;
  bool parity = false;
  for (int i = 0; i < 17; i++) {
    parity ^= decoded_bits[i];

    if ((i < 4) && decoded_bits[i]) {
      r = r | (1 << i);
    }

    if (decoded_bits[i] && (i > 4) && (i < 17)) {
      d_frame_bytes = d_frame_bytes | (1 << (i - 5));
    }
  }

  if (parity != decoded_bits[17]) {
    dout << "SIGNAL: wrong parity" << std::endl;
    return false;
  }

  switch (r) {
  case 11:
    d_frame_encoding = 0;
    d_frame_symbols = (int)ceil((16 + 8 * d_frame_bytes + 6) / (double)24);
    d_frame_mod = d_bpsk;
    // dout << "Encoding: 3 Mbit/s   ";
    break;
  case 15:
    d_frame_encoding = 1;
    d_frame_symbols = (int)ceil((16 + 8 * d_frame_bytes + 6) / (double)36);
    d_frame_mod = d_bpsk;
    // dout << "Encoding: 4.5 Mbit/s   ";
    break;
  case 10:
    d_frame_encoding = 2;
    d_frame_symbols = (int)ceil((16 + 8 * d_frame_bytes + 6) / (double)48);
    d_frame_mod = d_qpsk;
    // dout << "Encoding: 6 Mbit/s   ";
    break;
  case 14:
    d_frame_encoding = 3;
    d_frame_symbols = (int)ceil((16 + 8 * d_frame_bytes + 6) / (double)72);
    d_frame_mod = d_qpsk;
    // dout << "Encoding: 9 Mbit/s   ";
    break;
  case 9:
    d_frame_encoding = 4;
    d_frame_symbols = (int)ceil((16 + 8 * d_frame_bytes + 6) / (double)96);
    d_frame_mod = d_16qam;
    // dout << "Encoding: 12 Mbit/s   ";
    break;
  case 13:
    d_frame_encoding = 5;
    d_frame_symbols = (int)ceil((16 + 8 * d_frame_bytes + 6) / (double)144);
    d_frame_mod = d_16qam;
    // dout << "Encoding: 18 Mbit/s   ";
    break;
  case 8:
    d_frame_encoding = 6;
    d_frame_symbols = (int)ceil((16 + 8 * d_frame_bytes + 6) / (double)192);
    d_frame_mod = d_64qam;
    // dout << "Encoding: 24 Mbit/s   ";
    break;
  case 12:
    d_frame_encoding = 7;
    d_frame_symbols = (int)ceil((16 + 8 * d_frame_bytes + 6) / (double)216);
    d_frame_mod = d_64qam;
    // dout << "Encoding: 27 Mbit/s   ";
    break;
  default:
    // dout << "unknown encoding" << std::endl;
    return false;
  }

  // mylog(boost::format("encoding: %1% - length: %2% - symbols: %3%") %
  //       d_frame_encoding % d_frame_bytes % d_frame_symbols);
  return true;
}

const int packetize_frame_impl::interleaver_pattern[48] = {
    0, 3, 6, 9,  12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45,
    1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46,
    2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35, 38, 41, 44, 47};

} /* namespace wifigpu */
} /* namespace gr */
