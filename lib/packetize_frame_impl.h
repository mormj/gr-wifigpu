/* -*- c++ -*- */
/*
 * Copyright 2021 Josh Morman.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_WIFIGPU_PACKETIZE_FRAME_IMPL_H
#define INCLUDED_WIFIGPU_PACKETIZE_FRAME_IMPL_H

#include <wifigpu/constellations.h>
#include "equalizer/base.h"
#include "viterbi_decoder/viterbi_decoder.h"

#include <wifigpu/packetize_frame.h>

#include <pmt/pmt.h>

using namespace pmt;

namespace gr {
namespace wifigpu {
enum Equalizer {
	LS   = 0,
	LMS  = 1,
	COMB = 2,
	STA  = 3,
};


class packetize_frame_impl : public packetize_frame {
private:
  pmt_t d_samples;
  pmt_t d_pdu = nullptr;
  pmt_t d_dict;
  void process_symbol(gr_complex *in, gr_complex *symbols, uint8_t *out);
  bool parse_signal(uint8_t *signal);
	bool decode_signal_field(uint8_t *rx_bits);
	void deinterleave(uint8_t *rx_bits);

	equalizer::base *d_equalizer;
	// gr::thread::mutex d_mutex;
	std::vector<gr::tag_t> tags;
	bool d_debug;
	bool d_log;
	int  d_current_symbol;
	viterbi_decoder d_decoder;

	// freq offset
	double d_freq;  // Hz
	double d_freq_offset_from_synclong;  // Hz, estimation from "sync_long" block
	double d_bw;  // Hz
	double d_er;
	double d_epsilon0;
	gr_complex d_prev_pilots[4];

	int  d_frame_bytes;
	int  d_frame_symbols;
	int  d_frame_encoding;

	uint8_t d_deinterleaved[48];
	gr_complex symbols[48];

	std::shared_ptr<gr::digital::constellation> d_frame_mod;
	constellation_bpsk::sptr d_bpsk;
	constellation_qpsk::sptr d_qpsk;
	constellation_16qam::sptr d_16qam;
	constellation_64qam::sptr d_64qam;

static const int interleaver_pattern[48];

  int packet_cnt = 0;

public:
 enum { WAITING_FOR_TAG, FINISH_LAST_FRAME } d_state = WAITING_FOR_TAG;
  void set_algorithm(Equalizer algo);
  packetize_frame_impl(int algo, double freq, double bw);
  ~packetize_frame_impl();

  // Where all the action really happens
  void forecast(int noutput_items, gr_vector_int &ninput_items_required);

  int general_work(int noutput_items, gr_vector_int &ninput_items,
                   gr_vector_const_void_star &input_items,
                   gr_vector_void_star &output_items);
};

} // namespace wifigpu
} // namespace gr

#endif /* INCLUDED_WIFIGPU_PACKETIZE_FRAME_IMPL_H */
