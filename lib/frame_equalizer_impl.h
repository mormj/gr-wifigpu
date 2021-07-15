/* -*- c++ -*- */
/*
 * Copyright 2021 gr-wifigpu author.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_WIFIGPU_FRAME_EQUALIZER_IMPL_H
#define INCLUDED_WIFIGPU_FRAME_EQUALIZER_IMPL_H

#include <wifigpu/frame_equalizer.h>
#include <wifigpu/constellations.h>
#include "equalizer/base.h"
#include "viterbi_decoder/viterbi_decoder.h"

namespace gr {
namespace wifigpu {

class frame_equalizer_impl : public frame_equalizer {
private:
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

public:
  frame_equalizer_impl(int algo, double freq, double bw);

  // Where all the action really happens
  void forecast(int noutput_items, gr_vector_int &ninput_items_required);

  int general_work(int noutput_items, gr_vector_int &ninput_items,
                   gr_vector_const_void_star &input_items,
                   gr_vector_void_star &output_items);

	void set_algorithm(Equalizer algo);
	void set_bandwidth(double bw);
	void set_frequency(double freq);
};

} // namespace wifigpu
} // namespace gr

#endif /* INCLUDED_WIFIGPU_FRAME_EQUALIZER_IMPL_H */
