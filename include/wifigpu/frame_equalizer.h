/* -*- c++ -*- */
/*
 * Copyright 2021 gr-wifigpu author.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_WIFIGPU_FRAME_EQUALIZER_H
#define INCLUDED_WIFIGPU_FRAME_EQUALIZER_H

#include <gnuradio/block.h>
#include <wifigpu/api.h>

enum Equalizer {
	LS   = 0,
	LMS  = 1,
	COMB = 2,
	STA  = 3,
};


namespace gr {
namespace wifigpu {


/*!
 * \brief <+description of block+>
 * \ingroup wifigpu
 *
 */
class WIFIGPU_API frame_equalizer : virtual public gr::block {
public:
  typedef std::shared_ptr<frame_equalizer> sptr;

  /*!
   * \brief Return a shared_ptr to a new instance of wifigpu::frame_equalizer.
   *
   * To avoid accidental use of raw pointers, wifigpu::frame_equalizer's
   * constructor is in a private implementation
   * class. wifigpu::frame_equalizer::make is the public interface for
   * creating new instances.
   */
  static sptr make(int algo, double freq, double bw);
};

} // namespace wifigpu
} // namespace gr

#endif /* INCLUDED_WIFIGPU_FRAME_EQUALIZER_H */
