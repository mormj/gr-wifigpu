/* -*- c++ -*- */
/*
 * Copyright 2021 gr-wifigpu author.
 *
 * This is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3, or (at your option)
 * any later version.
 *
 * This software is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this software; see the file COPYING.  If not, write to
 * the Free Software Foundation, Inc., 51 Franklin Street,
 * Boston, MA 02110-1301, USA.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "sync_short_impl.h"
#include <gnuradio/io_signature.h>

namespace gr {
namespace wifigpu {

sync_short::sptr sync_short::make(float threshold, int min_plateau) {
  return gnuradio::get_initial_sptr(
      new sync_short_impl(threshold, min_plateau));
}

/*
 * The private constructor
 */
sync_short_impl::sync_short_impl(float threshold, int min_plateau)
    : gr::block("sync_short",
                gr::io_signature::make3(3, 3, sizeof(gr_complex),
                                        sizeof(gr_complex), sizeof(float)),
                gr::io_signature::make3(1, 3, sizeof(gr_complex), sizeof(uint8_t), sizeof(uint8_t))),
      d_threshold(threshold), d_min_plateau(min_plateau) {
  set_history(d_min_plateau);
}

/*
 * Our virtual destructor.
 */
sync_short_impl::~sync_short_impl() {}

void sync_short_impl::forecast(int noutput_items,
                               gr_vector_int &ninput_items_required) {
  ninput_items_required[0] = noutput_items;

  above_threshold.resize(8192);
  accum.resize(8192);
}

int sync_short_impl::general_work(int noutput_items,
                                  gr_vector_int &ninput_items,
                                  gr_vector_const_void_star &input_items,
                                  gr_vector_void_star &output_items) {
  const gr_complex *in = (const gr_complex *)input_items[0];
  const gr_complex *in_abs = (const gr_complex *)input_items[1];
  const float *in_cor = (const float *)input_items[2];
  gr_complex *out = (gr_complex *)output_items[0];
  // uint8_t *out_plateau = (uint8_t *)output_items[1];
  // uint8_t *out_accum = (uint8_t *)output_items[2];

  int h = history() - 1;
  if (noutput_items > above_threshold.size()) {
    above_threshold.resize(noutput_items + h);
    accum.resize(noutput_items);
  }

  for (int i = 0; i < noutput_items + h; i++) {
    above_threshold[i] = in_cor[i] > d_threshold;
  }

  accum[0] = 0;
  for (int j = 0; j < h + 1; j++) {
    if (above_threshold[j]) {
      accum[0]++;
    } else {
      accum[0] = 0;
    }
  }

  auto nread = nitems_read(0);
  auto nwritten = nitems_written(0);

  for (int i = 1; i < noutput_items; i++) {
    if (above_threshold[i]) {
      accum[i] = accum[i - 1] + 1;

      if (accum[i] >= d_min_plateau &&
          nread + i - d_last_tag_location > MIN_GAP) {
        d_last_tag_location = nread + i;
        d_freq_offset = arg(in_abs[i]) / 16;
        insert_tag(nwritten + i, d_freq_offset, nread + i);
      }

    } else {
      accum[i] = 0;
    }
  }

  for (int o = 0; o < noutput_items; o++) {
    out[o] = in[o+h] * exp(gr_complex(0, -d_freq_offset * (nwritten+o)));
  }

  // memcpy(out_plateau, above_threshold.data(), noutput_items);
  // memcpy(out_accum, accum.data(), noutput_items);

  // Tell runtime system how many input items we consumed on
  // each input stream.
  consume_each(noutput_items);

  // Tell runtime system how many output items we produced.
  return noutput_items;
}

void sync_short_impl::insert_tag(uint64_t item, double freq_offset, uint64_t input_item) {
  // mylog(boost::format("frame start at in: %2% out: %1%") % item %
  // input_item);

  const pmt::pmt_t key = pmt::string_to_symbol("wifi_start");
  const pmt::pmt_t value = pmt::from_double(freq_offset);
  const pmt::pmt_t srcid = pmt::string_to_symbol(name());
  add_item_tag(0, item, key, value, srcid);
}

} /* namespace wifigpu */
} // namespace gr
