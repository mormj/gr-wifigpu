/* -*- c++ -*- */
/*
 * Copyright 2021 gr-wifigpu author.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "frame_equalizer_impl.h"
#include "equalizer/base.h"
#include "equalizer/comb.h"
#include "equalizer/lms.h"
#include "equalizer/ls.h"
#include "equalizer/sta.h"
#include "utils.h"
#include <gnuradio/io_signature.h>

extern void exec_freq_correction(cuFloatComplex *in, cuFloatComplex *out,
                                 float freq_offset, float start_idx, int n,
                                 int grid_size, int block_size,
                                 cudaStream_t stream);
extern void get_block_and_grid_freq_correction(int *minGrid, int *minBlock);

extern void exec_multiply_const(cuFloatComplex *in, cuFloatComplex *out,
                                cuFloatComplex k, int n, int grid_size,
                                int block_size, cudaStream_t stream);
extern void exec_multiply_phase(cuFloatComplex *in, cuFloatComplex *out,
                                float *beta, int n, int grid_size,
                                int block_size, cudaStream_t stream);
extern void exec_calc_beta_err(cuFloatComplex *in, float *polarity,
                               int current_symbol_index,
                               cuFloatComplex *last_symbol, float bw,
                               float freq, float *beta, float *err, int n,
                               int grid_size, int block_size,
                               cudaStream_t stream);
extern void exec_correct_sampling_offset(cuFloatComplex *in,
                                         cuFloatComplex *out, int start_idx,
                                         float freq_offset, int n,
                                         int grid_size, int block_size,
                                         cudaStream_t stream);

extern void exec_ls_freq_domain_equalization(cuFloatComplex *in,
                                             cuFloatComplex *out,
                                             cuFloatComplex *H, int n,
                                             int grid_size, int block_size,
                                             cudaStream_t stream);

extern void exec_ls_freq_domain_chanest(cuFloatComplex *in, float *training_seq,
                                        cuFloatComplex *H, int grid_size,
                                        int block_size, cudaStream_t stream);

extern void exec_bpsk_decision_maker(cuFloatComplex *in, uint8_t *out, int n,
                                     int grid_size, int block_size,
                                     cudaStream_t stream);

namespace gr {
namespace wifigpu {

frame_equalizer::sptr frame_equalizer::make(int algo, double freq, double bw) {
  return gnuradio::make_block_sptr<frame_equalizer_impl>(algo, freq, bw);
}

/*
 * The private constructor
 */
frame_equalizer_impl::frame_equalizer_impl(int algo, double freq, double bw)
    : gr::block("frame_equalizer",
                gr::io_signature::make(1, 1, 64 * sizeof(gr_complex)),
                gr::io_signature::make(1, 1, 48)),
      d_current_symbol(0), d_equalizer(nullptr), d_freq(freq), d_bw(bw),
      d_frame_bytes(0), d_frame_symbols(0), d_freq_offset_from_synclong(0.0) {

  message_port_register_out(pmt::mp("symbols"));

  d_bpsk = constellation_bpsk::make();
  d_qpsk = constellation_qpsk::make();
  d_16qam = constellation_16qam::make();
  d_64qam = constellation_64qam::make();

  d_frame_mod = d_bpsk;

  set_tag_propagation_policy(block::TPP_DONT);
  set_algorithm((Equalizer)algo);

  get_block_and_grid_freq_correction(&d_min_grid_size, &d_block_size);
  cudaDeviceSynchronize();
  std::cerr << "minGrid: " << d_min_grid_size << ", blockSize: " << d_block_size
            << std::endl;

  cudaStreamCreate(&d_stream);

  // Temporary Buffers
  checkCudaErrors(cudaMalloc((void **)&d_dev_in, d_max_out_buffer));
  checkCudaErrors(cudaMalloc((void **)&d_dev_out, d_max_out_buffer));
  checkCudaErrors(
      cudaMalloc((void **)&d_dev_last_symbol, 64 * sizeof(cuFloatComplex)));
  checkCudaErrors(cudaMalloc((void **)&d_dev_beta,
                             sizeof(float) * (d_max_out_buffer / (8 * 64))));
  checkCudaErrors(cudaMalloc((void **)&d_dev_er,
                             sizeof(float) * (d_max_out_buffer / (8 * 64))));

  checkCudaErrors(cudaMalloc((void **)&d_dev_polarity, 127 * sizeof(float)));
  checkCudaErrors(
      cudaMalloc((void **)&d_dev_long_training, 64 * sizeof(float)));
  checkCudaErrors(cudaMalloc((void **)&d_dev_H, 64 * sizeof(cuFloatComplex)));

  for (int i = 0; i < 127; i++) {
    fPOLARITY[i] = equalizer::base::POLARITY[i].real();
  }

  for (int i = 0; i < 64; i++) {
    fLONG[i] = equalizer::base::LONG[i].real();
  }

  checkCudaErrors(cudaMemcpy(d_dev_polarity, fPOLARITY, 127 * sizeof(float),
                             cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_dev_long_training, fLONG, 64 * sizeof(float),
                             cudaMemcpyHostToDevice));

  cuFloatComplex *d_dev_prev_pilots;
  float *d_dev_beta;
  float *d_dev_err;
  float *d_dev_d_err;

  set_output_multiple(64);
}

void frame_equalizer_impl::set_algorithm(Equalizer algo) {
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

void frame_equalizer_impl::forecast(int noutput_items,
                                    gr_vector_int &ninput_items_required) {
  ninput_items_required[0] = noutput_items;
}

int frame_equalizer_impl::general_work(int noutput_items,
                                       gr_vector_int &ninput_items,
                                       gr_vector_const_void_star &input_items,
                                       gr_vector_void_star &output_items) {

  // gr::thread::scoped_lock lock(d_mutex);

  const gr_complex *in = (const gr_complex *)input_items[0];
  uint8_t *out = (uint8_t *)output_items[0];

  checkCudaErrors(cudaMemcpyAsync(d_dev_in, in,
                                  sizeof(gr_complex) * noutput_items * 64,
                                  cudaMemcpyHostToDevice, d_stream));

  int i = 0;
  int o = 0;
  gr_complex symbols[48];
  gr_complex current_symbol[64];

  auto nread = nitems_read(0);
  auto nwritten = nitems_written(0);
  // dout << "FRAME EQUALIZER: input " << ninput_items[0] << "  output " <<
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
        consume_each(0);
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

      auto gridSize = (symbols_to_process + d_block_size - 1) / d_block_size;

      // // // prior to the tag, we need to finish up the previous frame

      exec_correct_sampling_offset(
          (cuFloatComplex *)d_dev_in, (cuFloatComplex *)d_dev_in,
          d_current_symbol, 2.0 * M_PI * 80 * (d_epsilon0 + d_er),
          64 * symbols_to_process, gridSize, d_block_size, d_stream);

      exec_calc_beta_err((cuFloatComplex *)d_dev_in, d_dev_polarity,
                         d_current_symbol, (cuFloatComplex *)d_dev_last_symbol,
                         d_bw, d_freq, d_dev_beta, d_dev_er, symbols_to_process,
                         gridSize, d_block_size, d_stream);

      exec_multiply_phase(
          (cuFloatComplex *)d_dev_in, (cuFloatComplex *)d_dev_in, d_dev_beta,
          symbols_to_process * 64, gridSize, d_block_size, d_stream);

      exec_ls_freq_domain_equalization(d_dev_in, d_dev_in, d_dev_H,
                                       symbols_to_process * 64, gridSize,
                                       d_block_size, d_stream);

      exec_bpsk_decision_maker(d_dev_in, d_dev_out, symbols_to_process * 48,
                               gridSize, d_block_size, d_stream);

      checkCudaErrors(cudaMemcpyAsync(out, d_dev_out,
                                      symbols_to_process * sizeof(uint8_t) * 48,
                                      cudaMemcpyDeviceToHost, d_stream));

      // float host_beta[symbols_to_process];
      // float host_er[symbols_to_process];

      // cudaMemcpyAsync(host_beta, d_dev_beta,
      // symbols_to_process*sizeof(float),cudaMemcpyDeviceToHost, d_stream);
      // cudaMemcpyAsync(host_er, d_dev_er,
      // symbols_to_process*sizeof(float),cudaMemcpyDeviceToHost, d_stream);
      // cudaStreamSynchronize(d_stream);

      for (int o = 0; o < symbols_to_process; o++) {

        if (d_current_symbol == 3) {
          pmt::pmt_t dict = pmt::make_dict();
          dict = pmt::dict_add(dict, pmt::mp("frame_bytes"),
                               pmt::from_uint64(d_frame_bytes));
          dict = pmt::dict_add(dict, pmt::mp("encoding"),
                               pmt::from_uint64(d_frame_encoding));
          dict = pmt::dict_add(dict, pmt::mp("snr"),
                               pmt::from_double(d_equalizer->get_snr()));
          dict = pmt::dict_add(dict, pmt::mp("freq"), pmt::from_double(d_freq));
          dict = pmt::dict_add(dict, pmt::mp("freq_offset"),
                               pmt::from_double(d_freq_offset_from_synclong));
          add_item_tag(0, nitems_written(0) + o,
                       pmt::string_to_symbol("wifi_start"), dict,
                       pmt::string_to_symbol(alias()));
        }

        // // // update estimate of residual frequency offset
        // // if (d_current_symbol >= 2) {

        // //   double alpha = 0.1;
        // //   d_er = (1 - alpha) * d_er + alpha * er;
        // // }

        d_current_symbol++;
      }

      if (d_current_symbol > d_frame_symbols + 2) {
        d_state = WAITING_FOR_TAG;
      }

      consume(0, symbols_to_consume);
      return symbols_to_process;
    }
  } else {

    if (tags.size()) {
      // new frame

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
        consume(0, frame_start);
        return 0;
      }

      auto gridSize = (3 + d_block_size - 1) / d_block_size;

      exec_correct_sampling_offset((cuFloatComplex *)d_dev_in,
                                   (cuFloatComplex *)d_dev_in, 0,
                                   2.0 * M_PI * 80 * (d_epsilon0 + d_er),
                                   64 * 3, gridSize, d_block_size, d_stream);

      exec_calc_beta_err((cuFloatComplex *)d_dev_in, d_dev_polarity, 0,
                         (cuFloatComplex *)d_dev_last_symbol, d_bw, d_freq,
                         d_dev_beta, d_dev_er, 3, gridSize, d_block_size,
                         d_stream);

      exec_multiply_phase((cuFloatComplex *)d_dev_in,
                          (cuFloatComplex *)d_dev_in, d_dev_beta, 3 * 64,
                          gridSize, d_block_size, d_stream);

      exec_ls_freq_domain_chanest(d_dev_in, d_dev_long_training, d_dev_H,
                                  gridSize, d_block_size, d_stream);

      exec_ls_freq_domain_equalization(d_dev_in + 64 * 2, d_dev_in + 64 * 2,
                                       d_dev_H, 64, gridSize, d_block_size,
                                       d_stream);

      exec_bpsk_decision_maker(d_dev_in, d_dev_out, 3 * 48, gridSize,
                               d_block_size, d_stream);

      static uint8_t signal_field[48];
      checkCudaErrors(cudaMemcpyAsync(signal_field, d_dev_out,
                                      sizeof(uint8_t) * 48,
                                      cudaMemcpyDeviceToHost, d_stream));

      decode_signal_field(signal_field);

      for (int o = 0; o < 3; o++) {
        // // // update estimate of residual frequency offset
        // // if (d_current_symbol >= 2) {

        // //   double alpha = 0.1;
        // //   d_er = (1 - alpha) * d_er + alpha * er;
        // // }

        // if (d_current_symbol > 2) {
        //   o++;
        //   pmt::pmt_t pdu = pmt::make_dict();
        //   message_port_pub(
        //       pmt::mp("symbols"),
        //       pmt::cons(pmt::make_dict(), pmt::init_c32vector(48, symbols)));
        // }

        d_current_symbol++;
      }

      d_state = FINISH_LAST_FRAME;

      consume(0, 3);
      return 0;

    } else {
      consume(0, ninput_items[0]);
      return 0;
    }
  }
} // namespace wifigpu

bool frame_equalizer_impl::decode_signal_field(uint8_t *rx_bits) {

  static ofdm_param ofdm(BPSK_1_2);
  static frame_param frame(ofdm, 0);

  deinterleave(rx_bits);
  uint8_t *decoded_bits = d_decoder.decode(&ofdm, &frame, d_deinterleaved);

  return parse_signal(decoded_bits);
}

void frame_equalizer_impl::deinterleave(uint8_t *rx_bits) {
  for (int i = 0; i < 48; i++) {
    d_deinterleaved[i] = rx_bits[interleaver_pattern[i]];
  }
}

bool frame_equalizer_impl::parse_signal(uint8_t *decoded_bits) {

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

const int frame_equalizer_impl::interleaver_pattern[48] = {
    0, 3, 6, 9,  12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45,
    1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46,
    2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35, 38, 41, 44, 47};

} // namespace wifigpu
} /* namespace gr */
