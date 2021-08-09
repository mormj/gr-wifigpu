#include <volk/volk.h>

#include "equalizer/base.h"
#include "utils.h"
#include "viterbi_decoder/viterbi_decoder.h"
#include <boost/crc.hpp>
#include <pmt/pmt.h>
#include <vector>
#include <wifigpu/constellations.h>
#include <wifigpu/mapper.h>

#include "equalizer/base.h"
#include "equalizer/comb.h"
#include "equalizer/lms.h"
#include "equalizer/ls.h"
#include "equalizer/sta.h"

using namespace std;

namespace gr {
namespace wifigpu {

class burst_worker {
private:
  int m_id;
  int packet_cnt;
  bool d_debug = false;
  bool d_log = false;
  gr::logger_ptr d_logger;

  frame_param d_frame;
  ofdm_param d_ofdm;

  viterbi_decoder d_decoder;

  uint8_t d_rx_symbols[48 * MAX_SYM];
  uint8_t d_rx_bits[MAX_ENCODED_BITS];
  uint8_t d_deinterleaved_bits[MAX_ENCODED_BITS];
  uint8_t out_bytes[MAX_PSDU_SIZE + 2]; // 2 for signal field

  int d_frame_bytes = 0;
  int d_frame_symbols = 0;
  int d_frame_encoding = 0;

  uint8_t d_deinterleaved[48];
  gr_complex symbols[48];

  std::shared_ptr<gr::digital::constellation> d_frame_mod;
  constellation_bpsk::sptr d_bpsk;
  constellation_qpsk::sptr d_qpsk;
  constellation_16qam::sptr d_16qam;
  constellation_64qam::sptr d_64qam;

  equalizer::base *d_equalizer;
  gr_complex d_prev_pilots[4];

  double d_freq;
  double d_bw;
  double d_freq_offset;

  const int interleaver_pattern[48] = {
      0, 3, 6, 9,  12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45,
      1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46,
      2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35, 38, 41, 44, 47};

  void decode(uint8_t *rx_bits, uint8_t *rx_symbols,
              uint8_t *deinterleaved_bits, uint8_t *out_bytes,
              viterbi_decoder &decoder, frame_param &frame_info,
              ofdm_param &ofdm_info) {

    for (int i = 0; i < frame_info.n_sym * 48; i++) {
      for (int k = 0; k < ofdm_info.n_bpsc; k++) {
        rx_bits[i * ofdm_info.n_bpsc + k] = !!(rx_symbols[i] & (1 << k));
      }
    }

    deinterleave(rx_bits, deinterleaved_bits, frame_info.n_sym, ofdm_info);
    uint8_t *decoded =
        decoder.decode(&ofdm_info, &frame_info, deinterleaved_bits);

    descramble(frame_info.psdu_size, decoded, out_bytes);
    print_output(out_bytes, frame_info.psdu_size);

    // skip service field
    boost::crc_32_type result;
    result.process_bytes(out_bytes + 2, frame_info.psdu_size);
    if (result.checksum() != 558161692) {
      dout << "checksum wrong -- dropping" << std::endl;
      return;
    }
    else
    {
      crc_cnt++;
      if (crc_cnt % 1000 == 0)
      {
        std::cout << "burst_worker:" <<  crc_cnt << std::endl;
      }
    }

    mylog(boost::format("encoding: %1% - length: %2% - symbols: %3%") %
          ofdm_info.encoding % frame_info.psdu_size % frame_info.n_sym);

    // // create PDU
    // pmt::pmt_t blob = pmt::make_blob(out_bytes + 2, frame_info.psdu_size -
    // 4); d_meta = pmt::dict_add(d_meta, pmt::mp("dlt"),
    //                        pmt::from_long(LINKTYPE_IEEE802_11));

    // message_port_pub(pmt::mp("out"), pmt::cons(d_meta, blob));
  }

  void deinterleave(uint8_t *rx_bits, uint8_t *deinterleaved_bits, size_t n_sym,
                    ofdm_param &ofdm_info) {

    int n_cbps = ofdm_info.n_cbps;
    int first[n_cbps];
    int second[n_cbps];
    int s = std::max(ofdm_info.n_bpsc / 2, 1);

    for (int j = 0; j < n_cbps; j++) {
      first[j] = s * (j / s) + ((j + int(floor(16.0 * j / n_cbps))) % s);
    }

    for (int i = 0; i < n_cbps; i++) {
      second[i] = 16 * i - (n_cbps - 1) * int(floor(16.0 * i / n_cbps));
    }

    int count = 0;
    for (int i = 0; i < n_sym; i++) {
      for (int k = 0; k < n_cbps; k++) {
        deinterleaved_bits[i * n_cbps + second[first[k]]] =
            rx_bits[i * n_cbps + k];
      }
    }
  }

  void descramble(size_t psdu_size, uint8_t *decoded_bits, uint8_t *out_bytes) {

    int state = 0;
    std::memset(out_bytes, 0, psdu_size + 2);

    for (int i = 0; i < 7; i++) {
      if (decoded_bits[i]) {
        state |= 1 << (6 - i);
      }
    }
    out_bytes[0] = state;

    int feedback;
    int bit;

    for (int i = 7; i < psdu_size * 8 + 16; i++) {
      feedback = ((!!(state & 64))) ^ (!!(state & 8));
      bit = feedback ^ (decoded_bits[i] & 0x1);
      out_bytes[i / 8] |= bit << (i % 8);
      state = ((state << 1) & 0x7e) | feedback;
    }
  }

  void print_output(uint8_t *out_bytes, size_t psdu_size) {

    dout << std::endl;
    dout << "psdu size" << psdu_size << std::endl;
    for (int i = 2; i < psdu_size + 2; i++) {
      dout << std::setfill('0') << std::setw(2) << std::hex
           << ((unsigned int)out_bytes[i] & 0xFF) << std::dec << " ";
      if (i % 16 == 15) {
        dout << std::endl;
      }
    }
    dout << std::endl;
    for (int i = 2; i < psdu_size + 2; i++) {
      if ((out_bytes[i] > 31) && (out_bytes[i] < 127)) {
        dout << ((char)out_bytes[i]);
      } else {
        dout << ".";
      }
    }
    dout << std::endl;
  }

  void equalize_frame(const gr_complex *symbols, uint8_t *demapped_symbols) {

    double d_freq_offset_from_synclong = d_freq_offset * d_bw / (2 * M_PI);
    double d_epsilon0 = d_freq_offset * d_bw / (2 * M_PI * d_freq);
    double d_er = 0.0;

    static gr_complex current_symbol[64];
    static gr_complex tmp_symbols[48];

    for (size_t i = 0; i < d_frame_symbols; i++) {
      size_t d_current_symbol = i + 3;

      std::memcpy(current_symbol, symbols + i*64, 64 * sizeof(gr_complex));

      // compensate sampling offset
      for (int i = 0; i < 64; i++) {
        current_symbol[i] *=
            exp(gr_complex(0, 2 * M_PI * d_current_symbol * 80 *
                                  (d_epsilon0 + d_er) * (i - 32) / 64));
      }

      gr_complex p = equalizer::base::POLARITY[(d_current_symbol - 2) % 127];

      double beta;
      if (d_current_symbol < 2) {
        beta = arg(current_symbol[11] - current_symbol[25] +
                   current_symbol[39] + current_symbol[53]);

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
      d_equalizer->equalize(current_symbol, d_current_symbol, tmp_symbols,
                            demapped_symbols + i * 48, d_frame_mod);
    }

    volatile int x = 7;
  }

  // bool decode_signal_field(uint8_t *rx_bits) {

  //   static ofdm_param ofdm(BPSK_1_2);
  //   static frame_param frame(&ofdm, 0);

  //   deinterleave_signal(rx_bits);
  //   uint8_t *decoded_bits = d_decoder.decode(&ofdm, &frame, d_deinterleaved);

  //   return parse_signal(decoded_bits);
  // }

  // void deinterleave_signal(uint8_t *rx_bits) {
  //   for (int i = 0; i < 48; i++) {
  //     d_deinterleaved[i] = rx_bits[interleaver_pattern[i]];
  //   }
  // }

  // bool parse_signal(uint8_t *decoded_bits) {

  //   int r = 0;
  //   d_frame_bytes = 0;
  //   bool parity = false;
  //   for (int i = 0; i < 17; i++) {
  //     parity ^= decoded_bits[i];

  //     if ((i < 4) && decoded_bits[i]) {
  //       r = r | (1 << i);
  //     }

  //     if (decoded_bits[i] && (i > 4) && (i < 17)) {
  //       d_frame_bytes = d_frame_bytes | (1 << (i - 5));
  //     }
  //   }

  //   if (parity != decoded_bits[17]) {
  //     dout << "SIGNAL: wrong parity" << std::endl;
  //     return false;
  //   }

  //   switch (r) {
  //   case 11:
  //     d_frame_encoding = 0;
  //     d_frame_symbols = (int)ceil((16 + 8 * d_frame_bytes + 6) / (double)24);
  //     d_frame_mod = d_bpsk;
  //     // dout << "Encoding: 3 Mbit/s   ";
  //     break;
  //   case 15:
  //     d_frame_encoding = 1;
  //     d_frame_symbols = (int)ceil((16 + 8 * d_frame_bytes + 6) / (double)36);
  //     d_frame_mod = d_bpsk;
  //     // dout << "Encoding: 4.5 Mbit/s   ";
  //     break;
  //   case 10:
  //     d_frame_encoding = 2;
  //     d_frame_symbols = (int)ceil((16 + 8 * d_frame_bytes + 6) / (double)48);
  //     d_frame_mod = d_qpsk;
  //     // dout << "Encoding: 6 Mbit/s   ";
  //     break;
  //   case 14:
  //     d_frame_encoding = 3;
  //     d_frame_symbols = (int)ceil((16 + 8 * d_frame_bytes + 6) / (double)72);
  //     d_frame_mod = d_qpsk;
  //     // dout << "Encoding: 9 Mbit/s   ";
  //     break;
  //   case 9:
  //     d_frame_encoding = 4;
  //     d_frame_symbols = (int)ceil((16 + 8 * d_frame_bytes + 6) / (double)96);
  //     d_frame_mod = d_16qam;
  //     // dout << "Encoding: 12 Mbit/s   ";
  //     break;
  //   case 13:
  //     d_frame_encoding = 5;
  //     d_frame_symbols = (int)ceil((16 + 8 * d_frame_bytes + 6) /
  //     (double)144); d_frame_mod = d_16qam;
  //     // dout << "Encoding: 18 Mbit/s   ";
  //     break;
  //   case 8:
  //     d_frame_encoding = 6;
  //     d_frame_symbols = (int)ceil((16 + 8 * d_frame_bytes + 6) /
  //     (double)192); d_frame_mod = d_64qam;
  //     // dout << "Encoding: 24 Mbit/s   ";
  //     break;
  //   case 12:
  //     d_frame_encoding = 7;
  //     d_frame_symbols = (int)ceil((16 + 8 * d_frame_bytes + 6) /
  //     (double)216); d_frame_mod = d_64qam;
  //     // dout << "Encoding: 27 Mbit/s   ";
  //     break;
  //   default:
  //     // dout << "unknown encoding" << std::endl;
  //     return false;
  //   }

  //   // mylog(boost::format("encoding: %1% - length: %2% - symbols: %3%") %
  //   //       d_frame_encoding % d_frame_bytes % d_frame_symbols);
  //   return true;
  // }

public:
int crc_cnt = 0;
  burst_worker(int id) : d_ofdm(BPSK_1_2), d_frame(&d_ofdm, 0) {
    m_id = id;
    packet_cnt = 0;

    d_equalizer = new equalizer::ls();

    d_bpsk = constellation_bpsk::make();
    d_qpsk = constellation_qpsk::make();
    d_16qam = constellation_16qam::make();
    d_64qam = constellation_64qam::make();
  }

  ~burst_worker() {
    std::cout << "burst_worker:" <<  crc_cnt << std::endl;
  }

  pmt::pmt_t process(pmt::pmt_t msg) {
    pmt::pmt_t meta(pmt::car(msg));
    pmt::pmt_t data(pmt::cdr(msg));

    size_t num_pdu_samples = pmt::length(data);
    size_t len_bytes(0);

    const gr_complex *samples =
        (const gr_complex *)pmt::c32vector_elements(data, len_bytes);

    if (!samples) {
      std::cout << "ERROR: Invalid input type - must be a PMT vector"
                << std::endl;
      return pmt::PMT_NIL;
    }

    d_frame_bytes = pmt::to_uint64(
        pmt::dict_ref(meta, pmt::mp("frame_bytes"), pmt::PMT_NIL));
    d_frame_symbols = pmt::to_uint64(
        pmt::dict_ref(meta, pmt::mp("frame_symbols"), pmt::PMT_NIL));
    d_frame_encoding =
        pmt::to_uint64(pmt::dict_ref(meta, pmt::mp("encoding"), pmt::PMT_NIL));
    d_bw = pmt::to_double(pmt::dict_ref(meta, pmt::mp("bw"), pmt::PMT_NIL));
    d_freq = pmt::to_double(
        pmt::dict_ref(meta, pmt::mp("freq"), pmt::from_double(2412000000)));
    d_freq_offset = pmt::to_double(
        pmt::dict_ref(meta, pmt::mp("freq_offset"), pmt::from_double(0.0)));

    switch (d_frame_encoding) {
    case 0:
    case 1:
      d_frame_mod = d_bpsk;
      break;
    case 2:
    case 3:
      d_frame_mod = d_qpsk;
      break;
    case 4:
    case 5:
      d_frame_mod = d_16qam;
      break;
    case 6:
    case 7:
      d_frame_mod = d_64qam;
      break;
    default:
      throw new std::runtime_error("invalid encoding");
    }

    size_t len_h, len_prev;
    const gr_complex *H = pmt::c32vector_elements(
        pmt::dict_ref(meta, pmt::mp("H"), pmt::PMT_NIL), len_h);
    const gr_complex *tmp_prev_pilots = pmt::c32vector_elements(
        pmt::dict_ref(meta, pmt::mp("prev_pilots"), pmt::PMT_NIL), len_prev);
    memcpy(d_prev_pilots, tmp_prev_pilots, 4 * sizeof(gr_complex));
    d_equalizer->set_H(H);

    equalize_frame(samples, d_rx_symbols);


    d_ofdm = ofdm_param((Encoding)d_frame_encoding);
    d_frame = frame_param(&d_ofdm, d_frame_bytes);

    // need to equalize and demap the samples to d_rx_bits
    decode(d_rx_bits, d_rx_symbols, d_deinterleaved_bits, out_bytes, d_decoder,
           d_frame, d_ofdm);

    // Insert MAC Decode code here
    // std::cout << "Threadpool got new burst" << std::endl;

    // repackage as pmt and place on output queue
    // pmt::pmt_t vecpmt(pmt::init_c32vector(nproduced, &buffer1[0]));
    // pmt::pmt_t pdu(pmt::cons(pmt::PMT_NIL, vecpmt));
    pmt::pmt_t pdu = pmt::PMT_NIL;

    this->packet_cnt++;
    // std::cout << "worker: " << packet_cnt << std::endl;

    return pdu;
  }
}; // namespace wifigpu
} // namespace wifigpu
} // namespace gr