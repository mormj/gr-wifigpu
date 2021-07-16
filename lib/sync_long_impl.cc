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

#include "sync_long_impl.h"
#include <gnuradio/io_signature.h>
#include <helper_cuda.h>

extern void exec_multiply_kernel_ccc(cuFloatComplex *in1, cuFloatComplex *in2,
                                     cuFloatComplex *out, int n, int grid_size,
                                     int block_size, cudaStream_t stream);

extern void get_block_and_grid_multiply(int *minGrid, int *minBlock);

namespace gr {
namespace wifigpu {

sync_long::sptr sync_long::make(unsigned int sync_length) {
  return gnuradio::get_initial_sptr(new sync_long_impl(sync_length));
}

/*
 * The private constructor
 */
sync_long_impl::sync_long_impl(unsigned int sync_length)
    : gr::block("sync_long", gr::io_signature::make(2, 2, sizeof(gr_complex)),
                gr::io_signature::make(1, 1, sizeof(gr_complex))),
      d_sync_length(sync_length) {
  // set_output_multiple(d_fftsize); // make sure the fft size is sufficient for
  //                                 // freq domain convolution

  cudaStreamCreate(&d_stream);

  checkCudaErrors(cufftCreate(&d_plan));
  checkCudaErrors(cufftSetStream(d_plan, d_stream));

  size_t workSize;
  checkCudaErrors(cufftMakePlanMany(d_plan, 1, &d_fftsize, NULL, 1, 1, NULL, 1,
                                    1, CUFFT_C2C, 1, &workSize));

  checkCudaErrors(cudaMalloc((void **)&d_dev_training_freq,
                             sizeof(cufftComplex) * d_fftsize * 1));
  checkCudaErrors(
      cudaMalloc((void **)&d_dev_in, sizeof(cufftComplex) * d_fftsize * 1));
  // checkCudaErrors(cudaMalloc((void **)&d_dev_tail,
  //                            sizeof(float) * d_fftsize * 1));

  // d_xformed_taps.resize(d_fftsize);

  // Frequency domain the Training Sequency
  checkCudaErrors(
      cudaMemset(d_dev_training_freq, 0, d_fftsize * sizeof(cufftComplex)));

  checkCudaErrors(cudaMemcpy(d_dev_training_freq, &LONG[0],
                             64 * sizeof(cufftComplex),
                             cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_dev_training_freq + 64, &LONG[0],
                             64 * sizeof(cufftComplex),
                             cudaMemcpyHostToDevice));

  checkCudaErrors(cufftExecC2C(d_plan, d_dev_training_freq, d_dev_training_freq,
                               CUFFT_FORWARD));

  get_block_and_grid_multiply(&d_min_grid_size, &d_block_size);

  set_tag_propagation_policy(TPP_DONT);

  set_output_multiple(4096);
}

/*
 * Our virtual destructor.
 */
sync_long_impl::~sync_long_impl() {}

void sync_long_impl::forecast(int noutput_items,
                              gr_vector_int &ninput_items_required) {

  auto nreq = noutput_items;
  if (nreq < d_fftsize)
    nreq = d_fftsize;

  ninput_items_required[0] = nreq;
  ninput_items_required[1] = nreq;
}

int sync_long_impl::general_work(int noutput_items, gr_vector_int &ninput_items,
                                 gr_vector_const_void_star &input_items,
                                 gr_vector_void_star &output_items) {
  const gr_complex *in = (const gr_complex *)input_items[0];
  const gr_complex *in_delayed = (const gr_complex *)input_items[1];
  gr_complex *out = (gr_complex *)output_items[0];

  // Look at the tags
  int ninput = std::min(ninput_items[0], ninput_items[1]);
  // int ninput = std::min(std::min(ninput_items[0], ninput_items[1]), 8192);
  const uint64_t nread = nitems_read(0);
  const uint64_t nwritten = nitems_written(0);
  get_tags_in_range(d_tags, 0, nread, nread + ninput);
  int nconsumed = 0;
  int nproduced = 0;

  if (d_tags.size() > 1) {
    volatile int x = 7;
  }

  // std::cout << "work: " << ninput << " " << noutput_items;

  // if (0)
  { // finish copying up to the next tag
    auto max_inputs = ninput;
    if (d_tags.size()) {
      // std::cout << "   There is a tag in this work() call " << std::endl;
      // find the next tag
      // for (auto &t : d_tags) {
      if (d_tags[0].offset >= (nread + nconsumed)) {

        max_inputs = d_tags[0].offset - nread;
        // std::cout << "       " << max_inputs << "/" << ninput <<  std::endl;
        // break;
      }
      // }

      // std::cout << " " << max_inputs << " " << d_tags[0].offset << std::endl;
    }

    // std::cout << std::endl;

    if (d_state == COPY) {
      while (nconsumed + 80 <= max_inputs &&
             nproduced + 64 <=
                 noutput_items) { // or until i hit the next tag!!!!
        memcpy(out + nproduced, in + nconsumed + 16,
               sizeof(gr_complex) * 64); // throw away the cyclic prefix
        // std::cout << "sym2 " << d_num_syms << " " << nread + nconsumed << " "
        //           << nwritten + nproduced << std::endl;
        nproduced += 64;
        nconsumed += 80;

        d_num_syms++;
      }
      // if (nproduced != 0 && nconsumed != 0)
      // std::cout << "copy2: " << nproduced << "/" << nconsumed << "/" << nread
      // << "/" << nwritten +nproduced<< "/" << ninput << "/" << noutput_items
      // << " --- " << d_num_syms  << std::endl;
    }
  }

  if (d_tags.size()) {
    int tag_idx = 0;
    for (auto &t : d_tags) {
      auto offset = t.offset;
      if (offset >= nread) {
        d_freq_offset_short = pmt::to_double(t.value);

        if (offset - nread + d_fftsize <= ninput && noutput_items >= 128 &&
            ((d_state == SYNC) ||
             (d_state == COPY && (offset - nread - nconsumed < 64)))) {

#if 0
          checkCudaErrors(cudaMemcpyAsync(d_dev_in, &in[offset - nread],
                                          d_fftsize * sizeof(cufftComplex),
                                          cudaMemcpyHostToDevice, d_stream));

          checkCudaErrors(
              cufftExecC2C(d_plan, d_dev_in, d_dev_in, CUFFT_FORWARD));

          exec_multiply_kernel_ccc(d_dev_in, d_dev_training_freq, d_dev_in,
                                   d_fftsize, d_min_grid_size, d_block_size,
                                   d_stream);

          checkCudaErrors(
              cufftExecC2C(d_plan, d_dev_in, d_dev_in, CUFFT_INVERSE));

          // Find the peak
          std::vector<gr_complex> host_data(d_fftsize);
          checkCudaErrors(cudaMemcpyAsync(host_data.data(), d_dev_in,
                                          d_fftsize * sizeof(cufftComplex),
                                          cudaMemcpyDeviceToHost, d_stream));

          cudaStreamSynchronize(d_stream);
          std::vector<float> abs_corr(d_fftsize);
          // std::cout << "freq_corr = [";

          size_t max_index = 0;
          float max_value = 0.0;
          for (size_t i = 0; i < host_data.size(); i++) {
            abs_corr[i] = std::abs(host_data[i]);
            // std::cout << abs_corr[i];
            // if (i < host_data.size()-1)
            // std::cout << ",";

            if (abs_corr[i] > max_value) {
              max_value = abs_corr[i];
              max_index = i;
            }
          }
#endif

          size_t max_index = 297;
          // nproduced += d_fftsize;

          // Copy the LTF symbols
          // std::cout << max_index << std::endl;
          memcpy(out + nproduced,
                 in + (offset - nread + max_index - 160 + 32 + 1),
                 sizeof(gr_complex) * 128);

          const pmt::pmt_t key = pmt::string_to_symbol("wifi_start");
          const pmt::pmt_t value = // pmt::from_long(max_index);
              pmt::from_double(d_freq_offset_short - d_freq_offset);
          const pmt::pmt_t srcid = pmt::string_to_symbol(name());
          // add_item_tag(0, nwritten+nproduced+max_index, key, value, srcid);
          add_item_tag(0, nwritten + nproduced, key, value, srcid);
          // std::cout << "  *********** tag on item: " << nwritten+nproduced <<
          // " --- " << d_num_syms <<std::endl; std::cout << "ntags: " <<
          // ++ntags << std::endl; if (std::abs(out[nproduced] -
          // gr_complex(0.83205323,-1.66078582e-8)) > 0.0001 )
          // {
          //   volatile int x = 123;

          // }
          if (d_num_syms != 514)
          {
            volatile int x = 7;
          }

          d_num_syms = 0;
          std::cout << "tag0 " << d_num_syms << " "
                    << nread + (offset - nread + max_index - 160 + 32 + 1)
                    << " " << nwritten + nproduced << " " << nread << " "
                    << offset << std::endl;
          nproduced += 128;

          d_state = COPY;

          nconsumed = (offset - nread + max_index - 160 + 32 + 128 + 1);


          uint64_t nn = nread+nconsumed-298;
          if (nn % 41481 != 0)
          {
            volatile int x = 7;
          }

          // finish copying this burst
          {
            auto max_inputs = ninput;
            while (nconsumed + 80 <= max_inputs &&
                   nproduced + 64 <=
                       noutput_items) { // or until i hit the next tag!!!!

              memcpy(out + nproduced, in + nconsumed + 16,
                     sizeof(gr_complex) * 64); // throw away the cyclic prefix
              // std::cout << "sym1 " << d_num_syms << " " << nread + nconsumed
              //           << " " << nwritten + nproduced << std::endl;
              nproduced += 64;
              nconsumed += 80;
              d_num_syms++;
            }
            // std::cout << "copy1: " << nproduced << "/" << nconsumed << "/" <<
            // nread << "/" << nwritten << "/" << ninput << "/" << noutput_items
            // << " --- " << d_num_syms  << std::endl;
          }

        } else if (d_state == SYNC) {
          nconsumed = offset - nread;
        }
      }

      tag_idx++;
      break;
    }
  } else if (d_state == SYNC) {

    // If no tags and not currently processing a potential burst,
    //   then consume all the input and move on

    nconsumed = ninput;
    nproduced = 0;
  }

  assert((nwritten + nproduced) % 64 == 0);
  cudaStreamSynchronize(d_stream);
  consume_each(nconsumed);
  // Tell runtime system how many output items we produced.
  return nproduced;
}

const std::vector<gr_complex> sync_long_impl::LONG = {

    gr_complex(-0.0455, -1.0679), gr_complex(0.3528, -0.9865),
    gr_complex(0.8594, 0.7348),   gr_complex(0.1874, 0.2475),
    gr_complex(0.5309, -0.7784),  gr_complex(-1.0218, -0.4897),
    gr_complex(-0.3401, -0.9423), gr_complex(0.8657, -0.2298),
    gr_complex(0.4734, 0.0362),   gr_complex(0.0088, -1.0207),
    gr_complex(-1.2142, -0.4205), gr_complex(0.2172, -0.5195),
    gr_complex(0.5207, -0.1326),  gr_complex(-0.1995, 1.4259),
    gr_complex(1.0583, -0.0363),  gr_complex(0.5547, -0.5547),
    gr_complex(0.3277, 0.8728),   gr_complex(-0.5077, 0.3488),
    gr_complex(-1.1650, 0.5789),  gr_complex(0.7297, 0.8197),
    gr_complex(0.6173, 0.1253),   gr_complex(-0.5353, 0.7214),
    gr_complex(-0.5011, -0.1935), gr_complex(-0.3110, -1.3392),
    gr_complex(-1.0818, -0.1470), gr_complex(-1.1300, -0.1820),
    gr_complex(0.6663, -0.6571),  gr_complex(-0.0249, 0.4773),
    gr_complex(-0.8155, 1.0218),  gr_complex(0.8140, 0.9396),
    gr_complex(0.1090, 0.8662),   gr_complex(-1.3868, -0.0000),
    gr_complex(0.1090, -0.8662),  gr_complex(0.8140, -0.9396),
    gr_complex(-0.8155, -1.0218), gr_complex(-0.0249, -0.4773),
    gr_complex(0.6663, 0.6571),   gr_complex(-1.1300, 0.1820),
    gr_complex(-1.0818, 0.1470),  gr_complex(-0.3110, 1.3392),
    gr_complex(-0.5011, 0.1935),  gr_complex(-0.5353, -0.7214),
    gr_complex(0.6173, -0.1253),  gr_complex(0.7297, -0.8197),
    gr_complex(-1.1650, -0.5789), gr_complex(-0.5077, -0.3488),
    gr_complex(0.3277, -0.8728),  gr_complex(0.5547, 0.5547),
    gr_complex(1.0583, 0.0363),   gr_complex(-0.1995, -1.4259),
    gr_complex(0.5207, 0.1326),   gr_complex(0.2172, 0.5195),
    gr_complex(-1.2142, 0.4205),  gr_complex(0.0088, 1.0207),
    gr_complex(0.4734, -0.0362),  gr_complex(0.8657, 0.2298),
    gr_complex(-0.3401, 0.9423),  gr_complex(-1.0218, 0.4897),
    gr_complex(0.5309, 0.7784),   gr_complex(0.1874, -0.2475),
    gr_complex(0.8594, -0.7348),  gr_complex(0.3528, 0.9865),
    gr_complex(-0.0455, 1.0679),  gr_complex(1.3868, -0.0000),

};

} /* namespace wifigpu */
} /* namespace gr */
