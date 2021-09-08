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

#define USE_CUSTOM_BUFFERS 0

#include "sync_long_impl.h"
#include <cuda_buffer/cuda_buffer.h>
#include <gnuradio/io_signature.h>
#include <helper_cuda.h>

extern void exec_multiply_kernel_ccc(cuFloatComplex *in1, cuFloatComplex *in2,
                                     cuFloatComplex *out, int n, int grid_size,
                                     int block_size, cudaStream_t stream);

extern void get_block_and_grid_multiply(int *minGrid, int *minBlock);

extern void exec_remove_cp(cuFloatComplex *in, cuFloatComplex *out, int symlen,
                           int cplen, int n, int grid_size, int block_size,
                           cudaStream_t stream);

namespace gr {
namespace wifigpu {

sync_long::sptr sync_long::make(unsigned int sync_length) {
  return gnuradio::get_initial_sptr(new sync_long_impl(sync_length));
}

/*
 * The private constructor
 */
sync_long_impl::sync_long_impl(unsigned int sync_length)
    : gr::block(
          "sync_long",
#if USE_CUSTOM_BUFFERS
          gr::io_signature::make(1, 1, sizeof(gr_complex), cuda_buffer::type),
          gr::io_signature::make(1, 1, sizeof(gr_complex), cuda_buffer::type)),
#else
          gr::io_signature::make(1, 1, sizeof(gr_complex)),
          gr::io_signature::make(1, 1, sizeof(gr_complex))),
#endif
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
  cudaStreamSynchronize(d_stream);

  // get_block_and_grid_multiply(&d_min_grid_size, &d_block_size);
  d_block_size = 1024;

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
  // ninput_items_required[1] = nreq;
}

int sync_long_impl::general_work(int noutput_items, gr_vector_int &ninput_items,
                                 gr_vector_const_void_star &input_items,
                                 gr_vector_void_star &output_items) {
  const gr_complex *in = (const gr_complex *)input_items[0];
  // const gr_complex *in_delayed = (const gr_complex *)input_items[1];
  gr_complex *out = (gr_complex *)output_items[0];

  // Look at the tags
  int ninput = ninput_items[0]; // std::min(ninput_items[0], ninput_items[1]);
  // int ninput = std::min(std::min(ninput_items[0], ninput_items[1]), 8192);
  const uint64_t nread = nitems_read(0);
  const uint64_t nwritten = nitems_written(0);
  get_tags_in_range(tags, 0, nread, nread + ninput);

  // std::cout << tags.size() << std::endl;

  int nconsumed = 0;
  int nproduced = 0;
  auto noutput = noutput_items;

  size_t tag_idx = 0;
  while (true) {
    auto tag = &tags[tag_idx];
    tag_t *next_tag = nullptr;
    if (tag_idx < tags.size() - 1) {
      next_tag = &tags[tag_idx + 1];
    }

    if (d_state == FINISH_LAST_FRAME) {
      auto max_consume = ninput - nconsumed;
      auto max_produce = noutput - nproduced;
      if (tag_idx < tags.size()) {
        // only consume up to the next tag
        max_consume = tags[tag_idx].offset - (nread + nconsumed);
        if (max_consume < 80 ||
            max_produce <
                64) { // need an entire OFDM symbol to do anything here
          nconsumed = tags[tag_idx].offset - nread;
          d_state = WAITING_FOR_TAG;
          continue;
        }
      } else { // no more tags
        if (max_consume < 80 ||
            max_produce <
                64) { // need an entire OFDM symbol to do anything here
          nconsumed += max_consume;
          break;
        }
      }

#if USE_CUSTOM_BUFFERS

#if 1
      auto nsyms = std::min(max_consume / 80, max_produce / 64);
      auto gridSize = (80 * nsyms + d_block_size - 1) / d_block_size;
      exec_remove_cp((cuFloatComplex *)in + nconsumed, (cuFloatComplex *)out + nproduced, 80, 16,
                     80 * nsyms, gridSize, d_block_size, d_stream);
      cudaStreamSynchronize(d_stream);

      int i = 80 * nsyms;
      int o = 64 * nsyms;
      nconsumed += i;
      nproduced += o;
#else
      int i = 0;
      int o = 0;
      while (i + 80 <= max_consume && o + 64 <= max_produce) {
        cudaMemcpy(out + o + nproduced, in + i + nconsumed + 16,
               sizeof(gr_complex) * 64, cudaMemcpyDeviceToDevice); // throw away the cyclic prefix

        i += 80;
        o += 64;
      }

      // FILE *pFile;
      // pFile = fopen("/tmp/gr_sync_long.fc32", "wb");
      // fwrite(out, sizeof(gr_complex), o, pFile);
      nconsumed += i;
      nproduced += o;
#endif
#else
      int i = 0;
      int o = 0;
      while (i + 80 <= max_consume && o + 64 <= max_produce) {
        memcpy(out + o + nproduced, in + i + nconsumed + 16,
               sizeof(gr_complex) * 64); // throw away the cyclic prefix

        i += 80;
        o += 64;
      }

      // FILE *pFile;
      // pFile = fopen("/tmp/gr_sync_long.fc32", "wb");
      // fwrite(out, sizeof(gr_complex), o, pFile);
      nconsumed += i;
      nproduced += o;
#endif

    } else { // WAITING_FOR_TAG

      if (tag_idx < tags.size()) {
        auto offset = tags[tag_idx].offset;

        d_freq_offset_short = pmt::to_double(tags[0].value);

        if (offset - nread + d_fftsize <= ninput &&
            (noutput - nproduced) >= 128) {

#if USE_CUSTOM_BUFFERS
          checkCudaErrors(cufftExecC2C(d_plan,
                                       (cufftComplex *)&in[offset - nread],
                                       d_dev_in, CUFFT_FORWARD));
#else
          checkCudaErrors(cudaMemcpyAsync(d_dev_in, &in[offset - nread],
                                          d_fftsize * sizeof(cufftComplex),
                                          cudaMemcpyHostToDevice, d_stream));
          checkCudaErrors(
              cufftExecC2C(d_plan, d_dev_in, d_dev_in, CUFFT_FORWARD));
#endif

          auto gridSize = (d_fftsize + d_block_size - 1) / d_block_size;
          exec_multiply_kernel_ccc(d_dev_in, d_dev_training_freq, d_dev_in,
                                   d_fftsize, gridSize, d_block_size, d_stream);

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

          // size_t max_index = 297;

          // Copy the LTF symbols
          // offset and nread should always be equal
          size_t copy_index = 0;
          if (max_index > (160 - 32 - 1)) {
            copy_index = max_index - 160 + 32 + 1;
          }

          // std::cout << max_index << " / " << nread << " / " << offset << " / "
          //           << (offset - nread + copy_index) << std::endl;

#if USE_CUSTOM_BUFFERS

          checkCudaErrors(cudaMemcpyAsync(
              out + nproduced, in + (offset - nread + copy_index),
              sizeof(gr_complex) * 128, cudaMemcpyDeviceToDevice, d_stream));
          cudaStreamSynchronize(d_stream);
#else
          memcpy(out + nproduced, in + (offset - nread + copy_index),
                 sizeof(gr_complex) * 128);
#endif

          const pmt::pmt_t key = pmt::string_to_symbol("wifi_start");
          const pmt::pmt_t value = // pmt::from_long(max_index);
              pmt::from_double(d_freq_offset_short - d_freq_offset);
          const pmt::pmt_t srcid = pmt::string_to_symbol(name());
          add_item_tag(0, nwritten + nproduced, key, value, srcid);

          // std::cout << "adding tag at " << (nwritten + nproduced) / 64 << std::endl;

          d_num_syms = 0;
          d_state = FINISH_LAST_FRAME;
          nconsumed = (offset - nread + copy_index + 128);
          nproduced += 128;
          tag_idx++;

        } else {
          // not enough left with this tag
          // clear up to the current tag
          nconsumed = offset - nread;
          break;
        }

      } else // out of tags
      {
        nconsumed = ninput;
        break;
      }
    }
  }
  cudaStreamSynchronize(d_stream);
  consume_each(nconsumed);
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
} // namespace gr
