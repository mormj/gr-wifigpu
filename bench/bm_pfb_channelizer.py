#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: Not titled yet
# GNU Radio version: 3.9.0.0-git

from gnuradio import blocks
from gnuradio import gr
from gnuradio.filter import firdes
import sys
import signal
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
from gnuradio.fft import window
from gnuradio import fft
import time
import trt
import json
import datetime
import itertools
import bench
from gnuradio.filter import pfb
from gnuradio import filter
import wifigpu

class benchmark_copy(gr.top_block):

    def __init__(self, args):
        gr.top_block.__init__(self, "Benchmark Copy", catch_exceptions=True)

        ##################################################
        # Variables
        ##################################################
        nsamples = args.samples
        nchans = args.nchans
        attn = args.attenuation
        bufsize = args.buffer_size
                
        taps = filter.firdes.low_pass_2(1, nchans, 1 / 2, 1 / 10,  attenuation_dB=attn,window=fft.window.WIN_BLACKMAN_hARRIS)

        ##################################################
        # Blocks
        ##################################################
        self.nsrc = blocks.null_source(gr.sizeof_gr_complex*1)
        self.nsnk = blocks.null_sink(gr.sizeof_gr_complex*1)
        self.hd = blocks.head(gr.sizeof_gr_complex*1, int(nsamples))
        self.channelizer = wifigpu.gpu_channelizer(
            nchans,
            taps)
        self.nsrc.set_min_output_buffer(bufsize)
        self.hd.set_min_output_buffer(bufsize)
        self.channelizer.set_min_output_buffer(bufsize)

        ##################################################
        # Connections
        ##################################################

        for ii in range(nchans):
            self.connect((self.channelizer, ii), (self.nsnk, ii))
        # self.connect(self.channelizer, self.nsnk)
        self.connect((self.hd, 0), (self.channelizer, 0))
        self.connect((self.nsrc, 0), (self.hd, 0))



def main(top_block_cls=benchmark_copy, options=None):

    parser = ArgumentParser(description='Run a flowgraph iterating over parameters for benchmarking')
    parser.add_argument('--rt_prio', help='enable realtime scheduling', action='store_true')
    parser.add_argument('--samples', type=int, default=1e8)
    parser.add_argument('--nchans', type=int, default=4)
    parser.add_argument('--attenuation', type=float, default=70)
    parser.add_argument('--buffer_size', type=int, default=8192)

    args = parser.parse_args()
    print(args)

    if args.rt_prio and gr.enable_realtime_scheduling() != gr.RT_OK:
        print("Error: failed to enable real-time scheduling.")

    tb = top_block_cls(args)

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()
        sys.exit(0)

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    print("starting ...")
    startt = time.time()
    tb.start()

    tb.wait()
    endt = time.time()
    print(f'[PROFILE_TIME]{endt-startt}[PROFILE_TIME]')

if __name__ == '__main__':
    main()
