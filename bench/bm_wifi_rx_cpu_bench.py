#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: Wifi Rx Cpu Bench
# GNU Radio version: v3.10.0.0git-494-g5f36d4f4

from gnuradio import blocks
import pmt
from gnuradio import fft
from gnuradio.fft import window
from gnuradio import gr
from gnuradio.filter import firdes
import sys
import signal
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
import ieee802_11
import numpy as np
import time


def snipfcn_snippet_0(self):
    self.startt = time.time()

def snipfcn_snippet_0_0(self):
    self.endt = time.time()
    print(f'[PROFILE_TIME]{self.endt-self.startt}[PROFILE_TIME]')


def snippets_main_after_init(tb):
    snipfcn_snippet_0(tb)

def snippets_main_after_stop(tb):
    snipfcn_snippet_0_0(tb)


class wifi_rx_cpu_bench(gr.top_block):

    def __init__(self, args):
        gr.top_block.__init__(self, "Wifi Rx Cpu Bench", catch_exceptions=True)

        ##################################################
        # Parameters
        ##################################################
        filename = args.filename
        buffer_size = args.buffer_size
        stop_point = args.stop_point

        ##################################################
        # Variables
        ##################################################
        self.window_size = window_size = 48
        self.variable_0_0_0_0 = variable_0_0_0_0 = 0
        self.sync_length = sync_length = 320
        self.samp_rate = samp_rate = args.bandwidth_mhz * 1000000
        self.lo_offset = lo_offset = 0
        self.freq = freq = 2412000000
        self.chan_est = chan_est = 0

        ##################################################
        # Blocks
        ##################################################
        src = blocks.file_source(gr.sizeof_gr_complex*1, filename, False, 0, 0)
        src.set_begin_tag(pmt.PMT_NIL)
        src.set_min_output_buffer(buffer_size)
        

        mult_blk = blocks.multiply_vcc(1)
        mult_blk.set_min_output_buffer(buffer_size)
        mov_avg_cc = blocks.moving_average_cc(window_size, 1, 4096, 1)
        mov_avg_cc.set_min_output_buffer(buffer_size)
        mov_avg_ff = blocks.moving_average_ff(window_size + 16, 1, 4096, 1)
        mov_avg_ff.set_min_output_buffer(buffer_size)

        divide_blk = blocks.divide_ff(1)
        divide_blk.set_min_output_buffer(buffer_size)
        delay0 = blocks.delay(gr.sizeof_gr_complex*1, 16)
        delay0.set_min_output_buffer(buffer_size)
        delay1 = blocks.delay(gr.sizeof_gr_complex*1, sync_length)
        delay1.set_min_output_buffer(buffer_size)
        conj_blk = blocks.conjugate_cc()
        conj_blk.set_min_output_buffer(buffer_size)
        cplxm2 = blocks.complex_to_mag_squared(1)
        cplxm2.set_min_output_buffer(buffer_size)
        cplxmag = blocks.complex_to_mag(1)
        cplxmag.set_min_output_buffer(buffer_size)



        ##################################################
        # Connections
        ##################################################
        self.connect((src, 0), (cplxm2, 0))
        self.connect((src, 0), (delay0, 0))
        self.connect((src, 0), (mult_blk, 0))
        self.connect((mov_avg_ff, 0), (divide_blk, 1))
        self.connect((mov_avg_cc, 0), (cplxmag, 0))
        self.connect((cplxmag, 0), (divide_blk, 0))
        self.connect((cplxm2, 0), (mov_avg_ff, 0))
        self.connect((conj_blk, 0), (mult_blk, 1))
        
        self.connect((delay0, 0), (conj_blk, 0))
        self.connect((mult_blk, 0), (mov_avg_cc, 0))

        if stop_point == 0:
            ns1 = blocks.null_sink(gr.sizeof_gr_complex)
            ns2 = blocks.null_sink(gr.sizeof_gr_complex)
            ns3 = blocks.null_sink(gr.sizeof_float)
            self.connect((delay0, 0), (ns1, 0))
            self.connect((mov_avg_cc, 0), (ns2, 0))
            self.connect((divide_blk, 0), (ns3, 0)) 

        if (stop_point > 0):
            sync_short = ieee802_11.sync_short(0.56, 2, False, False)
            sync_short.set_min_output_buffer(buffer_size)
            self.connect((delay0, 0), (sync_short, 0))
            self.connect((mov_avg_cc, 0), (sync_short, 1))
            self.connect((divide_blk, 0), (sync_short, 2))

        if (stop_point == 1):
            ns1 = blocks.null_sink(gr.sizeof_gr_complex)
            self.connect((sync_short, 0), (ns1, 0))

        if (stop_point > 1):
            sync_long = ieee802_11.sync_long(sync_length, False, False)
            sync_long.set_min_output_buffer(buffer_size)
            self.connect((sync_short, 0), (delay1, 0))
            self.connect((delay1, 0), (sync_long, 1))
            self.connect((sync_short, 0), (sync_long, 0))

        if (stop_point == 2):
            ns1 = blocks.null_sink(gr.sizeof_gr_complex)
            self.connect((sync_long, 0), (ns1, 0))

        if (stop_point > 2):
            fft_blk = fft.fft_vcc(64, True, window.rectangular(64), True, 1)
            s2v = blocks.stream_to_vector(gr.sizeof_gr_complex*1, 64)
            frame_eq = ieee802_11.frame_equalizer(chan_est, freq, samp_rate, False, False)
            self.connect((sync_long, 0), (s2v, 0))
            self.connect((s2v, 0), (fft_blk, 0))
            self.connect((fft_blk, 0), (frame_eq, 0))

        if (stop_point == 3):
            ns1 = blocks.null_sink(48*gr.sizeof_char)
            self.connect((frame_eq, 0), (ns1, 0))
        
        if (stop_point == 4):
            decode_mac = ieee802_11.decode_mac(False, False)
            self.connect((frame_eq, 0), (decode_mac, 0))
            


def argument_parser():
    parser = ArgumentParser()
    parser.add_argument(
        "-f", "--filename", dest="filename", type=str, default=None,
        help="Set filename [default=%(default)r]")
    parser.add_argument('--encoding', type=int, default=0)
    parser.add_argument('--pdu_length', type=int, default=36)
    parser.add_argument('--pkt_space', type=int, default=1000)
    parser.add_argument('--bandwidth_mhz', type=int, default=20)
    
    parser.add_argument('--rt_prio', help='enable realtime scheduling', action='store_true')
    parser.add_argument('--stop_point', type=int, default=0, help='0=presync, 1=sync_short, 2=sync_long, 3=frame_eq, 4=decode_mac')
    parser.add_argument('--buffer_size', type=int, default=1024*1024)

    return parser


def main(top_block_cls=wifi_rx_cpu_bench, options=None):
    if options is None:
        options = argument_parser().parse_args()

    if options.rt_prio and gr.enable_realtime_scheduling() != gr.RT_OK:
        print("Error: failed to enable real-time scheduling.")

    tb = top_block_cls(options)
    snippets_main_after_init(tb)
    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()
        snippets_main_after_stop(tb)
        sys.exit(0)

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    tb.start()

    tb.wait()
    snippets_main_after_stop(tb)

if __name__ == '__main__':
    main()
