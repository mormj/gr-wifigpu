#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: Bm Gpu Packetize
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
import numpy as np
import time
import wifigpu
import os


def snipfcn_snippet_0(self):
    self.startt = time.time()

def snipfcn_snippet_0_0(self):
    self.endt = time.time()
    print(f'[PROFILE_TIME]{self.endt-self.startt}[PROFILE_TIME]')


def snippets_main_after_init(tb):
    snipfcn_snippet_0(tb)

def snippets_main_after_stop(tb):
    snipfcn_snippet_0_0(tb)


class bm_gpu_packetize(gr.top_block):

    def __init__(self, bandwidth_mhz=20, encoding=0, file_base='/data/data/cropcircles', filename=None, num_threads=1, pdu_length=36, pkt_space=1000, buffer_size=1024*1024):
        gr.top_block.__init__(self, "Bm Gpu Packetize", catch_exceptions=True)


        print(filename)
        if not filename:
            filename = 'wifi_synth_' + str(pdu_length) + '_' + str(pkt_space) + '_' + str(int(bandwidth_mhz)) + 'MHz_' + '10s_MCS' + str(encoding) + '.fc32'
            print(filename)

        filename = os.path.join(file_base, filename)

        ##################################################
        # Parameters
        ##################################################
        self.bandwidth_mhz = bandwidth_mhz
        self.encoding = encoding
        self.file_base = file_base
        self.filename = filename
        self.num_threads = num_threads
        self.pdu_length = pdu_length
        self.pkt_space = pkt_space
        self.buffer_size = buffer_size

        ##################################################
        # Variables
        ##################################################
        self.window_size = window_size = 48
        self.variable_0_0_0_0 = variable_0_0_0_0 = 0
        self.sync_length = sync_length = 320
        self.samp_rate = samp_rate = bandwidth_mhz * 1e6
        self.lo_offset = lo_offset = 0
        self.freq = freq = 2412000000
        self.chan_est = chan_est = 0

        ##################################################
        # Blocks
        ##################################################
        self.wifigpu_tp_decode_mac_0 = wifigpu.tp_decode_mac(num_threads, 100000, False, False)
        self.wifigpu_sync_short_0 = wifigpu.sync_short(.56, 2)
        self.wifigpu_sync_short_0.set_min_output_buffer(buffer_size)
        self.wifigpu_sync_long_0 = wifigpu.sync_long(320)
        # self.wifigpu_sync_long_0.set_min_output_buffer(buffer_size // 64)
        self.wifigpu_presync_0 = wifigpu.presync()
        self.wifigpu_presync_0.set_min_output_buffer(buffer_size)
        self.wifigpu_packetize_frame_0 = wifigpu.packetize_frame(0, freq, samp_rate)
        self.fft_vxx_0 = fft.fft_vcc(64, True, window.rectangular(64), True, 1)
        # self.blocks_stream_to_vector_0 = blocks.stream_to_vector(gr.sizeof_gr_complex*1, 64)
        self.blocks_file_source_0 = blocks.file_source(gr.sizeof_gr_complex*1, filename, False, 0, 0)
        self.blocks_file_source_0.set_begin_tag(pmt.PMT_NIL)
        self.blocks_file_source_0.set_min_output_buffer(buffer_size)



        ##################################################
        # Connections
        ##################################################
        self.msg_connect((self.wifigpu_packetize_frame_0, 'pdus'), (self.wifigpu_tp_decode_mac_0, 'pdus'))
        self.connect((self.blocks_file_source_0, 0), (self.wifigpu_presync_0, 0))
        # self.connect((self.blocks_stream_to_vector_0, 0), (self.fft_vxx_0, 0))
        self.connect((self.fft_vxx_0, 0), (self.wifigpu_packetize_frame_0, 0))
        self.connect((self.wifigpu_presync_0, 0), (self.wifigpu_sync_short_0, 0))
        self.connect((self.wifigpu_presync_0, 2), (self.wifigpu_sync_short_0, 2))
        self.connect((self.wifigpu_presync_0, 1), (self.wifigpu_sync_short_0, 1))
        self.connect((self.wifigpu_sync_long_0, 0), (self.fft_vxx_0, 0))
        self.connect((self.wifigpu_sync_short_0, 0), (self.wifigpu_sync_long_0, 0))



def argument_parser():
    parser = ArgumentParser()
    parser.add_argument(
        "--bandwidth_mhz", dest="bandwidth_mhz", type=eng_float, default=eng_notation.num_to_str(float(20)),
        help="Set bandwidth_mhz [default=%(default)r]")
    parser.add_argument(
        "--encoding", dest="encoding", type=intx, default=0,
        help="Set encoding [default=%(default)r]")
    parser.add_argument(
        "-d", "--file_base", dest="file_base", type=str, default='/data/data/cropcircles',
        help="Set file_base [default=%(default)r]")
    parser.add_argument(
        "-f", "--filename", dest="filename", type=str, default='',
        help="Set filename [default=%(default)r]")
    parser.add_argument(
        "--num_threads", dest="num_threads", type=intx, default=1,
        help="Set num_threads [default=%(default)r]")
    parser.add_argument(
        "--pdu_length", dest="pdu_length", type=intx, default=36,
        help="Set pdu_length [default=%(default)r]")
    parser.add_argument(
        "--pkt_space", dest="pkt_space", type=intx, default=1000,
        help="Set pkt_space [default=%(default)r]")
    parser.add_argument(
        "--buffer_size", dest="buffer_size", type=intx, default=1024*1024,
        help="Set num_threads [default=%(default)r]")
    return parser
    


def main(top_block_cls=bm_gpu_packetize, options=None):
    if options is None:
        options = argument_parser().parse_args()
    if gr.enable_realtime_scheduling() != gr.RT_OK:
        print("Error: failed to enable real-time scheduling.")
    tb = top_block_cls(bandwidth_mhz=options.bandwidth_mhz, encoding=options.encoding, file_base=options.file_base, filename=options.filename, num_threads=options.num_threads, 
        pdu_length=options.pdu_length, pkt_space=options.pkt_space, buffer_size=options.buffer_size)
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
