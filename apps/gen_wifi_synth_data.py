#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: Wifi Tx
# GNU Radio version: v3.10.0.0git-494-g5f36d4f4

import os
import sys
sys.path.append(os.environ.get('GRC_HIER_PATH', os.path.expanduser('~/.grc_gnuradio')))

from gnuradio import blocks
import pmt
from gnuradio import gr
from gnuradio.filter import firdes
from gnuradio.fft import window
import signal
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
from wifi_phy_hier import wifi_phy_hier  # grc-generated hier_block
import foo
import ieee802_11


def snipfcn_snippet_0(self):
    print(os.path.join(self.file_base ,'wifi_synth_' + str(self.pdu_length) + '_' + str(self.pktspace) + '_' + str(self.samp_rate_mhz) + 'MHz_' + str(self.len_collect) + 's_MCS' + str(self.encoding) + '.fc32'))


def snippets_main_after_init(tb):
    snipfcn_snippet_0(tb)


class wifi_tx(gr.top_block):

    def __init__(self, encoding=0, file_base='', interval=1, len_collect=10, pdu_length=36, pktspace=1000, samp_rate_mhz=20):
        gr.top_block.__init__(self, "Wifi Tx", catch_exceptions=True)

        ##################################################
        # Parameters
        ##################################################
        self.encoding = encoding
        self.file_base = file_base
        self.interval = interval
        self.len_collect = len_collect
        self.pdu_length = pdu_length
        self.pktspace = pktspace
        self.samp_rate_mhz = samp_rate_mhz

        ##################################################
        # Variables
        ##################################################
        self.samp_rate = samp_rate = int(samp_rate_mhz * 1e6)
        self.out_buf_size = out_buf_size = 960000
        self.freq = freq = 2412000000

        ##################################################
        # Blocks
        ##################################################
        self.wifi_phy_hier_0 = wifi_phy_hier(
            bandwidth=int(samp_rate_mhz * 1e6),
            chan_est=0,
            encoding=encoding,
            frequency=freq,
            sensitivity=0.56,
        )
        self.ieee802_11_mac_0 = ieee802_11.mac([0x23, 0x23, 0x23, 0x23, 0x23, 0x23], [0x42, 0x42, 0x42, 0x42, 0x42, 0x42], [0xff, 0xff, 0xff, 0xff, 0xff, 255])
        self.foo_packet_pad2_0 = foo.packet_pad2(False, False, .000001, pktspace//2, pktspace//2)
        self.foo_packet_pad2_0.set_min_output_buffer(960000)
        self.file1 = blocks.file_sink(gr.sizeof_gr_complex*1, os.path.join(file_base, 'wifi_synth_' + str(pdu_length) + '_' + str(pktspace) + '_' + str(samp_rate_mhz) + 'MHz_' + str(len_collect) + 's_MCS' + str(encoding) + '.fc32'), False)
        self.file1.set_unbuffered(False)
        self.blocks_vector_source_x_0 = blocks.vector_source_c((0,), False, 1, [])
        self.blocks_multiply_const_vxx_0 = blocks.multiply_const_cc(0.6)
        self.blocks_multiply_const_vxx_0.set_min_output_buffer(100000)
        self.blocks_message_strobe_0_0 = blocks.message_strobe(pmt.intern("".join("x" for i in range(pdu_length))), interval)
        self.blocks_head_0 = blocks.head(gr.sizeof_gr_complex*1, int(samp_rate_mhz * len_collect * 1000000))



        ##################################################
        # Connections
        ##################################################
        self.msg_connect((self.blocks_message_strobe_0_0, 'strobe'), (self.ieee802_11_mac_0, 'app in'))
        self.msg_connect((self.ieee802_11_mac_0, 'phy out'), (self.wifi_phy_hier_0, 'mac_in'))
        self.connect((self.blocks_head_0, 0), (self.file1, 0))
        self.connect((self.blocks_multiply_const_vxx_0, 0), (self.foo_packet_pad2_0, 0))
        self.connect((self.blocks_vector_source_x_0, 0), (self.wifi_phy_hier_0, 0))
        self.connect((self.foo_packet_pad2_0, 0), (self.blocks_head_0, 0))
        self.connect((self.wifi_phy_hier_0, 0), (self.blocks_multiply_const_vxx_0, 0))


    def get_encoding(self):
        return self.encoding

    def set_encoding(self, encoding):
        self.encoding = encoding
        self.file1.open(os.path.join(self.file_base, 'wifi_synth_' + str(self.pdu_length) + '_' + str(self.pktspace) + '_' + str(self.samp_rate_mhz) + 'MHz_' + str(self.len_collect) + 's_MCS' + str(self.encoding) + '.fc32'))
        self.wifi_phy_hier_0.set_encoding(self.encoding)

    def get_file_base(self):
        return self.file_base

    def set_file_base(self, file_base):
        self.file_base = file_base
        self.file1.open(os.path.join(self.file_base, 'wifi_synth_' + str(self.pdu_length) + '_' + str(self.pktspace) + '_' + str(self.samp_rate_mhz) + 'MHz_' + str(self.len_collect) + 's_MCS' + str(self.encoding) + '.fc32'))

    def get_interval(self):
        return self.interval

    def set_interval(self, interval):
        self.interval = interval
        self.blocks_message_strobe_0_0.set_period(self.interval)

    def get_len_collect(self):
        return self.len_collect

    def set_len_collect(self, len_collect):
        self.len_collect = len_collect
        self.blocks_head_0.set_length(int(self.samp_rate_mhz * self.len_collect * 1000000))
        self.file1.open(os.path.join(self.file_base, 'wifi_synth_' + str(self.pdu_length) + '_' + str(self.pktspace) + '_' + str(self.samp_rate_mhz) + 'MHz_' + str(self.len_collect) + 's_MCS' + str(self.encoding) + '.fc32'))

    def get_pdu_length(self):
        return self.pdu_length

    def set_pdu_length(self, pdu_length):
        self.pdu_length = pdu_length
        self.blocks_message_strobe_0_0.set_msg(pmt.intern("".join("x" for i in range(self.pdu_length))))
        self.file1.open(os.path.join(self.file_base, 'wifi_synth_' + str(self.pdu_length) + '_' + str(self.pktspace) + '_' + str(self.samp_rate_mhz) + 'MHz_' + str(self.len_collect) + 's_MCS' + str(self.encoding) + '.fc32'))

    def get_pktspace(self):
        return self.pktspace

    def set_pktspace(self, pktspace):
        self.pktspace = pktspace
        self.file1.open(os.path.join(self.file_base, 'wifi_synth_' + str(self.pdu_length) + '_' + str(self.pktspace) + '_' + str(self.samp_rate_mhz) + 'MHz_' + str(self.len_collect) + 's_MCS' + str(self.encoding) + '.fc32'))

    def get_samp_rate_mhz(self):
        return self.samp_rate_mhz

    def set_samp_rate_mhz(self, samp_rate_mhz):
        self.samp_rate_mhz = samp_rate_mhz
        self.set_samp_rate(int(self.samp_rate_mhz * 1e6))
        self.blocks_head_0.set_length(int(self.samp_rate_mhz * self.len_collect * 1000000))
        self.file1.open(os.path.join(self.file_base, 'wifi_synth_' + str(self.pdu_length) + '_' + str(self.pktspace) + '_' + str(self.samp_rate_mhz) + 'MHz_' + str(self.len_collect) + 's_MCS' + str(self.encoding) + '.fc32'))
        self.wifi_phy_hier_0.set_bandwidth(int(self.samp_rate_mhz * 1e6))

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate

    def get_out_buf_size(self):
        return self.out_buf_size

    def set_out_buf_size(self, out_buf_size):
        self.out_buf_size = out_buf_size

    def get_freq(self):
        return self.freq

    def set_freq(self, freq):
        self.freq = freq
        self.wifi_phy_hier_0.set_frequency(self.freq)



def argument_parser():
    parser = ArgumentParser()
    parser.add_argument(
        "--encoding", dest="encoding", type=intx, default=0,
        help="Set encoding [default=%(default)r]")
    parser.add_argument(
        "-d", "--file-base", dest="file_base", type=str, default='',
        help="Set file_base [default=%(default)r]")
    parser.add_argument(
        "--interval", dest="interval", type=intx, default=1,
        help="Set interval [default=%(default)r]")
    parser.add_argument(
        "-l", "--len-collect", dest="len_collect", type=intx, default=10,
        help="Set len_collect [default=%(default)r]")
    parser.add_argument(
        "--pdu-length", dest="pdu_length", type=intx, default=36,
        help="Set pdu_length [default=%(default)r]")
    parser.add_argument(
        "--pktspace", dest="pktspace", type=intx, default=1000,
        help="Set pktspace [default=%(default)r]")
    parser.add_argument(
        "--samp-rate-mhz", dest="samp_rate_mhz", type=intx, default=20,
        help="Set samp_rate_mhz [default=%(default)r]")
    return parser


def main(top_block_cls=wifi_tx, options=None):
    if options is None:
        options = argument_parser().parse_args()
    tb = top_block_cls(encoding=options.encoding, file_base=options.file_base, interval=options.interval, len_collect=options.len_collect, pdu_length=options.pdu_length, pktspace=options.pktspace, samp_rate_mhz=options.samp_rate_mhz)
    snippets_main_after_init(tb)
    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()

        sys.exit(0)

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    tb.start()

    tb.wait()


if __name__ == '__main__':
    main()
