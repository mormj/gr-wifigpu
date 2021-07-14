/* -*- c++ -*- */

#define WIFIGPU_API

%include "gnuradio.i"           // the common stuff

//load generated python docstrings
%include "wifigpu_swig_doc.i"

%{
#include "wifigpu/sync_short.h"
#include "wifigpu/presync.h"
#include "wifigpu/sync_long.h"
%}

%include "wifigpu/sync_short.h"
GR_SWIG_BLOCK_MAGIC2(wifigpu, sync_short);
%include "wifigpu/presync.h"
GR_SWIG_BLOCK_MAGIC2(wifigpu, presync);

%include "wifigpu/sync_long.h"
GR_SWIG_BLOCK_MAGIC2(wifigpu, sync_long);
