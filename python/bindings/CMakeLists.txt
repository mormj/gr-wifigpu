# Copyright 2020 Free Software Foundation, Inc.
#
# This file is part of GNU Radio
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

########################################################################
# Check if there is C++ code at all
########################################################################
if(NOT wifigpu_sources)
    MESSAGE(STATUS "No C++ sources... skipping python bindings")
    return()
endif(NOT wifigpu_sources)

########################################################################
# Check for pygccxml
########################################################################
GR_PYTHON_CHECK_MODULE_RAW(
    "pygccxml"
    "import pygccxml"
    PYGCCXML_FOUND
    )

include(GrPybind)

########################################################################
# Python Bindings
########################################################################

list(APPEND wifigpu_python_files
    presync_python.cc
    sync_short_python.cc
    sync_long_python.cc
    frame_equalizer_python.cc
    gpu_channelizer_python.cc python_bindings.cc)

GR_PYBIND_MAKE_OOT(wifigpu 
   ../..
   gr::wifigpu
   "${wifigpu_python_files}")

install(TARGETS wifigpu_python DESTINATION ${GR_PYTHON_DIR}/wifigpu COMPONENT pythonapi)
