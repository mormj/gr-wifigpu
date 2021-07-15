/*
 * Copyright 2021 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

/***********************************************************************************/
/* This file is automatically generated using bindtool and can be manually edited  */
/* The following lines can be configured to regenerate this file during cmake      */
/* If manual edits are made, the following tags should be modified accordingly.    */
/* BINDTOOL_GEN_AUTOMATIC(0)                                                       */
/* BINDTOOL_USE_PYGCCXML(0)                                                        */
/* BINDTOOL_HEADER_FILE(frame_equalizer.h)                                        */
/* BINDTOOL_HEADER_FILE_HASH(53955c8e5dfa64df90f50d4635ba6d5f)                     */
/***********************************************************************************/

#include <pybind11/complex.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

#include <wifigpu/frame_equalizer.h>
// pydoc.h is automatically generated in the build directory
#include <frame_equalizer_pydoc.h>

void bind_frame_equalizer(py::module& m)
{

    using frame_equalizer    = ::gr::wifigpu::frame_equalizer;


    py::class_<frame_equalizer, gr::block, gr::basic_block,
        std::shared_ptr<frame_equalizer>>(m, "frame_equalizer", D(frame_equalizer))

        .def(py::init(&frame_equalizer::make),
           py::arg("algo"),
           py::arg("freq"),
           py::arg("bw"),
           D(frame_equalizer,make)
        )
        



        ;




}







