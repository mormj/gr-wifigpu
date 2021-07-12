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

#ifndef INCLUDED_WIFIGPU_SYNC_SHORT_H
#define INCLUDED_WIFIGPU_SYNC_SHORT_H

#include <wifigpu/api.h>
#include <gnuradio/block.h>

namespace gr {
  namespace wifigpu {

    /*!
     * \brief <+description of block+>
     * \ingroup wifigpu
     *
     */
    class WIFIGPU_API sync_short : virtual public gr::block
    {
     public:
      typedef boost::shared_ptr<sync_short> sptr;

      /*!
       * \brief Return a shared_ptr to a new instance of wifigpu::sync_short.
       *
       * To avoid accidental use of raw pointers, wifigpu::sync_short's
       * constructor is in a private implementation
       * class. wifigpu::sync_short::make is the public interface for
       * creating new instances.
       */
      static sptr make(float threshold, int min_plateau);
    };

  } // namespace wifigpu
} // namespace gr

#endif /* INCLUDED_WIFIGPU_SYNC_SHORT_H */

