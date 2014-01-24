/******************************************************************************
 * Copyright (c) 2010-2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2013, NVIDIA CORPORATION.  All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/


/******************************************************************************
 * CTA tile-processing abstraction for multi-GPU BFS frontier
 * contraction+binning (downsweep scan+scatter)
 ******************************************************************************/

#pragma once

#include <b40c/util/basic_utils.cuh>
#include <b40c/partition/downsweep/cta.cuh>

#include <b40c/graph/bfs/partition_contract/downsweep/tile.cuh>

namespace b40c {
namespace graph {
namespace bfs {
namespace partition_contract {
namespace downsweep {


/**
 * CTA tile-processing abstraction for multi-GPU BFS frontier
 * contraction+binning (downsweep scan+scatter)
 *
 * Derives from partition::downsweep::Cta
 */
template <typename KernelPolicy>
struct Cta :
	partition::downsweep::Cta<
		KernelPolicy,
		Cta<KernelPolicy>,			// This class
		Tile>						// bfs::partition_contract::downsweep::Tile
{
	//---------------------------------------------------------------------
	// Typedefs and Constants
	//---------------------------------------------------------------------

	// Base class type
	typedef partition::downsweep::Cta<KernelPolicy, Cta, Tile> Base;

	typedef typename KernelPolicy::VertexId 				VertexId;
	typedef typename KernelPolicy::ValidFlag				ValidFlag;
	typedef typename KernelPolicy::SizeT 					SizeT;
	typedef typename KernelPolicy::SmemStorage				SmemStorage;
	typedef typename KernelPolicy::Grid::LanePartial		LanePartial;

	typedef typename KernelPolicy::KeyType 					KeyType;
	typedef typename KernelPolicy::ValueType				ValueType;


	//---------------------------------------------------------------------
	// Members
	//---------------------------------------------------------------------

	// Validity flags
	ValidFlag 			*&d_flags_in;

	// Number of GPUs to partition the frontier into
	int num_gpus;

	//---------------------------------------------------------------------
	// Methods
	//---------------------------------------------------------------------

	/**
	 * Constructor
	 */
	__device__ __forceinline__ Cta(
		SmemStorage 	&smem_storage,
		int 			num_gpus,
		VertexId 		*&d_in,
		VertexId 		*&d_out,
		VertexId 		*&d_predecessor_in,
		VertexId 		*&d_predecessor_out,
		ValidFlag		*&d_flags_in,
		SizeT 			*&d_spine,
		LanePartial		base_composite_counter,
		int				*raking_segment) :
			Base(
				smem_storage,
				d_in,							// d_in_keys
				d_out,							// d_out_keys
				(ValueType *&) d_predecessor_in,		// d_in_values
				(ValueType *&) d_predecessor_out,	// d_out_values
				d_spine,
				base_composite_counter,
				raking_segment),
			d_flags_in(d_flags_in),
			num_gpus(num_gpus)
	{}
};


} // namespace downsweep
} // namespace partition_contract
} // namespace bfs
} // namespace graph
} // namespace b40c

