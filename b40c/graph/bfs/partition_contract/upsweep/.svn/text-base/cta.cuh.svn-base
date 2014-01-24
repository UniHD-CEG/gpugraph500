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
 * contraction+binning (upsweep reduction)
 ******************************************************************************/

#pragma once

#include <b40c/partition/upsweep/cta.cuh>

#include <b40c/graph/bfs/partition_contract/upsweep/tile.cuh>

namespace b40c {
namespace graph {
namespace bfs {
namespace partition_contract {
namespace upsweep {


/**
 * CTA tile-processing abstraction for multi-GPU BFS frontier
 * contraction+binning (upsweep reduction)
 *
 * Derives from partition::upsweep::Cta
 */
template <typename KernelPolicy>
struct Cta :
	partition::upsweep::Cta<
		KernelPolicy,
		Cta<KernelPolicy>,			// This class
		Tile>						// radix_sort::upsweep::Tile
{
	//---------------------------------------------------------------------
	// Typedefs and Constants
	//---------------------------------------------------------------------

	// Base class type
	typedef partition::upsweep::Cta<KernelPolicy, Cta, Tile> Base;

	typedef typename KernelPolicy::SmemStorage 				SmemStorage;
	typedef typename KernelPolicy::VertexId 				VertexId;
	typedef typename KernelPolicy::ValidFlag				ValidFlag;
	typedef typename KernelPolicy::VisitedMask 			VisitedMask;

	typedef typename KernelPolicy::KeyType 					KeyType;
	typedef typename KernelPolicy::SizeT 					SizeT;


	//---------------------------------------------------------------------
	// Members
	//---------------------------------------------------------------------

	// Input and output device pointers
	ValidFlag				*d_flags_out;
	VisitedMask 			*d_visited_mask;

	// Smem storage for reduction tree and hashing scratch
	volatile VertexId 		(*vid_hashtable)[SmemStorage::WARP_HASH_ELEMENTS];
	volatile VertexId		*history;

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
		VertexId 		*d_in,
		ValidFlag		*d_flags_out,
		SizeT 			*d_spine,
		VisitedMask 	*d_visited_mask) :
			Base(
				smem_storage,
				d_in, 				// d_in_keys
				d_spine),
			d_flags_out(d_flags_out),
			d_visited_mask(d_visited_mask),
			vid_hashtable(smem_storage.vid_hashtable),
			history(smem_storage.history),
			num_gpus(num_gpus)
	{
		// Initialize history filter
		for (int offset = threadIdx.x; offset < SmemStorage::HISTORY_HASH_ELEMENTS; offset += KernelPolicy::THREADS) {
			history[offset] = -1;
		}
	}

};



} // namespace upsweep
} // namespace partition_contract
} // namespace bfs
} // namespace graph
} // namespace b40c

