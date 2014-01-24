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
 * Tile-processing functionality for BFS expansion kernels
 ******************************************************************************/

#pragma once

#include <b40c/util/device_intrinsics.cuh>
#include <b40c/util/cta_work_progress.cuh>
#include <b40c/util/scan/cooperative_scan.cuh>
#include <b40c/util/io/modified_load.cuh>
#include <b40c/util/io/modified_store.cuh>
#include <b40c/util/io/load_tile.cuh>
#include <b40c/util/io/initialize_tile.cuh>
#include <b40c/util/operators.cuh>

#include <b40c/util/soa_tuple.cuh>
#include <b40c/util/scan/cooperative_scan.cuh>

namespace b40c {
namespace graph {
namespace bfs {
namespace microbench {
namespace serial_gather {


texture<char, cudaTextureType1D, cudaReadModeElementType> bitmask_tex_ref;


/**
 * Cta
 */
template <typename KernelPolicy>
struct Cta
{
	//---------------------------------------------------------------------
	// Typedefs
	//---------------------------------------------------------------------

	typedef typename KernelPolicy::VertexId 		VertexId;
	typedef typename KernelPolicy::SizeT 			SizeT;
	typedef typename KernelPolicy::VisitedMask 	VisitedMask;

	typedef typename KernelPolicy::SmemStorage		SmemStorage;

	typedef typename KernelPolicy::RakingDetails 		RakingDetails;


	//---------------------------------------------------------------------
	// Members
	//---------------------------------------------------------------------

	// Current BFS queue index
	VertexId 				iteration;
	VertexId 				queue_index;

	// Input and output device pointers
	SizeT 					*d_in_row_offsets;
	VertexId 				*d_out;
	SizeT 					*d_in_row_lengths;
	VertexId				*d_column_indices;
	VisitedMask 			*d_visited_mask;
	VertexId 				*d_labels;

	// Work progress
	util::CtaWorkProgress	&work_progress;

	// Operational details for raking grid
	RakingDetails 			raking_details;

	// Smem storage
	SmemStorage 			&smem_storage;


	//---------------------------------------------------------------------
	// Helper Structures
	//---------------------------------------------------------------------

	/**
	 * Tile
	 */
	template <
		int LOG_LOADS_PER_TILE,
		int LOG_LOAD_VEC_SIZE>
	struct Tile
	{
		//---------------------------------------------------------------------
		// Typedefs and Constants
		//---------------------------------------------------------------------

		enum {
			LOADS_PER_TILE 		= 1 << LOG_LOADS_PER_TILE,
			LOAD_VEC_SIZE 		= 1 << LOG_LOAD_VEC_SIZE
		};


		//---------------------------------------------------------------------
		// Members
		//---------------------------------------------------------------------

		SizeT		row_offset[LOADS_PER_TILE][LOAD_VEC_SIZE];
		SizeT		row_length[LOADS_PER_TILE][LOAD_VEC_SIZE];

		SizeT		rank[LOADS_PER_TILE][LOAD_VEC_SIZE];

		//---------------------------------------------------------------------
		// Helper Structures
		//---------------------------------------------------------------------

		/**
		 * Iterate next vector element
		 */
		template <int LOAD, int VEC, int dummy = 0>
		struct Iterate
		{
			/**
			 * Expand serially
			 */
			static __device__ __forceinline__ void ExpandSerial(Cta *cta, Tile *tile)
			{
				for (int i = 0; i < tile->row_length[LOAD][VEC]; i++) {

					// Gather a neighbor
					VertexId neighbor_id;

					util::io::ModifiedLoad<KernelPolicy::COLUMN_READ_MODIFIER>::Ld(
						neighbor_id,
						cta->d_column_indices + tile->row_offset[LOAD][VEC] + i);

					cta->smem_storage.gathered = neighbor_id;

					if (!KernelPolicy::BENCHMARK) {

						// Scatter it into queue
						util::io::ModifiedStore<KernelPolicy::QUEUE_WRITE_MODIFIER>::St(
							neighbor_id,
							cta->d_out + tile->rank[LOAD][VEC] + i);
					}
				}

				// Next vector element
				Iterate<LOAD, VEC + 1>::ExpandSerial(cta, tile);
			}
		};


		/**
		 * Iterate next load
		 */
		template <int LOAD, int dummy>
		struct Iterate<LOAD, LOAD_VEC_SIZE, dummy>
		{
			/**
			 * ExpandSerial
			 */
			static __device__ __forceinline__ void ExpandSerial(Cta *cta, Tile *tile)
			{
				Iterate<LOAD + 1, 0>::ExpandSerial(cta, tile);
			}
		};

		/**
		 * Terminate
		 */
		template <int dummy>
		struct Iterate<LOADS_PER_TILE, 0, dummy>
		{
			// ExpandSerial
			static __device__ __forceinline__ void ExpandSerial(Cta *cta, Tile *tile) {}
		};


		//---------------------------------------------------------------------
		// Interface
		//---------------------------------------------------------------------

		/**
		 * Expands neighbor lists sequentially (non-cooperatively)
		 */
		__device__ __forceinline__ void ExpandSerial(Cta *cta)
		{
			Iterate<0, 0>::ExpandSerial(cta, this);
		}
	};


	//---------------------------------------------------------------------
	// Methods
	//---------------------------------------------------------------------

	/**
	 * Constructor
	 */
	__device__ __forceinline__ Cta(
		VertexId				iteration,
		VertexId 				queue_index,
		SmemStorage 			&smem_storage,
		SizeT 					*d_in_row_offsets,
		VertexId 				*d_out,
		SizeT 					*d_in_row_lengths,
		VertexId 				*d_column_indices,
		VisitedMask 			*d_visited_mask,
		VertexId 				*d_labels,
		util::CtaWorkProgress	&work_progress) :

			iteration(iteration),
			smem_storage(smem_storage),
			queue_index(queue_index),
			raking_details(
				smem_storage.raking_elements,
				smem_storage.warpscan,
				0),
			d_in_row_offsets(d_in_row_offsets),
			d_in_row_lengths(d_in_row_lengths),
			d_out(d_out),
			d_column_indices(d_column_indices),
			d_visited_mask(d_visited_mask),
			d_labels(d_labels),
			work_progress(work_progress) {}


	/**
	 * Process a single tile
	 */
	__device__ __forceinline__ void ProcessTile(
		SizeT cta_offset,
		SizeT guarded_elements = KernelPolicy::TILE_ELEMENTS)
	{
		Tile<
			KernelPolicy::LOG_LOADS_PER_TILE,
			KernelPolicy::LOG_LOAD_VEC_SIZE> tile;

		// Load row offsets
		util::io::LoadTile<
			KernelPolicy::LOG_LOADS_PER_TILE,
			KernelPolicy::LOG_LOAD_VEC_SIZE,
			KernelPolicy::THREADS,
			KernelPolicy::QUEUE_READ_MODIFIER,
			false>::LoadValid(
				tile.row_offset,
				d_in_row_offsets,
				cta_offset,
				guarded_elements);

		// Load row lengths
		util::io::LoadTile<
			KernelPolicy::LOG_LOADS_PER_TILE,
			KernelPolicy::LOG_LOAD_VEC_SIZE,
			KernelPolicy::THREADS,
			KernelPolicy::QUEUE_READ_MODIFIER,
			false>::LoadValid(
				tile.row_length,
				d_in_row_lengths,
				cta_offset,
				guarded_elements,
				(SizeT) 0);

		// Copy lengths into ranks
		util::io::InitializeTile<
			KernelPolicy::LOG_LOADS_PER_TILE,
			KernelPolicy::LOG_LOAD_VEC_SIZE>::Copy(tile.rank, tile.row_length);

		if (!KernelPolicy::BENCHMARK) {

			// Scan tile of ranks, using an atomic add to reserve
			// space in the compacted queue, seeding ranks
			util::Sum<SizeT> scan_op;
			util::scan::CooperativeTileScan<KernelPolicy::LOAD_VEC_SIZE>::ScanTileWithEnqueue(
				raking_details,
				tile.rank,
				work_progress.GetQueueCounter<SizeT>(queue_index + 1),
				scan_op);
		}

		// Enqueue valid edge lists into outgoing queue
		tile.ExpandSerial(this);
	}
};



} // namespace serial_gather
} // namespace microbench
} // namespace bfs
} // namespace graph
} // namespace b40c

