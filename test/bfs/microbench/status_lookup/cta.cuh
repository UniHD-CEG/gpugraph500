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
 * Tile-processing functionality for BFS compaction upsweep kernels
 ******************************************************************************/

#pragma once

#include <b40c/util/io/modified_load.cuh>
#include <b40c/util/io/modified_store.cuh>
#include <b40c/util/io/initialize_tile.cuh>
#include <b40c/util/io/load_tile.cuh>
#include <b40c/util/io/store_tile.cuh>
#include <b40c/util/io/scatter_tile.cuh>
#include <b40c/util/cta_work_distribution.cuh>

#include <b40c/util/operators.cuh>
#include <b40c/util/reduction/serial_reduce.cuh>
#include <b40c/util/reduction/tree_reduce.cuh>

namespace b40c {
namespace graph {
namespace bfs {
namespace microbench {
namespace status_lookup {


texture<char, cudaTextureType1D, cudaReadModeElementType> bitmask_tex_ref;


/**
 * Cta
 */
template <typename KernelPolicy>
struct Cta
{
	//---------------------------------------------------------------------
	// Typedefs and Constants
	//---------------------------------------------------------------------

	typedef typename KernelPolicy::VertexId 		VertexId;
	typedef typename KernelPolicy::ValidFlag		ValidFlag;
	typedef typename KernelPolicy::VisitedMask 	VisitedMask;
	typedef typename KernelPolicy::SizeT 			SizeT;
	typedef typename KernelPolicy::ThreadId			ThreadId;
	typedef typename KernelPolicy::RakingDetails 		RakingDetails;
	typedef typename KernelPolicy::SmemStorage		SmemStorage;

	//---------------------------------------------------------------------
	// Members
	//---------------------------------------------------------------------

	// Current BFS iteration
	VertexId 				iteration;

	// Current BFS queue index
	VertexId 				queue_index;

	// Input and output device pointers
	VertexId 				*d_in;						// Incoming vertex ids
	SizeT 					*d_out_row_offsets;			// Compacted row offsets
	SizeT 					*d_out_row_lengths;			// Compacted row lengths
	VisitedMask 			*d_visited_mask;
	SizeT					*d_row_offsets;
	VertexId				*d_labels;

	// Work progress
	util::CtaWorkProgress	&work_progress;

	// Operational details for raking scan grid
	RakingDetails 			raking_details;

	SmemStorage				&smem_storage;


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

		// Dequeued vertex ids
		VertexId 	vertex_ids[LOADS_PER_TILE][LOAD_VEC_SIZE];
		SizeT		row_lengths[LOADS_PER_TILE][LOAD_VEC_SIZE];
		SizeT		row_offsets[LOADS_PER_TILE][LOAD_VEC_SIZE];

		// Whether or not the corresponding vertex_ids is valid for exploring
		ValidFlag 	flags[LOADS_PER_TILE][LOAD_VEC_SIZE];

		// Tile of local scatter offsets
		SizeT 		ranks[LOADS_PER_TILE][LOAD_VEC_SIZE];

		//---------------------------------------------------------------------
		// Helper Structures
		//---------------------------------------------------------------------

		/**
		 * Iterate over vertex id
		 */
		template <int LOAD, int VEC, int dummy = 0>
		struct Iterate
		{
			/**
			 * InitFlags
			 */
			static __device__ __forceinline__ void InitFlags(Tile *tile)
			{
				// Initially valid if vertex-id is valid
				tile->flags[LOAD][VEC] = (tile->vertex_ids[LOAD][VEC] == -1) ? 0 : 1;

				// Next
				Iterate<LOAD, VEC + 1>::InitFlags(tile);
			}


			/**
			 * BitmaskCull
			 */
			static __device__ __forceinline__ void BitmaskCull(
				Cta *cta,
				Tile *tile)
			{
				if (tile->flags[LOAD][VEC]) {

					// Location of mask byte to read
					SizeT mask_byte_offset = (tile->vertex_ids[LOAD][VEC] & KernelPolicy::VERTEX_ID_MASK) >> 3;

					// Read byte from from visited mask (tex)
					VisitedMask mask_byte = tex1Dfetch(
						bitmask_tex_ref,
						mask_byte_offset);

					// Bit in mask byte corresponding to current vertex id
					VisitedMask mask_bit = 1 << (tile->vertex_ids[LOAD][VEC] & 7);

					if ((mask_bit & mask_byte) == 0) {

						util::io::ModifiedLoad<util::io::ld::cg>::Ld(
							mask_byte, cta->d_visited_mask + mask_byte_offset);

						if ((mask_bit & mask_byte) == 0) {

							// Update with best effort
							mask_byte |= mask_bit;
							util::io::ModifiedStore<util::io::st::cg>::St(
								mask_byte,
								cta->d_visited_mask + mask_byte_offset);

							// Load source path of node
							VertexId source_path;
							util::io::ModifiedLoad<util::io::ld::cg>::Ld(
								source_path,
								cta->d_labels + tile->vertex_ids[LOAD][VEC]);

							cta->smem_storage.gathered = source_path;
						}
					}
				}

				// Next
				Iterate<LOAD, VEC + 1>::BitmaskCull(cta, tile);
			}


			/**
			 * HistoryCull
			 */
			static __device__ __forceinline__ void HistoryCull(
				Cta *cta,
				Tile *tile)
			{
				if (tile->flags[LOAD][VEC]) {

					int hash = ((typename KernelPolicy::UnsignedBits) tile->vertex_ids[LOAD][VEC]) % SmemStorage::HISTORY_HASH_ELEMENTS;
					VertexId retrieved = cta->smem_storage.history[hash];

					if (retrieved == tile->vertex_ids[LOAD][VEC]) {
						// Seen it
						tile->flags[LOAD][VEC] = 0;

					} else {
						// Update it
						cta->smem_storage.history[hash] = tile->vertex_ids[LOAD][VEC];
					}
				}

				// Next
				Iterate<LOAD, VEC + 1>::HistoryCull(cta, tile);
			}


			/**
			 * WarpCull
			 */
			static __device__ __forceinline__ void WarpCull(
				Cta *cta,
				Tile *tile)
			{
				if (tile->flags[LOAD][VEC]) {

					int warp_id 		= threadIdx.x >> 5;
					int hash 			= tile->vertex_ids[LOAD][VEC] & (SmemStorage::WARP_HASH_ELEMENTS - 1);

					cta->smem_storage.state.vid_hashtable[warp_id][hash] = tile->vertex_ids[LOAD][VEC];
					VertexId retrieved = cta->smem_storage.state.vid_hashtable[warp_id][hash];

					if (retrieved == tile->vertex_ids[LOAD][VEC]) {

						cta->smem_storage.state.vid_hashtable[warp_id][hash] = threadIdx.x;
						VertexId tid = cta->smem_storage.state.vid_hashtable[warp_id][hash];
						if (tid != threadIdx.x) {
							tile->flags[LOAD][VEC] = 0;
						}
					}
				}

				// Next
				Iterate<LOAD, VEC + 1>::WarpCull(cta, tile);
			}


			/**
			 * AtomicCull
			 */
			static __device__ __forceinline__ void AtomicCull(
				Cta *cta,
				Tile *tile)
			{
				if (tile->flags[LOAD][VEC]) {

					VertexId distance = atomicCAS(
						cta->d_labels + tile->vertex_ids[LOAD][VEC],
						-1,
						cta->iteration);

					if (distance == -1) {

						// Unvisited: get row offset/length
						tile->row_offsets[LOAD][VEC] = cta->d_row_offsets[tile->vertex_ids[LOAD][VEC]];
						tile->row_lengths[LOAD][VEC] = cta->d_row_offsets[tile->vertex_ids[LOAD][VEC] + 1] -
							tile->row_offsets[LOAD][VEC];

					} else {
						// Visited
						tile->flags[LOAD][VEC] = 0;
					}
				}

				// Next
				Iterate<LOAD, VEC + 1>::AtomicCull(cta, tile);
			}
		};


		/**
		 * Iterate next load
		 */
		template <int LOAD, int dummy>
		struct Iterate<LOAD, LOAD_VEC_SIZE, dummy>
		{
			// InitFlags
			static __device__ __forceinline__ void InitFlags(Tile *tile)
			{
				Iterate<LOAD + 1, 0>::InitFlags(tile);
			}

			// BitmaskCull
			static __device__ __forceinline__ void BitmaskCull(Cta *cta, Tile *tile)
			{
				Iterate<LOAD + 1, 0>::BitmaskCull(cta, tile);
			}

			// HistoryCull
			static __device__ __forceinline__ void HistoryCull(Cta *cta, Tile *tile)
			{
				Iterate<LOAD + 1, 0>::HistoryCull(cta, tile);
			}

			// WarpCull
			static __device__ __forceinline__ void WarpCull(Cta *cta, Tile *tile)
			{
				Iterate<LOAD + 1, 0>::WarpCull(cta, tile);
			}

			// AtomicCull
			static __device__ __forceinline__ void AtomicCull(Cta *cta, Tile *tile)
			{
				Iterate<LOAD + 1, 0>::AtomicCull(cta, tile);
			}
		};



		/**
		 * Terminate iteration
		 */
		template <int dummy>
		struct Iterate<LOADS_PER_TILE, 0, dummy>
		{
			// InitFlags
			static __device__ __forceinline__ void InitFlags(Tile *tile) {}

			// BitmaskCull
			static __device__ __forceinline__ void BitmaskCull(Cta *cta, Tile *tile) {}

			// HistoryCull
			static __device__ __forceinline__ void HistoryCull(Cta *cta, Tile *tile) {}

			// WarpCull
			static __device__ __forceinline__ void WarpCull(Cta *cta, Tile *tile) {}

			// AtomicCull
			static __device__ __forceinline__ void AtomicCull(Cta *cta, Tile *tile) {}
		};


		//---------------------------------------------------------------------
		// Interface
		//---------------------------------------------------------------------

		/**
		 * Initializer
		 */
		__device__ __forceinline__ void InitFlags()
		{
			Iterate<0, 0>::InitFlags(this);
		}

		/**
		 * Culls vertices based upon whether or not we've set a bit for them
		 * in the d_visited_mask bitmask
		 */
		__device__ __forceinline__ void BitmaskCull(Cta *cta)
		{
			Iterate<0, 0>::BitmaskCull(cta, this);
		}

		/**
		 * Culls vertices based upon local duplicate collisions
		 */
		__device__ __forceinline__ void LocalCull(Cta *cta)
		{
			Iterate<0, 0>::HistoryCull(cta, this);
			Iterate<0, 0>::WarpCull(cta, this);
		}

		/**
		 * Does perfect vertex culling
		 */
		__device__ __forceinline__ void AtomicCull(Cta *cta)
		{
			Iterate<0, 0>::AtomicCull(cta, this);
		}
	};




	//---------------------------------------------------------------------
	// Methods
	//---------------------------------------------------------------------

	/**
	 * Constructor
	 */
	__device__ __forceinline__ Cta(
		VertexId 				iteration,
		VertexId				queue_index,
		SmemStorage 			&smem_storage,
		VertexId 				*d_in,
		SizeT 					*d_out_row_offsets,
		SizeT 					*d_out_row_lengths,
		VisitedMask 			*d_visited_mask,
		SizeT					*d_row_offsets,
		VertexId				*d_labels,
		util::CtaWorkProgress	&work_progress) :

			iteration(iteration),
			queue_index(queue_index),
			smem_storage(smem_storage),
			raking_details(
				smem_storage.state.raking_elements,
				smem_storage.state.warpscan,
				0),
			d_in(d_in),
			d_out_row_offsets(d_out_row_offsets),
			d_out_row_lengths(d_out_row_lengths),
			d_visited_mask(d_visited_mask),
			d_row_offsets(d_row_offsets),
			d_labels(d_labels),
			work_progress(work_progress)

	{
		// Initialize history duplicate-filter
		for (int offset = threadIdx.x; offset < SmemStorage::HISTORY_HASH_ELEMENTS; offset += KernelPolicy::THREADS) {
			smem_storage.history[offset] = -1;
		}
	}


	/**
	 * Process a single, full tile
	 */
	__device__ __forceinline__ void ProcessTile(
		SizeT cta_offset,
		const SizeT &guarded_elements = KernelPolicy::TILE_ELEMENTS)
	{
		Tile<KernelPolicy::LOG_LOADS_PER_TILE, KernelPolicy::LOG_LOAD_VEC_SIZE> tile;

		// Load tile
		util::io::LoadTile<
			KernelPolicy::LOG_LOADS_PER_TILE,
			KernelPolicy::LOG_LOAD_VEC_SIZE,
			KernelPolicy::THREADS,
			KernelPolicy::READ_MODIFIER,
			false>::LoadValid(
				tile.vertex_ids,
				d_in,
				cta_offset,
				guarded_elements,
				(VertexId) -1);

		// Init valid flags
		tile.InitFlags();

		// Cull using global visited mask
		tile.BitmaskCull(this);

/* Not needed for micro-benchmarking
		// Cull using local collision hashing
		tile.LocalCull(this);
*/

		if (!KernelPolicy::BENCHMARK) {

			// Cull using atomic CAS
			tile.AtomicCull(this);

			// Copy flags into ranks
			util::io::InitializeTile<
				KernelPolicy::LOG_LOADS_PER_TILE,
				KernelPolicy::LOG_LOAD_VEC_SIZE>::Copy(tile.ranks, tile.flags);

			// Protect repurposable storage that backs both raking lanes and local cull scratch
			__syncthreads();

			// Scan tile of ranks, using an atomic add to reserve
			// space in the compacted queue, seeding ranks
			util::Sum<SizeT> scan_op;
			util::scan::CooperativeTileScan<KernelPolicy::LOAD_VEC_SIZE>::ScanTileWithEnqueue(
				raking_details,
				tile.ranks,
				work_progress.GetQueueCounter<SizeT>(queue_index + 1),
				scan_op);

			// Protect repurposable storage that backs both raking lanes and local cull scratch
			__syncthreads();

			// Scatter valid row offsets
			util::io::ScatterTile<
				KernelPolicy::LOG_LOADS_PER_TILE,
				KernelPolicy::LOG_LOAD_VEC_SIZE,
				KernelPolicy::THREADS,
				KernelPolicy::WRITE_MODIFIER>::Scatter(
					d_out_row_offsets,
					tile.row_offsets,
					tile.flags,
					tile.ranks);

			// Scatter valid row lengths
			util::io::ScatterTile<
			KernelPolicy::LOG_LOADS_PER_TILE,
			KernelPolicy::LOG_LOAD_VEC_SIZE,
				KernelPolicy::THREADS,
				KernelPolicy::WRITE_MODIFIER>::Scatter(
					d_out_row_lengths,
					tile.row_lengths,
					tile.flags,
					tile.ranks);
		}
	}
};


} // namespace status_lookup
} // namespace microbench
} // namespace bfs
} // namespace graph
} // namespace b40c

