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
#include <b40c/util/operators.cuh>

#include <b40c/util/soa_tuple.cuh>
#include <b40c/util/scan/soa/cooperative_soa_scan.cuh>

namespace b40c {
namespace graph {
namespace bfs {
namespace microbench {
namespace neighbor_gather {


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
	typedef typename SmemStorage::State::WarpComm 	WarpComm;

	typedef typename KernelPolicy::SoaScanOp		SoaScanOp;
	typedef typename KernelPolicy::RakingSoaDetails 	RakingSoaDetails;
	typedef typename KernelPolicy::TileTuple 		TileTuple;

	typedef util::Tuple<
		SizeT (*)[KernelPolicy::LOAD_VEC_SIZE],
		SizeT (*)[KernelPolicy::LOAD_VEC_SIZE]> 	RankSoa;

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
	RakingSoaDetails 			raking_soa_details;

	// Smem storage
	SmemStorage 			&smem_storage;


	/**
	 * BitmaskCull
	 */
	static __device__ __forceinline__ void BitmaskCull(
		Cta *cta,
		VertexId vertex)
	{
/*
		// Location of mask byte to read
		SizeT mask_byte_offset = (vertex & KernelPolicy::VERTEX_ID_MASK) >> 3;

		// Read byte from from visited mask (tex)
		VisitedMask mask_byte = tex1Dfetch(
			bitmask_tex_ref,
			mask_byte_offset);

		// Bit in mask byte corresponding to current vertex id
		VisitedMask mask_bit = 1 << (vertex & 7);

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
					cta->d_labels + vertex);

				vertex = source_path;
			}
		}
*/
		cta->smem_storage.gathered = vertex;
	}

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

		typedef typename util::VecType<SizeT, 2>::Type Vec2SizeT;


		//---------------------------------------------------------------------
		// Members
		//---------------------------------------------------------------------

		SizeT		row_offset[LOADS_PER_TILE][LOAD_VEC_SIZE];
		SizeT		row_length[LOADS_PER_TILE][LOAD_VEC_SIZE];

		SizeT		coarse_row_rank[LOADS_PER_TILE][LOAD_VEC_SIZE];
		SizeT		fine_row_rank[LOADS_PER_TILE][LOAD_VEC_SIZE];
		SizeT		row_progress[LOADS_PER_TILE][LOAD_VEC_SIZE];

		SizeT 		fine_count;
		SizeT		progress;


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
			 * Init
			 */
			static __device__ __forceinline__ void Init(Tile *tile)
			{
				tile->row_length[LOAD][VEC] = 0;
				tile->row_progress[LOAD][VEC] = 0;

				Iterate<LOAD, VEC + 1>::Init(tile);
			}

			/**
			 * Categorize
			 */
			static __device__ __forceinline__ void Categorize(Cta *cta, Tile *tile)
			{
				tile->fine_row_rank[LOAD][VEC] = (tile->row_length[LOAD][VEC] < KernelPolicy::WARP_GATHER_THRESHOLD) ?
					tile->row_length[LOAD][VEC] : 0;

				tile->coarse_row_rank[LOAD][VEC] = (tile->row_length[LOAD][VEC] < KernelPolicy::WARP_GATHER_THRESHOLD) ?
					0 : tile->row_length[LOAD][VEC];
/*
				tile->fine_row_rank[LOAD][VEC] = tile->row_length[LOAD][VEC];
				tile->coarse_row_rank[LOAD][VEC] = 0;
*/
				Iterate<LOAD, VEC + 1>::Categorize(cta, tile);
			}


			/**
			 * Expand by CTA
			 */
			static __device__ __forceinline__ void ExpandByCta(Cta *cta, Tile *tile)
			{
				// CTA-based expansion/loading
				while (true) {

					if (threadIdx.x == 0) {
						cta->smem_storage.state.cta_comm = KernelPolicy::THREADS;
					}

					__syncthreads();

					if (tile->row_length[LOAD][VEC] >= KernelPolicy::CTA_GATHER_THRESHOLD) {
						cta->smem_storage.state.cta_comm = threadIdx.x;
					}

					__syncthreads();

					int owner = cta->smem_storage.state.cta_comm;
					if (owner == KernelPolicy::THREADS) {
						break;
					}

					if (owner == threadIdx.x) {

						// Got control of the CTA
						cta->smem_storage.state.warp_comm[0][0] = tile->row_offset[LOAD][VEC];										// start
						cta->smem_storage.state.warp_comm[0][1] = tile->coarse_row_rank[LOAD][VEC];								// queue rank
						cta->smem_storage.state.warp_comm[0][2] = tile->row_offset[LOAD][VEC] + tile->row_length[LOAD][VEC];		// oob

						// Unset row length
						tile->row_length[LOAD][VEC] = 0;
					}

					__syncthreads();

					SizeT coop_offset 	= cta->smem_storage.state.warp_comm[0][0] + threadIdx.x;
					SizeT coop_rank	 	= cta->smem_storage.state.warp_comm[0][1] + threadIdx.x;
					SizeT coop_oob 		= cta->smem_storage.state.warp_comm[0][2];

					while (coop_offset < coop_oob) {

						// Gather
						VertexId neighbor_id;
						util::io::ModifiedLoad<KernelPolicy::COLUMN_READ_MODIFIER>::Ld(
							neighbor_id,
							cta->d_column_indices + coop_offset);

							BitmaskCull(cta, neighbor_id);

						if (!KernelPolicy::BENCHMARK) {

							// Scatter neighbor
							util::io::ModifiedStore<KernelPolicy::QUEUE_WRITE_MODIFIER>::St(
								neighbor_id,
								cta->d_out + cta->smem_storage.state.coarse_enqueue_offset + coop_rank);
						}

						coop_offset += KernelPolicy::THREADS;
						coop_rank += KernelPolicy::THREADS;
					}
				}

				// Next vector element
				Iterate<LOAD, VEC + 1>::ExpandByCta(cta, tile);
			}

			/**
			 * Expand by warp
			 */
			static __device__ __forceinline__ void ExpandByWarp(Cta *cta, Tile *tile)
			{
				// Warp-based expansion/loading
				int warp_id = threadIdx.x >> B40C_LOG_WARP_THREADS(KernelPolicy::CUDA_ARCH);
				int lane_id = util::LaneId();

				while (__any(tile->row_length[LOAD][VEC] >= KernelPolicy::WARP_GATHER_THRESHOLD)) {

					if (tile->row_length[LOAD][VEC] >= KernelPolicy::WARP_GATHER_THRESHOLD) {
						// Vie for control of the warp
						cta->smem_storage.state.warp_comm[warp_id][0] = lane_id;
					}

					if (lane_id == cta->smem_storage.state.warp_comm[warp_id][0]) {

						// Got control of the warp
						cta->smem_storage.state.warp_comm[warp_id][0] = tile->row_offset[LOAD][VEC];									// start
						cta->smem_storage.state.warp_comm[warp_id][1] = tile->coarse_row_rank[LOAD][VEC];								// queue rank
						cta->smem_storage.state.warp_comm[warp_id][2] = tile->row_offset[LOAD][VEC] + tile->row_length[LOAD][VEC];		// oob

						// Unset row length
						tile->row_length[LOAD][VEC] = 0;
					}

					SizeT coop_offset 	= cta->smem_storage.state.warp_comm[warp_id][0] + lane_id;
					SizeT coop_rank 	= cta->smem_storage.state.warp_comm[warp_id][1] + lane_id;
					SizeT coop_oob 		= cta->smem_storage.state.warp_comm[warp_id][2];

					while (coop_offset < coop_oob) {

						// Gather
						VertexId neighbor_id;
						util::io::ModifiedLoad<KernelPolicy::COLUMN_READ_MODIFIER>::Ld(
							neighbor_id, cta->d_column_indices + coop_offset);

						BitmaskCull(cta, neighbor_id);

						if (!KernelPolicy::BENCHMARK) {

							// Scatter neighbor
							util::io::ModifiedStore<KernelPolicy::QUEUE_WRITE_MODIFIER>::St(
								neighbor_id, cta->d_out + cta->smem_storage.state.coarse_enqueue_offset + coop_rank);
						}

						coop_offset += B40C_WARP_THREADS(KernelPolicy::CUDA_ARCH);
						coop_rank += B40C_WARP_THREADS(KernelPolicy::CUDA_ARCH);
					}

				}

				// Next vector element
				Iterate<LOAD, VEC + 1>::ExpandByWarp(cta, tile);
			}


			/**
			 * Expand by scan
			 */
			static __device__ __forceinline__ void ExpandByScan(Cta *cta, Tile *tile)
			{
				// Attempt to make further progress on this dequeued item's neighbor
				// list if its current offset into local scratch is in range
				SizeT scratch_offset = tile->fine_row_rank[LOAD][VEC] + tile->row_progress[LOAD][VEC] - tile->progress;

				while ((tile->row_progress[LOAD][VEC] < tile->row_length[LOAD][VEC]) &&
					(scratch_offset < SmemStorage::OFFSET_ELEMENTS))
				{
					// Put gather offset into scratch space
					cta->smem_storage.offset_scratch[scratch_offset] = tile->row_offset[LOAD][VEC] + tile->row_progress[LOAD][VEC];

					tile->row_progress[LOAD][VEC]++;
					scratch_offset++;
				}

				// Next vector element
				Iterate<LOAD, VEC + 1>::ExpandByScan(cta, tile);
			}
		};


		/**
		 * Iterate next load
		 */
		template <int LOAD, int dummy>
		struct Iterate<LOAD, LOAD_VEC_SIZE, dummy>
		{
			/**
			 * Init
			 */
			static __device__ __forceinline__ void Init(Tile *tile)
			{
				Iterate<LOAD + 1, 0>::Init(tile);
			}

			/**
			 * Categorize
			 */
			static __device__ __forceinline__ void Categorize(Cta *cta, Tile *tile)
			{
				Iterate<LOAD + 1, 0>::Categorize(cta, tile);
			}

			/**
			 * Expand by CTA
			 */
			static __device__ __forceinline__ void ExpandByCta(Cta *cta, Tile *tile)
			{
				Iterate<LOAD + 1, 0>::ExpandByCta(cta, tile);
			}

			/**
			 * Expand by warp
			 */
			static __device__ __forceinline__ void ExpandByWarp(Cta *cta, Tile *tile)
			{
				Iterate<LOAD + 1, 0>::ExpandByWarp(cta, tile);
			}

			/**
			 * Expand by scan
			 */
			static __device__ __forceinline__ void ExpandByScan(Cta *cta, Tile *tile)
			{
				Iterate<LOAD + 1, 0>::ExpandByScan(cta, tile);
			}
		};

		/**
		 * Terminate
		 */
		template <int dummy>
		struct Iterate<LOADS_PER_TILE, 0, dummy>
		{
			// Init
			static __device__ __forceinline__ void Init(Tile *tile) {}

			// Categorize
			static __device__ __forceinline__ void Categorize(Cta *cta, Tile *tile) {}

			// ExpandByCta
			static __device__ __forceinline__ void ExpandByCta(Cta *cta, Tile *tile) {}

			// ExpandByWarp
			static __device__ __forceinline__ void ExpandByWarp(Cta *cta, Tile *tile) {}

			// ExpandByScan
			static __device__ __forceinline__ void ExpandByScan(Cta *cta, Tile *tile) {}
		};


		//---------------------------------------------------------------------
		// Interface
		//---------------------------------------------------------------------

		/**
		 * Constructor
		 */
		__device__ __forceinline__ Tile()
		{
			Iterate<0, 0>::Init(this);
		}

		/**
		 * Categorize dequeued vertices, updating source path if necessary and
		 * obtaining edge-list details
		 */
		__device__ __forceinline__ void Categorize(Cta *cta)
		{
			Iterate<0, 0>::Categorize(cta, this);
		}

		/**
		 * Expands neighbor lists for valid vertices at CTA-expansion granularity
		 */
		__device__ __forceinline__ void ExpandByCta(Cta *cta)
		{
			Iterate<0, 0>::ExpandByCta(cta, this);
		}

		/**
		 * Expands neighbor lists for valid vertices a warp-expansion granularity
		 */
		__device__ __forceinline__ void ExpandByWarp(Cta *cta)
		{
			Iterate<0, 0>::ExpandByWarp(cta, this);
		}

		/**
		 * Expands neighbor lists by local scan rank
		 */
		__device__ __forceinline__ void ExpandByScan(Cta *cta)
		{
			Iterate<0, 0>::ExpandByScan(cta, this);
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
			raking_soa_details(
				typename RakingSoaDetails::GridStorageSoa(
					smem_storage.coarse_raking_elements,
					smem_storage.fine_raking_elements),
				typename RakingSoaDetails::WarpscanSoa(
					smem_storage.state.coarse_warpscan,
					smem_storage.state.fine_warpscan),
				TileTuple(0, 0)),
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

		// Categorize dequeued vertices, updating source path and obtaining
		// edge-list details
		tile.Categorize(this);

		// Scan tile with carry update in raking threads
		SoaScanOp scan_op;
		TileTuple totals;
		util::scan::soa::CooperativeSoaTileScan<KernelPolicy::LOAD_VEC_SIZE>::ScanTile(
			totals,
			raking_soa_details,
			RankSoa(tile.coarse_row_rank, tile.fine_row_rank),
			scan_op);

		SizeT coarse_count = totals.t0;
		tile.fine_count = totals.t1;

		if (!KernelPolicy::BENCHMARK) {

			// Use a single atomic add to reserve room in the queue
			if (threadIdx.x == 0) {

				smem_storage.state.coarse_enqueue_offset = work_progress.Enqueue(
					coarse_count + tile.fine_count,
					queue_index + 1);

				smem_storage.state.fine_enqueue_offset = smem_storage.state.coarse_enqueue_offset + coarse_count;
			}
		}

		// Enqueue valid edge lists into outgoing queue
		tile.ExpandByCta(this);

		// Enqueue valid edge lists into outgoing queue
		tile.ExpandByWarp(this);


		//
		// Enqueue the adjacency lists of unvisited node-IDs by repeatedly
		// gathering edges into the scratch space, and then
		// having the entire CTA copy the scratch pool into the outgoing
		// frontier queue.
		//

		tile.progress = 0;
		while (tile.progress < tile.fine_count) {

			// Fill the scratch space with gather-offsets for neighbor-lists.
			tile.ExpandByScan(this);

			__syncthreads();

			// Copy scratch space into queue
			int scratch_remainder = B40C_MIN(SmemStorage::OFFSET_ELEMENTS, tile.fine_count - tile.progress);

			for (int scratch_offset = threadIdx.x;
				scratch_offset < scratch_remainder;
				scratch_offset += KernelPolicy::THREADS)
			{
				// Gather a neighbor
				VertexId neighbor_id;

				util::io::ModifiedLoad<KernelPolicy::COLUMN_READ_MODIFIER>::Ld(
					neighbor_id,
					d_column_indices + smem_storage.offset_scratch[scratch_offset]);

				BitmaskCull(this, neighbor_id);

				if (!KernelPolicy::BENCHMARK) {

					// Scatter it into queue
					util::io::ModifiedStore<KernelPolicy::QUEUE_WRITE_MODIFIER>::St(
						neighbor_id,
						d_out + smem_storage.state.fine_enqueue_offset + tile.progress + scratch_offset);
				}
			}

			tile.progress += SmemStorage::OFFSET_ELEMENTS;

			__syncthreads();
		}
	}
};



} // namespace neighbor_gather
} // namespace microbench
} // namespace bfs
} // namespace graph
} // namespace b40c

