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
 * Upsweep tile processing abstraction
 ******************************************************************************/

#pragma once

#include <b40c/util/io/modified_load.cuh>
#include <b40c/util/io/modified_store.cuh>
#include <b40c/util/io/load_tile.cuh>
#include <b40c/util/io/store_tile.cuh>
#include <b40c/util/io/initialize_tile.cuh>

#include <b40c/partition/upsweep/tile.cuh>

#include <b40c/graph/bfs/two_phase/contract_atomic/cta.cuh>

#include <b40c/radix_sort/sort_utils.cuh>

namespace b40c {
namespace graph {
namespace bfs {
namespace partition_contract {
namespace upsweep {



/**
 * Tile
 *
 * Derives from partition::upsweep::Tile
 */
template <
	int LOG_LOADS_PER_TILE,
	int LOG_LOAD_VEC_SIZE,
	typename KernelPolicy>
struct Tile :
	partition::upsweep::Tile<
		LOG_LOADS_PER_TILE,
		LOG_LOAD_VEC_SIZE,
		KernelPolicy,
		Tile<LOG_LOADS_PER_TILE, LOG_LOAD_VEC_SIZE, KernelPolicy> >					// This class
{
	//---------------------------------------------------------------------
	// Typedefs and Constants
	//---------------------------------------------------------------------

	typedef typename KernelPolicy::VertexId 		VertexId;
	typedef typename KernelPolicy::ValidFlag		ValidFlag;
	typedef typename KernelPolicy::VisitedMask 	VisitedMask;
	typedef typename KernelPolicy::KeyType 			KeyType;
	typedef typename KernelPolicy::SizeT 			SizeT;

	enum {
		LOAD_VEC_SIZE 		= Tile::LOAD_VEC_SIZE,
		LOADS_PER_TILE 		= Tile::LOADS_PER_TILE,
	};

	//---------------------------------------------------------------------
	// Members
	//---------------------------------------------------------------------

	// Whether or not the corresponding vertex_id is valid for exploring
	ValidFlag 	valid[Tile::LOADS_PER_TILE][LOAD_VEC_SIZE];


	//---------------------------------------------------------------------
	// Iteration Structures
	//---------------------------------------------------------------------

	/**
	 * Iterate over vertex id
	 */
	template <int LOAD, int VEC, int dummy = 0>
	struct Iterate
	{
		/**
		 * BitmaskCull
		 */
		template <typename Cta>
		static __device__ __forceinline__ void BitmaskCull(
			Cta *cta,
			Tile *tile)
		{
			if (tile->valid[LOAD][VEC]) {

				// Location of mask byte to read
				SizeT mask_byte_offset = (tile->keys[LOAD][VEC] & KernelPolicy::VERTEX_ID_MASK) >> 3;

				// Bit in mask byte corresponding to current vertex id
				VisitedMask mask_bit = 1 << (tile->keys[LOAD][VEC] & 7);

				// Read byte from from visited mask (tex)
				VisitedMask mask_byte = tex1Dfetch(
					two_phase::contract_atomic::BitmaskTex<VisitedMask>::ref,
					mask_byte_offset);

				if (mask_bit & mask_byte) {

					// Seen it
					tile->valid[LOAD][VEC] = 0;

				} else {

					util::io::ModifiedLoad<util::io::ld::cg>::Ld(
						mask_byte, cta->d_visited_mask + mask_byte_offset);

					if (mask_bit & mask_byte) {

						// Seen it
						tile->valid[LOAD][VEC] = 0;

					} else {

						// Update with best effort
						mask_byte |= mask_bit;
						util::io::ModifiedStore<util::io::st::cg>::St(
							mask_byte,
							cta->d_visited_mask + mask_byte_offset);
					}
				}

			}

			// Next
			Iterate<LOAD, VEC + 1>::BitmaskCull(cta, tile);
		}


		/**
		 * HistoryCull
		 */
		template <typename Cta>
		static __device__ __forceinline__ void HistoryCull(
			Cta *cta,
			Tile *tile)
		{
			if (tile->valid[LOAD][VEC]) {

				int hash = ((typename KernelPolicy::UnsignedBits) tile->keys[LOAD][VEC]) % Cta::SmemStorage::HISTORY_HASH_ELEMENTS;
				VertexId retrieved = cta->history[hash];

				if (retrieved == tile->keys[LOAD][VEC]) {
					// Seen it
					tile->valid[LOAD][VEC] = 0;
				} else {
					// Update it
					cta->history[hash] = tile->keys[LOAD][VEC];
				}
			}

			// Next
			Iterate<LOAD, VEC + 1>::HistoryCull(cta, tile);
		}


		/**
		 * WarpCull
		 */
		template <typename Cta>
		static __device__ __forceinline__ void WarpCull(
			Cta *cta,
			Tile *tile)
		{
			if (tile->valid[LOAD][VEC]) {

				int warp_id 		= threadIdx.x >> 5;
				int hash 			= tile->keys[LOAD][VEC] & (Cta::SmemStorage::WARP_HASH_ELEMENTS - 1);

				cta->vid_hashtable[warp_id][hash] = tile->keys[LOAD][VEC];
				VertexId retrieved = cta->vid_hashtable[warp_id][hash];

				if (retrieved == tile->keys[LOAD][VEC]) {

					cta->vid_hashtable[warp_id][hash] = threadIdx.x;
					VertexId tid = cta->vid_hashtable[warp_id][hash];
					if (tid != threadIdx.x) {
						tile->valid[LOAD][VEC] = 0;
					}
				}
			}

			// Next
			Iterate<LOAD, VEC + 1>::WarpCull(cta, tile);
		}
	};


	/**
	 * Iterate next load
	 */
	template <int LOAD, int dummy>
	struct Iterate<LOAD, LOAD_VEC_SIZE, dummy>
	{
		// BitmaskCull
		template <typename Cta>
		static __device__ __forceinline__ void BitmaskCull(Cta *cta, Tile *tile)
		{
			Iterate<LOAD + 1, 0>::BitmaskCull(cta, tile);
		}

		// HistoryCull
		template <typename Cta>
		static __device__ __forceinline__ void HistoryCull(Cta *cta, Tile *tile)
		{
			Iterate<LOAD + 1, 0>::HistoryCull(cta, tile);
		}

		// WarpCull
		template <typename Cta>
		static __device__ __forceinline__ void WarpCull(Cta *cta, Tile *tile)
		{
			Iterate<LOAD + 1, 0>::WarpCull(cta, tile);
		}
	};



	/**
	 * Terminate iteration
	 */
	template <int dummy>
	struct Iterate<LOADS_PER_TILE, 0, dummy>
	{
		// BitmaskCull
		template <typename Cta>
		static __device__ __forceinline__ void BitmaskCull(Cta *cta, Tile *tile) {}

		// HistoryCull
		template <typename Cta>
		static __device__ __forceinline__ void HistoryCull(Cta *cta, Tile *tile) {}

		// WarpCull
		template <typename Cta>
		static __device__ __forceinline__ void WarpCull(Cta *cta, Tile *tile) {}
	};


	//---------------------------------------------------------------------
	// Interface
	//---------------------------------------------------------------------

	/**
	 * Culls vertices based upon whether or not we've set a bit for them
	 * in the d_visited_mask bitmask
	 */
	template <typename Cta>
	__device__ __forceinline__ void BitmaskCull(Cta *cta)
	{
		Iterate<0, 0>::BitmaskCull(cta, this);
	}

	/**
	 * Culls vertices based upon whether or not we've set a bit for them
	 * in the d_visited_mask bitmask
	 */
	template <typename Cta>
	__device__ __forceinline__ void LocalCull(Cta *cta)
	{
		Iterate<0, 0>::HistoryCull(cta, this);
		Iterate<0, 0>::WarpCull(cta, this);
	}


	//---------------------------------------------------------------------
	// Derived Interface
	//---------------------------------------------------------------------

	/**
	 * Returns the bin into which the specified key (vertex-id) is to be placed
	 */
	template <typename Cta>
	__device__ __forceinline__ int DecodeBin(KeyType key, Cta *cta)
	{
		return ((typename KernelPolicy::UnsignedBits) key) >> KernelPolicy::GPU_MASK_SHIFT;
	}


	/**
	 * Returns whether or not the key (vertex-id) is valid.
	 *
	 * Can be overloaded.
	 */
	template <int LOAD, int VEC>
	__device__ __forceinline__ bool IsValid()
	{
		return valid[LOAD][VEC];
	}


	/**
	 * Loads keys (vertex-ids) into the tile
	 */
	template <typename Cta>
	__device__ __forceinline__ void LoadKeys(
		Cta *cta,
		SizeT cta_offset)
	{
		// Read tile of keys
		util::io::LoadTile<
			LOG_LOADS_PER_TILE,
			LOG_LOAD_VEC_SIZE,
			KernelPolicy::THREADS,
			KernelPolicy::READ_MODIFIER,
			false>::LoadValid(
				(KeyType (*)[LOAD_VEC_SIZE]) this->keys,
				cta->d_in_keys,
				cta_offset);

		// Initialize valid flags
		util::io::InitializeTile<
			LOG_LOADS_PER_TILE,
			LOG_LOAD_VEC_SIZE,
			KernelPolicy::THREADS>::Init(valid, 1);

		// Cull valid flags using global visited mask
		BitmaskCull(cta);

		// Cull valid flags using local collision hashing
		LocalCull(cta);
	}


	/**
	 * Stores flags from the tile
	 */
	template <typename Cta>
	__device__ __forceinline__ void StoreKeys(
		Cta *cta,
		SizeT cta_offset)
	{
		// Store flags
		util::io::StoreTile<
			LOG_LOADS_PER_TILE,
			LOG_LOAD_VEC_SIZE,
			KernelPolicy::THREADS,
			KernelPolicy::WRITE_MODIFIER,
			false>::Store(
				valid,
				cta->d_flags_out,
				cta_offset);
	}
};



} // namespace upsweep
} // namespace partition_contract
} // namespace bfs
} // namespace graph
} // namespace b40c
