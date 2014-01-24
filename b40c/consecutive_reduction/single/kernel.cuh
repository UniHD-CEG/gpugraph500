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
 * Consecutive removal single-CTA scan kernel
 ******************************************************************************/

#pragma once

#include <b40c/consecutive_reduction/downsweep/cta.cuh>

namespace b40c {
namespace consecutive_reduction {
namespace single {


/**
 *  Consecutive removal single-CTA scan pass
 */
template <typename KernelPolicy>
__device__ __forceinline__ void SinglePass(
	typename KernelPolicy::KeyType			*d_in_keys,
	typename KernelPolicy::KeyType			*d_out_keys,
	typename KernelPolicy::ValueType		*d_in_values,
	typename KernelPolicy::ValueType		*d_out_values,
	typename KernelPolicy::SizeT			*d_num_compacted,
	typename KernelPolicy::SizeT 			num_elements,
	typename KernelPolicy::ReductionOp 		reduction_op,
	typename KernelPolicy::EqualityOp		equality_op,
	typename KernelPolicy::SmemStorage		&smem_storage)
{
	typedef downsweep::Cta<KernelPolicy> 			Cta;
	typedef typename KernelPolicy::SizeT 			SizeT;
	typedef typename KernelPolicy::SoaScanOperator	SoaScanOperator;

	// Exit if we're not the first CTA
	if (blockIdx.x > 0) return;

	// CTA processing abstraction
	Cta cta(
		smem_storage,
		d_in_keys,
		d_out_keys,
		d_in_values,
		d_out_values,
		d_num_compacted,
		SoaScanOperator(reduction_op),
		equality_op);

	// Number of elements in (the last) partially-full tile (requires guarded loads)
	SizeT guarded_elements = num_elements & (KernelPolicy::TILE_ELEMENTS - 1);

	// Offset of final, partially-full tile (requires guarded loads)
	SizeT guarded_offset = num_elements - guarded_elements;

	util::CtaWorkLimits<SizeT> work_limits(
		0,					// Offset at which this CTA begins processing
		num_elements,		// Total number of elements for this CTA to process
		guarded_offset, 	// Offset of final, partially-full tile (requires guarded loads)
		guarded_elements,	// Number of elements in partially-full tile
		num_elements,		// Offset at which this CTA is out-of-bounds
		true);				// If this block is the last block in the grid with any work

	cta.ProcessWorkRange(work_limits);
}


/**
 * Consecutive removal single-CTA scan kernel entrypoint
 */
template <typename KernelPolicy>
__launch_bounds__ (KernelPolicy::THREADS, KernelPolicy::MIN_CTA_OCCUPANCY)
__global__ 
void Kernel(
	typename KernelPolicy::KeyType			*d_in_keys,
	typename KernelPolicy::KeyType			*d_out_keys,
	typename KernelPolicy::ValueType		*d_in_values,
	typename KernelPolicy::ValueType		*d_out_values,
	typename KernelPolicy::SizeT			*d_num_compacted,
	typename KernelPolicy::SizeT 			num_elements,
	typename KernelPolicy::ReductionOp 		reduction_op,
	typename KernelPolicy::EqualityOp		equality_op)
{
	__shared__ typename KernelPolicy::SmemStorage smem_storage;

	SinglePass<KernelPolicy>(
		d_in_keys,
		d_out_keys,
		d_in_values,
		d_out_values,
		d_num_compacted,
		num_elements,
		reduction_op,
		equality_op,
		smem_storage);
}

} // namespace single
} // namespace consecutive_reduction
} // namespace b40c

