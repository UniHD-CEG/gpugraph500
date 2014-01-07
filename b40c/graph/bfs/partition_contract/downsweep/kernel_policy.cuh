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
 * Downsweep kernel configuration policy
 ******************************************************************************/

#pragma once

#include <b40c/util/cuda_properties.cuh>
#include <b40c/util/cta_work_distribution.cuh>

#include <b40c/partition/downsweep/kernel_policy.cuh>

namespace b40c {
namespace graph {
namespace bfs {
namespace partition_contract {
namespace downsweep {

/**
 * A detailed downsweep kernel configuration policy type that specializes kernel
 * code for a specific contraction pass. It encapsulates tuning configuration
 * policy details derived from TuningPolicy.
 */
template <
	typename TuningPolicy,				// Partition policy

	// Behavioral control parameters
	bool _INSTRUMENT>					// Whether or not we want instrumentation logic generated
struct KernelPolicy :
	partition::downsweep::KernelPolicy<TuningPolicy>
{
	typedef partition::downsweep::KernelPolicy<TuningPolicy> 	Base;			// Base class
	typedef typename TuningPolicy::SizeT 						SizeT;


	/**
	 * Shared storage
	 */
	struct SmemStorage : Base::SmemStorage
	{
		// Shared work-processing limits
		util::CtaWorkDistribution<SizeT>	work_decomposition;
	};

	enum {
		INSTRUMENT								= _INSTRUMENT,

		CUDA_ARCH 								= KernelPolicy::CUDA_ARCH,
		LOG_THREADS 							= KernelPolicy::LOG_THREADS,
		THREAD_OCCUPANCY						= B40C_SM_THREADS(CUDA_ARCH) >> LOG_THREADS,
		SMEM_OCCUPANCY							= B40C_SMEM_BYTES(CUDA_ARCH) / sizeof(SmemStorage),

		MAX_CTA_OCCUPANCY						= B40C_MIN(B40C_SM_CTAS(CUDA_ARCH), B40C_MIN(THREAD_OCCUPANCY, SMEM_OCCUPANCY)),

		VALID									= (MAX_CTA_OCCUPANCY > 0),
	};
};
	


} // namespace downsweep
} // namespace partition_contract
} // namespace bfs
} // namespace graph
} // namespace b40c


