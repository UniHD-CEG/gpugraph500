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
 * Scan operator for consecutive reduction problems
 ******************************************************************************/

#pragma once

#include <b40c/util/basic_utils.cuh>

namespace b40c {
namespace consecutive_reduction {

/**
 * Structure-of-array scan operator
 */
template <typename ReductionOp, typename TileTuple>
struct SoaScanOperator
{
	typedef typename TileTuple::T0 	ValueType;
	typedef typename TileTuple::T1 	FlagType;

	enum {
		IDENTITY_STRIDES = false,			// There is no "identity" region of warpscan storage exists for strides to index into
	};

	// ValueType reduction operator
	ReductionOp reduction_op;

	// Constructor
	__device__ __forceinline__ SoaScanOperator(ReductionOp reduction_op) :
		reduction_op(reduction_op)
	{}

	// SOA scan operator
	__device__ __forceinline__ TileTuple operator()(
		const TileTuple &first,
		const TileTuple &second)
	{
/*		NVBUGS XXX
		return TileTuple(
			(second.t1) ? second.t0 : reduction_op(first.t0, second.t0),
			first.t1 + second.t1);
*/
		if (second.t1) {
			return TileTuple(second.t0, first.t1 + second.t1);
		} else {
			return TileTuple(reduction_op(first.t0, second.t0), first.t1 + second.t1);
		}
	}

	// SOA identity operator
	__device__ __forceinline__ TileTuple operator()()
	{
		TileTuple retval;
		retval.t1 = 0;			// Flag identity
		return retval;
	}
};


} // namespace consecutive_reduction
} // namespace b40c

