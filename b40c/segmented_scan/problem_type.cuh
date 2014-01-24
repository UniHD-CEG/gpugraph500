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
 * Segmented scan problem type
 ******************************************************************************/

#pragma once

namespace b40c {
namespace segmented_scan {

/**
 * Type of segmented scan problem
 */
template <
	typename _T,			// Partial type
	typename _Flag,			// Flag type
	typename _SizeT,
	typename _ReductionOp,
	typename _IdentityOp,
	bool _EXCLUSIVE>
struct ProblemType
{
	typedef _T 				T;				// The type of data we are operating upon
	typedef _Flag 			Flag;			// The type of flag we're using
	typedef _SizeT 			SizeT;			// The integer type we should use to index into data arrays (e.g., size_t, uint32, uint64, etc)
	typedef _ReductionOp 	ReductionOp;	// The function or functor type for binary reduction (implements "T op(const T&, const T&)")
	typedef _IdentityOp 	IdentityOp;		// Identity operator type

	enum {
		EXCLUSIVE 			= _EXCLUSIVE,
	};
};


} // namespace segmented_scan
} // namespace b40c

