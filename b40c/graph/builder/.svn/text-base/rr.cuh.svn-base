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
 * Random-regular(ish) Graph Construction Routines
 ******************************************************************************/

#pragma once

#include <math.h>
#include <time.h>
#include <stdio.h>

#include <b40c/graph/builder/utils.cuh>

namespace b40c {
namespace graph {
namespace builder {


/**
 * A random graph where each node has a guaranteed degree of random neighbors.
 * Does not meet definition of random-regular: loops, and multi-edges are 
 * possible, and in-degree is not guaranteed to be the same as out degree.   
 */
template<bool LOAD_VALUES, typename VertexId, typename Value, typename SizeT>
int BuildRandomRegularishGraph(
	SizeT nodes,
	int degree,
	CsrGraph<VertexId, Value, SizeT> &csr_graph)
{
	SizeT edges 				= nodes * degree;
	
	csr_graph.template FromScratch<LOAD_VALUES>(nodes, edges);

	time_t mark0 = time(NULL);
	printf("  Selecting %llu random edges in COO format... ",
		(unsigned long long) edges);
	fflush(stdout);

	SizeT total = 0;
    for (VertexId node = 0; node < nodes; node++) {
    	
    	csr_graph.row_offsets[node] = total;
    	
    	for (int edge = 0; edge < degree; edge++) {
    		
    		VertexId neighbor = RandomNode(csr_graph.nodes);
    		csr_graph.column_indices[total] = neighbor;
    		if (LOAD_VALUES) {
    			csr_graph.values[node] = 1;
    		}
    		
    		total++;
    	}
    }
    
    csr_graph.row_offsets[nodes] = total; 	// last offset is always num_entries

	time_t mark1 = time(NULL);
	printf("Done selecting (%ds).\n", (int) (mark1 - mark0));
	fflush(stdout);

	return 0;
}


} // namespace builder
} // namespace graph
} // namespace b40c
