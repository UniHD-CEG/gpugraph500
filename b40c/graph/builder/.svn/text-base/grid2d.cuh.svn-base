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
 * 2D Grid Graph Construction Routines
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
 * Builds a square 2D grid CSR graph.  Interior nodes have degree 5 (including
 * a self-loop)
 * 
 * Returns 0 on success, 1 on failure.
 */
template<bool LOAD_VALUES, typename VertexId, typename Value, typename SizeT>
int BuildGrid2dGraph(
	VertexId width,
	CsrGraph<VertexId, Value, SizeT> &csr_graph)
{ 
	if (width < 0) {
		fprintf(stderr, "Invalid width: %d", width);
		return -1;
	}
	
	SizeT interior_nodes 		= (width - 2) * (width - 2);
	SizeT edge_nodes 			= (width - 2) * 4;
	SizeT corner_nodes 			= 4;
	SizeT nodes 				= width * width;
	SizeT edges 				= (interior_nodes * 4) + (edge_nodes * 3) + (corner_nodes * 2) + nodes;

	csr_graph.template FromScratch<LOAD_VALUES>(nodes, edges);

	SizeT total = 0;
	for (VertexId j = 0; j < width; j++) {
		for (VertexId k = 0; k < width; k++) {
			
			VertexId me = (j * width) + k;
			csr_graph.row_offsets[me] = total; 

			VertexId neighbor = (j * width) + (k - 1);
			if (k - 1 >= 0) {
				csr_graph.column_indices[total] = neighbor; 
				total++;
			}

			neighbor = (j * width) + (k + 1);
			if (k + 1 < width) {
				csr_graph.column_indices[total] = neighbor; 
				total++;
			}
			
			neighbor = ((j - 1) * width) + k;
			if (j - 1 >= 0) {
				csr_graph.column_indices[total] = neighbor; 
				total++;
			}
			
			neighbor = ((j + 1) * width) + k;
			if (j + 1 < width) {
				csr_graph.column_indices[total] = neighbor; 
				total++;
			}

			neighbor = me;
			csr_graph.column_indices[total] = neighbor;
			total++;

			if (LOAD_VALUES) {
				csr_graph.values[me] = 1;
			}
		}
	}
	csr_graph.row_offsets[csr_graph.nodes] = total; 	// last offset is always total

	return 0;
}

} // namespace builder
} // namespace graph
} // namespace b40c
