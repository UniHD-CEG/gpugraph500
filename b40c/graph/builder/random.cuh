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
 * Random Graph Construction Routines
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
 * Builds a random CSR graph by adding edges edges to nodes nodes by randomly choosing
 * a pair of nodes for each edge.  There are possibilities of loops and multiple 
 * edges between pairs of nodes.    
 * 
 * Returns 0 on success, 1 on failure.
 */
template<bool LOAD_VALUES, typename VertexId, typename Value, typename SizeT>
int BuildRandomGraph(
	SizeT nodes,
	SizeT edges,
	CsrGraph<VertexId, Value, SizeT> &csr_graph,
	bool undirected)
{ 
	typedef CooEdgeTuple<VertexId, Value> EdgeTupleType;

	if ((nodes < 0) || (edges < 0)) {
		fprintf(stderr, "Invalid graph size: nodes=%d, edges=%d", nodes, edges);
		return -1;
	}

	time_t mark0 = time(NULL);
	printf("  Selecting %llu %s random edges in COO format... ", 
		(unsigned long long) edges, (undirected) ? "undirected" : "directed");
	fflush(stdout);

	// Construct COO graph

	VertexId directed_edges = (undirected) ? edges * 2 : edges;
	EdgeTupleType *coo = (EdgeTupleType*) malloc(sizeof(EdgeTupleType) * directed_edges);

	for (SizeT i = 0; i < edges; i++) {
		coo[i].row = RandomNode(nodes);
		coo[i].col = RandomNode(nodes);
		coo[i].val = 1;
		if (undirected) {
			// Reverse edge
			coo[edges + i].row = coo[i].col;
			coo[edges + i].col = coo[i].row;
			coo[edges + i].val = 1;
		}
	}

	time_t mark1 = time(NULL);
	printf("Done selecting (%ds).\n", (int) (mark1 - mark0));
	fflush(stdout);
	
	// Convert sorted COO to CSR
	csr_graph.template FromCoo<LOAD_VALUES>(coo, nodes, directed_edges);
	free(coo);

	return 0;
}


} // namespace builder
} // namespace graph
} // namespace b40c
