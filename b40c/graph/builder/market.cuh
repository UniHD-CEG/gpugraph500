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
 * MARKET Graph Construction Routines
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
 * Reads a MARKET graph from an input-stream into a CSR sparse format
 */
template<bool LOAD_VALUES, typename VertexId, typename Value, typename SizeT>
int ReadMarketStream(
	FILE *f_in,
	CsrGraph<VertexId, Value, SizeT> &csr_graph,
	bool undirected)
{
	typedef CooEdgeTuple<VertexId, Value> EdgeTupleType;
	
	SizeT edges_read = -1;
	SizeT nodes = 0;
	SizeT edges = 0;
	EdgeTupleType *coo = NULL;		// read in COO format
	
	time_t mark0 = time(NULL);
	printf("  Parsing MARKET COO format ");
	fflush(stdout);

	char line[1024];

	bool ordered_rows = true;

	while(true) {

		if (fscanf(f_in, "%[^\n]\n", line) <= 0) {
			break;
		}

		if (line[0] == '%') {

			// Comment

		} else if (edges_read == -1) {

			// Problem description
			long long ll_nodes_x, ll_nodes_y, ll_edges;
			if (sscanf(line, "%lld %lld %lld", &ll_nodes_x, &ll_nodes_y, &ll_edges) != 3) {
				fprintf(stderr, "Error parsing MARKET graph: invalid problem description\n");
				return -1;
			}

			if (ll_nodes_x != ll_nodes_y) {
				fprintf(stderr, "Error parsing MARKET graph: not square (%lld, %lld)\n", ll_nodes_x, ll_nodes_y);
				return -1;
			}

			nodes = ll_nodes_x;
			edges = (undirected) ? ll_edges * 2 : ll_edges;

			printf(" (%lld nodes, %lld directed edges)... ",
				(unsigned long long) ll_nodes_x, (unsigned long long) ll_edges);
			fflush(stdout);
			
			// Allocate coo graph
			coo = (EdgeTupleType*) malloc(sizeof(EdgeTupleType) * edges);

			edges_read++;

		} else {

			// Edge description (v -> w)
			if (!coo) {
				fprintf(stderr, "Error parsing MARKET graph: invalid format\n");
				return -1;
			}			
			if (edges_read >= edges) {
				fprintf(stderr, "Error parsing MARKET graph: encountered more than %d edges\n", edges);
				if (coo) free(coo);
				return -1;
			}

			long long ll_row, ll_col;
			if (sscanf(line, "%lld %lld", &ll_col, &ll_row) != 2) {
				fprintf(stderr, "Error parsing MARKET graph: badly formed edge\n", edges);
				if (coo) free(coo);
				return -1;
			}

			coo[edges_read].row = ll_row - 1;	// zero-based array
			coo[edges_read].col = ll_col - 1;	// zero-based array

			edges_read++;

			if (undirected) {
				// Go ahead and insert reverse edge
				coo[edges_read].row = ll_col - 1;	// zero-based array
				coo[edges_read].col = ll_row - 1;	// zero-based array

				ordered_rows = false;
				edges_read++;
			}
		}
	}
	
	if (coo == NULL) {
		fprintf(stderr, "No graph found\n");
		return -1;
	}

	if (edges_read != edges) {
		fprintf(stderr, "Error parsing MARKET graph: only %d/%d edges read\n", edges_read, edges);
		if (coo) free(coo);
		return -1;
	}
	
	time_t mark1 = time(NULL);
	printf("Done parsing (%ds).\n", (int) (mark1 - mark0));
	fflush(stdout);
	
	// Convert COO to CSR
	csr_graph.template FromCoo<LOAD_VALUES>(coo, nodes, edges, ordered_rows);
	free(coo);

	fflush(stdout);
	
	return 0;
}


/**
 * Loads a MARKET-formatted CSR graph from the specified file.  If
 * dimacs_filename == NULL, then it is loaded from stdin.
 */
template<bool LOAD_VALUES, typename VertexId, typename Value, typename SizeT>
int BuildMarketGraph(
	char *dimacs_filename, 
	CsrGraph<VertexId, Value, SizeT> &csr_graph,
	bool undirected)
{ 
	if (dimacs_filename == NULL) {

		// Read from stdin
		printf("Reading from stdin:\n");
		if (ReadMarketStream<LOAD_VALUES>(stdin, csr_graph, undirected) != 0) {
			return -1;
		}

	} else {
	
		// Read from file
		FILE *f_in = fopen(dimacs_filename, "r");
		if (f_in) {
			printf("Reading from %s:\n", dimacs_filename);
			if (ReadMarketStream<LOAD_VALUES>(f_in, csr_graph, undirected) != 0) {
				fclose(f_in);
				return -1;
			}
			fclose(f_in);
		} else {
			perror("Unable to open file");
			return -1;
		}
	}
	
	return 0;
}


} // namespace builder
} // namespace graph
} // namespace b40c
