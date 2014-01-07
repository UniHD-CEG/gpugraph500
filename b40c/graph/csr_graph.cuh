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
 * Simple CSR sparse graph data structure
 ******************************************************************************/

#pragma once

#include <time.h>
#include <stdio.h>

#include <algorithm>

#include <b40c/util/error_utils.cuh>

namespace b40c {
namespace graph {


/**
 * CSR sparse format graph
 */
template<typename VertexId, typename Value, typename SizeT>
struct CsrGraph
{
	SizeT nodes;
	SizeT edges;
	
	SizeT 		*row_offsets;
	VertexId	*column_indices;
	Value		*values;
	
	bool 		pinned;

	/**
	 * Constructor
	 */
	CsrGraph(bool pinned = false)
	{
		nodes = 0;
		edges = 0;
		row_offsets = NULL;
		column_indices = NULL;
		values = NULL;
		this->pinned = pinned;
	}

	template <bool LOAD_VALUES>
	void FromScratch(SizeT nodes, SizeT edges)
	{
		this->nodes = nodes;
		this->edges = edges;

		if (pinned) {

			// Put our graph in pinned memory
			int flags = cudaHostAllocMapped;
			if (b40c::util::B40CPerror(cudaHostAlloc((void **)&row_offsets, sizeof(SizeT) * (nodes + 1), flags),
				"CsrGraph cudaHostAlloc row_offsets failed", __FILE__, __LINE__)) exit(1);
			if (b40c::util::B40CPerror(cudaHostAlloc((void **)&column_indices, sizeof(VertexId) * edges, flags),
				"CsrGraph cudaHostAlloc column_indices failed", __FILE__, __LINE__)) exit(1);

			if (LOAD_VALUES) {
				if (b40c::util::B40CPerror(cudaHostAlloc((void **)&values, sizeof(Value) * edges, flags),
						"CsrGraph cudaHostAlloc values failed", __FILE__, __LINE__)) exit(1);
			}

		} else {

			// Put our graph in regular memory
			row_offsets 		= (SizeT*) malloc(sizeof(SizeT) * (nodes + 1));
			column_indices 		= (VertexId*) malloc(sizeof(VertexId) * edges);
			values 				= (LOAD_VALUES) ? (Value*) malloc(sizeof(Value) * edges) : NULL;
		}
	}


	/**
	 * Build CSR graph from sorted COO graph
	 */
	template <bool LOAD_VALUES, typename Tuple>
	void FromCoo(
		Tuple *coo,
		SizeT coo_nodes,
		SizeT coo_edges,
		bool ordered_rows = false)
	{
		printf("  Converting %d vertices, %d directed edges (%s tuples) to CSR format... ",
			coo_nodes, coo_edges, ordered_rows ? "ordered" : "unordered");
		time_t mark1 = time(NULL);
		fflush(stdout);

		FromScratch<LOAD_VALUES>(coo_nodes, coo_edges);
		
		// Sort COO by row
		if (!ordered_rows) {
			std::stable_sort(coo, coo + coo_edges, DimacsTupleCompare<Tuple>);
		}

		VertexId prev_row = -1;
		for (SizeT edge = 0; edge < edges; edge++) {
			
			VertexId current_row = coo[edge].row;
			
			// Fill in rows up to and including the current row
			for (VertexId row = prev_row + 1; row <= current_row; row++) {
				row_offsets[row] = edge;
			}
			prev_row = current_row;
			
			column_indices[edge] = coo[edge].col;
			if (LOAD_VALUES) {
				coo[edge].Val(values[edge]);
			}
		}

		// Fill out any trailing edgeless nodes (and the end-of-list element)
		for (VertexId row = prev_row + 1; row <= nodes; row++) {
			row_offsets[row] = edges;
		}

		time_t mark2 = time(NULL);
		printf("Done converting (%ds).\n", (int) (mark2 - mark1));
		fflush(stdout);
	}

	/**
	 * Print log-histogram
	 */
	void PrintHistogram()
	{
		fflush(stdout);

		// Initialize
		int log_counts[32];
		for (int i = 0; i < 32; i++) {
			log_counts[i] = 0;
		}

		// Scan
		int max_log_length = -1;
		for (VertexId i = 0; i < nodes; i++) {

			SizeT length = row_offsets[i + 1] - row_offsets[i];

			int log_length = -1;
			while (length > 0) {
				length >>= 1;
				log_length++;
			}
			if (log_length > max_log_length) {
				max_log_length = log_length;
			}

			log_counts[log_length + 1]++;
		}
		printf("\nDegree Histogram (%lld vertices, %lld directed edges):\n", (long long) nodes, (long long) edges);
		for (int i = -1; i < max_log_length + 1; i++) {
			printf("\tDegree 2^%i: %d (%.2f%%)\n", i, log_counts[i + 1], (float) log_counts[i + 1] * 100.0 / nodes);
		}
		printf("\n");
		fflush(stdout);
	}


	/**
	 * Display CSR graph to console
	 */
	void DisplayGraph()
	{
		printf("Input Graph:\n");
		for (VertexId node = 0; node < nodes; node++) {
			PrintValue(node);
			printf(": ");
			for (SizeT edge = row_offsets[node]; edge < row_offsets[node + 1]; edge++) {
				PrintValue(column_indices[edge]);
				printf(", ");
			}
			printf("\n");
		}

	}

	/**
	 * Deallocates graph
	 */
	void Free()
	{
		if (row_offsets) {
			if (pinned) {
				b40c::util::B40CPerror(cudaFreeHost(row_offsets), "CsrGraph cudaFreeHost row_offsets failed", __FILE__, __LINE__);
			} else {
				free(row_offsets);
			}
			row_offsets = NULL;
		}
		if (column_indices) {
			if (pinned) {
				b40c::util::B40CPerror(cudaFreeHost(column_indices), "CsrGraph cudaFreeHost column_indices failed", __FILE__, __LINE__);
			} else {
				free(column_indices);
			}
			column_indices = NULL;
		}
		if (values) { free (values); values = NULL; }

		nodes = 0;
		edges = 0;
	}
	
	/**
	 * Destructor
	 */
	~CsrGraph()
	{
		Free();
	}
};


} // namespace graph
} // namespace b40c
