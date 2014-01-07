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
 * METIS Graph Construction Routines
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
 * Reads a DIMACS graph from an input-stream into a CSR sparse format 
 */
template<bool LOAD_VALUES, typename VertexId, typename Value, typename SizeT>
int ReadMetisStream(
	FILE *f_in,
	CsrGraph<VertexId, Value, SizeT> &csr_graph)
{
	
	time_t mark0 = time(NULL);
	printf("  Parsing METIS CSR format ");
	fflush(stdout);

	char 		line[1024];
	int			c;
	SizeT 		edges_read = 0;
	VertexId 	current_node = -1;
	long long 	ll_edge;
	
	while ((c = fgetc(f_in)) != EOF) {

		// Try and match the blank line
		switch (c) {
		
		case '%':
			// Comment: consume up to and including newline
			while ((c = fgetc(f_in)) != EOF) {
				if (c == '\n') break;
			}
			break;

		case ' ':
		case '\t':
			// whitespace
			break;

		case '\n':
			// End of line: begin processing next current_node
			current_node++;

			// next current_node starts its edges at column_indices[edges_read];
			csr_graph.row_offsets[current_node] = edges_read;

			break;
		
		default:
			
			// Data still in line: put char back
			ungetc(c, f_in);

			if (current_node == -1) {

				// First line: problem description
				long long ll_nodes, ll_edges;
				if (fscanf(f_in, "%lld %lld%[^\n]", &ll_nodes, &ll_edges, line) > 0) {
					
					csr_graph.template FromScratch<LOAD_VALUES>(
						ll_nodes,
						ll_edges * 2); 	// Most METIS graphs report the count of M undirected edges, followed by 2M directed edges with the inclusion of backedges

					printf("%d nodes, %d directed edges\n", csr_graph.nodes, csr_graph.edges);
					fflush(stdout);
					
				} else {
					fprintf(stderr, "Error parsing METIS graph: invalid format\n");
					return -1;
				}

			} else {
			
				// Continue processing next edge in edge list
				if (fscanf(f_in, "%lld", &ll_edge) > 0) {

					if (!csr_graph.row_offsets) {
						fprintf(stderr, "Error parsing METIS graph: no graph yet allocated\n");
						return -1;
					}			
					if (edges_read >= csr_graph.edges) {
						fprintf(stderr, "Error parsing METIS graph: encountered more than %d edges\n", csr_graph.edges);
						if (csr_graph.row_offsets) csr_graph.Free();
						return -1;
					}
					if (ll_edge > csr_graph.nodes) {
						fprintf(stderr, "Error parsing METIS graph: edge to %lld is larger than vertices in graph\n", ll_edge);
						if (csr_graph.row_offsets) csr_graph.Free();
						return -1;
					}

					csr_graph.column_indices[edges_read] = ll_edge - 1;	// zero-based array

					edges_read++;
					
				} else {
					fprintf(stderr, "Error parsing METIS graph: invalid format\n");
					if (csr_graph.row_offsets) csr_graph.Free();
					return -1;
				}
			}
		};
	}

	if (csr_graph.row_offsets == NULL) {
		fprintf(stderr, "No graph found\n");
		return -1;
	}

	// Fill out any trailing rows that didn't have explicit lines in the file
	while (current_node < csr_graph.nodes) {
		current_node++;
		csr_graph.row_offsets[current_node] = edges_read;
	}

	if (csr_graph.edges == edges_read * 2) {
		// Quick fix in case edges was the count of all edges in the file (instead of half), and we mistakenly doubled it
		printf("Actual edges: %d\n", edges_read);
		csr_graph.edges = edges_read;

	} else if (edges_read != csr_graph.edges) {
		fprintf(stderr, "Error parsing METIS graph: only %d/%d edges read\n", edges_read, csr_graph.edges);
		if (csr_graph.row_offsets) csr_graph.Free();
		return -1;
	}
	
	time_t mark1 = time(NULL);
	printf("Done parsing (%ds).\n", (int) (mark1 - mark0));
	fflush(stdout);
	
	return 0;
}


/**
 * Loads a METIS-formatted CSR graph from the specified file.  If 
 * metis_filename == NULL, then it is loaded from stdin.
 */
template<bool LOAD_VALUES, typename VertexId, typename Value, typename SizeT>
int BuildMetisGraph(
	char *metis_filename, 
	CsrGraph<VertexId, Value, SizeT> &csr_graph)
{ 
	if (metis_filename == NULL) {

		// Read from stdin
		printf("Reading from stdin:\n");
		if (ReadMetisStream<LOAD_VALUES>(stdin, csr_graph) != 0) {
			return -1;
		}

	} else {
	
		// Read from file
		FILE *f_in = fopen(metis_filename, "r");
		if (f_in) {
			printf("Reading from %s:\n", metis_filename);
			if (ReadMetisStream<LOAD_VALUES>(f_in, csr_graph) != 0) {
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


/**
 *
 */
template<typename VertexId, typename Value, typename SizeT>
int WriteMetisStream(
	FILE *f_out,
	const CsrGraph<VertexId, Value, SizeT> &csr_graph)
{
	time_t mark0 = time(NULL);
	printf("  Writing METIS CSR format... ");
	fflush(stdout);

	fprintf(f_out, "%lld %lld\n", (long long) csr_graph.nodes, (long long) csr_graph.edges);

	for (VertexId node = 0; node < csr_graph.nodes; node++) {

		for (SizeT edge = csr_graph.row_offsets[node]; edge < csr_graph.row_offsets[node + 1]; edge++) {

			fprintf(f_out, "%lld ", (long long) csr_graph.column_indices[edge] + 1);
		}
		fprintf(f_out, "\n");
	}

	time_t mark1 = time(NULL);
	printf("Done writing (%ds).\n", (int) (mark1 - mark0));
	fflush(stdout);

	return 0;
}


/**
 * Writes a METIS-formatted CSR graph to the specified file.  If
 * metis_filename == NULL, then it is written to stdout.
 */
template<typename VertexId, typename Value, typename SizeT>
int WriteMetisGraph(
	char *metis_filename,
	CsrGraph<VertexId, Value, SizeT> &csr_graph)
{
	if (metis_filename == NULL) {

		// Write to stdout
		printf("Writing to stdout:\n");
		if (WriteMetisStream(stdout, csr_graph) != 0) {
			return -1;
		}

	} else {

		// Write to file
		FILE *f_out = fopen(metis_filename, "w");
		if (f_out) {
			printf("Reading from %s:\n", metis_filename);
			if (WriteMetisStream(f_out, csr_graph) != 0) {
				fclose(f_out);
				return -1;
			}
			fclose(f_out);
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
