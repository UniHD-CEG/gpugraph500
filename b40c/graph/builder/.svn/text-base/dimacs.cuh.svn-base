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
 * DIMACS Graph Construction Routines
 ******************************************************************************/

#pragma once

#include <math.h>
#include <time.h>
#include <stdio.h>

#include <vector>
#include <string>

#include <b40c/util/basic_utils.cuh>
#include <b40c/graph/builder/utils.cuh>

namespace b40c {
namespace graph {
namespace builder {

/**
 * Reads a DIMACS graph from an input-stream into a CSR sparse format 
 */
template<bool LOAD_VALUES, typename VertexId, typename Value, typename SizeT>
int ReadDimacsStream(
	std::vector<FILE *> files,
	CsrGraph<VertexId, Value, SizeT> &csr_graph,
	bool undirected)
{
	typedef typename util::If<LOAD_VALUES, Value, util::NullType>::Type TupleValue;
	typedef CooEdgeTuple<VertexId, TupleValue> EdgeTupleType;
	
	SizeT edges_read = 0;
	SizeT nodes = 0;
	SizeT edges = 0;
	SizeT directed_edges = 0;
	EdgeTupleType *coo = NULL;		// read in COO format
	
	time_t mark0 = time(NULL);
	printf("  Parsing DIMACS COO format ");
	fflush(stdout);

	char line[1024];
	char problem_type[1024];

	bool ordered_rows = true;

	for (int file = 0; file < files.size(); file++) {

		bool parsed = false;
		while(!parsed) {

			if (fscanf(files[file], "%[^\n]\n", line) <= 0) {
				break;
			}

			switch (line[0]) {
			case 'p':
			{
				// Problem description (nodes is nodes, edges is edges)
				long long ll_nodes, ll_edges;
				sscanf(line, "p %s %lld %lld", problem_type, &ll_nodes, &ll_edges);
				if (nodes && (nodes != ll_nodes)) {
					fprintf(stderr, "Error: splice files do not name the same number of vertices\n");
					return -1;
				} else {
					nodes = ll_nodes;
				}
				edges += ll_edges;
				parsed = true;

				break;
			}
			default:
				// read remainder of line
				break;
			}
		}
	}

	directed_edges = (undirected) ? edges * 2 : edges;
	if (!directed_edges) {
		fprintf(stderr, "No graph found\n");
		return -1;
	}

	printf(" (%lld vertices, %lld directed edges)... ",
		(unsigned long long) nodes, (unsigned long long) directed_edges);
	fflush(stdout);

	// Allocate coo graph
	coo = (EdgeTupleType*) malloc(sizeof(EdgeTupleType) * directed_edges);

	// Vector of latest tuples
	std::vector<EdgeTupleType> tuples(files.size(), EdgeTupleType(-1, 0, 0));

	int progress = 0;

	// Splice in ordered vertices
	while(true) {

		// Pick the smallest edge from the file set and add it to the COO edge list
		int smallest = -1;
		for (int i = 0; i < files.size(); i++) {
			
			// Read a tuple from this file if necessary
			while ((tuples[i].row < 0) && (fscanf(files[i], "%[^\n]\n", line) > 0)) {

				switch (line[0]) {
				case 'a':
				{
					// Edge description (v -> w) with value val
					if (!coo) {
						fprintf(stderr, "Error parsing DIMACS graph: invalid format\n");
						return -1;
					}
					if (edges_read >= directed_edges) {
						fprintf(stderr, "Error parsing DIMACS graph: encountered more than %d edges\n", directed_edges);
						if (coo) free(coo);
						return -1;
					}

					long long ll_row, ll_col, ll_val;
					sscanf(line, "a %lld %lld %lld", &ll_row, &ll_col, &ll_val);

					tuples[i] = EdgeTupleType(
						ll_row - 1,	// zero-based array
						ll_col - 1,	// zero-based array
						ll_val);

					if (undirected) {
						// Go ahead and insert reverse edge
						coo[edges_read] = EdgeTupleType(
							ll_col - 1,	// zero-based array
							ll_row - 1,	// zero-based array
							ll_val);

						ordered_rows = false;
						edges_read++;
					}

					if (edges_read > (directed_edges / 32) * (progress + 1)) {
						progress++;
						printf("%.2f%%\n", float(progress) * (100.0 / 32.0));
						fflush(stdout);
					}

					break;
				}

				default:
					// read remainder of line
					break;
				}
			}

			// Compare this tuple against the smallest one so far
			if ((tuples[i].row >= 0) && ((smallest < 0) || (tuples[i].row < tuples[smallest].row))) {
				smallest = i;
			}
		}

		// Insert smallest edge from the splice files (or quit if none)
		if (smallest < 0) {
			break;
		} else {
			if (edges_read && (tuples[smallest].row < coo[edges_read - 1].row)) {
				ordered_rows = false;
			}
			coo[edges_read] = tuples[smallest];
			tuples[smallest].row = -1;
			smallest = -1;
			edges_read++;
		}
	}

	if (edges_read != directed_edges) {
		fprintf(stderr, "Error parsing DIMACS graph: only %d/%d edges read\n", edges_read, directed_edges);
		if (coo) free(coo);
		return -1;
	}
	
	time_t mark1 = time(NULL);
	printf("Done parsing (%ds).\n", (int) (mark1 - mark0));
	fflush(stdout);

	// Convert COO to CSR
	csr_graph.template FromCoo<LOAD_VALUES>(coo, nodes, directed_edges, ordered_rows);
	free(coo);

	fflush(stdout);
	return 0;
}


/**
 * Loads a DIMACS-formatted CSR graph from the specified file.  If 
 * dimacs_filename == NULL, then it is loaded from stdin.
 */
template<bool LOAD_VALUES, typename VertexId, typename Value, typename SizeT>
int BuildDimacsGraph(
	char *dimacs_filename, 
	CsrGraph<VertexId, Value, SizeT> &csr_graph,
	bool undirected,
	int splice)
{ 
	int retval = 0;

	std::vector<FILE*> files;
	if (dimacs_filename == NULL) {

		// Read from stdin
		printf("Reading from stdin:\n");
		files.push_back(stdin);
		if (ReadDimacsStream<LOAD_VALUES>(files, csr_graph, undirected) != 0) {
			retval = -1;
		}
	} else {
	
		// Read from file(s)
		FILE *f_in;
		if (splice) {
			for (int i = 0; i < splice; i++) {
				std::stringstream formatter;
				formatter << dimacs_filename << "." << i;
				if ((f_in = fopen(formatter.str().c_str(), "r")) == NULL) {
					break;
				}
				files.push_back(f_in);
				printf("Opened %s\n", formatter.str().c_str());
			}
		} else {
			if ((f_in = fopen(dimacs_filename, "r")) != NULL) {
				files.push_back(f_in);
				printf("Opened %s:\n", dimacs_filename);
			}
		}
		if (files.size()) {
			retval = ReadDimacsStream<LOAD_VALUES>(files, csr_graph, undirected);
			for (int i = 0; i < files.size(); i++) {
				if (files[i]) fclose(files[i]);
			}
		} else {
			perror("Unable to open file");
			retval = -1;
		}
	}

	return retval;
}

} // namespace builder
} // namespace graph
} // namespace b40c
