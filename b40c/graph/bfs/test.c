
#include <stdio.h> 
#include <string>
#include <deque>
#include <vector>
#include <iostream>
#include <mpi.h>
// Utilities and correctness-checking
#include <b40c_test_util.h>

// Graph construction utils
#include <b40c/graph/builder/dimacs.cuh>
#include <b40c/graph/builder/grid2d.cuh>
#include <b40c/graph/builder/grid3d.cuh>
#include <b40c/graph/builder/market.cuh>
#include <b40c/graph/builder/metis.cuh>
#include <b40c/graph/builder/rmat.cuh>
#include <b40c/graph/builder/random.cuh>
#include <b40c/graph/builder/rr.cuh>

// BFS includes
#include <b40c/graph/bfs/csr_problem.cuh>
#include <b40c/graph/bfs/enactor_contract_expand.cuh>
#include <b40c/graph/bfs/enactor_expand_contract.cuh>
#include <b40c/graph/bfs/enactor_two_phase.cuh>
#include <b40c/graph/bfs/enactor_hybrid.cuh>
#include <b40c/graph/bfs/enactor_multi_gpu.cuh>
#include <b40c/graph/bfs/enactor_multi_node.cuh>

using namespace b40c;
using namespace graph;

int main(int argc, char** argv){
	VertexId vid=5;	
	SizeT sizet=10;

}
