/*
 * Matthias Hauck, 2013
 */
#include "mpi.h"
#include <cstring>
#if __cplusplus > 199711L  //C++11 active
#include <random>
#endif
#include <algorithm>

#include <generator/make_graph.h>
#include "distmatrix2d.h"
#include "simplecpubfs.h"

int main(int argc, char** argv)
{

      int scale =  21;
      int edgefactor = 16;
      int num_of_iterations = 64;

      MPI_Init(&argc, &argv);
      int   R,C;
      bool  R_set =false, C_set = false;
      int size, rank;

      long vertices ;
      long global_edges;
      MPI_Comm_size(MPI_COMM_WORLD, &size);
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);

      if(rank == 0){
        int i = 1;
        // simple Commandline parser
        while(i < argc){
            if(!strcmp(argv[i], "-s")){
                if(i+1 < argc){
                    int s_tmp = atoi(argv[i+1]);
                    if(s_tmp < 1){
                        printf("Invalid scale factor: %s\n", argv[i+1]);
                    } else{
                        scale = s_tmp;
                        i++;
                    }
                }
            }else if(!strcmp(argv[i], "-e")){
                if(i+1 < argc){
                    int e_tmp = atoi(argv[i+1]);
                    if(e_tmp < 1){
                       printf("Invalid edge factor: %s\n", argv[i+1]);
                    } else{
                        edgefactor = e_tmp;
                        i++;
                    }
                }
            }else if(!strcmp(argv[i], "-i")){
                if(i+1 < argc){
                    int i_tmp = atoi(argv[i+1]);
                    if(i_tmp < 1){
                       printf("Invalid number of iterations: %s\n", argv[i+1]);
                    } else{
                        num_of_iterations = i_tmp;
                        i++;
                    }
                }
            }else if(!strcmp(argv[i], "-R")){
                if(i+1 < argc){
                    int R_tmp = atoi(argv[i+1]);
                    if(R_tmp < 1){
                       printf("Invalid row slice number: %s\n", argv[i+1]);
                    } else{
                       R_set = true;
                       R = R_tmp;
                       i++;
                    }
                }
            }else if(!strcmp(argv[i], "-C")){
                if(i+1 < argc){
                    int C_tmp = atoi(argv[i+1]);
                    if(C_tmp < 1){
                        printf("Invalid column slice number: %s\n", argv[i+1]);
                    } else{
                        C_set = true;
                        C = C_tmp;
                        i++;
                    }
                 }
             }
             i++;
        }

        if(R_set &&  !C_set ){
            if(R > size){
                printf("Error not enought nodesRequested: %d available: %d\n", R , size);
                MPI_Abort(MPI_COMM_WORLD, 1);
            } else {
                C = size / R;
            }
        } else if(!R_set && C_set ){
            if(C > size){
                printf("Error not enought nodes. Requested: %d available: %d\n", C , size);
                MPI_Abort(MPI_COMM_WORLD, 1);
            } else {
                R = size / C;
            }
        } else {
            R = size;
            C = 1;
        }
        printf("column slices: %d, row slices: %d\n", C, R );
      }
      MPI_Bcast(&scale     ,1,MPI_INT,0,MPI_COMM_WORLD);
      MPI_Bcast(&edgefactor,1,MPI_INT,0,MPI_COMM_WORLD);
      MPI_Bcast(&num_of_iterations,1,MPI_INT,0,MPI_COMM_WORLD);
      MPI_Bcast(&R         ,1,MPI_INT,0,MPI_COMM_WORLD);
      MPI_Bcast(&C         ,1,MPI_INT,0,MPI_COMM_WORLD);

      // Close unnessecary nodes
      if(R*C != size){
          printf("Number of nodes and size of grid do not match.\n");
          MPI_Abort(MPI_COMM_WORLD, 1);
      }
      //new number of vertices
      vertices  = 1L << scale ;

      // to stop times
      double tstart,tstop;

      // Graph generation
      MPI_Barrier(MPI_COMM_WORLD);
      tstart = MPI_Wtime();
      int64_t number_of_edges;
      packed_edge* edgelist;
      make_graph(scale, edgefactor << scale, 1, 2, &number_of_edges, &edgelist);
      MPI_Barrier(MPI_COMM_WORLD);
      tstop = MPI_Wtime();

      /*
      for(int i = 0; i < number_of_edges; i++){
          printf("%ld:%ld\n", edgelist[i].v0, edgelist[i].v1 );
      }
    */

      long global_edges_wd;
      MPI_Reduce(&number_of_edges, &global_edges_wd, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
      if (rank == 0) {
          printf("Input list of edges genereted.\n");
          printf("%e edge(s) generated in %fs (%f Medges/s on %d processor(s))\n", static_cast<double>(global_edges_wd), (tstop - tstart), global_edges_wd / (tstop - tstart) * 1.e-6, size);
      }

      // Matrix generation
      MPI_Barrier(MPI_COMM_WORLD);
      tstart = MPI_Wtime();
      DistMatrix2d store(R, C, scale);
      SimpleCPUBFS runBfs(store);
      store.setupMatrix(edgelist,number_of_edges);
      tstop = MPI_Wtime();
      long local_edges = store.getEdgeCount();
      MPI_Reduce(&local_edges, &global_edges, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
      if (rank == 0) {
          printf("Adijacency Matrix setup.\n");
          printf("%e edge(s) removed, because they are duplicates or self loops.\n", static_cast<double>(global_edges_wd- global_edges/2));
          printf("%e unique edge(s) processed in %fs (%f Medges/s on %d processor(s))\n", static_cast<double>(global_edges), (tstop - tstart), global_edges / (tstop - tstart) * 1.e-6, size);
      }

//     // print matrix
//     vtxtype* rowp = store.getRowPointer();
//     vtxtype* columnp = store.getColumnIndex();
//     for(int i = 0; i < store.getLocRowLength(); i++){
//          printf("%d: ",  store.localtoglobalRow(i));
//          for(int j = rowp[i]; j < rowp[i+1]; j++){
//              printf("%ld ",  columnp[j]);
//          }
//          printf("\n");
//     }

      // init
      // random number generator
      #if __cplusplus > 199711L
      std::knuth_b generator;
      std::uniform_int_distribution<vtxtype> distribution(0,vertices-1);
      #else
      //fallback if c++11 is not avible
      srand(1);
      #endif
      // variables to control iterations
      int maxgenvtx = 32; // relativ number of maximum attempts to find a valid start vertix per posible attempt
      std::vector<vtxtype> alreadyTryed;
      int iteration =  0;
      int giteration = 0; // generated number of iterations

      while(iteration < num_of_iterations ){
          // generate start node
          vtxtype start;

          int found_new = 0;
          if(rank == 0){
              while(giteration < num_of_iterations*maxgenvtx) {
                #if __cplusplus > 199711L
                start = distribution(generator);
                #else
                start = rand()% vertices;
                #endif

                giteration++;
                if(std::find(alreadyTryed.begin(),alreadyTryed.end(),start)==alreadyTryed.end()){
                    found_new = 1;
                    alreadyTryed.push_back(start);
                    break;
                }
              }
          }
          // tell other nodes, if new found
          MPI_Bcast(&found_new,1,MPI_INT,0,MPI_COMM_WORLD);
          if(found_new == 0){
              break;
          }

          // BFS
          MPI_Barrier(MPI_COMM_WORLD);
          tstart = MPI_Wtime();
          if(rank == 0){
            runBfs.runBFS(start);
          }else{
            runBfs.runBFS();
          }
          tstop = MPI_Wtime();
          if (rank == 0) {
              printf("BFS Iteration %d: Finished in %fs\n", iteration,(tstop-tstart));
          }
          // Validation
          // Test also if this result should be rejected
          // runBfs.validate();
          int reject;
          //test if the result should be rejected
          if(rank==0){
              //because validate is not implemented
              reject = 0;
          }
          //tell other nodes, if this iteration should be rejected
          MPI_Bcast(&reject,1,MPI_INT,0,MPI_COMM_WORLD);
          if(reject == 0){
              iteration++;
          }
          giteration++;
      }
      free(edgelist);

      // Statistic

      MPI_Finalize();
}

