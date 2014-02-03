/*
 * Matthias Hauck, 2013
 */
#include "mpi.h"
#include <cstring>
#include <cmath>
#if __cplusplus > 199711L  //C++11 active
#include <random>
#endif
#include <algorithm>

#include <generator/make_graph.h>
#include <validate/validate.h>
#include "distmatrix2d.h"
#include "simplecpubfs.h"

struct statistic {
    double min;
    double firstquartile;
    double median;
    double thirdquartile;
    double max;

    double mean;
    double stddev;

    double hmean;
    double hstddev;
};

template<class T>
statistic getStatistics(std::vector<T>& input){
    statistic out;
    std::sort(input.begin(), input.end());

    out.min = static_cast<double>(input.front());
    if(1*input.size()%4 == 0)
        out.firstquartile = 0.5*(input[1*input.size()/4 - 1]+input[1*input.size()/4]);
    else
        out.firstquartile = static_cast<double>(input[1*input.size()/4]);
    if(2*input.size()%4 == 0)
        out.median = 0.5*(input[2*input.size()/4 - 1]+input[2*input.size()/4]);
    else
        out.median = static_cast<double>(input[2*input.size()/4]);
    if(3*input.size()%4 == 0)
        out.thirdquartile = 0.5*(input[3*input.size()/4 - 1]+input[3*input.size()/4]);
    else
        out.thirdquartile = static_cast<double>(input[3*input.size()/4]);
    out.max = static_cast<double>(input.back());

    double qsum = 0.0, sum = 0.0;
    for(typename std::vector<T>::iterator it= input.begin(); it != input.end(); it++){
        double it_val = static_cast<double>(*it);
        sum += it_val;
        qsum += it_val * it_val;
    }

    out.mean = sum / static_cast<double>(input.size());
    out.stddev = sqrt((qsum - sum *sum / static_cast<double>(input.size())) /
                      (static_cast<double>(input.size())-1)  );

    double iv_sum = 0;
    for(typename std::vector<T>::iterator it= input.begin(); it != input.end(); it++){
        double it_val = static_cast<double>(*it);
        iv_sum += 1/it_val;
    }

    out.hmean = static_cast<double>(input.size()) / iv_sum;
    // Harmonic standard deviation from:
    // Nilan Norris, The Standard Errors of the Geometric and Harmonic
    // Means and Their Application to Index Numbers, 1940.
    // http://www.jstor.org/stable/2235723
    double qiv_dif =0.;
    for(typename std::vector<T>::iterator it= input.begin(); it != input.end(); it++){
        double it_val = 1/static_cast<double>(*it) - 1/out.hmean;
        qiv_dif += it_val * it_val;
    }
    out.hstddev = (sqrt(qiv_dif) / (static_cast<double>(input.size())-1)) * out.hmean * out.hmean;

    return out;
}

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
      double make_graph_time,constr_time;

      // Graph generation
      MPI_Barrier(MPI_COMM_WORLD);
      tstart = MPI_Wtime();
      int64_t number_of_edges;
      packed_edge* edgelist;
      make_graph(scale, edgefactor << scale, 1, 2, &number_of_edges, &edgelist);
      MPI_Barrier(MPI_COMM_WORLD);
      tstop = MPI_Wtime();

      long global_edges_wd;
      MPI_Reduce(&number_of_edges, &global_edges_wd, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
      make_graph_time = tstop - tstart;
      if (rank == 0) {
          printf("Input list of edges genereted.\n");
          printf("%e edge(s) generated in %fs (%f Medges/s on %d processor(s))\n", static_cast<double>(global_edges_wd), make_graph_time, global_edges_wd / make_graph_time * 1.e-6, size);
      }

      // Matrix generation
      MPI_Barrier(MPI_COMM_WORLD);
      tstart = MPI_Wtime();
      DistMatrix2d store(R, C);
      store.setupMatrix2(edgelist,number_of_edges);
      SimpleCPUBFS runBfs(store);
      tstop = MPI_Wtime();

      long local_edges = store.getEdgeCount();
      MPI_Reduce(&local_edges, &global_edges, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
      constr_time = tstop - tstart;
      if (rank == 0) {
          printf("Adijacency Matrix setup.\n");
          printf("%e edge(s) removed, because they are duplicates or self loops.\n", static_cast<double>(global_edges_wd- global_edges/2));
          printf("%e unique edge(s) processed in %fs (%f Medges/s on %d processor(s))\n", static_cast<double>(global_edges), constr_time, global_edges / constr_time * 1.e-6, size);
      }
/*
      for(int i = 0; i < number_of_edges; i++){
          printf("(%ld:%ld) %ld:%ld\n", store.getLocalRowID(), store.getLocalColumnID(),edgelist[i].v0, edgelist[i].v1 );
      }
*/
     // print matrix
   /*  const vtxtype* rowp = store.getRowPointer();
     const vtxtype* columnp = store.getColumnIndex();
     for(int i = 0; i < store.getLocRowLength(); i++){
          printf("%d: ",  store.localtoglobalRow(i));
          for(int j = rowp[i]; j < rowp[i+1]; j++){
              printf("%ld ",  columnp[j]);
          }
          printf("\n");
     }
*/
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
      int giteration = 0; // number of tried iterations
      bool valid = true;

      // For statistic
      std::vector<double> bfs_time;
      std::vector<long>   nedge; //number of edges
      std::vector<double> teps;

      while(iteration < num_of_iterations ){
          // generate start node
          vtxtype start;

          if(rank == 0){
              while(giteration < num_of_iterations*maxgenvtx) {
                #if __cplusplus > 199711L
                start = distribution(generator);
                #else
                start = rand()% vertices;
                #endif

                giteration++;
                if(std::find(alreadyTryed.begin(),alreadyTryed.end(),start)==alreadyTryed.end()){
                    alreadyTryed.push_back(start);
                    break;
                } else {
                    start = -1;
                }
              }
          }
          // tell other nodes, if new found
          MPI_Bcast(&start,1,MPI_LONG,0,MPI_COMM_WORLD);
          if(start == -1){
              break;
          }

          // BFS
          double rtstart,rtstop;

          MPI_Barrier(MPI_COMM_WORLD);
          rtstart = MPI_Wtime();
          if(rank == 0){
            runBfs.runBFS(start);
          }else{
            runBfs.runBFS();
          }
          rtstop = MPI_Wtime();
          if (rank == 0) {
              printf("BFS Iteration %d: Finished in %fs\n", iteration,(rtstop-rtstart));
          }
          // Validation
          // Test also if this result should be rejected
          //    // manual validation ;)
          //    for(int i = 0; i < store.getLocColLength(); i++){
          //        printf("%d:[%ld]\n", store.localtoglobalCol(i),  predessor[i]);
          //    }
          int reject;
          int level;
          int this_valid;
          vtxtype num_edges;

          tstart = MPI_Wtime();
          this_valid = validate_bfs_result(store, edgelist, number_of_edges,
                                           vertices, start, runBfs.getPredessor(), &num_edges, &level);
          tstop = MPI_Wtime();

          if (rank == 0) {
              printf("Validation of iteration %d finished in %fs\n", iteration,(tstop-tstart));
          }
          valid = valid && this_valid;

          //test if the result should be rejected
          if(rank==0){
              //because validate is not implemented
              reject = level == 0;
              if(reject)
                  printf("Result rejected!\n");
              else {
                  printf("Result: %s %ld Edge(s) processed, %f MTeps\n", (this_valid)? "Valid":"Invalid", num_edges, num_edges/(rtstop-rtstart) * 1E-6 );
                  //for statistic
                  bfs_time.push_back(rtstop-rtstart);
                  nedge.push_back(num_edges);
                  teps.push_back(num_edges/(rtstop-rtstart));
              }
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
      if(rank==0){
        printf ("Validation: %s\n", (valid)? "passed":"failed!");
        printf ("SCALE: %d\n", scale);
        printf ("edgefactor: %d\n", edgefactor);
        printf ("NBFS: %d\n", iteration);
        printf ("graph_generation: %2.3e\n", make_graph_time);
        printf ("num_mpi_processes: %d\n", size);
        printf ("construction_time: %2.3e\n", constr_time);

        statistic bfs_time_s = getStatistics (bfs_time);
        printf ("min_time: %2.3e\n", bfs_time_s.min);
        printf ("firstquartile_time: %2.3e\n", bfs_time_s.firstquartile);
        printf ("median_time: %2.3e\n",bfs_time_s.median);
        printf ("thirdquartile_time: %2.3e\n", bfs_time_s.thirdquartile);
        printf ("max_time: %2.3e\n", bfs_time_s.max);
        printf ("mean_time: %2.3e\n", bfs_time_s.mean);
        printf ("stddev_time: %2.3e\n", bfs_time_s.stddev);

        statistic nedge_s = getStatistics (nedge);
        printf ("min_nedge: %2.3e\n", nedge_s.min);
        printf ("firstquartile_nedge: %2.3e\n", nedge_s.firstquartile);
        printf ("median_nedge: %2.3e\n", nedge_s.median);
        printf ("thirdquartile_nedge: %2.3e\n", nedge_s.thirdquartile);
        printf ("max_nedge: %2.3e\n", nedge_s.max);
        printf ("mean_nedge: %2.3e\n", nedge_s.mean);
        printf ("stddev_nedge: %2.3e\n", nedge_s.stddev);
/*
       TEPS = kernel_2_nedge ./ kernel_2_time;
       N = length (TEPS);
       S = statistics (TEPS);
       S(6) = mean (TEPS, 'h');
       %% Harmonic standard deviation from:
       %% Nilan Norris, The Standard Errors of the Geometric and Harmonic
       %% Means and Their Application to Index Numbers, 1940.
       %% http://www.jstor.org/stable/2235723
       tmp = zeros (N, 1);
       tmp(TEPS > 0) = 1./TEPS(TEPS > 0);
       tmp = tmp - 1/S(6);
       S(7) = (sqrt (sum (tmp.^2)) / (N-1)) * S(6)^2;
*/
        statistic teps_s = getStatistics (teps);
        printf ("min_TEPS: %2.3e\n", teps_s.min);
        printf ("firstquartile_TEPS: %2.3e\n",  teps_s.firstquartile);
        printf ("median_TEPS: %2.3e\n",  teps_s.median);
        printf ("thirdquartile_TEPS: %2.3e\n",  teps_s.thirdquartile);
        printf ("max_TEPS: %2.3e\n",  teps_s.max);
        printf ("harmonic_mean_TEPS: %2.3e\n",  teps_s.hmean);
        printf ("harmonic_stddev_TEPS: %2.3e\n",  teps_s.hstddev);

      }
      MPI_Finalize();
}

