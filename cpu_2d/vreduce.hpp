#include <mpi.h>
#include <omp.h>
#include <cmath>

#include <algorithm>
#include <functional>

#include <sstream>

#ifndef VREDUCE_HPP
#define VREDUCE_HPP

template<class T>
void vreduce(std::function<void(T, long, T*, int )>& reduce, //void (long start, long size, FQ_T* &startaddr, vtxtype& outsize)
             std::function<void(T, long, T*&, int& )>& get, //void (long start, long size, FQ_T* &startaddr, vtxtype& outsize)
             T* recv_buff,
             int& rsize,
             int ssize,
             MPI_Datatype type,
             MPI_Comm comm
             #ifdef INSTRUMENTED
             ,double& twork
             #endif
             ){
    //
    int size, rank, rank2, n , p2n, r;

    //time mesurement
    #ifdef INSTRUMENTED
    double start, end ;
    #endif

    //step 1
    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank2);
    n = ilogb(static_cast<double>(size)); //integer log_2 of size
    p2n = 1 << n; // 2^n
    r = size - (1 << n);

    //step 2
    if( rank < 2 * r){
        if((rank & 1) == 0){ // even
            MPI_Status status; int psize_from;

            MPI_Recv(recv_buff,ssize,type, rank+1, 1, comm, &status);
            MPI_Get_count(&status,type,&psize_from );
            #ifdef INSTRUMENTED
            start=MPI_Wtime();
            #endif
            reduce(0,ssize,recv_buff,psize_from);
            #ifdef INSTRUMENTED
            end=MPI_Wtime();
            twork += end -start;
            #endif

        } else { // odd
            int psize_to;
            T* send;
            #ifdef INSTRUMENTED
            start=MPI_Wtime();
            #endif
            get(0, ssize, send, psize_to);
            #ifdef INSTRUMENTED
            end=MPI_Wtime();
            twork += end -start;
            #endif
            MPI_Send(send, psize_to, type, rank-1,1, comm);
        }
     }
     const std::function<int (int)> newrank = [&r](int oldr) { return (oldr < 2*r)? oldr/2 : oldr -r; };
     const std::function<int (int)> oldrank = [&r](int newr) { return (newr <  r )? newr*2 : newr +r; };

     MPI_Status status;
     int psize_to, psize_from;
     T* send;

     if((((rank & 1)==0) &&(rank < 2*r))||(rank >= 2*r)){

         int vrank, csize, offset, lowers, uppers;

         vrank  = newrank(rank);
         csize  = ssize;
         offset = 0;

         get(offset, csize, send, psize_to);

         for(int it=0; it < n; it++){
             lowers = csize/2;
             uppers = csize - lowers;

             if(((vrank >> it)&1)==0){// even
                #ifdef INSTRUMENTED
                start=MPI_Wtime();
                #endif
                get(offset+lowers, uppers, send, psize_to);
                #ifdef INSTRUMENTED
                end=MPI_Wtime();
                twork += end -start;
                #endif
                MPI_Sendrecv(send+offset, psize_to, type,
                                 oldrank((vrank+(1<<it))&(p2n-1)), it+2,
                                 recv_buff, lowers, type,
                                 oldrank((vrank+(1<<it))&(p2n-1)), it+2,
                                 comm, &status);

                MPI_Get_count(&status,type,&psize_from );
                #ifdef INSTRUMENTED
                start=MPI_Wtime();
                #endif
                reduce(offset,lowers,recv_buff,psize_from);
                #ifdef INSTRUMENTED
                end=MPI_Wtime();
                twork += end -start;
                #endif
                csize = lowers;
             } else { // odd
                #ifdef INSTRUMENTED
                start=MPI_Wtime();
                #endif
                get(offset, lowers, send, psize_to);
                #ifdef INSTRUMENTED
                end=MPI_Wtime();
                twork += end -start;
                #endif
                MPI_Sendrecv(send+offset, psize_to, type,
                                  oldrank((p2n+vrank-(1<<it))&(p2n-1)), it+2,
                                  recv_buff, uppers, type,
                                  oldrank((p2n+vrank-(1<<it))&(p2n-1)), it+2,
                                  comm, &status);

                 MPI_Get_count(&status,type,&psize_from );
                 #ifdef INSTRUMENTED
                 start=MPI_Wtime();
                 #endif
                 reduce(offset+lowers,uppers,recv_buff,psize_from);
                 #ifdef INSTRUMENTED
                 end=MPI_Wtime();
                 twork += end -start;
                 #endif
                 offset += lowers;
                 csize = uppers;
              }
         }
        // Datas to send to the other nodes
        #ifdef INSTRUMENTED
        start=MPI_Wtime();
        #endif
        get(offset, csize, send, psize_to);
        #ifdef INSTRUMENTED
        end=MPI_Wtime();
        twork += end -start;
        #endif
     }else{
         psize_to = 0;
         send = 0;
     }

     // Transmission of the final results
     int sizes[size];
     int disps[size];

     // Transmission of the subslice sizes
     MPI_Allgather(&psize_to ,1,MPI_INT,sizes,1,MPI_INT,comm);
     //Computation of displacements
     disps[oldrank(0)] = 0;
     for(int it=1; it< p2n; it++){
         disps[oldrank(it)]=disps[oldrank(it-1)]+sizes[oldrank(it-1)];
     }
     for(int it=p2n; it< size; it++){
         disps[oldrank(it-p2n)+1]=0;
     }

     MPI_Allgatherv(send, sizes[rank],
         type, recv_buff, sizes,
         disps, type, comm);

     rsize = disps[oldrank(p2n-1)] + sizes[oldrank(p2n-1)];

}


#endif // VREDUCE_HPP
