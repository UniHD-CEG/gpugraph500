#include <mpi.h>
#include <omp.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <vector>
#include <array>
#include <tuple>

#include <functional>

const long iterations = 100;
const long psize = 1000;
const bool test = false;// true;

int main(int argc, char** argv)
{
    //int provided;
    //MPI_Init_thread(&argc, &argv,MPI_THREAD_MULTIPLE,&provided);
    MPI_Init(&argc, &argv);
    //if(provided==MPI_THREAD_MULTIPLE){
    //    printf("Wrong thread mode!\n");
    //}
    int size, rank;

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double start, end;
    if(rank == 0){
    printf("All-reduce test\n");
    printf("Test nodes: %d\n",size);
    printf("iterations: %ld\n",iterations);
    }

    //messure barrier to subtract its time
    MPI_Barrier(MPI_COMM_WORLD); //first time sometimes slow
    start = MPI_Wtime();
    for(long i=0; i < iterations; i++ ){
        MPI_Barrier(MPI_COMM_WORLD);
    }
    end = MPI_Wtime();
    double btime = (end-start) / iterations;

    if(rank == 0){
    printf("Barrier: %.3e\n",(end-start)/iterations);
    }

    //preperation
    for(long psize =20; psize < 1000000; psize*=4){
    if(rank==0)
        printf("problem size: %ld\n", psize);
    double* send = new double[psize];
    double* recv = new double[psize];

    for(long i=0; i<psize; i++){
        send[i] = 1.;
    }

    double ref;
    //mpi reference
    start = MPI_Wtime();
    for(long i=0; i < iterations; i++ ){
        MPI_Allreduce(send, recv, psize,
            MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        if(test){
            bool ok = true;
            for(long it=0; it<psize; it++){
                if(recv[it]!=size){
                    ok = false;
                    printf("%ld expect %d got %f\n",it,size,recv[it]);
                    break;
                }
            }
            if(!ok){
                printf("Error found!\n");
            }
        }
    }
    end = MPI_Wtime();

    ref = (end-start)/iterations;
    if(rank == 0){
        printf("default: %.3e\n",ref);
    }

    double red, dist, comp, s0, s1, s2, s3;
    red=0.; dist=0.; comp=0.;

    double own1;
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Datatype type = MPI_DOUBLE;
    double* tmp = new double[psize];
    start = MPI_Wtime();
    for(long i=0; i < iterations; i++ ){

        std::copy_n(send,psize,recv);

        int rounds = 0;
        s0= MPI_Wtime();
        while((1 << rounds) < size ){
            if((rank >> rounds)%2 == 1){
                //comute recv addr
                int recv_addr = (rank + size - (1 << rounds)) % size;
                //send fq
                MPI_Send(recv,psize,type,recv_addr,rounds,comm);
                break;
            } else if ( rank+ (1 << rounds) < size ){
                MPI_Status    status;
                //compute send addr
                int sender_addr = (rank +  (1 << rounds)) % size;
                //recv fq
                MPI_Recv(tmp, psize, type,sender_addr,rounds, comm, &status);
                //do reduce
                s1 = MPI_Wtime();
                #pragma omp parallel for schedule(static,256) if(psize>256)
                for(int i=0; i<psize;i++){
                    recv[i]+=tmp[i];
                }
                s2=MPI_Wtime();
                comp+=s2-s1;
            }
            rounds++;
        }
        s3=MPI_Wtime();
        red += s3-s0;
        MPI_Bcast(recv,psize,type,0 ,comm);
        MPI_Barrier(comm);
        s0=MPI_Wtime();
        dist += s0 -s3;

    }
    end = MPI_Wtime();

    red/=iterations; dist= dist/iterations - btime; comp/=iterations;
    own1 = (end-start)/iterations -btime;
    if(rank == 0){
        printf("own1 imp.: %.3e (%3.0f%%) red %.3e (%3.0f%%) dist %.3e (%3.0f%%) comp %.3e (%3.0f%%)\n",own1,(own1/ref)*100.,red,(red/own1)*100.,dist, (dist/own1)*100.,comp, (comp/own1)*100.);
    }

    double own2;
    //mpi reference
    //MPI_Comm comm = MPI_COMM_WORLD;
    //MPI_Datatype type = MPI_DOUBLE;
    //double* tmp = new double[psize];

    red=0.; dist=0.; comp=0.;

    start = MPI_Wtime();
    for(long i=0; i < iterations; i++ ){
        MPI_Status status;
        int size, rank, n , p2n, r;

        //step 1
        MPI_Comm_size(comm, &size);
        MPI_Comm_rank(comm, &rank);
        n = log2(size); //integer log_2 of size
        p2n = 1 << n; // 2^n
        r = size - (1 << n);

        std::copy_n(send,psize,recv);
        //step 2
        s0=MPI_Wtime();
        if( rank < 2 * r){
            /*
            int lowers = psize/2;       //size of the upper segment that will be tansported
            int uppers = psize - lowers;//size of the lower segment that will be tansported

            if((rank & 1) == 0){ // even

                MPI_Sendrecv(recv+lowers, uppers, type,
                                rank+1, 0,
                                tmp, lowers, type,
                                rank+1, 0,
                                comm, &status);
                for(int i=0; i < lowers; i++){
                    recv[i] += tmp[i];
                }

                MPI_Recv(recv+lowers,uppers,type, rank+1, 1, comm, &status);
            } else { // odd
                MPI_Sendrecv(recv, lowers, type,
                                rank-1, 0,
                                tmp, uppers, type,
                                rank-1, 0,
                                comm, &status);
                for(int i=0; i < uppers; i++){
                    recv[i+lowers] += tmp[i];
                }

                MPI_Send(tmp, uppers, type, rank-1,1, comm);
            }
            */
            if((rank & 1) == 0){ // even
                MPI_Recv(recv,psize,type, rank+1, 1, comm, &status);
                s1= MPI_Wtime();
                #pragma omp parallel for schedule(static,256) if(psize>256)
                for(int i=0; i < psize; i++){
                    recv[i] += tmp[i];
                }
                s2= MPI_Wtime();
                comp += s2-s1;


            } else { // odd
                MPI_Send(recv, psize, type, rank-1,1, comm);
            }
        }
        const std::function<int (int)> newrank = [&r](int oldr) { return (oldr < 2*r)? oldr/2 : oldr -r; };
        const std::function<int (int)> oldrank = [&r](int newr) { return (newr <  r )? newr*2 : newr +r; };

        if((((rank & 1)==0) &&(rank < 2*r))||(rank >= 2*r)){

            int vrank, csize, offset, lowers, uppers;

            vrank  = newrank(rank);
            csize  = psize;
            offset = 0;

            for(int it=0; it < n; it++){
                lowers = csize/2;
                uppers = csize - lowers;

                if(((vrank >> it)&1)==0){// even
                   MPI_Sendrecv(recv+offset+lowers, uppers, type,
                                    oldrank((vrank+(1<<it))&(p2n-1)), it+2,
                                    tmp, lowers, type,
                                    oldrank((vrank+(1<<it))&(p2n-1)), it+2,
                                    comm, &status);
                   s1= MPI_Wtime();
                   #pragma omp parallel for schedule(static,256) if(lowers>256)
                   for(int i=0; i < lowers; i++){
                       recv[i+offset] += tmp[i];
                   }
                   s2= MPI_Wtime();
                   comp += s2-s1;
                   csize = lowers;
                 } else { // odd
                    MPI_Sendrecv(recv+offset, lowers, type,
                                     oldrank((p2n+vrank-(1<<it))&(p2n-1)), it+2,
                                     tmp, uppers, type,
                                     oldrank((p2n+vrank-(1<<it))&(p2n-1)), it+2,
                                     comm, &status);
                    s1= MPI_Wtime();
                    #pragma omp parallel for schedule(static,256) if(uppers>256)
                    for(int i=0; i < uppers; i++){
                        recv[i+offset+lowers] += tmp[i];
                    }
                    s2= MPI_Wtime();
                    comp += s2-s1;
                    offset += lowers;
                    csize = uppers;
                 }
            }
        }
        s3 = MPI_Wtime();
        red += s3 -s0;
        // Transmission of the final results
        int* sizes = new int[size];
        int* disps = new int[size];

        for(int it=0; it< p2n; it++){
            int reverse = (p2n-1)-it;
            sizes[oldrank(it)]=(((reverse+1)*psize)>>n) -((reverse*psize)>>n);
            disps[oldrank(it)]=(reverse*psize)>>n;
        }
        for(int it=p2n; it< size; it++){
            sizes[oldrank(it-p2n)+1]=0;
            disps[oldrank(it-p2n)+1]=0;
        }

        MPI_Allgatherv(MPI_IN_PLACE, sizes[rank],
            type, recv, sizes,
            disps, type, comm);
        s0= MPI_Wtime();
        dist += s0 - s3;

    }
    end = MPI_Wtime();

    own2 = (end-start)/iterations;
    red/=iterations; dist/=iterations; comp/=iterations;
    if(rank == 0){
        printf("own2 imp.: %.3e (%3.0f%%) red %.3e (%3.0f%%) dist %.3e (%3.0f%%) comp %.3e (%3.0f%%)\n",own2,(own2/ref)*100.,red,(red/own2)*100.,dist, (dist/own2)*100.,comp, (comp/own2)*100.);
    }

    }
    MPI_Finalize();

    return 0;
}

