#include <mpi.h>
#include <omp.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <vector>


const long iterations = 1000;
const long psize = 1000;
const bool test = false;// true;

struct slice{
    long real_slice;
    long real_slice_size;
    long real_slice_start;
    MPI_Request    rq;
};

int main(int argc, char** argv)
{
    int provided;
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
    if(rank == 0){
        ref = (end-start)/iterations;
        printf("default: %.3e\n",ref);
    }
    //multiple slice
    for(long slices=1; slices<9 ;slices++){
    double p1,p2,p3,s1=0,s2=0;
    start = MPI_Wtime();
    for(long i=0; i < iterations; i++){

        int maxrounds = static_cast<int>(ceil(log2(size)));

        long slice_size = psize/slices;
        long slice_res  = psize%slices;
        double* rc_buff = new double[psize];

        for(long i=0; i<psize; i++){
            recv[i] = send[i];
        }

        std::vector<slice> statusb;
        p1=MPI_Wtime();
        for(int rounds=0; rounds<maxrounds; rounds++){
            //compute active number of slices in this round for this node
            int rslices = slices / (1 << rounds ) + ((slices % (1 << rounds ) < 1)? 0 : 1);
            for(int s=0; s<rslices; s++){
                int vrank = (rank + s*(1 << rounds)) % size; // virtual rank for slice

                long real_slice = (rank%(1 << rounds)+s*(1 << rounds))%slices ;
                long real_slice_size  = slice_size            + ((real_slice < slice_res)? 1 : 0) ;
                long real_slice_start = slice_size*real_slice + ((real_slice >= slice_res)? slice_res : real_slice);

                if((vrank >> rounds)%2 == 1){// determine send or recive
                    MPI_Request request;
                    //send
                    int recv_addr;
                    //check if meseges have to be send up or down
                    if((rank >> rounds)%2 == 1)
                        //down
                        recv_addr = (rank + size - (1 << rounds)) % size;
                    else
                        //up
                        recv_addr = (rank + (1 << rounds)) % size;
                    //send fq
           //         printf("%d,%d: %d send to %d\n",rounds, s, rank,  recv_addr );
                    MPI_Isend(recv+real_slice_start, real_slice_size , MPI_DOUBLE,recv_addr,s,MPI_COMM_WORLD, &request);
                }else
                if(vrank + (1 << rounds) < size){ // test something to recive
                    slice sd = {real_slice,real_slice_size,real_slice_start };
                    //recive
                    int sender_addr;
                    if((rank >> rounds)%2 == 0)
                        //up
                        sender_addr = (rank +  (1 << rounds)) % size;
                    else
                        //down
                        sender_addr = (rank + size - (1 << rounds)) % size;
                    //recv fq
           //         printf("%d,%d: %d recive from %d\n",rounds, s, rank, sender_addr );
                    MPI_Irecv(rc_buff+real_slice_start, real_slice_size, MPI_DOUBLE,sender_addr,s, MPI_COMM_WORLD, &sd.rq);
                    statusb.push_back(sd);
                }
            }
            #pragma omp parallel for
            for(auto s=statusb.begin(); s !=statusb.end(); s++){
                MPI_Wait(&s->rq,MPI_STATUS_IGNORE);
                //do reduce
                for(long it=s->real_slice_start;
                    it < s->real_slice_start+s->real_slice_size; it++){
                    recv[it] += rc_buff[it];
                }
            }
            statusb.clear();
        }
        delete[] rc_buff;
        MPI_Barrier(MPI_COMM_WORLD);
        p2=MPI_Wtime();
        s1+=p2-p1;

        int exsl = slices +((slices%size==0)? 0: size-slices%size);
        int* sendc = new int[exsl];
        int* displs = new int[exsl];
        for(long i=0; i < slices; i++){
            sendc[i] = slice_size   + ((i < slice_res)? 1 : 0) ;
            displs[i] =  slice_size*i + ((i >= slice_res)? slice_res : i);
        }
        for(long i=slices; i < exsl; i++){
            sendc[i] = 0;
            displs[i] = 0;
        }
        int ms = slices/size+((slices%size!=0)?1:0);
        for(long s=0; s< ms; s++){
            MPI_Allgatherv(recv+displs[size*s+rank], sendc[size*s+rank],
                MPI_DOUBLE, recv+displs[size*s], sendc+size*s,
                displs+size*s, MPI_DOUBLE, MPI_COMM_WORLD);
        }
        /*
       // MPI_Request* request = new MPI_Request[slices];
        //distribute solution
        for(long s=0; s<slices; s++){
            int root = s%size;
            long ssize  = slice_size   + ((s < slice_res)? 1 : 0) ;
            long sstart = slice_size*s + ((s >= slice_res)? slice_res : s);

     //       MPI_Ibcast(recv+sstart,ssize,MPI_DOUBLE,root,MPI_COMM_WORLD, request+s);
            MPI_Bcast(recv+sstart,ssize,MPI_DOUBLE,root,MPI_COMM_WORLD);
        }
        //MPI_Waitall(slices,request,MPI_STATUSES_IGNORE);
        */
        MPI_Barrier(MPI_COMM_WORLD);
        p3=MPI_Wtime();
        s2+=p3-p2;
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
    if(rank == 0){
        double sres = (end-start)/iterations-2*btime;
        double s1res = s1/iterations-btime;
        double s2res = s2/iterations-btime;
        printf("Multiple slices(%ld): %.3e(%3.0f) %.3e(%3.0f) %.3e(%3.0f)\n",slices,
           sres, (sres/ref)*100.,
           s1res, (s1res/sres)*100.,
           s2res, (s2res/sres)*100.);
    }
    }
    delete[] send;
    delete[] recv;
    }
    MPI_Finalize();

    return 0;
}

