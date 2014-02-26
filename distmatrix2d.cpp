#include "distmatrix2d.h"

#include <algorithm>
#include <iostream>

#include <cstdlib> // only for values of packed_edge typ!!!
#ifdef _OPENMP
    #include <omp.h>
#endif

/*
 *Computes the owener node of a specific edge
*/
long DistMatrix2d::computeOwner(unsigned long row, unsigned long column)
{
    long rowSlice, columnSlice;
    long r_residuum, c_residuum;
    long rSliceSize, cSliceSize;

    long num8 = globalNumberOfVertex/8 + ((globalNumberOfVertex%8>0)? 1:0) ;

    r_residuum = num8 % R;
    rSliceSize = num8 / R;

    if((rowSlice = row/(rSliceSize+1)) >= r_residuum ){//compute row slice, if the slice number is in the bigger intervals
        rowSlice =  (row- r_residuum)/ rSliceSize; //compute row slice, if the slice number is in the smaler intervals
    }

    c_residuum = num8 % C;
    cSliceSize = num8 / C;

    if((columnSlice = column/(cSliceSize+1)) >= c_residuum ){ //compute column slice, if the slice number is in the bigger intervals
        columnSlice =  (column- c_residuum)/ cSliceSize; //compute column slice, if the slice number is in the smaler interval
    }
    return rowSlice*C+columnSlice;
}

bool DistMatrix2d::comparePackedEdgeR(packed_edge i, packed_edge j){
    if(i.v0<j.v0){
        return true;
    }else if(i.v0>j.v0 ){
        return false;
    } else
        return (i.v1<j.v1);
}
bool DistMatrix2d::comparePackedEdgeC(packed_edge i, packed_edge j){
    if(i.v1<j.v1){
        return true;
    }else if(i.v1>j.v1 ){
        return false;
    } else
        return (i.v0<j.v0);
}



DistMatrix2d::DistMatrix2d(int _R, int _C):R(_R),C(_C),row_pointer(NULL),column_index(NULL)
{
    int rank;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    //Compute owen row and column id
    r = rank/C;
    c = rank%C;
}

DistMatrix2d::~DistMatrix2d()
{
   if(row_pointer!=NULL)    delete[] row_pointer;
   row_pointer=NULL;
   if(column_index!=NULL)   delete[] column_index;
   column_index=NULL;

}

/*
 * Setup of 2d partitioned adjacency matrix.
 * 1. Annonce elements in each row(ignore selfloops)
 * 2. Compute intermediate row pointer and storage consume for columns
 * 3. Send edge list to owner
 * 4. Sort column indices
 * 5. Remove duplicate column indices
 *
 * Should be optimised.
*/
void DistMatrix2d::setupMatrix(packed_edge *input, long numberOfEdges, bool undirected)
{
    //get max vtx
    vtxtype maxVertex = -1;

    #pragma omp parallel for reduction(max: maxVertex)
    for(long i = 0; i < numberOfEdges; i++){
        packed_edge read = input[i];

        maxVertex = (maxVertex > read.v0)? maxVertex :  read.v0;
        maxVertex = (maxVertex > read.v1)? maxVertex :  read.v1;
    }


    MPI_Allreduce(&maxVertex, &globalNumberOfVertex, 1, MPI_LONG, MPI_MAX, MPI_COMM_WORLD);
    //because start at 0
    globalNumberOfVertex = globalNumberOfVertex+1;
    long num8 = globalNumberOfVertex/8 + ((globalNumberOfVertex%8>0)? 1:0) ;
    row_start       = (r*(num8/R) + ((r < num8%R)? r : num8%R))*8;
    row_length      = (num8/R + ((r < num8%R)? 1 : 0))*8;
    column_start    = (c*(num8/C) + ((c < num8%C)? c : num8%C))*8;
    column_length   = (num8/C + ((c < num8%C)? 1 : 0))*8;

    row_pointer = new vtxtype[row_length+1];
    long* row_elem = new long[row_length];
    const int outstanding_sends=40;
    MPI_Request iqueue[outstanding_sends];

    // row_elem is in the first part a counter of nz columns per row
    for(long i=0; i< row_length; i++){
       row_elem[i] = 0;
    }
    // Bad implementation: To many comunications
    int count_elementssend;
    int freeRqBuf;
    MPI_Status status;
    int flag;

    //reset outgoing send status
    for(int i=0; i < outstanding_sends; i++){
	iqueue[i]= MPI_REQUEST_NULL;
    }

    count_elementssend = 0;
    while(count_elementssend<numberOfEdges){
        for(int i=0; i < 10 && count_elementssend<numberOfEdges ; i++){
        //find free send buff;
        freeRqBuf = -1;
        for(int j=0; j<outstanding_sends; j++){
            MPI_Test(&(iqueue[(j+i)%outstanding_sends]),&flag,&status);
            if(flag){
                freeRqBuf = (j+i)%outstanding_sends;
                break;
            }
        }
        
        //Send an element
        if(freeRqBuf>-1){
            //if(input[count_elementssend].v0 != input[count_elementssend].v1){
                long dest = computeOwner(input[count_elementssend].v0, input[count_elementssend].v1);
                MPI_Issend(&(input[count_elementssend].v0), 1, MPI_LONG, dest, 0, MPI_COMM_WORLD, &(iqueue[freeRqBuf]));
            //}
            count_elementssend++;
        }else{
            int gunfinished;
            int someting_unfinshed=1;

            //Tell others that there is something unfinshed
            MPI_Allreduce(&someting_unfinshed, &gunfinished, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
        }
        }

        while(true){
            //Test if there is something to recive
            MPI_Iprobe(MPI_ANY_SOURCE, 0, MPI_COMM_WORLD,
                       &flag, &status);
            if(!flag) // There is no package to recive
                break;

            long buf;
            MPI_Recv(&buf, 1,  MPI_LONG, MPI_ANY_SOURCE, 0,
                       MPI_COMM_WORLD, &status);
            row_elem[buf-row_start]++;

        }
    }

    if(undirected){
        count_elementssend =0;
        while(count_elementssend<numberOfEdges){
            for(int i=0; i < 10 && count_elementssend<numberOfEdges ; i++){
            //find free send buff;
            freeRqBuf = -1;
            for(int j=0; j<outstanding_sends; j++){
                MPI_Test(&(iqueue[(j+i)%outstanding_sends]),&flag,&status);
                if(flag){
                    freeRqBuf = (j+i)%outstanding_sends;
                    break;
                }
            }

            //Send an element
            if(freeRqBuf>-1){
                //if(input[count_elementssend].v0 != input[count_elementssend].v1){
                    long dest = computeOwner(input[count_elementssend].v1, input[count_elementssend].v0);
                    MPI_Issend(&(input[count_elementssend].v1), 1, MPI_LONG, dest, 0, MPI_COMM_WORLD, &(iqueue[freeRqBuf]));
                //}
                count_elementssend++;
            }else{
                int gunfinished;
                int someting_unfinshed=1;

                //Tell others that there is something unfinshed
                MPI_Allreduce(&someting_unfinshed, &gunfinished, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
            }
            }

            while(true){
                //Test if there is something to recive
                MPI_Iprobe(MPI_ANY_SOURCE, 0, MPI_COMM_WORLD,
                           &flag, &status);
                if(!flag) // There is no package to recive
                    break;

                long buf;
                MPI_Recv(&buf, 1,  MPI_LONG, MPI_ANY_SOURCE, 0,
                           MPI_COMM_WORLD, &status);
                row_elem[buf-row_start]++;

            }
        }
    }

    //All sends are started
    //Recive rest
    while(true){
        //find unfinished sends
        int someting_unfinshed=0;

        for(int i=0; i<outstanding_sends; i++){
            MPI_Test(&(iqueue[i]),&flag,&status);
            if(!flag){
                someting_unfinshed = 1;
                break;
            }
        }
        //Ask other if there is something unfinshed
        int gunfinished;
        MPI_Allreduce(&someting_unfinshed, &gunfinished, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
        if(gunfinished == 0)
            break;
        while(true){
        //Test if there is something to recive
        MPI_Iprobe(MPI_ANY_SOURCE, 0, MPI_COMM_WORLD,
                   &flag, &status);
        if(!flag) // There is no package to recive
            break;

        long buf;
        MPI_Recv(&buf, 1,  MPI_LONG, MPI_ANY_SOURCE, 0,
                   MPI_COMM_WORLD, &status);
        row_elem[buf-row_start]++;

        }
    };

//2. Step
    //compute row index array(prefix scan)
    row_pointer[0] = 0;
    for(long i=1; i< row_length+1; i++){
        row_pointer[i] = row_pointer[i-1]+row_elem[i-1];
    }
    // row_elem is now a pointer to the relativ position, where new elements of a row may be insert
    for(long i=0; i< row_length; i++){
        row_elem[i] = 0;
    }
    column_index = new vtxtype[row_pointer[row_length]];
//3.
    //reset outgoing send status
    for(int i=0; i < outstanding_sends; i++){
	iqueue[i]=MPI_REQUEST_NULL;
    }

    count_elementssend =0;
    while(count_elementssend<numberOfEdges){
        for(int i=0; i < 10 && count_elementssend<numberOfEdges ; i++){
        //find free send buff;
        freeRqBuf = -1;
        for(int j=0; j<outstanding_sends; j++){
            MPI_Test(&(iqueue[(j+i)%outstanding_sends]),&flag,&status);
            if(flag){
                freeRqBuf = (j+i)%outstanding_sends;
                break;
            }
        }
        //Send an edge
        if(freeRqBuf>-1){
            long send_buf[2];
            send_buf[0]=input[count_elementssend].v0;
            send_buf[1]=input[count_elementssend].v1;
            //if(send_buf[0]!=send_buf[1]){
                long dest = computeOwner(input[count_elementssend].v0, input[count_elementssend].v1);

                MPI_Issend(send_buf, 2, MPI_LONG, dest, 0, MPI_COMM_WORLD, &(iqueue[freeRqBuf]));
            //}
            count_elementssend++;
        }else{
            int gunfinished;
            int someting_unfinshed=1;

            //Tell others that there is something unfinshed
            MPI_Allreduce(&someting_unfinshed, &gunfinished, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
        }
        }

        while(true){
            //Test if there is something to recive
            MPI_Iprobe(MPI_ANY_SOURCE, 0, MPI_COMM_WORLD,
                           &flag, &status);
            if(!flag) // There is no package to recive
                 break;

            long buf[2];
            MPI_Recv(buf, 2,  MPI_LONG, MPI_ANY_SOURCE, 0,
                           MPI_COMM_WORLD, &status);
            column_index[row_pointer[buf[0]-row_start]+row_elem[buf[0]-row_start]] = buf[1];
            row_elem[buf[0]-row_start]++;

        }
    }

    if(undirected){
    count_elementssend =0;
    while(count_elementssend<numberOfEdges){
        for(int i=0; i < 10 && count_elementssend<numberOfEdges ; i++){
        //find free send buff;
        freeRqBuf = -1;
        for(int i=0; i<outstanding_sends; i++){
            MPI_Test(&(iqueue[i]),&flag,&status);
            if(flag){
                freeRqBuf = i;
                break;
            }
        }

        //Send an edge
        if(freeRqBuf>-1){
            long send_buf[2];
            send_buf[0]=input[count_elementssend].v1;
            send_buf[1]=input[count_elementssend].v0;
            //if(send_buf[0]!=send_buf[1]){
                long dest = computeOwner(input[count_elementssend].v1, input[count_elementssend].v0);
                MPI_Issend(send_buf, 2, MPI_LONG, dest, 0, MPI_COMM_WORLD, &(iqueue[freeRqBuf]));
            //}
            count_elementssend++;
        }else{
            int gunfinished;
            int someting_unfinshed=1;

            //Tell others that there is something unfinshed
            MPI_Allreduce(&someting_unfinshed, &gunfinished, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
        }
        }
        while(true){
            //Test if there is something to recive
            MPI_Iprobe(MPI_ANY_SOURCE, 0, MPI_COMM_WORLD,
                       &flag, &status);
            if(!flag) // There is no package to recive
              break;

            long buf[2];
            MPI_Recv(buf, 2,  MPI_LONG, MPI_ANY_SOURCE, 0,
                      MPI_COMM_WORLD, &status);
            column_index[row_pointer[buf[0]-row_start]+row_elem[buf[0]-row_start]] = buf[1];
            row_elem[buf[0]-row_start]++;

        }
        }
     }
     //All sends are started
     //Recive rest
     while(true){
        //find unfinished sends
        int someting_unfinshed=0;

        for(int i=0; i<outstanding_sends; i++){
            MPI_Test(&(iqueue[i]),&flag,&status);
            if(!flag){
                someting_unfinshed = 1;
                break;
            }
        }
        //Ask other if there is something unfinshed
        int gunfinished;
        MPI_Allreduce(&someting_unfinshed, &gunfinished, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
        if(gunfinished==0)
             break;

        while(true){
        //Test if there is something to recive
        MPI_Iprobe(MPI_ANY_SOURCE, 0, MPI_COMM_WORLD,
                &flag, &status);
        if(!flag) // There is no package to recive
          break;

        long buf[2];
        MPI_Recv(buf, 2,  MPI_LONG, MPI_ANY_SOURCE, 0,
                 MPI_COMM_WORLD, &status);
        column_index[row_pointer[buf[0]-row_start]+row_elem[buf[0]-row_start]] = buf[1];
        row_elem[buf[0]-row_start]++;

        }
    }


    //sanity check
    #pragma omp parallel for
    for(long i=0; i< row_length; i++){
        if(row_elem[i]!=row_pointer[i+1]-row_pointer[i]){
            fprintf(stderr,"Number of nz-column mismatch in row %ld on node (%d:%d). Annonced: %ld Recived: %ld\n",
                    (row_start+i),r,c,row_elem[i],row_pointer[i+1]-row_pointer[i]);
        }
    }
//4.
    //sort edge list
    #pragma omp parallel for
    for(int i=0; i< row_length; i++){
        std::sort(column_index+row_pointer[i],column_index+row_pointer[i+1]);
    }
//5.
    // remove duplicates
    // The next section is very bad, because it use too much memory.
    #pragma omp parallel for
    for(long i=0; i< row_length; i++){
        long tmp_row_num = row_elem[i];
        //Search for duplicates in every row
        for(long j = row_pointer[i]+1; j < row_pointer[i+1]; j++){
            if(column_index[j-1]==column_index[j])
                tmp_row_num--;
        }
        row_elem[i] = tmp_row_num;
    }

    vtxtype* tmp_row_pointer = new vtxtype[row_length+1];
    vtxtype* tmp_column_index;

    tmp_row_pointer[0] = 0;
    for(long i=1; i< row_length+1; i++){
        tmp_row_pointer[i] = tmp_row_pointer[i-1]+row_elem[i-1];
    }

    tmp_column_index = new vtxtype[tmp_row_pointer[row_length]];
    //Copy unique entries in every row
    #pragma omp parallel for
    for(long i=0; i< row_length; i++){
        long next_elem = tmp_row_pointer[i];
        //skip empty row
        if(next_elem == tmp_row_pointer[i+1]) continue;
        tmp_column_index[next_elem] = column_index[row_pointer[i]];
       next_elem++;
        for(long j = row_pointer[i]+1; j < row_pointer[i+1]; j++){
            if(column_index[j-1]!=column_index[j]){
                tmp_column_index[next_elem] = column_index[j];
                next_elem++;
            }
        }
    }

    delete[] row_elem;
    delete[] row_pointer;
    delete[] column_index;
    row_pointer = tmp_row_pointer;
    column_index = tmp_column_index;
 }

void DistMatrix2d::setupMatrix2(packed_edge *&input, long &numberOfEdges, bool undirected){

    vtxtype maxVertex = -1;
    if(undirected == true){
        //generate other direction to be undirected
        input=(packed_edge *)realloc(input, 2*numberOfEdges*sizeof(packed_edge));

        #pragma omp parallel for reduction(max: maxVertex)
        for(long i = 0; i < numberOfEdges; i++){
            packed_edge read = input[i];

            input[numberOfEdges+i].v0 = read.v1;
            input[numberOfEdges+i].v1 = read.v0;

            maxVertex  = (maxVertex > read.v0)? maxVertex :  read.v0;
            maxVertex  = (maxVertex > read.v1)? maxVertex :  read.v1;
        }

        numberOfEdges= 2*numberOfEdges;
    } else {
        #pragma omp parallel for reduction(max: maxVertex)
        for(long i = 0; i < numberOfEdges; i++){
            packed_edge read = input[i];

            maxVertex  = (maxVertex > read.v0)? maxVertex:  read.v0;
            maxVertex  = (maxVertex > read.v1)? maxVertex :  read.v1;
        }
    }

    MPI_Allreduce(&maxVertex, &globalNumberOfVertex, 1, MPI_LONG, MPI_MAX, MPI_COMM_WORLD);
    //because start at 0
    globalNumberOfVertex = globalNumberOfVertex+1;
    long num8 = globalNumberOfVertex/8 + ((globalNumberOfVertex%8>0)? 1:0) ;
    row_start       = (r*(num8/R) + ((r < num8%R)? r : num8%R))*8;
    row_length      = (num8/R + ((r < num8%R)? 1 : 0))*8;
    column_start    = (c*(num8/C) + ((c < num8%C)? c : num8%C))*8;
    column_length   = (num8/C + ((c < num8%C)? 1 : 0))*8;

    // Split communicator into row and column communicator
    MPI_Comm row_comm, col_comm;
    // Split by row, rank by column
    MPI_Comm_split(MPI_COMM_WORLD, r, c, &row_comm);
    // Split by column, rank by row
    MPI_Comm_split(MPI_COMM_WORLD, c, r, &col_comm);

    MPI_Datatype packedEdgeMPI;
    int tupsize[] = {2};
    MPI_Aint tupoffset[] = {0};
    MPI_Datatype tuptype[] = {MPI_LONG};
    MPI_Type_struct(1,tupsize,tupoffset,tuptype,&packedEdgeMPI);
    MPI_Type_commit(&packedEdgeMPI);

    //column comunication
    std::sort(input,input+numberOfEdges,DistMatrix2d::comparePackedEdgeR);

    int* owen_send_size = new int[R];
    int* owen_offset = new int[R+1];
    int* other_size = new int[R];
    int* other_offset =  new int[R+1];

    long res = num8 % R;
    long ua_sl_size = num8 / R;

    // offset for transmission
    long sl_start = 0;
    owen_offset[0] = 0;
    for(int i = 1; i < R; i++){
        if(res > 0) {
            sl_start += (ua_sl_size +1)*8;
            res--;
        }else{
            sl_start += ua_sl_size*8;
        }
        packed_edge startEdge = {sl_start,0};
        owen_offset[i] = std::lower_bound(input+owen_offset[i-1],input+numberOfEdges, startEdge, DistMatrix2d::comparePackedEdgeR) - input;
    }
    owen_offset[R] = numberOfEdges;

    // compute transmission sizes
    for(int i = 0; i < R; i++){
        owen_send_size[i] = owen_offset[i+1]- owen_offset[i];
    }
    // send others sizes to recive sizes
    MPI_Alltoall(owen_send_size,1,MPI_INT,other_size,1,MPI_INT, col_comm);
    // compute transmission offsets
    other_offset[0] = 0;
    for(int i = 1; i < R+1; i++){
        other_offset[i] = other_size [i-1]+ other_offset[i-1];
    }
    numberOfEdges= other_offset[R];

    // allocate recive buffer
    packed_edge * coltransBuf = (packed_edge *) malloc(numberOfEdges*sizeof(packed_edge));

    // transmit data
    MPI_Alltoallv(input, owen_send_size, owen_offset, packedEdgeMPI, coltransBuf, other_size, other_offset, packedEdgeMPI, col_comm );


    //not nessecary any more
    free(input);
    input = coltransBuf;

    delete[] owen_send_size;
    delete[] owen_offset;
    delete[] other_size ;
    delete[] other_offset;

    //row comunication
    //sort
    std::sort(input,input+numberOfEdges,DistMatrix2d::comparePackedEdgeC);

    owen_send_size = new int[C];
    owen_offset = new int[C+1];
    other_size = new int[C];
    other_offset =  new int[C+1];

    res = num8 % C;
    ua_sl_size = num8 / C;

    // offset for transmission
    sl_start = 0;
    owen_offset[0] = 0;
    for(int i = 1; i < C; i++){
        if(res > 0) {
            sl_start += (ua_sl_size +1)*8;
            res--;
        }else{
            sl_start += ua_sl_size*8;
        }

        packed_edge startEdge = {0,sl_start};
        owen_offset[i] = std::lower_bound(input+owen_offset[i-1],input+numberOfEdges, startEdge , DistMatrix2d::comparePackedEdgeC) - input;       
    }
    owen_offset[C] = numberOfEdges;

    // compute transmission sizes
    for(int i = 0; i < C; i++){
        owen_send_size[i] = owen_offset[i+1]- owen_offset[i];
    }
    // send others sizes to recive sizes
    MPI_Alltoall(owen_send_size,1,MPI_INT,other_size,1,MPI_INT, row_comm);
    // compute transmission offsets
    other_offset[0] = 0;
    for(int i = 1; i < C+1; i++){
        other_offset[i] = other_size [i-1]+ other_offset[i-1];
    }
    numberOfEdges= other_offset[C];

    // allocate recive buffer
    packed_edge * rowtransBuf = (packed_edge *) malloc(other_offset[C]*sizeof(packed_edge));

    // transmit data
    MPI_Alltoallv(input, owen_send_size, owen_offset, packedEdgeMPI, rowtransBuf, other_size, other_offset, packedEdgeMPI, row_comm );

    //not nessecary any more
    free(input);
    input = rowtransBuf;

    delete[] owen_send_size;
    delete[] owen_offset;
    delete[] other_size ;
    delete[] other_offset;

    MPI_Type_free(&packedEdgeMPI);
    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&col_comm);

    std::sort(input,input+numberOfEdges,DistMatrix2d::comparePackedEdgeR);

    vtxtype* row_elm = new vtxtype[row_length];
         row_pointer = new vtxtype[row_length+1];

    #pragma omp parallel for
    for(long i=0; i < row_length; i++){
        row_elm[i]=0;
    }

    #ifdef _OPENMP
    #pragma omp parallel
    {
        long this_thread = omp_get_thread_num(), num_threads = omp_get_num_threads();
        long start = (this_thread  ) * row_length / num_threads;
        long end   = (this_thread+1) * row_length / num_threads;

        packed_edge startEdge = {start+row_start,0};
        long j = std::lower_bound(input, input+numberOfEdges, startEdge, DistMatrix2d::comparePackedEdgeR)-input;


        for(long i=start; i < end && j < numberOfEdges; i++){
            vtxtype last_valid = -1;

            while(j < numberOfEdges && input[j].v0-row_start == i ){
                if(input[j].v0 != input[j].v1){
                    last_valid = input[j].v1;
                    row_elm[i]++;
                    j++;
                    break;
                }
                j++;
            }
            while(j < numberOfEdges && input[j].v0-row_start == i ){
                if(input[j].v0 != input[j].v1 && last_valid != input[j].v1){
                    row_elm[i]++;
                }
                last_valid = input[j].v1;
                j++;
            }
        }

    }
    #else
    for(long i=0, j=0; i < row_length && j < numberOfEdges; i++){
        vtxtype last_valid = -1;

        while(j < numberOfEdges && input[j].v0-row_start == i ){
            if(input[j].v0 != input[j].v1){
                last_valid = input[j].v1;
                row_elm[i]++;
                j++;
                break;
            }
            j++;
        }
        while(j < numberOfEdges && input[j].v0-row_start == i ){
            if(input[j].v0 != input[j].v1 && last_valid != input[j].v1){
                row_elm[i]++;
            }
            last_valid = input[j].v1;
            j++;
        }
    }    
    #endif

    // prefix scan to compute row pointer
    row_pointer[0] = 0;
    for(long i = 0; i < row_length; i++){
        row_pointer[i+1] = row_pointer[i] + row_elm[i];
    }
    delete[] row_elm;
    column_index = new vtxtype[row_pointer[row_length]];

    //build columns
    #ifdef _OPENMP
    #pragma omp parallel
    {
        long this_thread = omp_get_thread_num(), num_threads = omp_get_num_threads();
        long start = (this_thread  ) * row_length / num_threads;
        long end   = (this_thread+1) * row_length / num_threads;


        packed_edge startEdge = {start+row_start,0};
        long j = std::lower_bound(input, input+numberOfEdges, startEdge, DistMatrix2d::comparePackedEdgeR)-input;

        for(long i=start; i < end && j < numberOfEdges; i++){
            vtxtype last_valid = -1;
            long inrow = 0;

            while(j < numberOfEdges && input[j].v0-row_start == i ){
                if(input[j].v0 != input[j].v1){
                    last_valid = input[j].v1;
                    column_index[row_pointer[i]+inrow] = input[j].v1;
                    inrow++;
                    j++;
                    break;
                }
                j++;
            }
            while(j < numberOfEdges && input[j].v0-row_start == i ){
                if(input[j].v0 != input[j].v1 && last_valid != input[j].v1){
                    column_index[row_pointer[i]+inrow] = input[j].v1;
                    inrow++;
                }
                last_valid = input[j].v1;
                j++;
            }
        }
    }
    #else
    for(long i=0, j=0; i < row_length && j < numberOfEdges; i++){
        vtxtype last_valid = -1;
        long inrow = 0;

        while(j < numberOfEdges && input[j].v0-row_start == i ){
            if(input[j].v0 != input[j].v1){
                last_valid = input[j].v1;
                column_index[row_pointer[i]+inrow] = input[j].v1;
                inrow++;
                j++;
                break;
            }
            j++;
        }
        while(j < numberOfEdges && input[j].v0-row_start == i ){
            if(input[j].v0 != input[j].v1 && last_valid != input[j].v1){
                column_index[row_pointer[i]+inrow] = input[j].v1;
                inrow++;
            }
            last_valid = input[j].v1;
            j++;
        }
    }
    #endif

}


std::vector<DistMatrix2d::fold_prop> DistMatrix2d::getFoldProperties() const
{
    // compute the properties of global folding
    struct fold_prop newprop;
    std::vector<struct fold_prop> fold_fq_props;
    long num8 = globalNumberOfVertex/8 + ((globalNumberOfVertex%8>0)? 1:0) ;

    // first fractions of first fq
    vtxtype ua_col_size= num8 /C; // non adjusted coumn size
    vtxtype col_size_res = num8 % C;

    vtxtype a_quot = row_start / ((ua_col_size+1)*8);

    if(a_quot >=  col_size_res){
        newprop.sendColSl = (row_start/8 - col_size_res) / ua_col_size;
        newprop.startvtx = row_start;
        newprop.size = (ua_col_size-((row_start/8 - col_size_res) % ua_col_size ))*8;
        newprop.size = (newprop.size < row_length)? newprop.size :  row_length;
        col_size_res = 0;
        //column end
        vtxtype colnextstart = ((col_size_res > 0)? (newprop.sendColSl+1)*(ua_col_size+1):
        (newprop.sendColSl+1)*ua_col_size+num8 % C)*8;
        newprop.size = (newprop.startvtx + newprop.size <= colnextstart)? newprop.size :  colnextstart-newprop.startvtx;
    } else {
        newprop.sendColSl = a_quot;
        newprop.startvtx = row_start;
        newprop.size = (ua_col_size +1)*8;
        newprop.size = (newprop.size < row_length)? newprop.size :  row_length;
        col_size_res -= a_quot +1;
        //column end
        vtxtype colnextstart = ((col_size_res > 0)? (newprop.sendColSl+1)*(ua_col_size+1):
        (newprop.sendColSl+1)*ua_col_size+num8 % C)*8;
        newprop.size = (newprop.startvtx + newprop.size <= colnextstart)? newprop.size :  colnextstart-newprop.startvtx;

    }
    fold_fq_props.push_back(newprop);
    // other
    const vtxtype row_end = row_start + row_length;
    while(newprop.startvtx+newprop.size<row_end){
        newprop.sendColSl++;
        newprop.startvtx+=newprop.size;
        newprop.size = ((col_size_res > 0)? ua_col_size+1 : ua_col_size)*8;
        col_size_res -= (col_size_res > 0)? 1 : 0;
        newprop.size = (newprop.startvtx+newprop.size < row_end )? newprop.size: row_end-newprop.startvtx;
        //column end
        vtxtype colnextstart = ((col_size_res > 0)? (newprop.sendColSl+1)*(ua_col_size+1):
        (newprop.sendColSl+1)*ua_col_size+num8 % C)*8;
        newprop.size = (newprop.startvtx + newprop.size <= colnextstart)? newprop.size :  colnextstart-newprop.startvtx;
        fold_fq_props.push_back(newprop);
    }

    return fold_fq_props;
}
/*
 * For Validator
 * Computes the column and the local pointer of the verteces in an array
*/
void DistMatrix2d::get_vertex_distribution_for_pred(size_t count, const int64_t *vertex_p, int *owner_p, size_t *local_p) const
{
  long num8 = globalNumberOfVertex/8 + ((globalNumberOfVertex%8>0)? 1:0 );
  long c_residuum = num8 % C;
  long c_SliceSize = num8 / C;

  //#pragma omp parallel for
    for (long i = 0; i < (ptrdiff_t)count; ++i) {
        if( vertex_p[i]/((c_SliceSize+1)*8) >= c_residuum){
            owner_p[i] = (vertex_p[i]-c_residuum*8)/(c_SliceSize*8);
            local_p[i] = (vertex_p[i]-c_residuum*8)%(c_SliceSize*8);
        } else {
            owner_p[i] = vertex_p[i]/((c_SliceSize+1)*8);
            local_p[i] = vertex_p[i]%((c_SliceSize+1)*8);
        }
    }
}
