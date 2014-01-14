#include "distmatrix2d.h"

#include <algorithm>
#include <iostream>

/*
 *Computes the owener node of a specific edge
*/
long DistMatrix2d::computeOwner(unsigned long row, unsigned long column)
{
    int rowSlice, columnSlice;
    int r_residuum, c_residuum;
    int rSliceSize, cSliceSize;

    r_residuum = globalNumberOfVertex % R;
    rSliceSize = globalNumberOfVertex / R;

    if((rowSlice = row/(rSliceSize+1)) >= r_residuum ){//compute row slice, if the slice number is in the bigger intervals
        rowSlice =  (row- r_residuum)/ rSliceSize; //compute row slice, if the slice number is in the smaler intervals
    }

    c_residuum = globalNumberOfVertex % C;
    cSliceSize = globalNumberOfVertex / C;

    if((columnSlice = column/(cSliceSize+1)) >= c_residuum ){ //compute column slice, if the slice number is in the bigger intervals
        columnSlice =  (column- c_residuum)/ cSliceSize; //compute column slice, if the slice number is in the smaler interval
    }
    return rowSlice*C+columnSlice;
}

DistMatrix2d::DistMatrix2d(int _R, int _C, unsigned long scale):R(_R),C(_C),row_pointer(NULL),column_index(NULL)
{
    int rank;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    //Compute owen row and column id
    r = rank/C;
    c = rank%C;

    globalNumberOfVertex   = 1 << scale;

    row_start       = r*(globalNumberOfVertex/R) + ((r < globalNumberOfVertex%R)? r : globalNumberOfVertex%R);
    row_length      = globalNumberOfVertex/R + ((r < globalNumberOfVertex%R)? 1 : 0);
    column_start    = c*(globalNumberOfVertex/C) + ((c < globalNumberOfVertex%C)? c : globalNumberOfVertex%C);
    column_length   = globalNumberOfVertex/C + ((c < globalNumberOfVertex%C)? 1 : 0);
}

DistMatrix2d::~DistMatrix2d()
{
   if(row_pointer!=NULL) delete[] row_pointer;
   row_pointer=NULL;
   if(column_index!=NULL) delete[] column_index;
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
void DistMatrix2d::setupMatrix(packed_edge *input, int numberOfEdges, bool undirected)
{

    row_pointer = new vtxtype[row_length+1];
    vtxtype* row_elem = new vtxtype[row_length];
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
        for(int i=0; i<outstanding_sends; i++){
            MPI_Test(&(iqueue[i]),&flag,&status);
            if(flag){
                freeRqBuf = i;
                break;
            }
        }
        
        //Send an element
        if(freeRqBuf>-1){
            if(input[count_elementssend].v0 != input[count_elementssend].v1){
                long dest = computeOwner(input[count_elementssend].v0, input[count_elementssend].v1);
                MPI_Issend(&(input[count_elementssend].v0), 1, MPI_LONG, dest, 0, MPI_COMM_WORLD, &(iqueue[freeRqBuf]));
            }
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
            for(int i=0; i<outstanding_sends; i++){
                MPI_Test(&(iqueue[i]),&flag,&status);
                if(flag){
                    freeRqBuf = i;
                    break;
                }
            }

            //Send an element
            if(freeRqBuf>-1){
                if(input[count_elementssend].v0 != input[count_elementssend].v1){
                    long dest = computeOwner(input[count_elementssend].v1, input[count_elementssend].v0);
                    MPI_Issend(&(input[count_elementssend].v1), 1, MPI_LONG, dest, 0, MPI_COMM_WORLD, &(iqueue[freeRqBuf]));
                }
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
            send_buf[0]=input[count_elementssend].v0;
            send_buf[1]=input[count_elementssend].v1;
            if(send_buf[0]!=send_buf[1]){
                long dest = computeOwner(input[count_elementssend].v0, input[count_elementssend].v1);

                MPI_Issend(send_buf, 2, MPI_LONG, dest, 0, MPI_COMM_WORLD, &(iqueue[freeRqBuf]));
            }
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
            if(send_buf[0]!=send_buf[1]){
                long dest = computeOwner(input[count_elementssend].v1, input[count_elementssend].v0);
                MPI_Issend(send_buf, 2, MPI_LONG, dest, 0, MPI_COMM_WORLD, &(iqueue[freeRqBuf]));
            }
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
    delete[] row_elem;

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

    delete[] row_pointer;
    delete[] column_index;
    row_pointer = tmp_row_pointer;
    column_index = tmp_column_index;
 }

std::vector<DistMatrix2d::fold_prop> DistMatrix2d::getFoldProperties()
{
    // compute the properties of global folding
    struct fold_prop newprop;
    std::vector<struct fold_prop> fold_fq_props;

    // first fractions of first fq
    vtxtype ua_col_size= globalNumberOfVertex /C; // non adjusted coumn size
    vtxtype col_size_res = globalNumberOfVertex % C;

    vtxtype a_quot = row_start / (ua_col_size+1);

    if(a_quot >=  col_size_res){
        newprop.sendColSl = (row_start - col_size_res) / ua_col_size;
        newprop.startvtx = row_start;
        newprop.size = ua_col_size-((row_start - col_size_res) % ua_col_size );
        newprop.size = (newprop.size < row_length)? newprop.size :  row_length;
        col_size_res = 0;
    } else {
        newprop.sendColSl = a_quot;
        newprop.startvtx = row_start;
        newprop.size = ua_col_size +1 - a_quot ;
        newprop.size = (newprop.size < row_length)? newprop.size :  row_length;
        col_size_res -= a_quot +1;
    }
    fold_fq_props.push_back(newprop);
    // other
    const vtxtype row_end = row_start + row_length;
    while(newprop.startvtx+newprop.size<row_end){
        newprop.sendColSl++;
        newprop.startvtx+=newprop.size;
        newprop.size = (col_size_res > 0)? ua_col_size+1 : ua_col_size;
        col_size_res -= (col_size_res > 0)? 1 : 0;
        newprop.size = (newprop.startvtx+newprop.size < row_end )? newprop.size: row_end-newprop.startvtx;
        fold_fq_props.push_back(newprop);
    }
    return fold_fq_props;
}
