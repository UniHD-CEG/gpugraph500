#include "generator/graph_generator.h"
#include <mpi.h>
#include <vector>
#include <algorithm>

#include <cstdlib> // only for values of packed_edge typ!!!
#ifdef _OPENMP
    #include <omp.h>
#endif

#ifndef DISTMATRIX2D_HH
#define DISTMATRIX2D_HH
/*
*   This class is a representation of a distributed 2d partitioned adjacency Matrix
*   in CRS style. The edges are not weighted, so there is no value, because an row
*   and column index indicate the edge. It uses MPI.
*
*/

// WOLO: without local offset; ALG: vertex alligment; PAD: pad the row slices, so that they are equal
template<class vertextyp, class rowoffsettyp, bool WOLO=false, int ALG=1, bool PAD=false>
class DistMatrix2d
{
public:
    typedef vertextyp    vtxtyp;
    typedef rowoffsettyp rowtyp;
private:

    int64_t R; //Row slices
    int64_t C; //Column slices
    int64_t r; //Row id of this node
    int64_t c; //Column id of this node

    vtxtyp globalNumberOfVertex;

    vtxtyp row_start, row_length; // global index of first row; size of local row slice
    vtxtyp column_start, column_length; // global index of first (potential) column; size of local column slice
    rowtyp* row_pointer;  //Row pointer to columns
    vtxtyp* column_index; //Column index

    /*
     *  Computes the owner node of a row/column pair.
     */
    int computeOwner(int64_t row, int64_t column);

    static bool comparePackedEdgeR(packed_edge i, packed_edge j);
    static bool comparePackedEdgeC(packed_edge i, packed_edge j);
public:
    struct fold_prop{
        int64_t     sendColSl;
        vtxtyp startvtx;
        long size;
    };

    DistMatrix2d(int64_t _R, int64_t _C);
    ~DistMatrix2d();

    void setupMatrix(packed_edge* input, int64_t numberOfEdges,bool undirected= true);
    void setupMatrix2(packed_edge* &input, int64_t &numberOfEdges, bool undirected= true);
    inline rowtyp getEdgeCount()const{return (row_pointer!=0)?row_pointer[row_length]:0;}
    std::vector<struct fold_prop> getFoldProperties() const;

    inline const rowtyp* getRowPointer() const {return row_pointer;}
    inline const vtxtyp* getColumnIndex() const { return column_index;}

    //inline bool isSym(){return C==R;}

    inline int64_t getNumRowSl() const{return R;}
    inline int64_t getNumColumnSl()const{return C;}

    inline int64_t getLocalRowID() const{return r;}
    inline int64_t getLocalColumnID() const{return c;}

    inline bool isLocalRow(vtxtyp vtx)const {return (row_start<=vtx) && (vtx < row_start+row_length);}
    inline bool isLocalColumn(vtxtyp vtx)const { return (column_start<=vtx) && (vtx < column_start+column_length);}


    inline vtxtyp globaltolocalRow(vtxtyp vtx)const {return vtx-row_start;}
    inline vtxtyp globaltolocalCol(vtxtyp vtx)const {return vtx-column_start;}
    inline vtxtyp localtoglobalRow(vtxtyp vtx)const {return vtx+row_start;}
    inline vtxtyp localtoglobalCol(vtxtyp vtx)const {return vtx+column_start;}

    inline vtxtyp getLocRowLength()const {return row_length;}
    inline vtxtyp getLocColLength()const {return column_length;}

    //For Validator
    void get_vertex_distribution_for_pred(size_t count, const vtxtyp* vertex_p, int* owner_p, size_t* local_p) const;

};


/*
 *Computes the owener node of a specific edge
*/
template<class vertextyp, class rowoffsettyp, bool WOLO, int ALG, bool PAD>
int DistMatrix2d<vertextyp,rowoffsettyp,WOLO,ALG,PAD>::computeOwner(int64_t row, int64_t column)
{
    int64_t rowSlice, columnSlice;
    int64_t r_residuum, c_residuum;
    int64_t rSliceSize, cSliceSize;

    int64_t numAlg = globalNumberOfVertex/ALG + ((globalNumberOfVertex%ALG>0)? 1:0) ;

    r_residuum = numAlg % R;
    rSliceSize = numAlg / R;

    if((rowSlice = row/(rSliceSize+1)) >= r_residuum ){//compute row slice, if the slice number is in the bigger intervals
        rowSlice =  (row- r_residuum)/ rSliceSize; //compute row slice, if the slice number is in the smaler intervals
    }

    c_residuum = numAlg % C;
    cSliceSize = numAlg / C;

    if((columnSlice = column/(cSliceSize+1)) >= c_residuum ){ //compute column slice, if the slice number is in the bigger intervals
        columnSlice =  (column- c_residuum)/ cSliceSize; //compute column slice, if the slice number is in the smaler interval
    }
    return rowSlice*C+columnSlice;
}
template<class vertextyp, class rowoffsettyp, bool WOLO, int ALG, bool PAD>
bool DistMatrix2d<vertextyp,rowoffsettyp,WOLO,ALG,PAD>::comparePackedEdgeR(packed_edge i, packed_edge j){
    if(i.v0<j.v0){
        return true;
    }else if(i.v0>j.v0 ){
        return false;
    } else
        return (i.v1<j.v1);
}
template<class vertextyp, class rowoffsettyp, bool WOLO, int ALG, bool PAD>
bool DistMatrix2d<vertextyp,rowoffsettyp,WOLO,ALG,PAD>::comparePackedEdgeC(packed_edge i, packed_edge j){
    if(i.v1<j.v1){
        return true;
    }else if(i.v1>j.v1 ){
        return false;
    } else
        return (i.v0<j.v0);
}

template<class vertextyp, class rowoffsettyp, bool WOLO, int ALG, bool PAD>
DistMatrix2d<vertextyp,rowoffsettyp,WOLO,ALG,PAD>::DistMatrix2d(int64_t _R, int64_t _C):R(_R),C(_C),row_pointer(NULL),column_index(NULL)
{
    int rank;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    //Compute owen row and column id
    r = rank/C;
    c = rank%C;
}

template<class vertextyp, class rowoffsettyp,bool WOLO, int ALG, bool PAD>
DistMatrix2d<vertextyp,rowoffsettyp,WOLO,ALG,PAD>::~DistMatrix2d()
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
template<class vertextyp, class rowoffsettyp, bool WOLO, int ALG, bool PAD>
void DistMatrix2d<vertextyp,rowoffsettyp,WOLO,ALG,PAD>::setupMatrix(packed_edge *input, int64_t numberOfEdges, bool undirected)
{
    //get max vtx
    vtxtyp maxVertex = -1;

    #ifdef OPENMP
    #pragma omp parallel for reduction(max: maxVertex)
    #endif
    for(vtxtyp i = 0; i < numberOfEdges; i++){
        packed_edge read = input[i];

        maxVertex = (maxVertex > read.v0)? maxVertex :  read.v0;
        maxVertex = (maxVertex > read.v1)? maxVertex :  read.v1;
    }

    MPI_Allreduce(&maxVertex, &globalNumberOfVertex, 1, MPI_LONG, MPI_MAX, MPI_COMM_WORLD);
    //because start at 0
    globalNumberOfVertex += 1;
    //row slice padding
    if(PAD){
        globalNumberOfVertex +=
                (globalNumberOfVertex%R>0)?R-globalNumberOfVertex%R : 0;
    }
    int64_t numAlg = globalNumberOfVertex/ALG + ((globalNumberOfVertex%ALG>0)? 1:0) ;
    row_start       = (r*(numAlg/R) + ((r < numAlg%R)? r : numAlg%R))*ALG;
    row_length      = (numAlg/R + ((r < numAlg%R)? 1 : 0))*ALG;
    column_start    = (c*(numAlg/C) + ((c < numAlg%C)? c : numAlg%C))*ALG;
    column_length   = (numAlg/C + ((c < numAlg%C)? 1 : 0))*ALG;

    row_pointer = new rowtyp[row_length+1];
    rowtyp* row_elem = new rowtyp[row_length];
    const int outstanding_sends=40;
    MPI_Request iqueue[outstanding_sends];

    // row_elem is in the first part a counter of nz columns per row
    for(vtxtyp i=0; i< row_length; i++){
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
                int64_t dest = computeOwner(input[count_elementssend].v0, input[count_elementssend].v1);
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

            int64_t buf;
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
                    int64_t dest = computeOwner(input[count_elementssend].v1, input[count_elementssend].v0);
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

                int64_t buf;
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

        int64_t buf;
        MPI_Recv(&buf, 1,  MPI_LONG, MPI_ANY_SOURCE, 0,
                   MPI_COMM_WORLD, &status);
        row_elem[buf-row_start]++;

        }
    };

//2. Step
    //compute row index array(prefix scan)
    row_pointer[0] = 0;
    for(vtxtyp i=1; i< row_length+1; i++){
        row_pointer[i] = row_pointer[i-1]+row_elem[i-1];
    }
    // row_elem is now a pointer to the relativ position, where new elements of a row may be insert
    for(vtxtyp i=0; i< row_length; i++){
        row_elem[i] = 0;
    }
    column_index = new vtxtyp[row_pointer[row_length]];
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
            int64_t send_buf[2];
            send_buf[0]=input[count_elementssend].v0;
            send_buf[1]=input[count_elementssend].v1;
            //if(send_buf[0]!=send_buf[1]){
                int64_t dest = computeOwner(input[count_elementssend].v0, input[count_elementssend].v1);

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

            int64_t buf[2];
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
            int64_t send_buf[2];
            send_buf[0]=input[count_elementssend].v1;
            send_buf[1]=input[count_elementssend].v0;
            //if(send_buf[0]!=send_buf[1]){
                int64_t dest = computeOwner(input[count_elementssend].v1, input[count_elementssend].v0);
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

            int64_t buf[2];
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

        int64_t buf[2];
        MPI_Recv(buf, 2,  MPI_LONG, MPI_ANY_SOURCE, 0,
                 MPI_COMM_WORLD, &status);
        column_index[row_pointer[buf[0]-row_start]+row_elem[buf[0]-row_start]] = buf[1];
        row_elem[buf[0]-row_start]++;

        }
    }


    //sanity check
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for(vtxtyp i=0; i< row_length; i++){
        if(row_elem[i]!=row_pointer[i+1]-row_pointer[i]){
            fprintf(stderr,"Number of nz-column mismatch in row %ld on node (%d:%d). Annonced: %ld Recived: %ld\n",
                    (row_start+i),r,c,row_elem[i],row_pointer[i+1]-row_pointer[i]);
        }
    }
//4.
    //sort edge list
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for(vtxtyp i=0; i< row_length; i++){
        std::sort(column_index+row_pointer[i],column_index+row_pointer[i+1]);
    }
//5.
    // remove duplicates
    // The next section is very bad, because it use too much memory.
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for(vtxtyp i=0; i< row_length; i++){
       rowtyp tmp_row_num = row_elem[i];
        //Search for duplicates in every row
        for(rowtyp j = row_pointer[i]+1; j < row_pointer[i+1]; j++){
            if(column_index[j-1]==column_index[j])
                tmp_row_num--;
        }
        row_elem[i] = tmp_row_num;
    }

    rowtyp* tmp_row_pointer = new rowtyp[row_length+1];
    vtxtyp* tmp_column_index;

    tmp_row_pointer[0] = 0;
    for(vtxtyp i=1; i< row_length+1; i++){
        tmp_row_pointer[i] = tmp_row_pointer[i-1]+row_elem[i-1];
    }

    tmp_column_index = new vtxtyp[tmp_row_pointer[row_length]];
    //Copy unique entries in every row
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for(vtxtyp i=0; i< row_length; i++){
       rowtyp next_elem = tmp_row_pointer[i];
       //skip empty row
       if(next_elem == tmp_row_pointer[i+1]) continue;
       if(WOLO){
            tmp_column_index[next_elem] = column_index[row_pointer[i]]-column_start;
       }else{
            tmp_column_index[next_elem] = column_index[row_pointer[i]];
       }
       next_elem++;
        for(rowtyp j = row_pointer[i]+1; j < row_pointer[i+1]; j++){
            if(column_index[j-1]!=column_index[j]){
                if(WOLO){
                    tmp_column_index[next_elem] = column_index[j]-column_start;
                }
                else {
                    tmp_column_index[next_elem] = column_index[j];
                }
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

template<class vertextyp, class rowoffsettyp, bool WOLO, int ALG, bool PAD>
void DistMatrix2d<vertextyp,rowoffsettyp,WOLO,ALG,PAD>::setupMatrix2(packed_edge *&input, int64_t &numberOfEdges, bool undirected){

    vtxtyp maxVertex = -1;
    if(undirected == true){
        //generate other direction to be undirected
        input=(packed_edge *)realloc(input, 2*numberOfEdges*sizeof(packed_edge));

        #ifdef _OPENMP
        #pragma omp parallel for reduction(max: maxVertex)
        #endif
        for(int64_t i = 0; i < numberOfEdges; i++){
            packed_edge read = input[i];

            input[numberOfEdges+i].v0 = read.v1;
            input[numberOfEdges+i].v1 = read.v0;

            maxVertex  = (maxVertex > read.v0)? maxVertex :  read.v0;
            maxVertex  = (maxVertex > read.v1)? maxVertex :  read.v1;
        }

        numberOfEdges= 2*numberOfEdges;
    } else {
        #ifdef _OPENMP
        #pragma omp parallel for reduction(max: maxVertex)
        #endif
        for(int64_t i = 0; i < numberOfEdges; i++){
            packed_edge read = input[i];

            maxVertex  = (maxVertex > read.v0)? maxVertex:  read.v0;
            maxVertex  = (maxVertex > read.v1)? maxVertex :  read.v1;
        }
    }
    MPI_Allreduce(&maxVertex, &globalNumberOfVertex, 1, MPI_LONG, MPI_MAX, MPI_COMM_WORLD);
    //because start at 0
    globalNumberOfVertex += 1;
    //row slice padding
    if(PAD){
        globalNumberOfVertex +=
                (globalNumberOfVertex%R>0)?R-globalNumberOfVertex%R : 0;
    }
    int64_t numAlg = globalNumberOfVertex/ALG + ((globalNumberOfVertex%ALG>0)? 1:0) ;
    row_start       = (r*(numAlg/R) + ((r < numAlg%R)? r : numAlg%R))*ALG;
    row_length      = (numAlg/R + ((r < numAlg%R)? 1 : 0))*ALG;
    column_start    = (c*(numAlg/C) + ((c < numAlg%C)? c : numAlg%C))*ALG;
    column_length   = (numAlg/C + ((c < numAlg%C)? 1 : 0))*ALG;

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
    MPI_Type_create_struct(1,tupsize,tupoffset,tuptype,&packedEdgeMPI);
    MPI_Type_commit(&packedEdgeMPI);

    //column comunication
    std::sort(input,input+numberOfEdges,DistMatrix2d::comparePackedEdgeR);

    int* owen_send_size = new int[R];
    int* owen_offset = new int[R+1];
    int* other_size = new int[R];
    int* other_offset =  new int[R+1];

    int64_t res = numAlg % R;
    int64_t ua_sl_size = numAlg / R;

    // offset for transmission
    int64_t sl_start = 0;
    owen_offset[0] = 0;
    for(int64_t i = 1; i < R; i++){
        if(res > 0) {
            sl_start += (ua_sl_size +1)*ALG;
            res--;
        }else{
            sl_start += ua_sl_size*ALG;
        }
        packed_edge startEdge = {sl_start,0};
        owen_offset[i] = std::lower_bound(input+owen_offset[i-1],input+numberOfEdges, startEdge, DistMatrix2d::comparePackedEdgeR) - input;
    }
    owen_offset[R] = numberOfEdges;

    // compute transmission sizes
    for(int64_t i = 0; i < R; i++){
        owen_send_size[i] = owen_offset[i+1]- owen_offset[i];
    }
    // send others sizes to recive sizes
    MPI_Alltoall(owen_send_size,1,MPI_INT,other_size,1,MPI_INT, col_comm);
    // compute transmission offsets
    other_offset[0] = 0;
    for(int64_t i = 1; i < R+1; i++){
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

    res = numAlg % C;
    ua_sl_size = numAlg / C;

    // offset for transmission
    sl_start = 0;
    owen_offset[0] = 0;
    for(int64_t i = 1; i < C; i++){
        if(res > 0) {
            sl_start += (ua_sl_size +1)*ALG;
            res--;
        }else{
            sl_start += ua_sl_size*ALG;
        }

        packed_edge startEdge = {0,sl_start};
        owen_offset[i] = std::lower_bound(input+owen_offset[i-1],input+numberOfEdges, startEdge , DistMatrix2d::comparePackedEdgeC) - input;
    }
    owen_offset[C] = numberOfEdges;

    // compute transmission sizes
    for(int64_t i = 0; i < C; i++){
        owen_send_size[i] = owen_offset[i+1]- owen_offset[i];
    }
    // send others sizes to recive sizes
    MPI_Alltoall(owen_send_size,1,MPI_INT,other_size,1,MPI_INT, row_comm);
    // compute transmission offsets
    other_offset[0] = 0;
    for(int64_t i = 1; i < C+1; i++){
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

    rowtyp* row_elm = new rowtyp[row_length];
         row_pointer = new rowtyp[row_length+1];

    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for(int64_t i=0; i < row_length; i++){
        row_elm[i]=0;
    }

    #ifdef _OPENMP
    #pragma omp parallel
    {
        int this_thread = omp_get_thread_num(), num_threads = omp_get_num_threads();
        vtxtyp start = (this_thread  ) * row_length / num_threads;
        vtxtyp end   = (this_thread+1) * row_length / num_threads;

        packed_edge startEdge = {start+row_start,0};
        vtxtyp j = std::lower_bound(input, input+numberOfEdges, startEdge, DistMatrix2d::comparePackedEdgeR)-input;


        for(vtxtyp i=start; i < end && j < numberOfEdges; i++){
            vtxtyp last_valid = -1;

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
    for(vtxtyp i=0, j=0; i < row_length && j < numberOfEdges; i++){
        vtxtyp last_valid = -1;

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
    for(vtxtyp i = 0; i < row_length; i++){
        row_pointer[i+1] = row_pointer[i] + row_elm[i];
    }
    delete[] row_elm;
    column_index = new vtxtyp[row_pointer[row_length]];

    //build columns
    #ifdef _OPENMP
    #pragma omp parallel
    {
        int this_thread = omp_get_thread_num(), num_threads = omp_get_num_threads();
        vtxtyp start = (this_thread  ) * row_length / num_threads;
        vtxtyp end   = (this_thread+1) * row_length / num_threads;


        packed_edge startEdge = {start+row_start,0};
        vtxtyp j = std::lower_bound(input, input+numberOfEdges, startEdge, DistMatrix2d::comparePackedEdgeR)-input;

        for(vtxtyp i=start; i < end && j < numberOfEdges; i++){
            vtxtyp last_valid = -1;
            rowtyp inrow = 0;

            while(j < numberOfEdges && input[j].v0-row_start == i ){
                if(input[j].v0 != input[j].v1){
                    last_valid = input[j].v1;
                    if(WOLO){
                        column_index[row_pointer[i]+inrow] = input[j].v1-column_start;
                    } else {
                        column_index[row_pointer[i]+inrow] = input[j].v1;
                    }
                    inrow++;
                    j++;
                    break;
                }
                j++;
            }
            while(j < numberOfEdges && input[j].v0-row_start == i ){
                if(input[j].v0 != input[j].v1 && last_valid != input[j].v1){
                    if(WOLO){
                        column_index[row_pointer[i]+inrow] = input[j].v1-column_start;
                    } else {
                        column_index[row_pointer[i]+inrow] = input[j].v1;
                    }
                    inrow++;
                }
                last_valid = input[j].v1;
                j++;
            }
        }
    }
    #else
    for(vtxtyp i=0, j=0; i < row_length && j < numberOfEdges; i++){
        vtxtyp last_valid = -1;
        rowtyp inrow = 0;

        while(j < numberOfEdges && input[j].v0-row_start == i ){
            if(input[j].v0 != input[j].v1){
                last_valid = input[j].v1;
                if(WOLO){
                    column_index[row_pointer[i]+inrow] = input[j].v1-column_start;
                } else {
                    column_index[row_pointer[i]+inrow] = input[j].v1;
                }
                inrow++;
                j++;
                break;
            }
            j++;
        }
        while(j < numberOfEdges && input[j].v0-row_start == i ){
            if(input[j].v0 != input[j].v1 && last_valid != input[j].v1){
                if(WOLO){
                    column_index[row_pointer[i]+inrow] = input[j].v1-column_start;
                } else {
                    column_index[row_pointer[i]+inrow] = input[j].v1;
                }
                inrow++;
            }
            last_valid = input[j].v1;
            j++;
        }
    }
    #endif

}

template<class vertextyp, class rowoffsettyp, bool WOLO, int ALG, bool PAD>
std::vector<typename DistMatrix2d<vertextyp,rowoffsettyp,WOLO,ALG,PAD>::fold_prop> DistMatrix2d<vertextyp,rowoffsettyp,WOLO,ALG,PAD>::getFoldProperties() const
{
    // compute the properties of global folding
    struct fold_prop newprop;
    std::vector<struct fold_prop> fold_fq_props;
    int64_t numAlg = globalNumberOfVertex/ALG + ((globalNumberOfVertex%ALG>0)? 1:0) ;

    // first fractions of first fq
    vtxtyp ua_col_size= numAlg /C; // non adjusted coumn size
    vtxtyp col_size_res = numAlg % C;

    vtxtyp a_quot = row_start / ((ua_col_size+1)*ALG);

    if(a_quot >=  col_size_res){
        newprop.sendColSl = (row_start/ALG - col_size_res) / ua_col_size;
        newprop.startvtx = row_start;
        newprop.size = (ua_col_size-((row_start/ALG - col_size_res) % ua_col_size ))*ALG;
        newprop.size = (newprop.size < row_length)? newprop.size :  row_length;
        col_size_res = 0;
        //column end
        vtxtyp colnextstart = ((col_size_res > 0)? (newprop.sendColSl+1)*(ua_col_size+1):
        (newprop.sendColSl+1)*ua_col_size+numAlg % C)*ALG;
        newprop.size = (newprop.startvtx + newprop.size <= colnextstart)? newprop.size :  colnextstart-newprop.startvtx;
    } else {
        newprop.sendColSl = a_quot;
        newprop.startvtx = row_start;
        newprop.size = (ua_col_size +1)*ALG;
        newprop.size = (newprop.size < row_length)? newprop.size :  row_length;
        col_size_res -= a_quot +1;
        //column end
        vtxtyp colnextstart = ((col_size_res > 0)? (newprop.sendColSl+1)*(ua_col_size+1):
        (newprop.sendColSl+1)*ua_col_size+numAlg % C)*ALG;
        newprop.size = (newprop.startvtx + newprop.size <= colnextstart)? newprop.size :  colnextstart-newprop.startvtx;

    }
    fold_fq_props.push_back(newprop);
    // other
    const vtxtyp row_end = row_start + row_length;
    while(newprop.startvtx+newprop.size<row_end){
        newprop.sendColSl++;
        newprop.startvtx+=newprop.size;
        newprop.size = ((col_size_res > 0)? ua_col_size+1 : ua_col_size)*ALG;
        col_size_res -= (col_size_res > 0)? 1 : 0;
        newprop.size = (newprop.startvtx+newprop.size < row_end )? newprop.size: row_end-newprop.startvtx;
        //column end
        vtxtyp colnextstart = ((col_size_res > 0)? (newprop.sendColSl+1)*(ua_col_size+1):
        (newprop.sendColSl+1)*ua_col_size+numAlg % C)*ALG;
        newprop.size = (newprop.startvtx + newprop.size <= colnextstart)? newprop.size :  colnextstart-newprop.startvtx;
        fold_fq_props.push_back(newprop);
    }

    return fold_fq_props;
}
/*
 * For Validator
 * Computes the column and the local pointer of the verteces in an array
*/
template<class vertextyp, class rowoffsettyp, bool WOLO, int ALG, bool PAD>
void DistMatrix2d<vertextyp,rowoffsettyp,WOLO,ALG,PAD>::get_vertex_distribution_for_pred(size_t count, const vtxtyp *vertex_p, int *owner_p, size_t *local_p) const
{
  int64_t numAlg = globalNumberOfVertex/ALG + ((globalNumberOfVertex%ALG>0)? 1:0 );
  int64_t c_residuum = numAlg % C;
  int64_t c_SliceSize = numAlg / C;

  //#pragma omp parallel for
    for (int64_t i = 0; i < (ptrdiff_t)count; ++i) {
        if( vertex_p[i]/((c_SliceSize+1)*ALG) >= c_residuum){
            owner_p[i] = (vertex_p[i]-c_residuum*ALG)/(c_SliceSize*ALG);
            local_p[i] = (vertex_p[i]-c_residuum*ALG)%(c_SliceSize*ALG);
        } else {
            owner_p[i] = vertex_p[i]/((c_SliceSize+1)*ALG);
            local_p[i] = vertex_p[i]%((c_SliceSize+1)*ALG);
        }
    }
}

#endif // DISTMATRIX2D_HH
