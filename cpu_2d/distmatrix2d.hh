#ifndef DISTMATRIX2D_HH
#define DISTMATRIX2D_HH

/**
 *   This class is a representation of a distributed 2d partitioned adjacency Matrix
 *   in CRS style. The edges are not weighted, so there is no value, because an row
 *   and column index indicate the edge. It uses MPI.
 *
 */

#include "comp_opt.h"
#include "generator/graph_generator.h"
#include <mpi.h>
#include <vector>
#include <algorithm>
#include <cstddef> // needed by MPICH implem.
#include <cstdlib> // only for values of packed_edge typ!!!
#include "config.h"

#ifdef _OPENMP
#include <omp.h>
#define OMP_CHUNK 4
#endif

#if _OPENMP
#include <parallel/algorithm>
#endif

#ifndef ALIGNMENT
#if HAVE_AVX
#define ALIGNMENT 32UL
#else
#define ALIGNMENT 16UL
#endif
#endif

// WOLO: without local offset; ALG: vertex alligment; PAD: pad the row slices, so that they are equal
template<typename vertextyp, typename rowoffsettyp, bool WOLO = false, int64_t ALG = 1LL, bool PAD = false>
class DistMatrix2d
{
public:
    typedef vertextyp vertexType;
    typedef rowoffsettyp rowtyp;

    const static bool WOLOFF = WOLO;
private:

    int64_t R; //Row slices
    int64_t C; //Column slices
    int64_t r; //Row id of this node
    int64_t c; //Column id of this node

    vertexType globalNumberOfVertex;

    vertexType row_start, row_length; // global index of first row; size of local row slice
    vertexType column_start, column_length; // global index of first (potential) column; size of local column slice
    rowtyp *row_pointer;  //Row pointer to columns
    vertexType *column_index; //Column index

    /*
     *  Computes the owner node of a row/column pair.
     */
    int computeOwner(int64_t row, int64_t column);

    static bool comparePackedEdgeR(packed_edge i, packed_edge j);

    static bool comparePackedEdgeC(packed_edge i, packed_edge j);

public:
    struct fold_prop
    {
        int64_t sendColSl;
        vertexType startvtx;
        vertexType gstartvtx;
        long size;
    };

    DistMatrix2d(int64_t _R, int64_t _C);

    ~DistMatrix2d();

    void setupMatrix(packed_edge *input, int64_t numberOfEdges, bool undirected = true);

    void setupMatrix2(packed_edge *&input, int64_t &numberOfEdges, bool undirected = true);

    inline rowtyp getEdgeCount() const
    {
        return (row_pointer != 0U) ? row_pointer[row_length] : 0U;
    }

    std::vector<struct fold_prop> getFoldProperties() const;

    inline const rowtyp *getRowPointer() const { return row_pointer; }

    inline const vertexType *getColumnIndex() const { return column_index; }

    inline int64_t getNumRowSl() const { return R; }

    inline int64_t getNumColumnSl() const { return C; }

    inline int64_t getLocalRowID() const { return r; }

    inline int64_t getLocalColumnID() const { return c; }

    inline bool isLocalRow(vertexType vtx) const
    {
        return (row_start <= vtx) && (vtx < row_start + row_length);
    }

    inline bool isLocalColumn(vertexType vtx) const
    {
        return (column_start <= vtx) && (vtx < column_start + column_length);
    }

    inline vertexType globaltolocalRow(vertexType vtx) const { return vtx - row_start; }

    inline vertexType globaltolocalCol(vertexType vtx) const { return vtx - column_start; }

    inline vertexType localtoglobalRow(vertexType vtx) const { return vtx + row_start; }

    inline vertexType localtoglobalCol(vertexType vtx) const { return vtx + column_start; }

    inline vertexType getLocRowLength() const { return row_length; }

    inline vertexType getLocColLength() const { return column_length; }

    void getVertexDistributionForPred(size_t count, const vertexType *vertex_p,
                                      int *owner_p,
                                      size_t *local_p) const;

    bool allValuesSmallerThan32Bits() const;
    bool allValuesPositive() const;
};

/**
 *
 *Computes the owner node of a specific edge
 *
 */
template<typename vertextyp, typename rowoffsettyp, bool WOLO, int64_t ALG, bool PAD>
int DistMatrix2d<vertextyp, rowoffsettyp, WOLO, ALG, PAD>::computeOwner(int64_t row, int64_t column)
{
    int64_t rowSlice, columnSlice;
    int64_t r_residuum, c_residuum;
    int64_t rSliceSize, cSliceSize;
    int64_t numAlg = globalNumberOfVertex / ALG + ((globalNumberOfVertex % ALG > 0LL) ? 1LL : 0LL);

    r_residuum = numAlg % R;
    rSliceSize = numAlg / R;
    if ((rowSlice = row / (rSliceSize + 1)) >= r_residuum)
    {
        //compute row slice, if the slice number is in the bigger intervals
        rowSlice = (row - r_residuum) / rSliceSize; //compute row slice, if the slice number is in the smaler intervals
    }

    c_residuum = numAlg % C;
    cSliceSize = numAlg / C;
    if ((columnSlice = column / (cSliceSize + 1LL)) >= c_residuum)
    {
        //compute column slice, if the slice number is in the bigger intervals
        columnSlice = (column - c_residuum) /
                      cSliceSize; //compute column slice, if the slice number is in the smaler interval
    }
    return static_cast<int32_t>(rowSlice * C + columnSlice); // <--- overflow
}

template<typename vertextyp, typename rowoffsettyp, bool WOLO, int64_t ALG, bool PAD>
bool DistMatrix2d<vertextyp, rowoffsettyp, WOLO, ALG, PAD>::comparePackedEdgeR(packed_edge i, packed_edge j)
{

    if (i.v0 < j.v0)
    {
        return true;
    }
    else if (i.v0 > j.v0)
    {
        return false;
    }
    else
    {
        return (i.v1 < j.v1);
    }
}

template<typename vertextyp, typename rowoffsettyp, bool WOLO, int64_t ALG, bool PAD>
bool DistMatrix2d<vertextyp, rowoffsettyp, WOLO, ALG, PAD>::comparePackedEdgeC(packed_edge i, packed_edge j)
{

    if (i.v1 < j.v1)
    {
        return true;
    }
    else if (i.v1 > j.v1)
    {
        return false;
    }
    else
    {
        return (i.v0 < j.v0);
    }
}

template<typename vertextyp, typename rowoffsettyp, bool WOLO, int64_t ALG, bool PAD>
DistMatrix2d<vertextyp, rowoffsettyp, WOLO, ALG, PAD>::DistMatrix2d(int64_t _R, int64_t _C) : R(_R), C(_C),
    row_pointer(NULL),
    column_index(NULL)
{
    int rank;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // Compute own row and column id
    r = rank / C;
    c = rank % C;
}

template<typename vertextyp, typename rowoffsettyp, bool WOLO, int64_t ALG, bool PAD>
DistMatrix2d<vertextyp, rowoffsettyp, WOLO, ALG, PAD>::~DistMatrix2d()
{

    if (row_pointer != NULL)
    {
        delete[] row_pointer;
    }
    row_pointer = NULL;
    if (column_index != NULL)
    {
        delete[] column_index;
    }
    column_index = NULL;
}

/*
 * Setup of 2d partitioned adjacency matrix.
 * 1. Announce elements in each row (ignore selfloops)
 * 2. Compute intermediate row pointer and storage consume for columns
 * 3. Send edge list to owner
 * 4. Sort column indices
 * 5. Remove duplicate column indices
 *
 * Should be optimised.
*/
template<typename vertextyp, typename rowoffsettyp, bool WOLO, int64_t ALG, bool PAD>
void DistMatrix2d<vertextyp, rowoffsettyp, WOLO, ALG, PAD>::setupMatrix(packed_edge *input, int64_t numberOfEdges,
        bool undirected)
{

    int64_t numAlg;
    int64_t dest;
    int64_t buf;
    int64_t buf2[2];
    int64_t send_buf[2];
    vertexType maxVertex = -1LL;
    const int32_t outstanding_sends = 40;
    int32_t count_elementssend, freeRqBuf, flag;
    int32_t gunfinished, someting_unfinshed;
    MPI_Status status;

    /**
     * omp reduction max for omp 3.1+
     * commented out due to dedlocks with high SF
     */

    /*
    #ifdef _DISABLED_OPENMP
        #pragma omp parallel for schedule(guided, OMP_CHUNK) reduction(max:maxVertex)
    #endif
    */
    //get max vtx
    for (vertexType i = 0LL; i < numberOfEdges; ++i)
    {
        const packed_edge read = input[i];
        maxVertex = (maxVertex > read.v0) ? maxVertex : read.v0;
        maxVertex = (maxVertex > read.v1) ? maxVertex : read.v1;
    }

    MPI_Allreduce(&maxVertex, &globalNumberOfVertex, 1, MPI_INT64_T, MPI_MAX, MPI_COMM_WORLD);
    // because start at 0
    globalNumberOfVertex += 1LL;
    // row slice padding
    if (PAD)
    {
        globalNumberOfVertex +=
            (globalNumberOfVertex % R > 0LL) ? R - globalNumberOfVertex % R : 0LL;
    }

    numAlg = globalNumberOfVertex / ALG + ((globalNumberOfVertex % ALG > 0LL) ? 1LL : 0LL);
    row_start = (r * (numAlg / R) + ((r < numAlg % R) ? r : numAlg % R)) * ALG;
    row_length = (numAlg / R + ((r < numAlg % R) ? 1LL : 0LL)) * ALG;
    column_start = (c * (numAlg / C) + ((c < numAlg % C) ? c : numAlg % C)) * ALG;
    column_length = (numAlg / C + ((c < numAlg % C) ? 1LL : 0LL)) * ALG;

    row_pointer = new rowtyp[row_length + 1U];
    rowtyp *row_elem = new rowtyp[row_length];
    MPI_Request iqueue[outstanding_sends];
    const int64_t crow_length = row_length;

    // row_elem is in the first part a counter of nz columns per row
    #ifdef _OPENMP
        #pragma omp parallel
    #endif
    for (vertexType i = 0LL; i < crow_length; ++i)
    {
        row_elem[i] = 0LL;
    }

    // Bad implementation: To many communications
    // reset outgoing send status
    #ifdef _OPENMP
        #pragma omp parallel
    #endif
    for (int32_t i = 0; i < outstanding_sends; ++i)
    {
        iqueue[i] = MPI_REQUEST_NULL;
    }

    count_elementssend = 0;
    while (count_elementssend < numberOfEdges)
    {
        for (int32_t i = 0; i < 10 && count_elementssend < numberOfEdges; ++i)
        {
            // find free send buff;
            freeRqBuf = -1;
            for (int32_t j = 0; j < outstanding_sends; ++j)
            {
                MPI_Test(&(iqueue[(j + i) % outstanding_sends]), &flag, &status);
                if (flag)
                {
                    freeRqBuf = (j + i) % outstanding_sends;
                    break;
                }
            }

            // Send an element
            if (freeRqBuf > -1)
            {
                int64_t dest = computeOwner(input[count_elementssend].v0, input[count_elementssend].v1);
                MPI_Issend(&(input[count_elementssend].v0), 1, MPI_INT64_T, dest, 0, MPI_COMM_WORLD, &(iqueue[freeRqBuf]));
                ++count_elementssend;
            }
            else
            {
                int32_t gunfinished;
                int32_t someting_unfinshed = 1;

                // Tell others that there is something unfinished
                MPI_Allreduce(&someting_unfinshed, &gunfinished, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
            }
        }

        while (true)
        {
            // Test if there is something to receive
            MPI_Iprobe(MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &flag, &status);
            if (!flag)   // There is no package to receive
            {
                break;
            }

            MPI_Recv(&buf, 1, MPI_INT64_T, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
            ++row_elem[buf - row_start];
        }
    }

    if (undirected)
    {
        count_elementssend = 0;
        while (count_elementssend < numberOfEdges)
        {
            for (int32_t i = 0; i < 10 && count_elementssend < numberOfEdges; ++i)
            {
                // find free send buff;
                freeRqBuf = -1;
                for (int32_t j = 0; j < outstanding_sends; ++j)
                {
                    MPI_Test(&(iqueue[(j + i) % outstanding_sends]), &flag, &status);
                    if (flag)
                    {
                        freeRqBuf = (j + i) % outstanding_sends;
                        break;
                    }
                }

                // Send an element
                if (freeRqBuf > -1)
                {
                    int64_t dest = computeOwner(input[count_elementssend].v1, input[count_elementssend].v0);
                    MPI_Issend(&(input[count_elementssend].v1), 1, MPI_INT64_T, dest, 0, MPI_COMM_WORLD,
                               &(iqueue[freeRqBuf]));
                    ++count_elementssend;
                }
                else
                {
                    someting_unfinshed = 1;

                    // Tell others that there is something unfinished
                    MPI_Allreduce(&someting_unfinshed, &gunfinished, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
                }
            }

            while (true)
            {
                // Test if there is something to receive
                MPI_Iprobe(MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &flag, &status);
                if (!flag)   // There is no package to receive
                {
                    break;
                }

                MPI_Recv(&buf, 1, MPI_INT64_T, MPI_ANY_SOURCE, 0,
                         MPI_COMM_WORLD, &status);
                ++row_elem[buf - row_start];
            }
        }
    }

    // All sends are started
    // Receive rest
    while (true)
    {
        //find unfinished sends
        someting_unfinshed = 0;
        for (int i = 0; i < outstanding_sends; ++i)
        {
            MPI_Test(&(iqueue[i]), &flag, &status);
            if (!flag)
            {
                someting_unfinshed = 1;
                break;
            }
        }
        //Ask other if there is something unfinished
        MPI_Allreduce(&someting_unfinshed, &gunfinished, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
        if (gunfinished == 0)
        {
            break;
        }

        while (true)
        {
            //Test if there is something to receive
            MPI_Iprobe(MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &flag, &status);
            if (!flag)
            {
                // There is no package to receive
                break;
            }

            MPI_Recv(&buf, 1, MPI_INT64_T, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
            ++row_elem[buf - row_start];
        }
    };

//2. Compute intermediate row pointer and storage consume for columns
    //compute row index array(prefix scan)
    row_pointer[0] = 0U;
    for (vertexType i = 1LL; i <= row_length; ++i)
    {
        row_pointer[i] = row_pointer[i - 1] + row_elem[i - 1];
    }
    // row_elem is now a pointer to the relative position, where new elements of a row may be insert
    #ifdef _OPENMP
        #pragma omp parallel
    #endif
    for (vertexType i = 0LL; i < row_length; ++i)
    {
        row_elem[i] = 0U;
    }
    column_index = new vertexType[row_pointer[row_length]];

//3. Send edge list to owner
    //reset outgoing send status
    #ifdef _OPENMP
        #pragma omp parallel
    #endif
    for (int i = 0; i < outstanding_sends; ++i)
    {
        iqueue[i] = MPI_REQUEST_NULL;
    }

    count_elementssend = 0;
    while (count_elementssend < numberOfEdges)
    {
        for (int i = 0; i < 10 && count_elementssend < numberOfEdges; ++i)
        {
            // find free send buff;
            freeRqBuf = -1;
            for (int j = 0; j < outstanding_sends; ++j)
            {
                MPI_Test(&(iqueue[(j + i) % outstanding_sends]), &flag, &status);
                if (flag)
                {
                    freeRqBuf = (j + i) % outstanding_sends;
                    break;
                }
            }
            // Send an edge
            if (freeRqBuf > -1)
            {
                send_buf[0] = input[count_elementssend].v0;
                send_buf[1] = input[count_elementssend].v1;
                //if(send_buf[0]!=send_buf[1]){
                dest = computeOwner(input[count_elementssend].v0, input[count_elementssend].v1);

                MPI_Issend(send_buf, 2, MPI_INT64_T, dest, 0, MPI_COMM_WORLD, &(iqueue[freeRqBuf]));
                //}
                ++count_elementssend;
            }
            else
            {
                someting_unfinshed = 1;

                // Tell others that there is something unfinished
                MPI_Allreduce(&someting_unfinshed, &gunfinished, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
            }
        }

        while (true)
        {
            // Test if there is something to receive
            MPI_Iprobe(MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &flag, &status);
            if (!flag)   // There is no package to receive
            {
                break;
            }


            MPI_Recv(buf2, 2, MPI_INT64_T, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
            int rstart = buf2[0] - row_start;
            column_index[row_pointer[rstart] + row_elem[rstart]] = buf2[1];
            ++row_elem[rstart];
        }
    }

    if (undirected)
    {
        count_elementssend = 0;
        while (count_elementssend < numberOfEdges)
        {
            for (int i = 0; i < 10 && count_elementssend < numberOfEdges; ++i)
            {
                // find free send buff;
                freeRqBuf = -1;
                for (int i = 0; i < outstanding_sends; ++i)
                {
                    MPI_Test(&(iqueue[i]), &flag, &status);
                    if (flag)
                    {
                        freeRqBuf = i;
                        break;
                    }
                }

                // Send an edge
                if (freeRqBuf > -1)
                {
                    send_buf[0] = input[count_elementssend].v1;
                    send_buf[1] = input[count_elementssend].v0;
                    //if(send_buf[0]!=send_buf[1]){
                    int64_t dest = computeOwner(input[count_elementssend].v1, input[count_elementssend].v0);
                    MPI_Issend(send_buf, 2, MPI_INT64_T, dest, 0, MPI_COMM_WORLD, &(iqueue[freeRqBuf]));
                    //}
                    ++count_elementssend;
                }
                else
                {
                    someting_unfinshed = 1;
                    // Tell others that there is something unfinished
                    MPI_Allreduce(&someting_unfinshed, &gunfinished, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
                }
            }
            while (true)
            {
                // Test if there is something to receive
                MPI_Iprobe(MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &flag, &status);
                if (!flag)   // There is no package to receive
                {
                    break;
                }

                MPI_Recv(buf2, 2, MPI_INT64_T, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
                int rstart = buf2[0] - row_start;
                column_index[row_pointer[rstart] + row_elem[rstart]] = buf2[1];
                ++row_elem[rstart];
            }
        }
    }
    // All sends are started
    // Receive rest
    while (true)
    {
        // find unfinished sends
        int32_t someting_unfinshed = 0;

        for (int32_t i = 0; i < outstanding_sends; ++i)
        {
            MPI_Test(&(iqueue[i]), &flag, &status);
            if (!flag)
            {
                someting_unfinshed = 1;
                break;
            }
        }
        // Ask other if there is something unfinished
        int32_t gunfinished;
        MPI_Allreduce(&someting_unfinshed, &gunfinished, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
        if (gunfinished == 0)
            break;

        while (true)
        {
            // Test if there is something to receive
            MPI_Iprobe(MPI_ANY_SOURCE, 0, MPI_COMM_WORLD,
                       &flag, &status);
            if (!flag)   // There is no package to receive
            {
                break;
            }

            MPI_Recv(buf2, 2, MPI_INT64_T, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
            int32_t rstart = buf2[0] - row_start;
            column_index[row_pointer[rstart] + row_elem[rstart]] = buf2[1];
            ++row_elem[rstart];
        }
    }


    //sanity check

#ifdef _DISABLED_OPENMP
    #pragma omp parallel for schedule(guided, OMP_CHUNK)
#endif

    for (vertexType i = 0LL; i < row_length; ++i)
    {
        if (row_elem[i] != row_pointer[i + 1] - row_pointer[i])
        {
            fprintf(stderr, "Number of nz-column mismatch in row %ld on node (%d:%d). Annonced: %ld Recived: %ld\n",
                    (row_start + i), r, c, row_elem[i], row_pointer[i + 1] - row_pointer[i]);
        }
    }

//4. Sort column indices
    //sort edge list

#ifdef _DISABLED_OPENMP
    #pragma omp parallel for schedule(guided, OMP_CHUNK)
#endif

    for (vertexType i = 0LL; i < row_length; ++i)
    {
#ifdef _OPENMP
        __gnu_parallel::sort(column_index + row_pointer[i], column_index + row_pointer[i + 1]);
#else
        std::sort(column_index + row_pointer[i], column_index + row_pointer[i + 1]);
#endif
    }

//5. Remove duplicate column indices
    // remove duplicates
    // The next section is very bad, because it use too much memory.

#ifdef _DISABLED_OPENMP
    #pragma omp parallel for schedule(guided, OMP_CHUNK)
#endif

    for (vertexType i = 0LL; i < row_length; ++i)
    {
        rowtyp tmp_row_num = row_elem[i];
        //Search for duplicates in every row
        const rowtyp rpointer = row_pointer[i + 1];
        for (rowtyp j = row_pointer[i] + 1; j < rpointer; ++j)
        {
            if (column_index[j - 1] == column_index[j])
            {
                --tmp_row_num;
            }
        }
        row_elem[i] = tmp_row_num;
    }

    rowtyp *tmp_row_pointer = new rowtyp[row_length + 1];
    vertexType *tmp_column_index;

    tmp_row_pointer[0] = 0U;
    for (vertexType i = 1LL; i <= row_length; ++i)
    {
        tmp_row_pointer[i] = tmp_row_pointer[i - 1] + row_elem[i - 1];
    }

    tmp_column_index = new vertexType[tmp_row_pointer[row_length]];

    //Copy unique entries in every row
    /**
     * comment out openmp if getting deadlocks
     * performance increase worthed?
     * race condition on tmp_column_index?
     */
#ifdef _DISABLED_OPENMP
    #pragma omp parallel for schedule(guided, OMP_CHUNK)
#endif

    for (vertexType i = 0LL; i < row_length; ++i)
    {
        rowtyp next_elem = tmp_row_pointer[i];
        //skip empty row
        if (next_elem == tmp_row_pointer[i + 1]) { continue; }
        if (WOLO)
        {
            tmp_column_index[next_elem] = column_index[row_pointer[i]] - column_start;
        }
        else
        {
            tmp_column_index[next_elem] = column_index[row_pointer[i]];
        }
        ++next_elem;
        const int32_t rpointer = row_pointer[i + 1];
        for (rowtyp j = row_pointer[i] + 1; j < rpointer; ++j)
        {
            if (column_index[j - 1] != column_index[j])
            {
                if (WOLO)
                {
                    tmp_column_index[next_elem] = column_index[j] - column_start;
                }
                else
                {
                    tmp_column_index[next_elem] = column_index[j];
                }
                ++next_elem;
            }
        }
    }

    delete[] row_elem;
    delete[] row_pointer;
    delete[] column_index;
    row_pointer = tmp_row_pointer;
    column_index = tmp_column_index;
}

template<typename vertextyp, typename rowoffsettyp, bool WOLO, int64_t ALG, bool PAD>
void DistMatrix2d<vertextyp, rowoffsettyp, WOLO, ALG, PAD>::setupMatrix2(packed_edge *&input,
        int64_t &numberOfEdges,
        bool undirected)
{

    int64_t maxVertex = -1LL;

    if (undirected)
    {
        //generate other direction to be undirected
        input = (packed_edge *) realloc(input, 2 * numberOfEdges * sizeof(packed_edge));


        /**
         * omp reduction max for omp 3.1+
         * performance increasement worthed?
         */

#ifdef _DISABLED_OPENMP
        #pragma omp parallel for schedule(guided, OMP_CHUNK) reduction(max : maxVertex)
#endif

        for (int64_t i = 0LL; i < numberOfEdges; ++i)
        {
            const packed_edge read = input[i];

            input[numberOfEdges + i].v0 = read.v1;
            input[numberOfEdges + i].v1 = read.v0;

            maxVertex = (maxVertex > read.v0) ? maxVertex : read.v0;
            maxVertex = (maxVertex > read.v1) ? maxVertex : read.v1;

        }
        numberOfEdges = 2LL * numberOfEdges;
    }
    else
    {

        /**
         * omp reduction max for omp 3.1+
         * comment out if experimenting deadlocks with high SF
         * performance increasement worthed?
         */
#ifdef _DISABLED_OPENMP
        #pragma omp parallel for schedule(guided, OMP_CHUNK) reduction(max : maxVertex)
#endif

        for (int64_t i = 0LL; i < numberOfEdges; ++i)
        {
            const packed_edge read = input[i];

            maxVertex = (maxVertex > read.v0) ? maxVertex : read.v0;
            maxVertex = (maxVertex > read.v1) ? maxVertex : read.v1;
        }
    }

    MPI_Allreduce(&maxVertex, &globalNumberOfVertex, 1, MPI_INT64_T, MPI_MAX, MPI_COMM_WORLD);
    //because start at 0
    globalNumberOfVertex += 1LL;
    //row slice padding
    if (PAD)
    {
        globalNumberOfVertex +=
            (globalNumberOfVertex % R > 0LL) ? R - globalNumberOfVertex % R : 0LL;
    }
    int64_t numAlg = globalNumberOfVertex / ALG + ((globalNumberOfVertex % ALG > 0LL) ? 1LL : 0LL);
    row_start = (r * (numAlg / R) + ((r < numAlg % R) ? r : numAlg % R)) * ALG;
    row_length = (numAlg / R + ((r < numAlg % R) ? 1LL : 0LL)) * ALG;
    column_start = (c * (numAlg / C) + ((c < numAlg % C) ? c : numAlg % C)) * ALG;
    column_length = (numAlg / C + ((c < numAlg % C) ? 1LL : 0LL)) * ALG;

    // Split communicator into row and column communicator
    MPI_Comm row_comm, col_comm;
    // Split by row, rank by column
    MPI_Comm_split(MPI_COMM_WORLD, r, c, &row_comm);
    // Split by column, rank by row
    MPI_Comm_split(MPI_COMM_WORLD, c, r, &col_comm);

    MPI_Datatype packedEdgeMPI;
    int32_t tupsize[] = {2};
    MPI_Aint tupoffset[] = {0};
    MPI_Datatype tuptype[] = {MPI_INT64_T};
    MPI_Type_create_struct(1, tupsize, tupoffset, tuptype, &packedEdgeMPI);
    MPI_Type_commit(&packedEdgeMPI);

    //column comunication
#ifdef _OPENMP
    __gnu_parallel::sort(input, input + numberOfEdges, DistMatrix2d::comparePackedEdgeR);
#else
    std::sort(input, input + numberOfEdges, DistMatrix2d::comparePackedEdgeR);
#endif

    int32_t *owen_send_size = new int32_t[R];
    int32_t *owen_offset = new int32_t[R + 1U];
    int32_t *other_size = new int32_t[R];
    int32_t *other_offset = new int32_t[R + 1U];

    int64_t res = numAlg % R;
    int64_t ua_sl_size = numAlg / R;

    // offset for transmission
    int64_t sl_start = 0LL;
    owen_offset[0U] = 0;
    for (int64_t i = 1LL; i < R; ++i)
    {
        if (res > 0LL)
        {
            sl_start += (ua_sl_size + 1LL) * ALG;
            --res;
        }
        else
        {
            sl_start += ua_sl_size * ALG;
        }
        packed_edge startEdge = {sl_start, 0};
        owen_offset[i] = std::lower_bound(input + owen_offset[i - 1], input + numberOfEdges, startEdge,
                                          DistMatrix2d::comparePackedEdgeR) - input;
    }
    owen_offset[R] = numberOfEdges;

    // compute transmission sizes
    for (int64_t i = 0LL; i < R; ++i)
    {
        owen_send_size[i] = owen_offset[i + 1U] - owen_offset[i];
    }
    // send others sizes to receive sizes
    MPI_Alltoall(owen_send_size, 1, MPI_INT, other_size, 1, MPI_INT, col_comm);
    // compute transmission offsets
    other_offset[0] = 0;
    for (int64_t i = 1LL; i <= R ; ++i)
    {
        other_offset[i] = other_size[i - 1U] + other_offset[i - 1U];
    }
    numberOfEdges = other_offset[R];

    // allocate receive buffer
    // packed_edge *coltransBuf = (packed_edge *) malloc(numberOfEdges * sizeof(packed_edge));
    packed_edge *coltransBuf = NULL;
    const int err1 = posix_memalign((void **)&coltransBuf, ALIGNMENT, numberOfEdges * sizeof(packed_edge));
    if (err1) {
        throw "Memory error.";
    }

    // transmit data
    MPI_Alltoallv(input, owen_send_size, owen_offset, packedEdgeMPI, coltransBuf, other_size, other_offset,
                  packedEdgeMPI, col_comm);

    //not necessary any more
    free(input);
    input = coltransBuf;

    delete[] owen_send_size;
    delete[] owen_offset;
    delete[] other_size;
    delete[] other_offset;

    //row communication
    //sort
#ifdef _OPENMP
    __gnu_parallel::sort(input, input + numberOfEdges, DistMatrix2d::comparePackedEdgeC);
#else
    std::sort(input, input + numberOfEdges, DistMatrix2d::comparePackedEdgeC);
#endif

    owen_send_size = new int32_t[C];
    owen_offset = new int32_t[C + 1U];
    other_size = new int32_t[C];
    other_offset = new int32_t[C + 1U];

    res = numAlg % C;
    ua_sl_size = numAlg / C;

    // offset for transmission
    sl_start = 0;
    owen_offset[0] = 0;
    for (int64_t i = 1LL; i < C; ++i)
    {
        if (res > 0LL)
        {
            sl_start += (ua_sl_size + 1) * ALG;
            --res;
        }
        else
        {
            sl_start += ua_sl_size * ALG;
        }

        packed_edge startEdge = {0, sl_start};
        owen_offset[i] = std::lower_bound(input + owen_offset[i - 1], input + numberOfEdges, startEdge,
                                          DistMatrix2d::comparePackedEdgeC) - input;
    }
    owen_offset[C] = numberOfEdges;

    // compute transmission sizes
    for (int64_t i = 0LL; i < C; ++i)
    {
        owen_send_size[i] = owen_offset[i + 1] - owen_offset[i];
    }
    // send others sizes to receive sizes
    MPI_Alltoall(owen_send_size, 1, MPI_INT, other_size, 1, MPI_INT, row_comm);
    // compute transmission offsets
    other_offset[0] = 0;
    for (int64_t i = 1LL; i <= C ; ++i)
    {
        other_offset[i] = other_size[i - 1U] + other_offset[i - 1U];
    }
    numberOfEdges = other_offset[C];

    // allocate receive buffer
    // packed_edge *rowtransBuf = (packed_edge *) malloc(other_offset[C] * sizeof(packed_edge));
    packed_edge *rowtransBuf = NULL;
    const int err2 = posix_memalign((void **)&rowtransBuf, ALIGNMENT, other_offset[C] * sizeof(packed_edge));
    if (err2) {
        throw "Memory error.";
    }

    // transmit data
    MPI_Alltoallv(input, owen_send_size, owen_offset, packedEdgeMPI, rowtransBuf, other_size, other_offset,
                  packedEdgeMPI, row_comm);

    //not necessary any more
    free(input);
    input = rowtransBuf;

    delete[] owen_send_size;
    delete[] owen_offset;
    delete[] other_size;
    delete[] other_offset;

    MPI_Type_free(&packedEdgeMPI);
    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&col_comm);

#ifdef _OPENMP
    __gnu_parallel::sort(input, input + numberOfEdges, DistMatrix2d::comparePackedEdgeR);
#else
    std::sort(input, input + numberOfEdges, DistMatrix2d::comparePackedEdgeR);
#endif

    rowtyp *row_elm = new rowtyp[row_length];
    row_pointer = new rowtyp[row_length + 1U];

    /**
     * omp worthed?
     */
    /*
    #ifdef _DISABLED_OPENMP
        #pragma omp parallel for schedule(static, OMP_CHUNK)
    #endif
    */

    /*
        const int64_t t_chunk = 8LL;
        for (int64_t ii = 0LL; ii < row_length; ii+=t_chunk) {
            for (int64_t i = ii; i < ii+t_chunk; ++i)
            {
                row_elm[i - ii] = 0;
            }
        }
    */
    #ifdef _OPENMP
        #pragma omp parallel for
    #endif
    for (int64_t i = 0LL; i < row_length; ++i)
    {
        row_elm[i] = 0U;
    }

    /**
     * The commented out omp region hangs up the app (OMP 3.1)
     */

#ifdef _DISABLED_OPENMP
    /*
        #pragma omp parallel
        {
            int this_thread = omp_get_thread_num(), num_threads = omp_get_num_threads();
            vertexType start = (this_thread) * row_length / num_threads;
            vertexType end   = (this_thread + 1) * row_length / num_threads;
            packed_edge startEdge = {start + row_start, 0};
            vertexType j = std::lower_bound(input, input + numberOfEdges, startEdge, DistMatrix2d::comparePackedEdgeR) - input;

            for (vertexType i = start; i < end && j < numberOfEdges; ++i)
    */
    for (vertexType i = 0LL, j = 0LL; i < row_length && j < numberOfEdges; ++i)
    {
        vertexType last_valid = -1LL;

        while (j < numberOfEdges && input[j].v0 - row_start == i)
        {
            if (input[j].v0 != input[j].v1)
            {
                last_valid = input[j].v1;
                ++row_elm[i];
                ++j;
                break;
            }
            ++j;
        }
        while (j < numberOfEdges && input[j].v0 - row_start == i)
        {
            if (input[j].v0 != input[j].v1 && last_valid != input[j].v1)
            {
                ++row_elm[i];
            }
            last_valid = input[j].v1;
            ++j;
        }
    }
//    }
#endif

#ifndef _DISABLED_OPENMP
    for (vertexType i = 0LL, j = 0LL; i < row_length && j < numberOfEdges; ++i)
    {
        vertexType last_valid = -1LL;

        while (j < numberOfEdges && input[j].v0 - row_start == i)
        {
            if (input[j].v0 != input[j].v1)
            {
                last_valid = input[j].v1;
                ++row_elm[i];
                ++j;
                break;
            }
            ++j;
        }
        while (j < numberOfEdges && input[j].v0 - row_start == i)
        {
            if (input[j].v0 != input[j].v1 && last_valid != input[j].v1)
            {
                ++row_elm[i];
            }
            last_valid = input[j].v1;
            ++j;
        }
    }
#endif

    // prefix scan to compute row pointer
    row_pointer[0] = 0;
    for (vertexType i = 0LL; i < row_length; ++i)
    {
        row_pointer[i + 1U] = row_pointer[i] + row_elm[i];
    }
    delete[] row_elm;
    column_index = new vertexType[row_pointer[row_length]];

    //build columns


    /**
     * This commented out omp region hangs up the app (OMP 3.1)
     */

#ifdef _DISABLED_OPENMP

    /*
       #pragma omp parallel
       {
           const int this_thread = omp_get_thread_num();
           const int num_threads = omp_get_num_threads();
           const vertexType start = (this_thread) * row_length / num_threads;
           const vertexType end   = (this_thread + 1) * row_length / num_threads;
           const packed_edge startEdge = {start + row_start, 0};

           vertexType jj = std::lower_bound(input, input + numberOfEdges, startEdge, DistMatrix2d::comparePackedEdgeR) - input;

            for (vertexType i = start, j = jj; i < end && j < numberOfEdges; ++i)
            {
    */
    for (vertexType i = 0LL, j = 0LL; i < row_length && j < numberOfEdges; ++i)
    {
        vertexType last_valid = -1LL;
        rowtyp inrow = 0U;
        const rowtyp r_pointer = row_pointer[i];

        while (j < numberOfEdges && input[j].v0 - row_start == i)
        {
            if (input[j].v0 != input[j].v1)
            {
                last_valid = input[j].v1;

                if (WOLO)
                {
                    column_index[r_pointer + inrow] = input[j].v1 - column_start;
                }
                else
                {
                    column_index[r_pointer + inrow] = input[j].v1;
                }
                ++inrow;
                ++j;
                break;
            }
            ++j;
        }
        while (j < numberOfEdges && input[j].v0 - row_start == i)
        {
            if (input[j].v0 != input[j].v1 && last_valid != input[j].v1)
            {

                if (WOLO)
                {
                    column_index[r_pointer + inrow] = input[j].v1 - column_start;
                }
                else
                {
                    column_index[r_pointer + inrow] = input[j].v1;
                }
                ++inrow;
            }
            last_valid = input[j].v1;
            ++j;
        }
    }
//    }
#endif


#ifndef _DISABLED_OPENMP
    for (vertexType i = 0LL, j = 0LL; i < row_length && j < numberOfEdges; ++i)
    {
        vertexType last_valid = -1LL;
        rowtyp inrow = 0U;
        const rowtyp r_pointer = row_pointer[i];

        while (j < numberOfEdges && input[j].v0 - row_start == i)
        {
            if (input[j].v0 != input[j].v1)
            {
                last_valid = input[j].v1;
                if (WOLO)
                {
                    column_index[r_pointer + inrow] = input[j].v1 - column_start;
                }
                else
                {
                    column_index[r_pointer + inrow] = input[j].v1;
                }
                ++inrow;
                ++j;
                break;
            }
            ++j;
        }
        while (j < numberOfEdges && input[j].v0 - row_start == i)
        {
            if (input[j].v0 != input[j].v1 && last_valid != input[j].v1)
            {
                if (WOLO)
                {
                    column_index[r_pointer + inrow] = input[j].v1 - column_start;
                }
                else
                {
                    column_index[r_pointer + inrow] = input[j].v1;
                }
                ++inrow;
            }
            last_valid = input[j].v1;
            ++j;
        }
    }
#endif
}

template<typename vertextyp, typename rowoffsettyp, bool WOLO, int64_t ALG, bool PAD>
std::vector <typename DistMatrix2d<vertextyp, rowoffsettyp, WOLO, ALG, PAD>::fold_prop>
DistMatrix2d<vertextyp, rowoffsettyp, WOLO, ALG, PAD>::getFoldProperties() const
{
    // compute the properties of global folding
    struct fold_prop newprop;
    std::vector<struct fold_prop> fold_fq_props;
    int64_t numAlg = globalNumberOfVertex / ALG + ((globalNumberOfVertex % ALG > 0LL) ? 1LL : 0LL);

    // first fractions of first fq
    vertexType ua_col_size = numAlg / C; // non adjusted coumn size
    vertexType col_size_res = numAlg % C;

    vertexType a_quot = row_start / ((ua_col_size + 1LL) * ALG);

    if (a_quot >= col_size_res)
    {
        newprop.sendColSl = (row_start / ALG - col_size_res) / ua_col_size;
        newprop.gstartvtx = row_start;

        if (WOLO)
        {
            newprop.startvtx = globaltolocalRow(newprop.gstartvtx);
        }
        else
        {
            newprop.startvtx = newprop.gstartvtx;
        }

        newprop.size = (ua_col_size - ((row_start / ALG - col_size_res) % ua_col_size)) * ALG;
        newprop.size = (newprop.size < row_length) ? newprop.size : row_length;
        col_size_res = 0LL;
        //column end
        vertexType colnextstart = ((col_size_res > 0) ? (newprop.sendColSl + 1) * (ua_col_size + 1) :
                                   (newprop.sendColSl + 1) * ua_col_size + numAlg % C) * ALG;
        newprop.size = (newprop.gstartvtx + newprop.size <= colnextstart) ? newprop.size : colnextstart -
                       newprop.gstartvtx;

    }
    else
    {
        newprop.sendColSl = a_quot;
        newprop.startvtx = row_start;

        if (WOLO)
        {
            newprop.startvtx = globaltolocalRow(newprop.gstartvtx);
        }
        else
        {
            newprop.startvtx = newprop.gstartvtx;
        }

        newprop.size = (ua_col_size + 1) * ALG;
        newprop.size = (newprop.size < row_length) ? newprop.size : row_length;
        col_size_res -= a_quot + 1LL;
        //column end
        vertexType colnextstart = ((col_size_res > 0LL) ? (newprop.sendColSl + 1) * (ua_col_size + 1) :
                                   (newprop.sendColSl + 1) * ua_col_size + numAlg % C) * ALG;
        newprop.size = (newprop.gstartvtx + newprop.size <= colnextstart) ? newprop.size : colnextstart -
                       newprop.gstartvtx;

    }
    fold_fq_props.push_back(newprop);
    // other
    const vertexType row_end = row_start + row_length;
    while (newprop.gstartvtx + newprop.size < row_end)
    {
        ++newprop.sendColSl;
        newprop.gstartvtx += newprop.size;

        if (WOLO)
        {
            newprop.startvtx = globaltolocalRow(newprop.gstartvtx);
        }
        else
        {
            newprop.startvtx = newprop.gstartvtx;
        }

        newprop.size = ((col_size_res > 0LL) ? ua_col_size + 1 : ua_col_size) * ALG;
        col_size_res -= (col_size_res > 0LL) ? 1LL : 0LL;
        newprop.size = (newprop.gstartvtx + newprop.size < row_end) ? newprop.size : row_end - newprop.gstartvtx;
        //column end
        vertexType colnextstart = ((col_size_res > 0LL) ? (newprop.sendColSl + 1) * (ua_col_size + 1) :
                                   (newprop.sendColSl + 1) * ua_col_size + numAlg % C) * ALG;
        newprop.size = (newprop.gstartvtx + newprop.size <= colnextstart) ? newprop.size : colnextstart -
                       newprop.gstartvtx;
        fold_fq_props.push_back(newprop);
    }

    return fold_fq_props;
}

/**
 * For Validator
 * Computes the column and the local pointer of the verteces in an array
 */
template<typename vertextyp, typename rowoffsettyp, bool WOLO, int64_t ALG, bool PAD>
void DistMatrix2d<vertextyp, rowoffsettyp, WOLO, ALG, PAD>::getVertexDistributionForPred(size_t count,
        const vertexType *vertex_p,
        int *owner_p,
        size_t *local_p) const
{
    int64_t numAlg = globalNumberOfVertex / ALG + ((globalNumberOfVertex % ALG > 0LL) ? 1LL : 0LL);
    int64_t c_residuum = numAlg % C;
    int64_t c_SliceSize = numAlg / C;
    ptrdiff_t maxCount = (ptrdiff_t) count;

#ifdef _DISABLED_OPENMP
    #pragma omp parallel for schedule(guided, OMP_CHUNK)
#endif

    for (ptrdiff_t i = 0; i < maxCount; ++i)
    {
        vertexType temporalVertex = vertex_p[i];
        if (temporalVertex / ((c_SliceSize + 1LL) * ALG) >= c_residuum)
        {
            owner_p[i] = (temporalVertex - c_residuum * ALG) / (c_SliceSize * ALG);
            local_p[i] = (temporalVertex - c_residuum * ALG) % (c_SliceSize * ALG);
        }
        else
        {
            owner_p[i] = temporalVertex / ((c_SliceSize + 1LL) * ALG);
            local_p[i] = temporalVertex % ((c_SliceSize + 1LL) * ALG);
        }
    }
}

/**
 * 32bits Check
 * Checks if all values in the Matrix are smaller than 32bits
 */
template<typename vertextyp, typename rowoffsettyp, bool WOLO, int64_t ALG, bool PAD>
bool DistMatrix2d<vertextyp, rowoffsettyp, WOLO, ALG, PAD>::allValuesSmallerThan32Bits() const
{

    const rowtyp *rowp = this->getRowPointer();
    const vertexType *columnp = this->getColumnIndex();
    bool allSmaller = true;
    int64_t val64;
    size_t i = 0, rowLength = this->getLocRowLength();
    rowtyp j, max_j;

    while (allSmaller && i < rowLength)
    {
        j = rowp[i];
        max_j = rowp[i + 1U];
        while (allSmaller && j < max_j)
        {
            val64 = static_cast<int64_t>(columnp[j]);
            if (val64 > 0xFFFFFFFFLL)
            {
                allSmaller = false;
            }
            ++j;
        }
        ++i;
    }

    return allSmaller;
}

/**
 * allPositive Check
 * Checks if all values in the Matrix are possitive
 */
template<typename vertextyp, typename rowoffsettyp, bool WOLO, int64_t ALG, bool PAD>
bool DistMatrix2d<vertextyp, rowoffsettyp, WOLO, ALG, PAD>::allValuesPositive() const
{

    const rowtyp *rowp = this->getRowPointer();
    const vertexType *columnp = this->getColumnIndex();
    bool allPositive = true;
    size_t i = 0, rowLength = this->getLocRowLength();
    rowtyp j, max_j;

    while (allPositive && i < rowLength)
    {
        j = rowp[i];
        max_j = rowp[i + 1U];
        while (allPositive && j < max_j)
        {
	    const vertexType x = columnp[j];
            const bool isPositive = (x>=0);
            if (!isPositive)
            {
                allPositive = false;
            }
            ++j;
        }
        ++i;
    }

    return allPositive;
}

#endif // DISTMATRIX2D_HH
