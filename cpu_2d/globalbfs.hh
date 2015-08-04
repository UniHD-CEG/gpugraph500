#ifndef GLOBALBFS_HH
#define GLOBALBFS_HH


#include "distmatrix2d.hh"
#include "comp_opt.h"
#include "bitlevelfunctions.h"
#include <vector>
#include <cstdio>
#include <assert.h>
#include "vreduce.hpp"
#include <ctgmath>
#include <string.h>
#include <functional>

/*
 * This classs implements a distributed level synchronus BFS on global scale.
 */
template<class Derived,
        class FQ_T,  // Queue Type
        class MType, // Bitmap mask
        class STORE> //Storage of Matrix
class GlobalBFS {
private:
    MPI_Comm row_comm, col_comm;
    // sending node column slice, startvtx, size
    std::vector <typename STORE::fold_prop> fold_fq_props;

    void allReduceBitCompressed(typename STORE::vtxtyp *&owen, typename STORE::vtxtyp *&tmp,
                                MType *&owenmap, MType *&tmpmap);

protected:
    const STORE &store;
    typename STORE::vtxtyp *predecessor;
    MPI_Datatype fq_tp_type; //Frontier Queue Transport Type
    MPI_Datatype bm_type;    // Bitmap Type
    //FQ_T*  __restrict__ recv_fq_buff; - conflicts with void* ref
    FQ_T *recv_fq_buff;
    long recv_fq_buff_length;
    MType *owenmask;
    MType *tmpmask;
    int64_t mask_size;

    // Functions that have to be implemented by the children
    // void reduce_fq_out(FQ_T* startaddr, long insize)=0;    //Global Reducer of the local outgoing frontier queues.  Have to be implemented by the children.
    // void getOutgoingFQ(FQ_T* &startaddr, vtxtype& outsize)=0;
    // void setModOutgoingFQ(FQ_T* startaddr, long insize)=0; //startaddr: 0, self modification
    // void getOutgoingFQ(vtxtype globalstart, vtxtype size, FQ_T* &startaddr, vtxtype& outsize)=0;
    // void setIncommingFQ(vtxtype globalstart, vtxtype size, FQ_T* startaddr, vtxtype& insize_max)=0;
    // bool istheresomethingnew()=0;           //to detect if finished
    // void setStartVertex(const vtxtype start)=0;
    // void runLocalBFS()=0;
    // For accelerators with own memory

    void getBackPredecessor(); // expected to be used afet the application finished

    void getBackOutqueue();

    void setBackInqueue();

    void generatOwenMask();

public:
    GlobalBFS(STORE &_store);

    ~GlobalBFS();

#ifdef INSTRUMENTED
    void runBFS(typename STORE::vtxtyp startVertex, double& lexp, double &lqueue, double& rowcom, double& colcom, double& predlistred);
#else
    void runBFS(typename STORE::vtxtyp startVertex);
#endif

    typename STORE::vtxtyp *getPredecessor();
};

/*
 * Bitmap compression on predecessor reduction
 *
 */
template<class Derived, class FQ_T, class MType, class STORE>
void GlobalBFS<Derived, FQ_T, MType, STORE>::allReduceBitCompressed(typename STORE::vtxtyp *&owen,
                                                                    typename STORE::vtxtyp *&tmp, MType *&owenmap,
                                                                    MType *&tmpmap) {
    MPI_Status status;
    int communicatorSize, communicatorRank, intLdSize, power2intLdSize, residuum;
    int psize = mask_size;
    int mtypesize = 8 * sizeof(MType);
    //step 1
    MPI_Comm_size(col_comm, &communicatorSize);
    MPI_Comm_rank(col_comm, &communicatorRank);

    intLdSize = ilogb(static_cast<double>(communicatorSize)); //integer log_2 of size
    power2intLdSize = 1 << intLdSize; // 2^n
    residuum = communicatorSize - (1 << intLdSize);

    //step 2
    if (communicatorRank < 2 * residuum) {
        if ((communicatorRank & 1) == 0) { // even
            MPI_Sendrecv(owenmap, psize, bm_type, communicatorRank + 1, 0,
                         tmpmap, psize, bm_type, communicatorRank + 1, 0,
                         col_comm, &status);
            for (int i = 0; i < psize; ++i) {
                tmpmap[i] = tmpmap[i] & ~owenmap[i];
                owenmap[i] = owenmap[i] | tmpmap[i];
            }

            MPI_Recv(tmp, store.getLocColLength(), fq_tp_type, communicatorRank + 1, 1, col_comm, &status);
            int p = 0;
            for (int i = 0; i < psize; ++i) {
                MType tmpm = tmpmap[i];
                int size = i * mtypesize;
                while (tmpm != 0) {
                    int last = ffsl(tmpm) - 1;
                    owen[size + last] = tmp[p];
                    ++p;
                    tmpm ^= (1 << last);
                }
            }

        } else { // odd
            MPI_Sendrecv(owenmap, psize, bm_type, communicatorRank - 1, 0,
                         tmpmap, psize, bm_type, communicatorRank - 1, 0,
                         col_comm, &status);
            for (int i = 0; i < psize; ++i) {
                tmpmap[i] = ~tmpmap[i] & owenmap[i];
            }
            int p = 0;
            for (int i = 0; i < psize; ++i) {
                MType tmpm = tmpmap[i];
                int size = i * mtypesize;
                while (tmpm != 0) {
                    int last = ffsl(tmpm) - 1;
                    tmp[p] = owen[size + last];
                    ++p;
                    tmpm ^= (1 << last);
                }
            }
            MPI_Send(tmp, p, fq_tp_type, communicatorRank - 1, 1, col_comm);
        }
    }
    const std::function <int(int)> newRank = [&residuum](int oldr) {
        return (oldr < 2 * residuum) ? oldr / 2 : oldr - residuum;
    };
    const std::function <int(int)> oldRank = [&residuum](int newr) {
        return (newr < residuum) ? newr * 2 : newr + residuum;
    };

    if ((((communicatorRank & 1) == 0) && (communicatorRank < 2 * residuum)) || (communicatorRank >= 2 * residuum)) {
        int ssize, vrank, offset, lowers, uppers, size;

        ssize = psize;
        vrank = newRank(communicatorRank);
        offset = 0;


        for (int it = 0; it < intLdSize; ++it) {
            lowers = ssize / 2; //lower slice size
            uppers = ssize - lowers; //upper slice size
            size = lowers * mtypesize;

            if (((vrank >> it) & 1) == 0) {// even
                //Transmission of the the bitmap
                MPI_Sendrecv(owenmap + offset, ssize, bm_type, oldRank((vrank + (1 << it)) & (power2intLdSize - 1)),
                             (it << 1) + 2,
                             tmpmap + offset, ssize, bm_type, oldRank((vrank + (1 << it)) & (power2intLdSize - 1)),
                             (it << 1) + 2,
                             col_comm, &status);
                for (int i = 0; i < lowers; ++i) {
                    int ioffset = i + offset;
                    tmpmap[ioffset] = tmpmap[ioffset] & ~owenmap[ioffset];
                    owenmap[ioffset] = owenmap[ioffset] | tmpmap[ioffset];
                }
                for (int i = lowers; i < ssize; ++i) {
                        int ioffset = i + offset;
                    tmpmap[ioffset] = (~tmpmap[ioffset]) & owenmap[ioffset];
                }
                //Generation of foreign updates
                int p = 0;
                for (int i = 0; i < uppers; ++i) {
                    MType tmpm = tmpmap[i + offset + lowers];
                    int index = (i + offset + lowers) * mtypesize;
                    while (tmpm != 0) {
                        int last = ffsl(tmpm) - 1;
                        tmp[size + p] = owen[index + last];
                        ++p;
                        tmpm ^= (1 << last);
                    }
                }
                //Transmission of updates
                MPI_Sendrecv(tmp + size, p, fq_tp_type,
                             oldRank((vrank + (1 << it)) & (power2intLdSize - 1)), (it << 1) + 3,
                             tmp, size, fq_tp_type,
                             oldRank((vrank + (1 << it)) & (power2intLdSize - 1)), (it << 1) + 3,
                             col_comm, &status);
                //Updates for own data
                p = 0;
                for (int i = 0; i < lowers; ++i) {
                    MType tmpm = tmpmap[offset + i];
                    int index = (i + offset) * mtypesize;
                    while (tmpm != 0) {
                        int last = ffsl(tmpm) - 1;
                        owen[index + last] = tmp[p];
                        ++p;
                        tmpm ^= (1 << last);
                    }
                }
                ssize = lowers;
            } else { // odd
                //Transmission of the the bitmap
                MPI_Sendrecv(owenmap + offset, ssize, bm_type,
                             oldRank((power2intLdSize + vrank - (1 << it)) & (power2intLdSize - 1)), (it << 1) + 2,
                             tmpmap + offset, ssize, bm_type,
                             oldRank((power2intLdSize + vrank - (1 << it)) & (power2intLdSize - 1)), (it << 1) + 2,
                             col_comm, &status);
                for (int i = 0; i < lowers; ++i) {
                    tmpmap[i + offset] = (~tmpmap[i + offset]) & owenmap[i + offset];
                }
                for (int i = lowers; i < ssize; ++i) {
                    tmpmap[i + offset] = tmpmap[i + offset] & ~owenmap[i + offset];
                    owenmap[i + offset] = owenmap[i + offset] | tmpmap[i + offset];
                }
                //Generation of foreign updates
                int p = 0;
                for (int i = 0; i < lowers; ++i) {
                    MType tmpm = tmpmap[i + offset];
                    while (tmpm != 0) {
                        int last = ffsl(tmpm) - 1;
                        tmp[p] = owen[(i + offset) * mtypesize + last];
                        ++p;
                        tmpm ^= (1 << last);
                    }
                }
                //Transmission of updates
                MPI_Sendrecv(tmp, p, fq_tp_type,
                             oldRank((power2intLdSize + vrank - (1 << it)) & (power2intLdSize - 1)), (it << 1) + 3,
                             tmp + size, uppers * mtypesize, fq_tp_type,
                             oldRank((power2intLdSize + vrank - (1 << it)) & (power2intLdSize - 1)), (it << 1) + 3,
                             col_comm, &status);

                //Updates for own data
                p = 0;
                for (int i = 0; i < uppers; ++i) {
                    MType tmpm = tmpmap[offset + lowers + i];
                    int lindex = (i + offset + lowers) * mtypesize;
                    while (tmpm != 0) {
                        int last = ffsl(tmpm) - 1;
                        owen[lindex + last] = tmp[p + size];
                        ++p;
                        tmpm ^= (1 << last);
                    }
                }
                offset += lowers;
                ssize = uppers;
            }
        }
    }
    //Computation of displacements
    std::vector<int> sizes(communicatorSize);
    std::vector<int> disps(communicatorSize);

    unsigned int lastReversedSliceIDs = 0;
    unsigned int lastTargetNode = oldRank(lastReversedSliceIDs);

    sizes[lastTargetNode] = ((psize) >> intLdSize) * mtypesize;
    disps[lastTargetNode] = 0;

    for (unsigned int slice = 1; slice < power2intLdSize; ++slice) {
        unsigned int reversedSliceIDs = reverse(slice, intLdSize);
        unsigned int targetNode = oldRank(reversedSliceIDs);
        sizes[targetNode] = (psize >> intLdSize) * mtypesize;
        disps[targetNode] = ((slice * psize) >> intLdSize) * mtypesize;
        lastTargetNode = targetNode;
    }
    //nodes without a partial resulty
    for (unsigned int node = 0; node < residuum; ++node) {
        int index = 2 * node + 1;
        sizes[index] = 0;
        disps[index] = 0;
    }
    // Transmission of the final results
    MPI_Allgatherv(MPI_IN_PLACE, sizes[communicatorRank], fq_tp_type,
                   owen, &sizes[0], &disps[0], fq_tp_type, col_comm);
}

template<class Derived, class FQ_T, class MType, class STORE>
void GlobalBFS<Derived, FQ_T, MType, STORE>::getBackPredecessor() { }

template<class Derived, class FQ_T, class MType, class STORE>
void GlobalBFS<Derived, FQ_T, MType, STORE>::getBackOutqueue() { }

template<class Derived, class FQ_T, class MType, class STORE>
void GlobalBFS<Derived, FQ_T, MType, STORE>::setBackInqueue() { }

/*
 * Generates a map of the vertex with predecessor
 */
template<class Derived, class FQ_T, class MType, class STORE>
void GlobalBFS<Derived, FQ_T, MType, STORE>::generatOwenMask() {
    int mtypesize = 8 * sizeof(MType);

#ifdef _OPENMP
    #pragma omp parallel for
#endif

    for (long i = 0; i < mask_size; ++i) {
        MType tmp = 0;
        int index = i * mtypesize;
        for (long j = 0; j < mtypesize; ++j) {
            if ((predecessor[index + j] != -1) &&
                ((index + j) < store.getLocColLength()))
                tmp |= 1 << j;
        }
        owenmask[i] = tmp;
    }
}

template<class Derived, class FQ_T, class MType, class STORE>
GlobalBFS<Derived, FQ_T, MType, STORE>::GlobalBFS(STORE &_store) : store(_store) {
    int mtypesize = 8 * sizeof(MType);
    // Split communicator into row and column communicator
    // Split by row, rank by column
    MPI_Comm_split(MPI_COMM_WORLD, store.getLocalRowID(), store.getLocalColumnID(), &row_comm);
    // Split by column, rank by row
    MPI_Comm_split(MPI_COMM_WORLD, store.getLocalColumnID(), store.getLocalRowID(), &col_comm);

    fold_fq_props = store.getFoldProperties();

    mask_size = (store.getLocColLength() / mtypesize) +
                ((store.getLocColLength() % mtypesize > 0) ? 1 : 0);
    owenmask = new MType[mask_size];
    tmpmask = new MType[mask_size];
}


template<class Derived, class FQ_T, class MType, class STORE>
GlobalBFS<Derived, FQ_T, MType, STORE>::~GlobalBFS() {
    delete[] owenmask;
    delete[] tmpmask;
}

template<class Derived, class FQ_T, class MType, class STORE>
typename STORE::vtxtyp *GlobalBFS<Derived, FQ_T, MType, STORE>::getPredecessor() {
    return predecessor;
}


#endif // GLOBALBFS_HH
