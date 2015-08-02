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

		int ssize, vrank, offset, lowers, uppers;

		ssize = psize;
		vrank = newRank(communicatorRank);
		offset = 0;

		for (int it = 0; it < intLdSize; ++it) {
			lowers = ssize / 2; //lower slice size
			uppers = ssize - lowers; //upper slice size

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
					int size = lowers * mtypesize;
					int index = (i + offset + lowers) * mtypesize;
					while (tmpm != 0) {
						int last = ffsl(tmpm) - 1;
						tmp[size + p] = owen[index + last];
						++p;
						tmpm ^= (1 << last);
					}
				}
				//Transmission of updates
				MPI_Sendrecv(tmp + lowers * mtypesize, p, fq_tp_type,
							 oldRank((vrank + (1 << it)) & (power2intLdSize - 1)), (it << 1) + 3,
							 tmp, lowers * mtypesize, fq_tp_type,
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
							 tmp + lowers * mtypesize, uppers * mtypesize, fq_tp_type,
							 oldRank((power2intLdSize + vrank - (1 << it)) & (power2intLdSize - 1)), (it << 1) + 3,
							 col_comm, &status);

				//Updates for own data
				p = 0;
				for (int i = 0; i < uppers; ++i) {
					MType tmpm = tmpmap[offset + lowers + i];
					int lindex = (i + offset + lowers) * mtypesize;
					int rindex = lowers * mtypesize;
					while (tmpm != 0) {
						int last = ffsl(tmpm) - 1;
						owen[lindex + last] = tmp[p + rindex];
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
		sizes[2 * node + 1] = 0;
		disps[2 * node + 1] = 0;
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

/**********************************************************************************
 * BFS search:
 * 0) Node 0 sends start vertex to all nodes
 * 1) Nodes test, if they are responsible for this vertex and push it,
 *    if they are in there fq
 * 2) Local expansion
 * 3) Test if anything is done
 * 4) global expansion
 * 5) global fold
 **********************************************************************************/

#ifdef INSTRUMENTED
	template<class Derived,class FQ_T,class MType,class STORE>
	void GlobalBFS<Derived,FQ_T,MType,STORE>::runBFS(typename STORE::vtxtyp startVertex, double& lexp, double& lqueue, double& rowcom, double& colcom, double& predlistred)
#else
	template<class Derived, class FQ_T, class MType, class STORE>
	void GlobalBFS<Derived, FQ_T, MType, STORE>::runBFS(typename STORE::vtxtyp startVertex)
#endif

{

#ifdef INSTRUMENTED
	double tstart, tend;
	lexp =0;
	lqueue =0;
	double comtstart, comtend;
	rowcom = 0;
	colcom = 0;
#endif

// 0) Node 0 sends start vertex to all nodes
	MPI_Bcast(&startVertex, 1, MPI_LONG, 0, MPI_COMM_WORLD);

// 1) Nodes test, if they are responsible for this vertex and push it, if they are in there fq
#ifdef INSTRUMENTED
	tstart = MPI_Wtime();
#endif

	static_cast<Derived *>(this)->setStartVertex(startVertex);

#ifdef INSTRUMENTED
	tend = MPI_Wtime();
	lqueue +=tend-tstart;
#endif

// 2) Local expansion
	int iter = 0;
	while (true) {

#ifdef INSTRUMENTED
	tstart = MPI_Wtime();
#endif

		static_cast<Derived *>(this)->runLocalBFS();

#ifdef INSTRUMENTED
	tend = MPI_Wtime();
	lexp +=tend-tstart;
#endif

// 3) Test if anything is done
		int anynewnodes, anynewnodes_global;

#ifdef INSTRUMENTED
	tstart = MPI_Wtime();
#endif

		anynewnodes = static_cast<Derived *>(this)->istheresomethingnew();

#ifdef INSTRUMENTED
	tend = MPI_Wtime();
	lqueue +=tend-tstart;
#endif

		MPI_Allreduce(&anynewnodes, &anynewnodes_global, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
		if (!anynewnodes_global) {

#ifdef INSTRUMENTED
	tstart = MPI_Wtime();
#endif

			static_cast<Derived *>(this)->getBackPredecessor();

#ifdef INSTRUMENTED
	tend = MPI_Wtime();
	lqueue += tend-tstart;
#endif

			// MPI_Allreduce(MPI_IN_PLACE, predecessor ,store.getLocColLength(),MPI_LONG,MPI_MAX,col_comm);
			static_cast<Derived *>(this)->generatOwenMask();
			allReduceBitCompressed(predecessor,
								   recv_fq_buff, // have to be changed for bitmap queue
								   owenmask, tmpmask);

#ifdef INSTRUMENTED
	tend = MPI_Wtime();
	predlistred = tend-tstart;
#endif

			return; // There is nothing too do. Finish iteration.
		}

// 4) global expansion
#ifdef INSTRUMENTED
	comtstart = MPI_Wtime();
	tstart = MPI_Wtime();
#endif

		static_cast<Derived *>(this)->getBackOutqueue();

#ifdef INSTRUMENTED
	tend = MPI_Wtime();
	lqueue +=tend-tstart;
#endif

		int _outsize; //really int, because mpi supports no long message sizes :(
		using namespace std::placeholders;
		std::function <void(FQ_T, long, FQ_T *, int)> reduce =
				std::bind(static_cast<void (Derived::*)(FQ_T, long, FQ_T *, int)>(&Derived::reduce_fq_out),
						  static_cast<Derived *>(this), _1, _2, _3, _4);
		std::function <void(FQ_T, long, FQ_T *&, int &)> get =
				std::bind(static_cast<void (Derived::*)(FQ_T, long, FQ_T *&, int &)>(&Derived::getOutgoingFQ),
						  static_cast<Derived *>(this), _1, _2, _3, _4);

		vreduce(reduce, get,
				recv_fq_buff,
				_outsize,
				store.getLocColLength(),
				fq_tp_type,
				col_comm

#ifdef INSTRUMENTED
				 ,lqueue
#endif
		);

		static_cast<Derived *>(this)->setModOutgoingFQ(recv_fq_buff, _outsize);

#ifdef INSTRUMENTED
	comtend = MPI_Wtime();
	colcom += comtend-comtstart;
#endif



// 4) global expansion
/* Failed to integrate this code*/

// Regarding a later implementation: In globalbfs.hh commit 59ab747 lines
// 395-475 is the column reduction code that I used in my thesis. This code
// is much simpler then the new implementation.

/*
#ifdef INSTRUMENTED
	comtstart = MPI_Wtime();
	#endif
#ifdef INSTRUMENTED
	tstart = MPI_Wtime();
#endif

	static_cast<Derived*>(this)->getBackOutqueue();

#ifdef INSTRUMENTED
	tend = MPI_Wtime();
	lqueue +=tend-tstart;
#endif

	// tree based reduce operation with messages of variable size
	// root 0
	int rounds = 0;
	while((1 << rounds) < store.getNumRowSl() ) {
		if((store.getLocalRowID() >> rounds) % 2 == 1){
			FQ_T* startaddr_fq;
			int _outsize;
			//comute recv addr
			int recv_addr = (store.getLocalRowID() + store.getNumRowSl() - (1 << rounds)) % store.getNumRowSl();
			//get fq to send

#ifdef INSTRUMENTED
			tstart = MPI_Wtime();
#endif

			static_cast<Derived*>(this)->getOutgoingFQ(startaddr_fq, _outsize);

#ifdef INSTRUMENTED
			tend = MPI_Wtime();
			lqueue +=tend-tstart;
#endif

			//send fq
			MPI_Ssend(startaddr_fq,_outsize ,fq_tp_type,recv_addr,rounds,col_comm);
			break;
		} else if ( store.getLocalRowID() + (1 << rounds) < store.getNumRowSl() ){
			MPI_Status    status;
			int count;
			//compute send addr
			int sender_addr = (store.getLocalRowID() +  (1 << rounds)) % store.getNumRowSl();
			//recv fq
			MPI_Recv(recv_fq_buff, recv_fq_buff_length, fq_tp_type,sender_addr,rounds, col_comm, &status);
			MPI_Get_count(&status, fq_tp_type, &count);
			//do reduce

#ifdef INSTRUMENTED
			tstart = MPI_Wtime();
#endif

			// @TODO:
			static_cast<Derived*>(this)->reduce_fq_out(recv_fq_buff,static_cast<long>(count), startaddr_fq, recv_fq_buff_length);

#ifdef INSTRUMENTED
			tend = MPI_Wtime();
			lqueue +=tend-tstart;
#endif

		}
		++rounds;
	}

	//distribute solution
	if(0 == store.getLocalRowID())
	{
		FQ_T* startaddr_fq;
		int _outsize;
		//get fq to send

#ifdef INSTRUMENTED
		tstart = MPI_Wtime();
#endif

		static_cast<Derived*>(this)->getOutgoingFQ(startaddr_fq, _outsize);

#ifdef INSTRUMENTED
		tend = MPI_Wtime();
		lqueue +=tend-tstart;
#endif

		MPI_Bcast(&_outsize, 1, MPI_LONG, 0, col_comm);
		MPI_Bcast(startaddr_fq, _outsize, fq_tp_type, 0, col_comm);

#ifdef INSTRUMENTED
		tstart = MPI_Wtime();
#endif

		static_cast<Derived*>(this)->setModOutgoingFQ(0,_outsize);

#ifdef INSTRUMENTED
		tend = MPI_Wtime();
		lqueue += tend-tstart;
#endif

	} else {
		int _outsize;
		MPI_Bcast(&_outsize, 1, MPI_LONG, 0, col_comm);
		assert(_outsize <= recv_fq_buff_length);
		MPI_Bcast(recv_fq_buff, _outsize, fq_tp_type, 0, col_comm);

#ifdef INSTRUMENTED
		tstart = MPI_Wtime();
#endif

		static_cast<Derived*>(this)->setModOutgoingFQ(recv_fq_buff,_outsize);

#ifdef INSTRUMENTED
		tend = MPI_Wtime();
		lqueue +=tend-tstart;
#endif

	}

	int _outsize; //really int, because mpi supports no long message sizes :(
	using namespace std::placeholders;
	std::function<void(FQ_T, long, FQ_T*, int )> reduce =
			std::bind(static_cast<void (Derived::*)(FQ_T, long, FQ_T*, int )>(&Derived::reduce_fq_out),static_cast<Derived*>(this), _1, _2, _3, _4);
	std::function<void(FQ_T, long, FQ_T*&, int& )> get =
			std::bind(static_cast<void (Derived::*)(FQ_T, long, FQ_T*&, int& )>(&Derived::getOutgoingFQ),static_cast<Derived*>(this), _1, _2, _3, _4);

	vreduce( reduce, get,
				 recv_fq_buff,
				 _outsize,
				 store.getLocColLength(),
				 fq_tp_type,
				 col_comm
#ifdef INSTRUMENTED
				 ,lqueue
#endif
				 );

	static_cast<Derived*>(this)->setModOutgoingFQ(recv_fq_buff,_outsize);


#ifdef INSTRUMENTED
	comtend = MPI_Wtime();
	colcom += comtend-comtstart;
#endif
*/

// 5) global fold
#ifdef INSTRUMENTED
	comtstart = MPI_Wtime();
#endif

		for (typename std::vector<typename STORE::fold_prop>::iterator it = fold_fq_props.begin();
			 it != fold_fq_props.end(); ++it) {
			if (it->sendColSl == store.getLocalColumnID()) {
				FQ_T *startaddr;
				int outsize;

#ifdef INSTRUMENTED
	tstart = MPI_Wtime();
#endif

				static_cast<Derived *>(this)->getOutgoingFQ(it->startvtx, it->size, startaddr, outsize);

#ifdef INSTRUMENTED
	tend = MPI_Wtime();
	lqueue +=tend-tstart;
#endif

				MPI_Bcast(&outsize, 1, MPI_LONG, it->sendColSl, row_comm);
				MPI_Bcast(startaddr, outsize, fq_tp_type, it->sendColSl, row_comm);

#ifdef INSTRUMENTED
	tstart = MPI_Wtime();
#endif

				static_cast<Derived *>(this)->setIncommingFQ(it->startvtx, it->size, startaddr, outsize);

#ifdef INSTRUMENTED
	tend = MPI_Wtime();
	lqueue +=tend-tstart;
#endif

			} else {
				int outsize;
				MPI_Bcast(&outsize, 1, MPI_LONG, it->sendColSl, row_comm);
				assert(outsize <= recv_fq_buff_length);
				MPI_Bcast(recv_fq_buff, outsize, fq_tp_type, it->sendColSl, row_comm);

#ifdef INSTRUMENTED
	tstart = MPI_Wtime();
#endif

				static_cast<Derived *>(this)->setIncommingFQ(it->startvtx, it->size, recv_fq_buff, outsize);

#ifdef INSTRUMENTED
	tend = MPI_Wtime();
	lqueue +=tend-tstart;
#endif
			}
		}

#ifdef INSTRUMENTED
	tstart = MPI_Wtime();
#endif

		static_cast<Derived *>(this)->setBackInqueue();

#ifdef INSTRUMENTED
	tend = MPI_Wtime();
	lqueue +=tend-tstart;
#endif

#ifdef INSTRUMENTED
	comtend = MPI_Wtime();
	rowcom += comtend - comtstart;
#endif
		++iter;
	}
}

template<class Derived, class FQ_T, class MType, class STORE>
typename STORE::vtxtyp *GlobalBFS<Derived, FQ_T, MType, STORE>::getPredecessor() {
	return predecessor;
}

#endif // GLOBALBFS_HH
