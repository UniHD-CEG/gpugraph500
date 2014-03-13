#include <algorithm>
#include "simplecpubfs.h"

SimpleCPUBFS::SimpleCPUBFS(MatrixT &_store):GlobalBFS<SimpleCPUBFS,void,MatrixT>(_store)
{
    fq_tp_type = MPI_INT64_T; //Frontier Queue Transport Type
    predessor = new vtxtype[store.getLocColLength()];

    //allocate recive buffer
    recv_fq_buff_length = std::max(store.getLocRowLength(), store.getLocColLength());
    recv_fq_buff = static_cast<void*>( new vtxtype[recv_fq_buff_length]);

}

SimpleCPUBFS::~SimpleCPUBFS()
{
    delete[] static_cast<vtxtype*>(recv_fq_buff);
    delete[] predessor;
}

void SimpleCPUBFS::reduce_fq_out(void *startaddr, long insize)
{
    std::vector<vtxtype> out_buff(store.getLocColLength());
    std::vector<vtxtype>::iterator it;

    it=std::set_union(fq_out.begin(), fq_out.end(), static_cast<vtxtype*>(startaddr), static_cast<vtxtype*>(startaddr)+insize, out_buff.begin());
    out_buff.resize(it-out_buff.begin());
    fq_out=out_buff;
}


void SimpleCPUBFS::runLocalBFS()
{
    fq_out.clear();

    while(!fq_in.empty()){
        vtxtype actual_vtx = fq_in.back();
        vtxtype actual_vtx_loc = store.globaltolocalRow(actual_vtx);
        fq_in.pop_back();

        for(vtxtype i = store.getRowPointer()[actual_vtx_loc]; i < store.getRowPointer()[actual_vtx_loc+1]; i++){
            vtxtype visit_vtx     = store.getColumnIndex()[i];
            vtxtype visit_vtx_loc = store.globaltolocalCol(visit_vtx);
            if(!visited[visit_vtx_loc]){
                fq_out.push_back(visit_vtx);
                visited[visit_vtx_loc] = true;
                predessor[visit_vtx_loc] = actual_vtx;
            }
        }
    }
    std::sort(fq_out.begin(), fq_out.end());
}

void SimpleCPUBFS::setStartVertex(const vtxtype start)
{
    //reset predessor list
    for(int i = 0; i < store.getLocColLength(); i++){
        predessor[i] = -1;
    }

    visited.assign(store.getLocColLength(),false);
    fq_out.clear();
    fq_in.clear();

    if(store.isLocalColumn(start)){
        visited[store.globaltolocalCol(start)]   = true;
        predessor[store.globaltolocalCol(start)] = start;
    }

    if(store.isLocalRow(start)){
        fq_in.push_back(start);
    }
}

bool SimpleCPUBFS::istheresomethingnew()
{
    return (fq_out.size() > 0);
}

void SimpleCPUBFS::setIncommingFQ(vtxtype globalstart, vtxtype size, void *startaddr, vtxtype &insize_max)
{
    assert(store.isLocalRow(globalstart));
    assert(insize_max >= size);
    long* buffer = static_cast<long *>(startaddr);
    for(long i=0; i < insize_max; i++){
        fq_in.push_back(buffer[i]);
    }
}

void SimpleCPUBFS::getOutgoingFQ(vtxtype globalstart, vtxtype size, void* &startaddr, vtxtype &outsize)
{
    long sidx= 	static_cast<long>(std::lower_bound(fq_out.begin(),fq_out.end(), globalstart) - fq_out.begin());
    long eidx= static_cast<long>(std::upper_bound(fq_out.begin()+sidx,fq_out.end(), globalstart+size-1) - fq_out.begin());

    startaddr   = static_cast<void*>(fq_out.data()+sidx);
    outsize     = eidx-sidx;
}

void SimpleCPUBFS::setModOutgoingFQ(void *startaddr, long insize)
{
    if(startaddr != 0){
        fq_out.clear();
        fq_out.resize(insize);
        std::copy(static_cast<vtxtype*>(startaddr),static_cast<vtxtype*>(startaddr)+insize, fq_out.begin());
     }

    for(std::vector<vtxtype>::iterator it = fq_out.begin(); it != fq_out.end(); it++){
        visited[store.globaltolocalCol(*it)] = true;
    }
}

void SimpleCPUBFS::getOutgoingFQ(void *&startaddr, vtxtype &outsize)
{
    outsize     = fq_out.size();
    startaddr   = static_cast<void *>(fq_out.data());
}
