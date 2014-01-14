#include <generator/graph_generator.h>
#include "mpi.h"
#include <vector>

#ifndef DISTMATRIX2D_H
#define DISTMATRIX2D_H

/*
*   This class is a representation of a distributed 2d partitioned adjacency Matrix
*   in CRS style. The edges are not weighted, so there is no value, because an row
*   and column index indicate the edge. It uses MPI.
*
*/
typedef long vtxtype;

class DistMatrix2d
{
    int R; //Row slices
    int C; //Column slices
    int r; //Row id of this node
    int c; //Column id of this node

    vtxtype globalNumberOfVertex;

    vtxtype row_start, row_length; // global index of first row; size of local row slice
    vtxtype column_start, column_length; // global index of first (potential) column; size of local column slice
    vtxtype* row_pointer;  //Row pointer to columns
    vtxtype* column_index; //Column index

    /*
     *  Computes the owner node of a row/column pair.
     */
    long computeOwner(unsigned long row, unsigned long column);

public:
    struct fold_prop{
        int     sendColSl;
        vtxtype startvtx;
        vtxtype size;
    };

    DistMatrix2d(int _R, int _C, unsigned long scale);
    ~DistMatrix2d();

    void setupMatrix(packed_edge* input, int numberOfEdges,bool undirected= true);
    inline long getEdgeCount(){return (row_pointer!=0)?row_pointer[row_length]:0;}
    std::vector<struct fold_prop> getFoldProperties();

    inline vtxtype* getRowPointer() {return row_pointer;}
    inline vtxtype* getColumnIndex(){ return column_index;}

    inline bool isSym(){return C==R;}

    inline int getNumRowSl(){return R;}
    inline int getNumColumnSl(){return C;}

    inline int getLocalRowID(){return r;}
    inline int getLocalColumnID(){return c;}

    inline bool isLocalRow(vtxtype vtx){return (row_start<=vtx) && (vtx < row_start+row_length);}
    inline bool isLocalColumn(vtxtype vtx){ return (column_start<=vtx) && (vtx < column_start+column_length);}


    inline vtxtype globaltolocalRow(vtxtype vtx){return vtx-row_start;}
    inline vtxtype globaltolocalCol(vtxtype vtx){return vtx-column_start;}
    inline vtxtype localtoglobalRow(vtxtype vtx){return vtx+row_start;}
    inline vtxtype localtoglobalCol(vtxtype vtx){return vtx+column_start;}

    inline vtxtype getLocRowLength(){return row_length;}
    inline vtxtype getLocColLength(){return column_length;}

};

#endif // DISTMATRIX2D_H
