/* Copyright (C) 2010-2011 The Trustees of Indiana University.             */
/*                                                                         */
/* Use, modification and distribution is subject to the Boost Software     */
/* License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at */
/* http://www.boost.org/LICENSE_1_0.txt)                                   */
/*                                                                         */
/*  Authors: Jeremiah Willcock                                             */
/*           Andrew Lumsdaine                                              */
/* modification for 2d partitioning: Matthias Hauck 2014 */
#define __STDC_LIMIT_MACROS
#include "../distmatrix2d.hh"
#include "mpi_workarounds.h"
#include "../generator/utils.h"
#include "onesided.h"
#include <mpi.h>
#include <stdint.h>
#include <inttypes.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>

#ifndef VALIDATE_H
#define VALIDATE_H

template<class MatrixT>
int validate_bfs_result(const MatrixT& store, packed_edge *edgelist, int64_t number_of_edges,
                        const int64_t nglobalverts, const int64_t root, int64_t* const pred, int64_t* const edge_visit_count_ptr, int *level);

static inline size_t size_min(size_t a, size_t b) {
  return a < b ? a : b;
}

static inline ptrdiff_t ptrdiff_min(ptrdiff_t a, ptrdiff_t b) {
  return a < b ? a : b;
}

/* Chunk size for blocks of one-sided operations; a fence is inserted after (at
 * most) each CHUNKSIZE one-sided operations. */
/* It seams as there is a limit of the number of MPI_Get in an epoche in OpenMPI.
 * An incresed Chunksize can in this way problematic. */
#define CHUNKSIZE (1 << 20)
#define HALF_CHUNKSIZE ((CHUNKSIZE) / 2)


/* This code assumes signed shifts are arithmetic, which they are on
 * practically all modern systems but is not guaranteed by C. */
int64_t get_pred_from_pred_entry(int64_t val);

uint16_t get_depth_from_pred_entry(int64_t val);

//void write_pred_entry_depth(int64_t* loc, uint16_t depth);

template<class T>
void write_pred_entry_depth(T* loc, uint16_t depth) {
   assert(sizeof(T)==8);
  *loc = (*loc & static_cast<T>(0xFFFFFFFFFFFF)) | ((T)(depth & 0xFFFF) << 48);
}


/* Returns true if all values are in range. */
template<class MatrixT>
static int check_value_ranges(const MatrixT& store, const int64_t nglobalverts, const typename MatrixT::vtxtyp* const pred) {
  int any_range_errors = 0;
  {
    for (size_t ii = 0; ii < static_cast<size_t>(store.getLocColLength()); ii += CHUNKSIZE) {
      ptrdiff_t i_start = ii;
      ptrdiff_t i_end = ptrdiff_min(ii + CHUNKSIZE, store.getLocColLength());
      ptrdiff_t i;
      assert (i_start >= 0 && i_start <= (ptrdiff_t)store.getLocColLength());
      assert (i_end >= 0 && i_end <= (ptrdiff_t)store.getLocColLength());
#pragma omp parallel for reduction(||:any_range_errors)
      for (i = i_start; i < i_end; ++i) {
        int64_t p = get_pred_from_pred_entry(pred[i]);
        if (p < -1 || p >= nglobalverts) {
          fprintf(stderr, "(%ld:%ld): Validation error: parent of vertex %" PRId64 " is out-of-range value %" PRId64 ".\n", store.getLocalRowID(), store.getLocalColumnID() , store.localtoglobalCol(i), p);
          any_range_errors = 1;
        }
      }
    }
  }
  MPI_Allreduce(MPI_IN_PLACE, &any_range_errors, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
  return !any_range_errors;
}

/* Use the predecessors in the given map to write the BFS levels to the high 16
 * bits of each element in pred; this also catches some problems in pred
 * itself.  Returns true if the predecessor map is valid. */
template<class MatrixT>
static int build_bfs_depth_map(const MatrixT& store, const int64_t nglobalverts, const typename MatrixT::vtxtyp nlocalverts, const size_t maxlocalverts, const typename MatrixT::vtxtyp root, typename MatrixT::vtxtyp* const pred, int* level) {
  (void)nglobalverts;
  int validation_passed = 1;
  int root_owner;
  size_t root_local;
  store.get_vertex_distribution_for_pred(1, &root, &root_owner, &root_local);
  int root_is_mine = (root_owner == store.getLocalColumnID());
  if (root_is_mine) assert (root_local < nlocalverts);

  {
#pragma omp parallel for
    for (ptrdiff_t i = 0; i < (ptrdiff_t)nlocalverts; ++i) write_pred_entry_depth(&pred[i], UINT16_MAX);
    if (root_is_mine) write_pred_entry_depth(&pred[root_local], 0);
  }
  MPI_Comm row_comm; //, col_comm;
  MPI_Comm_split(MPI_COMM_WORLD, store.getLocalRowID(), store.getLocalColumnID(), &row_comm);
  // Split by column, rank by row
  //MPI_Comm_split(MPI_COMM_WORLD, store.getLocalColumnID(), store.getLocalRowID(), &col_comm);

  int64_t* pred_pred = (int64_t*)xMPI_Alloc_mem(size_min(CHUNKSIZE, nlocalverts) * sizeof(int64_t)); /* Predecessor info of predecessor vertex for each local vertex */
  gather* pred_win = init_gather((void*)pred, nlocalverts, sizeof(int64_t), pred_pred, size_min(CHUNKSIZE, nlocalverts), size_min(CHUNKSIZE, nlocalverts), MPI_INT64_T, row_comm);
  typename MatrixT::vtxtyp* pred_vtx = (typename MatrixT::vtxtyp*)xmalloc(size_min(CHUNKSIZE, nlocalverts) * sizeof(int64_t)); /* Vertex (not depth) part of pred map */
  int* pred_owner = (int*)xmalloc(size_min(CHUNKSIZE, nlocalverts) * sizeof(int));
  size_t* pred_local = (size_t*)xmalloc(size_min(CHUNKSIZE, nlocalverts) * sizeof(size_t));
  int iter_number = 0;

  {
    /* Iteratively update depth[v] = min(depth[v], depth[pred[v]] + 1) [saturating at UINT16_MAX] until no changes. */
   while (1) {
      int any_changes = 0;
      for (ptrdiff_t ii = 0; ii < (ptrdiff_t)maxlocalverts; ii += CHUNKSIZE) {
        ptrdiff_t i_start = ptrdiff_min(ii, nlocalverts);
        ptrdiff_t i_end = ptrdiff_min(ii + CHUNKSIZE, nlocalverts);
        begin_gather(pred_win);
        assert (i_start >= 0 && i_start <= (ptrdiff_t)nlocalverts);
        assert (i_end >= 0 && i_end <= (ptrdiff_t)nlocalverts);
#pragma omp parallel for
        for (ptrdiff_t i = i_start; i < i_end; ++i) {
          pred_vtx[i - i_start] = get_pred_from_pred_entry(pred[i]);
        }

        store.get_vertex_distribution_for_pred(i_end - i_start, pred_vtx, pred_owner, pred_local);
#pragma omp parallel for
        for (ptrdiff_t i = i_start; i < i_end; ++i) {
          if (pred[i] != -1) {
            add_gather_request(pred_win, i - i_start, pred_owner[i - i_start], pred_local[i - i_start], i - i_start);
          } else {
            pred_pred[i - i_start] = -1;
          }
        }
        end_gather(pred_win);
#pragma omp parallel for reduction(&&:validation_passed) reduction(||:any_changes)
        for (ptrdiff_t i = i_start; i < i_end; ++i) {
          if (store.getLocalColumnID() == root_owner && (size_t)i == root_local) continue;
          if (get_depth_from_pred_entry(pred_pred[i - i_start]) != UINT16_MAX) {
            if (get_depth_from_pred_entry(pred[i]) != UINT16_MAX && get_depth_from_pred_entry(pred[i]) != get_depth_from_pred_entry(pred_pred[i - i_start]) + 1) {
              validation_passed = 0;
            } else if (get_depth_from_pred_entry(pred[i]) == get_depth_from_pred_entry(pred_pred[i - i_start]) + 1) {
              /* Nothing to do */
            } else {
              write_pred_entry_depth(&pred[i], get_depth_from_pred_entry(pred_pred[i - i_start]) + 1);
              any_changes = 1;
            }
          }
        }
      }
      MPI_Allreduce(MPI_IN_PLACE, &any_changes, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
      if (!any_changes) break;
      ++iter_number;
    }
  }

  *level = iter_number;

  destroy_gather(pred_win);
  MPI_Free_mem(pred_pred);
  free(pred_owner);
  free(pred_local);
  free(pred_vtx);
  MPI_Comm_free(&row_comm);
  //MPI_Comm_free(&col_comm);
  return validation_passed;
}

/* Returns true if result is valid.  Also, updates high 16 bits of each element
 * of pred to contain the BFS level number (or -1 if not visited) of each
 * vertex; this is based on the predecessor map if the user didn't provide it.
 * */
template<class MatrixT>
int validate_bfs_result(const MatrixT &store, packed_edge* edgelist, int64_t number_of_edges,
                        const int64_t nglobalverts, const typename MatrixT::vtxtyp root,typename MatrixT::vtxtyp* const pred, typename MatrixT::vtxtyp* const edge_visit_count_ptr, int* level) {

  assert (pred);
  *edge_visit_count_ptr = 0; /* Ensure it is a valid pointer */
  int ranges_ok = check_value_ranges(store,nglobalverts, pred);
  if (root < 0 || root >= nglobalverts) {
    fprintf(stderr, "(%ld:%ld): Validation error: root vertex %" PRId64 " is invalid.\n", store.getLocalRowID(), store.getLocalColumnID(), root);
    ranges_ok = 0;
  }
  if (!ranges_ok) return 0; /* Fail */

  assert (pred);

  int validation_passed = 1;
  int root_owner;
  size_t root_local;
  store.get_vertex_distribution_for_pred(1, &root, &root_owner, &root_local);
  int root_is_mine = (root_owner == store.getLocalColumnID());

  /* Get maximum values so loop counts are consistent across ranks. */
  uint64_t maxlocalverts_ui = store.getLocColLength();
  MPI_Allreduce(MPI_IN_PLACE, &maxlocalverts_ui, 1, MPI_UINT64_T, MPI_MAX, MPI_COMM_WORLD);
  size_t maxlocalverts = (size_t)maxlocalverts_ui;

  assert (pred);

  /* Check that root is its own parent. */
  if (root_is_mine) {
    assert (root_local < static_cast<size_t>(store.getLocColLength()));
    if (get_pred_from_pred_entry(pred[root_local]) != root) {
      fprintf(stderr, "(%ld:%ld): Validation error: parent of root vertex %" PRId64 " is %" PRId64 " and not itself as it should be as root.\n", store.getLocalRowID(), store.getLocalColumnID(), root, get_pred_from_pred_entry(pred[root_local]));
      validation_passed = 0;
    }
  }

  assert (pred);

  /* Check that nothing else is its own parent. */
  {
    int* pred_owner = (int*)xmalloc(size_min(CHUNKSIZE,store.getLocColLength()) * sizeof(int));
    size_t* pred_local = (size_t*)xmalloc(size_min(CHUNKSIZE, store.getLocColLength()) * sizeof(size_t));
    typename MatrixT::vtxtyp* pred_vtx = (typename MatrixT::vtxtyp*)xmalloc(size_min(CHUNKSIZE, store.getLocColLength()) * sizeof(int64_t)); /* Vertex (not depth) part of pred map */
    for (ptrdiff_t ii = 0; ii < (ptrdiff_t)store.getLocColLength(); ii += CHUNKSIZE) {
      ptrdiff_t i_start = ii;
      ptrdiff_t i_end = ptrdiff_min(ii + CHUNKSIZE, store.getLocColLength());

      assert (i_start >= 0 && i_start <= (ptrdiff_t)store.getLocColLength());
      assert (i_end >= 0 && i_end <= (ptrdiff_t)store.getLocColLength());
#pragma omp parallel for
      for (ptrdiff_t i = i_start; i < i_end; ++i) {
        pred_vtx[i - i_start] = get_pred_from_pred_entry(pred[i]);
      }
      store.get_vertex_distribution_for_pred(i_end - i_start, pred_vtx, pred_owner, pred_local);
#pragma omp parallel for reduction(&&:validation_passed)
      for (ptrdiff_t i = i_start; i < i_end; ++i) {
        if ((!root_is_mine || (size_t)i != root_local) &&
            get_pred_from_pred_entry(pred[i]) != -1 &&
            pred_owner[i - i_start] == store.getLocalColumnID() &&
            pred_local[i - i_start] == (size_t)i) {
                 fprintf(stderr, "(%ld:%ld): Validation error: parent of non-root vertex %" PRId64 " is itself.\n", store.getLocalRowID(), store.getLocalColumnID(), store.localtoglobalCol(i));
                 validation_passed = 0;
        }
      }
    }
    free(pred_owner);
    free(pred_local);
    free(pred_vtx);
  }

  assert (pred);
  if(validation_passed == 0)
      printf("other tests faild\n");
  {
    /* Create a vertex depth map to use for later validation. */
    int pred_ok = build_bfs_depth_map(store, nglobalverts, store.getLocColLength(), maxlocalverts, root, pred, level);
    if (!pred_ok) validation_passed = 0;
  }
  if(validation_passed == 0)
      printf("depth map creation faild\n");
  {
      MPI_Comm row_comm, col_comm;
      MPI_Comm_split(MPI_COMM_WORLD, store.getLocalRowID(), store.getLocalColumnID(), &row_comm);
      // Split by column, rank by row
      MPI_Comm_split(MPI_COMM_WORLD, store.getLocalColumnID(), store.getLocalRowID(), &col_comm);

    /* Check that all edges connect vertices whose depths differ by at most
     * one, and check that there is an edge from each vertex to its claimed
     * predecessor.  Also, count visited edges (including duplicates and
     * self-loops).  */

      long pred_visited_size = store.getLocColLength()/64+((store.getLocColLength()%64>0)? 1:0);
      uint64_t* pred_visited = (uint64_t*) malloc(pred_visited_size* sizeof(uint64_t));
      memset(pred_visited, 0, pred_visited_size * sizeof(uint64_t));
      int64_t* rowPred = (int64_t*) malloc(store.getLocRowLength()*sizeof(int64_t));
      const std::vector<typename MatrixT::fold_prop> rowFractions = store.getFoldProperties();
      int64_t edge_visit_count = 0;
      int    valid_level = 1;
      int    all_visited = 1;

      for(typename std::vector<typename MatrixT::fold_prop>::const_iterator it = rowFractions.begin(); it  != rowFractions.end(); it++){
          if(it->sendColSl == store.getLocalColumnID() ){
              MPI_Bcast(&pred[store.globaltolocalCol(it->startvtx)],it->size,MPI_INT64_T,it->sendColSl,row_comm);
              memcpy(&rowPred[store.globaltolocalRow(it->startvtx)],&pred[store.globaltolocalCol(it->startvtx)],it->size*sizeof(int64_t));
          }else{
              MPI_Bcast(&rowPred[store.globaltolocalRow(it->startvtx)],it->size,MPI_INT64_T,it->sendColSl,row_comm);
          }
      }

      #pragma omp parallel for reduction(+: edge_visit_count) reduction(&&: valid_level)
      for(long i=0; i < number_of_edges; i++){
          packed_edge edge = edgelist[i];

          int64_t p_vertex0 = rowPred[store.globaltolocalRow(edge.v0)];
          int64_t p_vertex1 = pred[store.globaltolocalCol(edge.v1)];

          uint16_t d0 = get_depth_from_pred_entry(p_vertex0);
          uint16_t d1 = get_depth_from_pred_entry(p_vertex1);
          // test if level diverenz is max. 1
          if((d0 - d1 > 1 || d1 - d0 > 1) ){
              valid_level = 0;
              fprintf(stderr,"(%ld:%ld) Edge [%ld(%ld):%ld(%ld)] with wrong levels: %d %d\n",store.getLocalRowID(),
                      store.getLocalColumnID(),edge.v0,store.globaltolocalRow(edge.v0),edge.v1,store.globaltolocalCol(edge.v1),d0,d1);
          }

          // mark if there is an edge from each vertex to its claimed
          // predecessor
          if(edge.v0 == get_pred_from_pred_entry(p_vertex1)){
              uint64_t vertex1_loc = store.globaltolocalCol(edge.v1);
              uint64_t word_index = vertex1_loc/64;
              uint64_t bit_index  = vertex1_loc%64;

              if((pred_visited[word_index] & (1 << bit_index)) == 0 ){
                #pragma omp atomic
                pred_visited[word_index]|= 1 << bit_index;
              }
          }
          //count "visited" edge
          if(get_pred_from_pred_entry(p_vertex0)!= -1)
              edge_visit_count++;
       }
      if(valid_level==0)
          printf("invalid_level\n");

       MPI_Allreduce(MPI_IN_PLACE,pred_visited, pred_visited_size, MPI_UINT64_T, MPI_BOR, col_comm);

       #pragma omp parallel for reduction(&&: all_visited)
       for(long i=0; i < store.getLocColLength(); i++){
           // check that there is a mark for each vertex that there is an edge
           // to its claimed predecessor
           if((pred_visited[i/64] & 1 << (i%64)) == 0){
               //execept if there is no predecessor
               if(get_pred_from_pred_entry(pred[i])!= -1 &&
                  get_pred_from_pred_entry(pred[i])!= store.localtoglobalCol(i)){
                   fprintf(stderr,"predecessor(%ld) of Vertex %ld is invalid!\n", get_pred_from_pred_entry(pred[i]), store.localtoglobalCol(i));
                   all_visited = 0;
                }
           }
       }

       free(rowPred);
       free(pred_visited);
       MPI_Comm_free(&row_comm);
       MPI_Comm_free(&col_comm);

       if(all_visited==0)
           fprintf(stderr,"Not all vertexes with an edge are visited.\n");

       validation_passed = validation_passed && valid_level && all_visited;

       MPI_Allreduce(MPI_IN_PLACE, &edge_visit_count, 1, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
       *edge_visit_count_ptr = edge_visit_count/2;

  }

  /* Collect the global validation result. */
  MPI_Allreduce(MPI_IN_PLACE, &validation_passed, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
  return validation_passed;
}
#endif // VALIDATE_H
