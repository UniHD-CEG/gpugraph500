#include "checkqueue.h"

CheckQueue::CheckQueue(const QType minRowId, const QType maxRowId, const QType minColumnId, const QType maxColumnId): minRowId(minRowId),maxRowId(maxRowId),minColumnId(minColumnId),maxColumnId(maxColumnId)
{
}

bool CheckQueue::checkRow(const QType *queue, long length)
{
    //check if length is in a valid range
    if(length>(maxRowId-minRowId+1) || length < 0)
        return false;

    //check values of queue
    if(length >= 1){
        // test pointwise value if it is in expected range
        if(queue[0]< minRowId || queue[0]>maxRowId)
            return false;
    }

    if(int i=1; i< length; i++){
        // test pointwise value if it is in expected range
        if(queue[i]< minRowId || queue[i]>maxRowId)
            return false;

        //test if not sorted
        if(queue[i]<=queue[i-1])
            return false;
    }

    return true;
}

bool CheckQueue::checkCol(const QType* queue, long length)
{
    //check if length is in a valid range
    if(length>(maxColumnId-minColumnId+1) || length < 0)
        return false;

    //check values of queue
    if(length >= 1){
        // test pointwise value if it is in expected range
        if(queue[0]< minColumnId || queue[0]>maxColumnId)
            return false;
    }

    if(int i=1; i< length; i++){
        // test pointwise value if it is in expected range
        if(queue[i]< minColumnId || queue[i]>maxColumnId)
            return false;

        //test if not sorted
        if(queue[i]<=queue[i-1])
            return false;
    }

    return true;
}
