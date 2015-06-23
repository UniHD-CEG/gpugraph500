/**

  @author Matthias Hauck
  @description
*/
#ifndef CHECKQUEUE_H
#define CHECKQUEUE_H

template< typename QType >
class CheckQueue
{
    const QType minRowId;
    const QType maxRowId;
    const QType minColumnId;
    const QType maxColumnId;
public:
    enum class ErrorCode{
        Valid = 0,
        InvalidLength,
        IdsOutOfRange,
        NotSorted,
        DuplicteIds
    };

    CheckQueue(const QType minRowId, const QType maxRowId,
               const QType minColumnId, const QType maxColumnId): minRowId(minRowId),maxRowId(maxRowId),minColumnId(minColumnId),maxColumnId(maxColumnId){}

    ErrorCode checkRowLength(long length){
        //check if length is in a valid range
        if(length>(maxRowId-minRowId+1) || length < 0)
            return ErrorCode::InvalidLength;
        return ErrorCode::Valid;
    }
    ErrorCode checkColLength(long length){
        //check if length is in a valid range
        if(length>(maxColumnId-minColumnId+1) || length < 0)
            return ErrorCode::InvalidLength;
        return ErrorCode::Valid;
    }

    ErrorCode checkRow(const QType* queue, long length){
        //check if length is in a valid range
        if(length>(maxRowId-minRowId+1) || length < 0)
            return ErrorCode::InvalidLength;;

        //check values of queue
        if(length >= 1){
            // test pointwise value if it is in expected range
            if(queue[0]< minRowId || queue[0]>maxRowId)
                return ErrorCode::IdsOutOfRange;
        }

        for(long i=1; i< length; i++){
            // test pointwise value if it is in expected range
            if(queue[i]< minRowId || queue[i]>maxRowId)
                return ErrorCode::IdsOutOfRange;

            //test if not unique
            if(queue[i]==queue[i-1])
                return ErrorCode::DuplicteIds;

            //test if not sorted
            if(queue[i]<queue[i-1])
                return ErrorCode::NotSorted;
        }

        return ErrorCode::Valid;
    }

    ErrorCode checkCol(const QType* queue, long length){
        //check if length is in a valid range
        if(length>(maxColumnId-minColumnId+1) || length < 0)
            return ErrorCode::InvalidLength;;

        //check values of queue
        if(length >= 1){
            // test pointwise value if it is in expected range
            if(queue[0]< minColumnId || queue[0]>maxColumnId)
                return ErrorCode::IdsOutOfRange;
        }

        for(long i=1; i< length; i++){
            // test pointwise value if it is in expected range
            if(queue[i]< minColumnId || queue[i]>maxColumnId)
                return ErrorCode::IdsOutOfRange;

            //test if not unique
            if(queue[i]==queue[i-1])
                return ErrorCode::DuplicteIds;

            //test if not sorted
            if(queue[i]<queue[i-1])
                return ErrorCode::NotSorted;
        }

        return ErrorCode::Valid;
    }
};

#endif // CHECKQUEUE_H
