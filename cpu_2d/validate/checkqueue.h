/**

  @author Matthias Hauck
  @description
*/
#ifndef CHECKQUEUE_H
#define CHECKQUEUE_H

template< typename QType >
class CheckQueue
{
    const long maxLength;
    const QType minRowId;
    const QType maxRowId;
    const QType minColumnId;
    const QType maxColumnId;
public:
    CheckQueue(const QType minRowId, const QType maxRowId,
               const QType minColumnId, const QType maxColumnId);

    bool checkRow(const QType* queue, long length);

    bool checkCol(const QType* queue, long length);
};

#endif // CHECKQUEUE_H
