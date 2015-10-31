#include "bitlevelfunctions.h"

/* see "Reverse bits the obvious way" on
 * http://graphics.stanford.edu/~seander/bithacks.html
 */
unsigned int reverse(unsigned int value, const int bitSize)
{
    unsigned int result = 0; // r will be reversed bits of v
    int bits = bitSize;

    //first get LSB of v
    for (; value; value >>= 1)
    {
        result <<= 1;
        result |= value & 1;
        bits--; // extra shift needed at end
    }
    result <<= bits; // shift when v's highest bits are zero
    return result & ((1 << bitSize) - 1);  //mask out out of range bits
}


