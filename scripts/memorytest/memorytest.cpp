#include <iostream>
#include <cstdint>
#include <cuda.h>

using namespace std;


size_t getCudaFreeMemory(size_t *total_bytes)
{
    size_t free_bytes;
    cudaError_t err;
    err = cudaMemGetInfo(&free_bytes, total_bytes);
    if (err != cudaSuccess) {
        cout << " cudaMemGetInfo returned the error: " << cudaGetErrorString(err) << endl;
        free_bytes = 0;
    }
    return free_bytes;
}


void synchronizeCudaDevice()
{
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        cout << "cudaDeviceSynchronize returned the error " << cudaGetErrorString(err) << endl;
        exit(1);
    }
}

int main(void)
{

    cudaError_t err;
    bool finished;
    size_t memorySize;
    size_t availableFreeSize, allocableSize, availableFreeSizeBefore, availableFreeSizeAfter, availableFreeSizeAfterFreeing;
    int i;

    finished = false;
    err = cudaMemGetInfo(&availableFreeSize, &allocableSize);
    if (err != cudaSuccess) {
        cout << " cudaMemGetInfo error: " << cudaGetErrorString(err) << endl;
        finished = true;
    }

    // cout << "Maximum total CUDA memory: " << allocableSize << " bytes" << endl;
    // cout << "Free CUDA memory: " << availableFreeSize << " bytes" << endl;
    err = cudaSetDevice(0);
    i = 0;

    memorySize = allocableSize;
    while (memorySize > 0 && !finished) {
        synchronizeCudaDevice();
        while (memorySize > 0 && !finished) {
            uint8_t *dynArray;
            uint8_t *cudaDynArray;
            dynArray = new (nothrow) uint8_t [memorySize];
            err = cudaMemGetInfo(&availableFreeSizeBefore, &allocableSize);
            err = cudaMalloc(&cudaDynArray, memorySize);
            if (err == cudaSuccess) {
                finished = true;
                err = cudaMemGetInfo(&availableFreeSizeAfter, &allocableSize);
                if (err != cudaSuccess) {
                    cout << " cudaMemGetInfo error: " << cudaGetErrorString(err) << endl;
                }
                cout << "CUDA allocated memory chunk " << memorySize << endl;
                cout << "Free CUDA memory. Before allocated " << availableFreeSizeBefore << " bytes" << endl;
                cout << "Free CUDA memory. After allocated " << availableFreeSizeAfter << " bytes" << endl;
            }
            ++i;
            cudaFree(cudaDynArray);
            if (finished) {
                err = cudaMemGetInfo(&availableFreeSizeAfterFreeing, &allocableSize);
                cout << "Free CUDA memory. After allocated and Freed " << availableFreeSizeAfterFreeing << " bytes" <<
                endl;
            }
            memorySize -= i;
            delete dynArray;
        }
        synchronizeCudaDevice();
        // cout << "iteraction: " << i << endl;
    }
    /*
    cout << "Maximum allocable CUDA memory: " << memorySize << " bytes" << endl;
    err = cudaMemGetInfo(&availableFreeSize, &memorySize);
    if (err != cudaSuccess) {
        cout << " cudaMemGetInfo error: " << cudaGetErrorString(err) << endl;
    }
    cout << "Free CUDA memory: " << availableFreeSize << " bytes" << endl;
     */

}