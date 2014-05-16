#ifndef CUDA_SUPPORT_HH
#define CUDA_SUPPORT_HH

#include <stdint.h>
#include <cstdio>
#include <typeinfo>


#include <b40c/util/basic_utils.cuh>

template<class T>
void PrintValue(T& val){
    fprintf(stderr,"Unable to print value of type %s.\n", typeid(T).name());
}

template<>
void PrintValue<uint64_t>(uint64_t& val){
    printf("%lX",val);
}

template<>
void PrintValue<int64_t>(int64_t& val){
    printf("%lX",val);
}

//template<>
//void PrintValue<long>(long& val){
//    printf("%ld",val);
//}

template<>
void PrintValue<long long>(long long& val){
    printf("%llX",val);
}

template<>
void PrintValue<unsigned long long>(unsigned long long& val){
    printf("%llX",val);
}

template<>
void PrintValue<int>(int& val){
    printf("%X",val);
}

template<>
void PrintValue<unsigned int>(unsigned int& val){
    printf("%X",val);
}

#endif // CUDA_SUPPORT_HH
