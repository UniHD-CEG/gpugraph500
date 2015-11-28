/*
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#ifndef ADD_H_GUARD
#define ADD_H_GUARD


#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/count.h>
#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/adjacent_difference.h>
#include <thrust/binary_search.h>
#include <thrust/transform.h>
#include <thrust/partition.h>
#include <thrust/fill.h>
#include <thrust/scan.h>
#include <thrust/unique.h>
#include <thrust/gather.h>
#include <thrust/sort.h>
#include <thrust/merge.h>
#include <thrust/functional.h>
#include <thrust/system/cuda/experimental/pinned_allocator.h>
#include <queue>
#include <string>
#include <map>
#include <unordered_map>
#include <set>
#include <vector>
#include <stack>
#include <ctime>
#include <limits>
#include <fstream>

typedef long long int int_type;
typedef unsigned int int32_type;
typedef unsigned short int int16_type;
typedef unsigned char int8_type;
typedef double float_type;

using namespace std;
using namespace thrust::system::cuda::experimental;


extern size_t int_size;
extern size_t float_size;
extern bool delta;

/*
extern unsigned int hash_seed;
extern queue<string> op_type;
extern bool op_case;
extern queue<string> op_sort;
extern queue<string> op_presort;
extern queue<string> op_value;
extern queue<int_type> op_nums;
extern queue<float_type> op_nums_f;
extern queue<unsigned int> op_nums_precision; //decimals' precision
extern queue<string> col_aliases;
extern size_t total_count, oldCount, total_max, totalRecs, alloced_sz;
extern unsigned int total_segments;
extern size_t process_count;
extern bool fact_file_loaded;
extern void* alloced_tmp;
extern unsigned int partition_count;
extern map<string,string> setMap; //map to keep track of column names and set names
extern std::clock_t tot;
extern std::clock_t tot_disk;
extern bool verbose;
extern bool save_dict;
extern bool interactive;
extern bool ssd;
extern bool star;
extern map<string, char*> index_buffers;
extern map<string, char*> buffers;
extern map<string, size_t> buffer_sizes;
extern queue<string> buffer_names;
extern size_t total_buffer_size;
//extern thrust::device_vector<unsigned char> scratch;
extern thrust::device_vector<int> ranj;
extern size_t alloced_sz;
//extern ContextPtr context;
extern map<unsigned int, map<unsigned long long int, size_t> > char_hash; // mapping between column's string hashes and string positions
extern bool scan_state;
extern unsigned int statement_count;
extern map<string, map<string, bool> > used_vars;
//extern map<string, unsigned int> cpy_bits;
//extern map<string, long long int> cpy_init_val;
extern bool phase_copy;
extern map<string,bool> min_max_eq;
extern vector<void*> alloced_mem;
extern map<string, string> filter_var;


template<typename T>
struct uninitialized_host_allocator
        : std::allocator<T>
{
    // note that construct is annotated as
    __host__
    void construct(T *p)
    {
        // no-op
    }
};


template<typename T>
struct uninitialized_allocator
        : thrust::device_malloc_allocator<T>
{
    // note that construct is annotated as
    // a __host__ __device__ function
    __host__ __device__
    void construct(T *p)
    {
        // no-op
    }
};


struct set_minus : public binary_function<int,bool,int>
{
    //! Function call operator. The return value is <tt>lhs + rhs</tt>.
    __host__ __device__ int operator()(const int &lhs, const bool &rhs) const {
        if (rhs) return lhs;
        else return -1;
    }
};



template <typename HeadFlagType>
struct head_flag_predicate
        : public thrust::binary_function<HeadFlagType,HeadFlagType,bool>
{
    __host__ __device__
    bool operator()(HeadFlagType left, HeadFlagType right) const
    {
        return !left;
    }
};

struct float_to_long
{
    __device__
    long long int operator()(const float_type x)
    {
        return __double2ll_rn(x*100);
    }
};

struct float_equal_to
{
    __device__
    bool operator()(const float_type &lhs, const float_type &rhs) const {
        return (__double2ll_rn(lhs*100) == __double2ll_rn(rhs*100));
    }
};


struct int_upper_equal_to
{
    //! Function call operator. The return value is <tt>lhs == rhs</tt>.
    __host__ __device__ bool operator()(const int_type &lhs, const int_type &rhs) const {
        return (lhs >> 32)  == (rhs >> 32);
    }
};

struct float_upper_equal_to
{
    //! Function call operator. The return value is <tt>lhs == rhs</tt>.
    __host__ __device__ bool operator()(const float_type &lhs, const float_type &rhs) const {
        return ((int_type)lhs >> 32)  == ((int_type)rhs >> 32);
    }
};


struct long_to_float
{
    __host__ __device__
    float_type operator()(const long long int x)
    {
        return (((float_type)x)/100.0);
    }
};


struct is_break
{
 __host__ __device__
 bool operator()(const char x)
 {
   return x == '\n';
 }
};

struct gpu_date
{
    const char *source;
    long long int *dest;

    gpu_date(const char *_source, long long int *_dest):
              source(_source), dest(_dest) {}
    template <typename IndexType>
    __host__ __device__
    void operator()(const IndexType & i) {
        const char *s;
        long long int acc;
        int z = 0, c;

        s = source + 15*i;
        c = (unsigned char) *s++;

        for (acc = 0; z < 10; c = (unsigned char) *s++) {
            if(c != '-') {
                c -= '0';
                acc *= 10;
                acc += c;
            };
            z++;
        }
        dest[i] = acc;
    }
};


struct gpu_atof
{
    const char *source;
    double *dest;
    const unsigned int *len;

    gpu_atof(const char *_source, double *_dest, const unsigned int *_len):
              source(_source), dest(_dest), len(_len) {}
    template <typename IndexType>
    __host__ __device__
    void operator()(const IndexType & i) {
        const char *p;
        int frac;
        double sign, value, scale;

        p = source + len[0]*i;

        while (*p == ' ') {
            p += 1;
        }

        sign = 1.0;
        if (*p == '-') {
            sign = -1.0;
            p += 1;
        } else if (*p == '+') {
            p += 1;
        }

        for (value = 0.0; *p >= '0' && *p <= '9'; p += 1) {
            value = value * 10.0 + (*p - '0');
        }

        if (*p == '.') {
            double pow10 = 10.0;
            p += 1;
            while (*p >= '0' && *p <= '9') {
                value += (*p - '0') / pow10;
                pow10 *= 10.0;
                p += 1;
            }
        }

        frac = 0;
        scale = 1.0;

        dest[i] = sign * (frac ? (value / scale) : (value * scale));
    }
};



struct gpu_atod
{
    const char *source;
    int_type *dest;
    const unsigned int *len;
    const unsigned int *sc;

    gpu_atod(const char *_source, int_type *_dest, const unsigned int *_len, const unsigned int *_sc):
              source(_source), dest(_dest), len(_len), sc(_sc) {}
    template <typename IndexType>
    __host__ __device__
    void operator()(const IndexType & i) {
        const char *p;
        int frac;
        double sign, value, scale;

        p = source + len[0]*i;

        while (*p == ' ') {
            p += 1;
        }

        sign = 1.0;
        if (*p == '-') {
            sign = -1.0;
            p += 1;
        } else if (*p == '+') {
            p += 1;
        }

        for (value = 0.0; *p >= '0' && *p <= '9'; p += 1) {
            value = value * 10.0 + (*p - '0');
        }

        if (*p == '.') {
            double pow10 = 10.0;
            p += 1;
            while (*p >= '0' && *p <= '9') {
                value += (*p - '0') / pow10;
                pow10 *= 10.0;
                p += 1;
            }
        }

        frac = 0;
        scale = 1.0;

        dest[i] = (sign * (frac ? (value / scale) : (value * scale)))*sc[0];
    }
};


struct gpu_atold
{
    const char *source;
    long long int *dest;
    const unsigned int *len;
    const unsigned int *sc;

    gpu_atold(const char *_source, long long int *_dest, const unsigned int *_len, const unsigned int *_sc):
              source(_source), dest(_dest), len(_len), sc(_sc) {}
    template <typename IndexType>
    __host__ __device__
    void operator()(const IndexType & i) {
        const char *s;
        long long int acc;
        int c;
        int neg;
        int point = 0;
        bool cnt = 0;

        s = source + len[0]*i;

        do {
            c = (unsigned char) *s++;
        } while (c == ' ');

        if (c == '-') {
            neg = 1;
            c = *s++;
        } else {
            neg = 0;
            if (c == '+')
                c = *s++;
        }

        for (acc = 0;; c = (unsigned char) *s++) {
            if (c >= '0' && c <= '9')
                c -= '0';
            else {
                if(c != '.')
                    break;
                cnt = 1;
                continue;
            };
            if (c >= 10)
                break;
            if (neg) {
                acc *= 10;
                acc -= c;
            }
            else {
                acc *= 10;
                acc += c;
            }
            if(cnt)
                point++;
            if(point == sc[0])
                break;
        }
        dest[i] = acc * (unsigned int)exp10((double)sc[0]- point);
    }
};


struct gpu_atoll
{
    const char *source;
    long long int *dest;
    const unsigned int *len;

    gpu_atoll(const char *_source, long long int *_dest, const unsigned int *_len):
              source(_source), dest(_dest), len(_len) {}
    template <typename IndexType>
    __host__ __device__
    void operator()(const IndexType & i) {
        const char *s;
        long long int acc;
        int c;
        int neg;

        s = source + len[0]*i;

        do {
            c = (unsigned char) *s++;
        } while (c == ' ');

        if (c == '-') {
            neg = 1;
            c = *s++;
        } else {
            neg = 0;
            if (c == '+')
                c = *s++;
        }

        for (acc = 0;; c = (unsigned char) *s++) {
            if (c >= '0' && c <= '9')
                c -= '0';
            else
                break;
            if (c >= 10)
                break;
            if (neg) {
                acc *= 10;
                acc -= c;
            }
            else {
                acc *= 10;
                acc += c;
            }
        }
        dest[i] = acc;
    }
};


struct parse_functor
{
    const char *source;
    char **dest;
    const unsigned int *ind;
    const unsigned int *cnt;
    const char *separator;
    const long long int *src_ind;
    const unsigned int *dest_len;

    parse_functor(const char* _source, char** _dest, const unsigned int* _ind, const unsigned int* _cnt, const char* _separator,
                  const long long int* _src_ind, const unsigned int* _dest_len):
        source(_source), dest(_dest), ind(_ind), cnt(_cnt),  separator(_separator), src_ind(_src_ind), dest_len(_dest_len) {}

    template <typename IndexType>
    __host__ __device__
    void operator()(const IndexType & i) {
        unsigned int curr_cnt = 0, dest_curr = 0, j = 0, t, pos;
        pos = src_ind[i]+1;

        while(dest_curr < *cnt) {
            if(ind[dest_curr] == curr_cnt) { //process
                t = 0;
                while(source[pos+j] != *separator) {
                    //printf("REG %d ", j);
                    if(source[pos+j] != 0) {
                        dest[dest_curr][dest_len[dest_curr]*i+t] = source[pos+j];
                        t++;
                    };
                    j++;
                };
                j++;
                dest_curr++;
            }
            else {
                //printf("Skip %d \n", j);
                while(source[pos+j] != *separator) {
                    j++;
                    //printf("CONT Skip %d \n", j);
                };
                j++;
            };
            curr_cnt++;
            //printf("DEST CURR %d %d %d %d \n" , j, dest_curr, ind[dest_curr], curr_cnt);
        }

    }
};
*/

#endif // ADD_H_GUARD
