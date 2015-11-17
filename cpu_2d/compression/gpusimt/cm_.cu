/*
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

#include <cctype>
#include <algorithm>
#include <functional>
#include <numeric>
#include "cm.h"
#include "compress.cu"

/*
#include "atof.h"
#include "compress.cu"
#include "sorts.cu"
#include "filter.h"
#include "callbacks.h"
#include "zone_map.h"
*/
#include <unistd.h>

using namespace std;
using namespace thrust::placeholders;

size_t total_count = 0, total_max;
clock_t tot;
unsigned int total_segments = 0;
size_t process_count;
size_t alloced_sz = 0;
bool fact_file_loaded = 1;
bool verbose;
bool interactive, ssd, delta, star;
void* d_v = nullptr;
void* s_v = nullptr;
queue<string> op_sort;
queue<string> op_presort;
queue<string> op_type;
bool op_case = 0;
queue<string> op_value;
queue<int_type> op_nums;
queue<float_type> op_nums_f;
queue<unsigned int> op_nums_precision;
queue<string> col_aliases;
//map<string, map<string, col_data> > data_dict;
map<unsigned int, map<unsigned long long int, size_t> > char_hash;

map<string, char*> index_buffers;
map<string, char*> buffers;
map<string, size_t> buffer_sizes;
size_t total_buffer_size;
queue<string> buffer_names;

void* alloced_tmp;
bool alloced_switch = 0;
//map<string,CudaSet*> varNames; //  STL map to manage CudaSet variables
map<string, unsigned int> cpy_bits;
map<string, long long int> cpy_init_val;
char* readbuff = nullptr;


struct f_equal_to
{
    __host__ __device__
    bool operator()(const float_type x, const float_type y)
    {
        return (((x-y) < EPSILON) && ((x-y) > -EPSILON));
    }
};


struct f_less
{
    __host__ __device__
    bool operator()(const float_type x, const float_type y)
    {
        return ((y-x) > EPSILON);
    }
};

struct f_greater
{
    __host__ __device__
    bool operator()(const float_type x, const float_type y)
    {
        return ((x-y) > EPSILON);
    }
};

struct f_greater_equal_to
{
    __host__ __device__
    bool operator()(const float_type x, const float_type y)
    {
        return (((x-y) > EPSILON) || (((x-y) < EPSILON) && ((x-y) > -EPSILON)));
    }
};

struct f_less_equal
{
    __host__ __device__
    bool operator()(const float_type x, const float_type y)
    {
        return (((y-x) > EPSILON) || (((x-y) < EPSILON) && ((x-y) > -EPSILON)));
    }
};

struct f_not_equal_to
{
    __host__ __device__
    bool operator()(const float_type x, const float_type y)
    {
        return ((x-y) > EPSILON) || ((x-y) < -EPSILON);
    }
};


struct long_to_float_type
{
    __host__ __device__
    float_type operator()(const int_type x)
    {
        return (float_type)x;
    }
};

template <typename T>
struct power_functor : public thrust::unary_function<T,T>
{
    unsigned int a;

    __host__ __device__
    power_functor(unsigned int a_) { a = a_; }

    __host__ __device__
    T operator()(T x)
    {
		return x*(unsigned int)pow((double)10,(double)a);
	}
};

/*class power_functor {

    unsigned int a;

    public:

        power_functor(unsigned int a_) { a = a_; }

        __host__ __device__ int_type operator()(int_type x) const
        {
            return x*(unsigned int)pow((double)10,(double)a);
        }
};
*/


/*
void allocColumns(CudaSet* a, queue<string> fields);
void copyColumns(CudaSet* a, queue<string> fields, unsigned int segment, size_t& count, bool rsz, bool flt);
void mygather(unsigned int tindex, unsigned int idx, CudaSet* a, CudaSet* t, size_t count, size_t g_size);
void mycopy(unsigned int tindex, unsigned int idx, CudaSet* a, CudaSet* t, size_t count, size_t g_size);
void write_compressed_char(string file_name, unsigned int index, size_t mCount);
size_t getFreeMem();
size_t getTotalSystemMemory();
void process_error(int severity, string err);
*/





































/*
int_type reverse_op(int_type op_type)
{
    if (op_type == 2) // >
        return 5;
    else if (op_type == 1)  // <
        return 6;
    else if (op_type == 6) // >=
        return 1;
    else if (op_type == 5)  // <=
        return 2;
    else return op_type;
}


size_t getFreeMem()
{
    size_t available, total;
    cudaMemGetInfo(&available, &total);
    return available;
} ;



void allocColumns(CudaSet* a, queue<string> fields)
{
    if(a->filtered) {
        CudaSet* t;
        if(a->filtered)
            t = varNames[a->source_name];
        else
            t = a;

        if(int_size*t->maxRecs > alloced_sz) {
            if(alloced_sz) {
                cudaFree(alloced_tmp);
            };
            cudaMalloc((void **) &alloced_tmp, int_size*t->maxRecs);
            alloced_sz = int_size*t->maxRecs;
        }
    }
    else {
        while(!fields.empty()) {
            if(var_exists(a, fields.front()) && !a->onDevice(fields.front())) {
                a->allocColumnOnDevice(fields.front(), a->maxRecs);
            }
            fields.pop();
        };
    };
}



void gatherColumns(CudaSet* a, CudaSet* t, string field, unsigned int segment, size_t& count)
{
    if(!a->onDevice(field)) {
        a->allocColumnOnDevice(field, a->maxRecs);
    };
    if(a->prm_index == 'R') {
        mygather(field, a, t, count, a->mRecCount);
    }
    else {
        mycopy(field, a, t, count, t->mRecCount);
        a->mRecCount = t->mRecCount;
    };
}


void copyFinalize(CudaSet* a, queue<string> fields)
{
	set<string> uniques;
	if(scratch.size() < a->mRecCount*8)
		scratch.resize(a->mRecCount*8);
	thrust::device_ptr<int_type> tmp((int_type*)thrust::raw_pointer_cast(scratch.data()));

	while(!fields.empty()) {
        if (uniques.count(fields.front()) == 0 && var_exists(a, fields.front()) && cpy_bits.find(fields.front()) != cpy_bits.end())	{

			if(cpy_bits[fields.front()] == 8) {
				if(a->type[fields.front()] != 1) {
					thrust::device_ptr<char> src((char*)thrust::raw_pointer_cast(a->d_columns_int[fields.front()].data()));
					thrust::transform(src, src+a->mRecCount, tmp, to_int64<char>());
				}
				else {
					thrust::device_ptr<unsigned char> src((unsigned char*)thrust::raw_pointer_cast(a->d_columns_float[fields.front()].data()));
					thrust::transform(src, src+a->mRecCount, tmp, to_int64<unsigned char>());
				};
			}
			else if(cpy_bits[fields.front()] == 16) {
				if(a->type[fields.front()] != 1) {
					thrust::device_ptr<unsigned short int> src((unsigned short int*)thrust::raw_pointer_cast(a->d_columns_int[fields.front()].data()));
					thrust::transform(src, src+a->mRecCount, tmp, to_int64<unsigned short>());
				}
				else {
					thrust::device_ptr<unsigned short int> src((unsigned short int*)thrust::raw_pointer_cast(a->d_columns_float[fields.front()].data()));
					thrust::transform(src, src+a->mRecCount, tmp, to_int64<unsigned short>());
				};
			}
			else if(cpy_bits[fields.front()] == 32) {
				if(a->type[fields.front()] != 1) {
					thrust::device_ptr<unsigned int> src((unsigned int*)thrust::raw_pointer_cast(a->d_columns_int[fields.front()].data()));
					thrust::transform(src, src+a->mRecCount, tmp, to_int64<unsigned int>());
				}
				else {
					thrust::device_ptr<unsigned int> src((unsigned int*)thrust::raw_pointer_cast(a->d_columns_float[fields.front()].data()));
					thrust::transform(src, src+a->mRecCount, tmp, to_int64<unsigned int>());
				};
			}
			else {
				if(a->type[fields.front()] != 1) {
					thrust::device_ptr<int_type> src((int_type*)thrust::raw_pointer_cast(a->d_columns_int[fields.front()].data()));
					thrust::copy(src, src+a->mRecCount, tmp);
				}
				else {
					thrust::device_ptr<int_type> src((int_type*)thrust::raw_pointer_cast(a->d_columns_float[fields.front()].data()));
					thrust::copy(src, src+a->mRecCount, tmp);
				};
			};
			thrust::constant_iterator<int_type> iter(cpy_init_val[fields.front()]);
			if(a->type[fields.front()] != 1) {
				thrust::transform(tmp, tmp + a->mRecCount, iter, a->d_columns_int[fields.front()].begin(), thrust::plus<int_type>());
			}
			else {
				thrust::device_ptr<int_type> dest((int_type*)thrust::raw_pointer_cast(a->d_columns_float[fields.front()].data()));
				thrust::transform(tmp, tmp + a->mRecCount, iter, dest, thrust::plus<int_type>());
                thrust::transform(dest, dest+a->mRecCount, a->d_columns_float[fields.front()].begin(), long_to_float());
			};
		};
		uniques.insert(fields.front());
        fields.pop();
    };
}


void copyColumns(CudaSet* a, queue<string> fields, unsigned int segment, size_t& count, bool rsz, bool flt)
{
    set<string> uniques;
    if(a->filtered) { //filter the segment
        if(flt) {
            filter_op(a->fil_s, a->fil_f, segment);
        };
        if(rsz && a->mRecCount) {
            queue<string> fields1(fields);
            while(!fields1.empty()) {
                a->resizeDeviceColumn(a->devRecCount + a->mRecCount, fields1.front());
                fields1.pop();
            };
            a->devRecCount = a->devRecCount + a->mRecCount;
        };
    };
	cpy_bits.clear();
	cpy_init_val.clear();
	auto f(fields);

    while(!fields.empty()) {
        if (uniques.count(fields.front()) == 0 && var_exists(a, fields.front()))	{
            if(a->filtered) {
                if(a->mRecCount) {
                    CudaSet *t = varNames[a->source_name];
                    alloced_switch = 1;
                    t->CopyColumnToGpu(fields.front(), segment);
                    gatherColumns(a, t, fields.front(), segment, count);
                    alloced_switch = 0;
                };
            }
            else {
                if(a->mRecCount) {
                    a->CopyColumnToGpu(fields.front(), segment, count);
                };
            };
            uniques.insert(fields.front());
        };
        fields.pop();
    };
}


void mygather(string colname, CudaSet* a, CudaSet* t, size_t offset, size_t g_size)
{
    if(t->type[colname] != 1 ) {
		if(cpy_bits.find(colname) != cpy_bits.end()) { // non-delta compression
			if(cpy_bits[colname] == 8) {
					thrust::device_ptr<unsigned char> d_col_source((unsigned char*)alloced_tmp);
					thrust::device_ptr<unsigned char> d_col_dest((unsigned char*)thrust::raw_pointer_cast(a->d_columns_int[colname].data()));
					thrust::gather(a->prm_d.begin(), a->prm_d.begin() + g_size, d_col_source, d_col_dest + offset);
			}
			else if(cpy_bits[colname] == 16) {
					thrust::device_ptr<unsigned short int> d_col_source((unsigned short int*)alloced_tmp);
					thrust::device_ptr<unsigned short int> d_col_dest((unsigned short int*)thrust::raw_pointer_cast(a->d_columns_int[colname].data()));
					thrust::gather(a->prm_d.begin(), a->prm_d.begin() + g_size, d_col_source, d_col_dest + offset);
			}
			else if(cpy_bits[colname] == 32) {
					thrust::device_ptr<unsigned int> d_col_source((unsigned int*)alloced_tmp);
					thrust::device_ptr<unsigned int> d_col_dest((unsigned int*)thrust::raw_pointer_cast(a->d_columns_int[colname].data()));
					thrust::gather(a->prm_d.begin(), a->prm_d.begin() + g_size, d_col_source, d_col_dest + offset);
			}
			else if(cpy_bits[colname] == 64) {
					thrust::device_ptr<int_type> d_col((int_type*)alloced_tmp);
					thrust::gather(a->prm_d.begin(), a->prm_d.begin() + g_size, d_col, a->d_columns_int[colname].begin() + offset);
			};
		}
		else {
			thrust::device_ptr<int_type> d_col((int_type*)alloced_tmp);
			thrust::gather(a->prm_d.begin(), a->prm_d.begin() + g_size, d_col, a->d_columns_int[colname].begin() + offset);
		};

    }
    else  {
		if(cpy_bits.find(colname) != cpy_bits.end()) { // non-delta compression
			if(cpy_bits[colname] == 8) {
					thrust::device_ptr<unsigned char> d_col_source((unsigned char*)alloced_tmp);
					thrust::device_ptr<unsigned char> d_col_dest((unsigned char*)thrust::raw_pointer_cast(a->d_columns_float[colname].data()));
					thrust::gather(a->prm_d.begin(), a->prm_d.begin() + g_size, d_col_source, d_col_dest + offset);
			}
			else if(cpy_bits[colname] == 16) {
					thrust::device_ptr<unsigned short int> d_col_source((unsigned short int*)alloced_tmp);
					thrust::device_ptr<unsigned short int> d_col_dest((unsigned short int*)thrust::raw_pointer_cast(a->d_columns_float[colname].data()));
					thrust::gather(a->prm_d.begin(), a->prm_d.begin() + g_size, d_col_source, d_col_dest + offset);
			}
			else if(cpy_bits[colname] == 32) {
					thrust::device_ptr<unsigned int> d_col_source((unsigned int*)alloced_tmp);
					thrust::device_ptr<unsigned int> d_col_dest((unsigned int*)thrust::raw_pointer_cast(a->d_columns_float[colname].data()));
					thrust::gather(a->prm_d.begin(), a->prm_d.begin() + g_size, d_col_source, d_col_dest + offset);
			}
			else if(cpy_bits[colname] == 64) {
					thrust::device_ptr<int_type> d_col((int_type*)alloced_tmp);
					thrust::gather(a->prm_d.begin(), a->prm_d.begin() + g_size, d_col, a->d_columns_float[colname].begin() + offset);
			};
		}
		else {
			thrust::device_ptr<float_type> d_col((float_type*)alloced_tmp);
			thrust::gather(a->prm_d.begin(), a->prm_d.begin() + g_size, d_col, a->d_columns_float[colname].begin() + offset);
		};
    }
};


void mycopy(string colname, CudaSet* a, CudaSet* t, size_t offset, size_t g_size)
{
    if(t->type[colname] != 1) {
		if(cpy_bits.find(colname) != cpy_bits.end()) { // non-delta compression
			if(cpy_bits[colname] == 8) {
					thrust::device_ptr<unsigned char> d_col_source((unsigned char*)alloced_tmp);
					thrust::device_ptr<unsigned char> d_col_dest((unsigned char*)thrust::raw_pointer_cast(a->d_columns_int[colname].data()));
					thrust::copy(d_col_source, d_col_source + g_size, d_col_dest + offset);
			}
			else if(cpy_bits[colname] == 16) {
					thrust::device_ptr<short int> d_col_source((short int*)alloced_tmp);
					thrust::device_ptr<short int> d_col_dest((short int*)thrust::raw_pointer_cast(a->d_columns_int[colname].data()+offset));
					thrust::copy(d_col_source, d_col_source + g_size, d_col_dest + offset);
			}
			else if(cpy_bits[colname] == 32) {
					thrust::device_ptr<unsigned int> d_col_source((unsigned int*)alloced_tmp);
					thrust::device_ptr<unsigned int> d_col_dest((unsigned int*)thrust::raw_pointer_cast(a->d_columns_int[colname].data()));
					thrust::copy(d_col_source, d_col_source + g_size, d_col_dest + offset);
			}
			else if(cpy_bits[colname] == 64) {
					thrust::device_ptr<int_type> d_col_source((int_type*)alloced_tmp);
					thrust::copy(d_col_source, d_col_source + g_size, a->d_columns_int[colname].begin() + offset);
			};
		}
		else {
			thrust::device_ptr<int_type> d_col((int_type*)alloced_tmp);
			thrust::copy(d_col, d_col + g_size, a->d_columns_int[colname].begin() + offset);
		};
    }
    else {
		if(cpy_bits.find(colname) != cpy_bits.end()) { // non-delta compression
			if(cpy_bits[colname] == 8) {
					thrust::device_ptr<unsigned char> d_col_source((unsigned char*)alloced_tmp);
					thrust::device_ptr<unsigned char> d_col_dest((unsigned char*)thrust::raw_pointer_cast(a->d_columns_float[colname].data()));
					thrust::copy(d_col_source, d_col_source + g_size, d_col_dest + offset);
			}
			else if(cpy_bits[colname] == 16) {
					thrust::device_ptr<short int> d_col_source((short int*)alloced_tmp);
					thrust::device_ptr<short int> d_col_dest((short int*)thrust::raw_pointer_cast(a->d_columns_float[colname].data()+offset));
					thrust::copy(d_col_source, d_col_source + g_size, d_col_dest + offset);
			}
			else if(cpy_bits[colname] == 32) {
					thrust::device_ptr<unsigned int> d_col_source((unsigned int*)alloced_tmp);
					thrust::device_ptr<unsigned int> d_col_dest((unsigned int*)thrust::raw_pointer_cast(a->d_columns_float[colname].data()));
					thrust::copy(d_col_source, d_col_source + g_size, d_col_dest + offset);
			}
			else if(cpy_bits[colname] == 64) {
					thrust::device_ptr<int_type> d_col_source((int_type*)alloced_tmp);
					thrust::copy(d_col_source, d_col_source + g_size, a->d_columns_float[colname].begin() + offset);
			};
		}
		else {
			thrust::device_ptr<float_type> d_col((float_type*)alloced_tmp);
			thrust::copy(d_col, d_col + g_size,	a->d_columns_float[colname].begin() + offset);
		};
	};
};



size_t load_queue(queue<string> c1, CudaSet* right, string f2, size_t &rcount,
                  unsigned int start_segment, unsigned int end_segment, bool rsz, bool flt)
{
    queue<string> cc;
    while(!c1.empty()) {
        if(std::find(right->columnNames.begin(), right->columnNames.end(), c1.front()) !=  right->columnNames.end()) {
            if(f2 != c1.front() ) {
                cc.push(c1.front());
            };
        };
        c1.pop();
    };
    if(std::find(right->columnNames.begin(), right->columnNames.end(), f2) !=  right->columnNames.end()) {
        cc.push(f2);
    };

    if(right->filtered) {
        allocColumns(right, cc);
    };

    rcount = right->maxRecs;
    queue<string> ct(cc);

    while(!ct.empty()) {
        if(right->filtered && rsz) {
            right->mRecCount = 0;
        }
        else {
            right->allocColumnOnDevice(ct.front(), rcount*right->segCount);
        };
        ct.pop();
    };

    size_t cnt_r = 0;
    right->devRecCount = 0;
    for(unsigned int i = start_segment; i < end_segment; i++) {
        if(!right->filtered)
            copyColumns(right, cc, i, cnt_r, rsz, 0);
        else
            copyColumns(right, cc, i, cnt_r, rsz, flt);
        cnt_r = cnt_r + right->mRecCount;
    };

    right->mRecCount = cnt_r;
    return cnt_r;

}

size_t max_char(CudaSet* a)
{
    size_t max_char1 = 8;
    for(unsigned int i = 0; i < a->columnNames.size(); i++) {
        if(a->type[a->columnNames[i]] == 2) {
            if (a->char_size[a->columnNames[i]] > max_char1)
                max_char1 = a->char_size[a->columnNames[i]];
        }
        else if(a->type[a->columnNames[i]] == 0 && a->string_map.find(a->columnNames[i]) != a->string_map.end()) {
            auto s = a->string_map[a->columnNames[i]];
            auto pos = s.find_first_of(".");
            auto len = data_dict[s.substr(0, pos)][s.substr(pos+1)].col_length;
            if (len > max_char1)
                max_char1 = len;
        };
    };
    return max_char1;
};


size_t max_char(CudaSet* a, queue<string> field_names)
{
    size_t max_char = 8;
    while (!field_names.empty()) {
        if (a->type[field_names.front()] == 2) {
            if (a->char_size[field_names.front()] > max_char)
                max_char = a->char_size[field_names.front()];
        };
        field_names.pop();
    };
    return max_char;
};


void setSegments(CudaSet* a, queue<string> cols)
{
    size_t mem_available = getFreeMem();
    size_t tot_sz = 0;
    while(!cols.empty()) {
        if(a->type[cols.front()] != 2)
            tot_sz = tot_sz + int_size;
        else
            tot_sz = tot_sz + a->char_size[cols.front()];
        cols.pop();
    };
    if(a->mRecCount*tot_sz > mem_available/3) { //default is 3
        a->segCount = (a->mRecCount*tot_sz)/(mem_available/5) + 1;
        a->maxRecs = (a->mRecCount/a->segCount)+1;
    };
};


void update_permutation_char_host(char* key, unsigned int* permutation, size_t RecCount, string SortType, char* tmp, unsigned int len)
{
    str_gather_host(permutation, RecCount, (void*)key, (void*)tmp, len);

    if (SortType.compare("DESC") == 0 )
        str_sort_host(tmp, RecCount, permutation, 1, len);
    else
        str_sort_host(tmp, RecCount, permutation, 0, len);
}


void apply_permutation_char(char* key, unsigned int* permutation, size_t RecCount, char* tmp, unsigned int len)
{
    // copy keys to temporary vector
    cudaMemcpy( (void*)tmp, (void*) key, RecCount*len, cudaMemcpyDeviceToDevice);
    // permute the keys
    str_gather((void*)permutation, RecCount, (void*)tmp, (void*)key, len);
}


void apply_permutation_char_host(char* key, unsigned int* permutation, size_t RecCount, char* res, unsigned int len)
{
    str_gather_host(permutation, RecCount, (void*)key, (void*)res, len);
}


void filter_op(const char *s, const char *f, unsigned int segment)
{
    CudaSet *a, *b;

    a = varNames.find(f)->second;
    a->name = f;
    //std::clock_t start1 = std::clock();

    if(a->mRecCount == 0 && !a->filtered) {
        b = new CudaSet(0,1);
    }
    else {
        if(verbose)
            cout << "FILTER " << s << " " << f << " " << getFreeMem() << '\xd';

        b = varNames[s];
        b->name = s;
        b->string_map = a->string_map;
        size_t cnt = 0;
        allocColumns(a, b->fil_value);

        if (b->prm_d.size() == 0) {
            b->prm_d.resize(a->maxRecs);
		};

        //cout << endl << "MAP CHECK start " << segment <<  endl;
        char map_check = zone_map_check(b->fil_type,b->fil_value,b->fil_nums, b->fil_nums_f, b->fil_nums_precision, a, segment);
        //cout << endl << "MAP CHECK segment " << segment << " " << map_check <<  endl;

        if(map_check == 'R') {
			auto old_ph = phase_copy;
			phase_copy = 0;
            copyColumns(a, b->fil_value, segment, cnt);
			phase_copy = old_ph;
            bool* res = filter(b->fil_type,b->fil_value,b->fil_nums, b->fil_nums_f, b->fil_nums_precision, a, segment);
            thrust::device_ptr<bool> bp((bool*)res);
            b->prm_index = 'R';
            b->mRecCount = thrust::count(bp, bp + (unsigned int)a->mRecCount, 1);
            thrust::copy_if(thrust::make_counting_iterator((unsigned int)0), thrust::make_counting_iterator((unsigned int)a->mRecCount),
                            bp, b->prm_d.begin(), thrust::identity<bool>());
            cudaFree(res);
        }
        else  {
			b->prm_index = map_check;
			if(map_check == 'A')
				b->mRecCount = a->mRecCount;
			else
				b->mRecCount = 0;
        };
        if(segment == a->segCount-1)
            a->deAllocOnDevice();
    }
    if(verbose)
        cout << endl << "filter res " << b->mRecCount << " " << phase_copy << endl;
}




size_t load_right(CudaSet* right, string f2, queue<string> op_g, queue<string> op_alt, size_t& rcount, unsigned int start_seg, unsigned int end_seg) {

    size_t cnt_r = 0;
    //if join is on strings then add integer columns to left and right tables and modify colInd1 and colInd2

    // need to allocate all right columns
    if(right->not_compressed) {
        queue<string> op_alt1;
        op_alt1.push(f2);
        cnt_r = load_queue(op_alt1, right, "", rcount, start_seg, end_seg, 1, 1);

        queue<string> op_alt2;
        while(!op_alt.empty()) {
            if(f2.compare(op_alt.front())) {
                if (std::find(right->columnNames.begin(), right->columnNames.end(), op_alt.front()) != right->columnNames.end()) {
                    op_alt2.push(op_alt.front());
                };
            };
            op_alt.pop();
        };
        if(!op_alt2.empty())
            cnt_r = load_queue(op_alt2, right, "", rcount, start_seg, end_seg, 0, 0);
    }
    else {
        cnt_r = load_queue(op_alt, right, f2, rcount, start_seg, end_seg, 1, 1);
    };

    return cnt_r;
};



void insert_records(const char* f, const char* s) {
    char buf[4096];
    size_t size, maxRecs, cnt = 0;
    string str_s, str_d;

    if(varNames.find(s) == varNames.end()) {
        process_error(3, "couldn't find " + string(s) );
    };
    CudaSet *a;
    a = varNames.find(s)->second;
    a->name = s;

    if(varNames.find(f) == varNames.end()) {
        process_error(3, "couldn't find " + string(f) );
    };

    CudaSet *b;
    b = varNames.find(f)->second;
    b->name = f;

    // if both source and destination are on disk
    cout << "SOURCES " << a->source << ":" << b->source << endl;
    if(a->source && b->source) {
        for(unsigned int i = 0; i < a->segCount; i++) {
            for(unsigned int z = 0; z < a->columnNames.size(); z++) {

				if(a->type[a->columnNames[z]] != 2) {
					str_s = a->load_file_name + "." + a->columnNames[z] + "." + to_string(i);
					str_d = b->load_file_name + "." + a->columnNames[z] + "." + to_string(b->segCount + i);
					cout << str_s << " " << str_d << endl;
					FILE* source = fopen(str_s.c_str(), "rb");
					FILE* dest = fopen(str_d.c_str(), "wb");
					while (size = fread(buf, 1, BUFSIZ, source)) {
						fwrite(buf, 1, size, dest);
					}
					fclose(source);
					fclose(dest);
				}
				else { //merge strings
					//read b's strings
					str_s = b->load_file_name + "." + b->columnNames[z];
					FILE* dest = fopen(str_s.c_str(), "rb");
					auto len = b->char_size[b->columnNames[z]];
					map<string, unsigned long long int> map_d;
					buf[len] = 0;
					unsigned long long cnt = 0;
					while (fread(buf, len, 1, dest)) {
						map_d[buf] = cnt;
						cnt++;
					};
					fclose(dest);
					unsigned long long int cct = cnt;

					str_s = a->load_file_name + "." + a->columnNames[z] + "." + to_string(i) + ".hash";
					str_d = b->load_file_name + "." + b->columnNames[z] + "." + to_string(b->segCount + i) + ".hash";
					FILE* source = fopen(str_s.c_str(), "rb");
					dest = fopen(str_d.c_str(), "wb");
					while (size = fread(buf, 1, BUFSIZ, source)) {
						fwrite(buf, 1, size, dest);
					}
					fclose(source);
					fclose(dest);

					str_s = a->load_file_name + "." + a->columnNames[z];
					source = fopen(str_s.c_str(), "rb");
					map<unsigned long long int, string> map_s;
					buf[len] = 0;
					cnt = 0;
					while (fread(buf, len, 1, source)) {
						map_s[cnt] = buf;
						cnt++;
					};
					fclose(source);

					queue<string> op_vx;
					op_vx.push(a->columnNames[z]);
					allocColumns(a, op_vx);
					a->resize(a->maxRecs);
					a->CopyColumnToGpu(a->columnNames[z], z, 0);
					a->CopyColumnToHost(a->columnNames[z]);

					str_d = b->load_file_name + "." + b->columnNames[z];
                    fstream f_file;
                    f_file.open(str_d.c_str(), ios::out|ios::app|ios::binary);

					for(auto j = 0; j < a->mRecCount; j++) {
						auto ss = map_s[a->h_columns_int[a->columnNames[z]][j]];
						if(map_d.find(ss) == map_d.end()) { //add
							f_file.write((char *)ss.c_str(), len);
							a->h_columns_int[a->columnNames[z]][j] = cct;
							cct++;
						}
						else {
							a->h_columns_int[a->columnNames[z]][j] = map_d[ss];
						};
					};
					f_file.close();

					thrust::device_vector<int_type> d_col(a->mRecCount);
					thrust::copy(a->h_columns_int[a->columnNames[z]].begin(), a->h_columns_int[a->columnNames[z]].begin() + a->mRecCount, d_col.begin());
					auto i_name = b->load_file_name + "." + b->columnNames[z] + "." + to_string(b->segCount + i) + ".idx";
					pfor_compress(thrust::raw_pointer_cast(d_col.data()), a->mRecCount*int_size, i_name, a->h_columns_int[a->columnNames[z]], 0);
				};
            };
        };

        if(a->maxRecs > b->maxRecs)
            maxRecs = a->maxRecs;
        else
            maxRecs = b->maxRecs;

        for(unsigned int i = 0; i < b->columnNames.size(); i++) {
            b->reWriteHeader(b->load_file_name, b->columnNames[i], a->segCount + b->segCount, a->totalRecs + b->totalRecs, maxRecs);
        };
    }
    else if(!a->source && !b->source) { //if both source and destination are in memory
        size_t oldCount = b->mRecCount;
        b->resize(a->mRecCount);
        for(unsigned int z = 0; z< b->mColumnCount; z++) {
            if(b->type[a->columnNames[z]] == 0) {
                thrust::copy(a->h_columns_int[a->columnNames[z]].begin(), a->h_columns_int[a->columnNames[z]].begin() + a->mRecCount, b->h_columns_int[b->columnNames[z]].begin() + oldCount);
            }
            else if(b->type[a->columnNames[z]] == 1) {
                thrust::copy(a->h_columns_float[a->columnNames[z]].begin(), a->h_columns_float[a->columnNames[z]].begin() + a->mRecCount, b->h_columns_float[b->columnNames[z]].begin() + oldCount);
            }
            else {
                cudaMemcpy(b->h_columns_char[b->columnNames[z]] + b->char_size[b->columnNames[z]]*oldCount, a->h_columns_char[a->columnNames[z]], a->char_size[a->columnNames[z]]*a->mRecCount, cudaMemcpyHostToHost);
            };
        };
    }
    else if(!a->source && b->source) {

        total_segments = b->segCount;
        total_count = b->mRecCount;
        total_max = b->maxRecs;;

        queue<string> op_vx;
        for(unsigned int i=0; i < a->columnNames.size(); i++)
            op_vx.push(a->columnNames[i]);

        allocColumns(a, op_vx);
        a->resize(a->maxRecs);
        for(unsigned int i = 0; i < a->segCount; i++) {
            if (a->filtered) {
                copyColumns(a, op_vx, i, cnt);
                a->CopyToHost(0, a->mRecCount);
            };
            a->compress(b->load_file_name, 0, 1, i - (a->segCount-1), a->mRecCount);
        };
        for(unsigned int i = 0; i < b->columnNames.size(); i++) {
            b->writeHeader(b->load_file_name, b->columnNames[i], total_segments);
        };
    };
};



void delete_records(const char* f) {

    CudaSet *a;
    a = varNames.find(f)->second;
    a->name = f;
    size_t totalRemoved = 0;
    size_t maxRecs = 0;

    if(!a->keep) { // temporary variable
        process_error(2, "Delete operator is only applicable to disk based sets\nfor deleting records from derived sets please use filter operator ");
    }
    else {  // read matching segments, delete, compress and write on a disk replacing the original segments

        string str, str_old;
        queue<string> op_vx;
        size_t cnt;
        for ( auto it=data_dict[a->load_file_name].begin() ; it != data_dict[a->load_file_name].end(); ++it ) {
            op_vx.push((*it).first);
            if (std::find(a->columnNames.begin(), a->columnNames.end(), (*it).first) == a->columnNames.end()) {

                if ((*it).second.col_type == 0) {
                    a->type[(*it).first] = 0;
                    a->decimal[(*it).first] = 0;
                    //a->h_columns_int[(*it).first] = thrust::host_vector<int_type, pinned_allocator<int_type> >();
                    a->h_columns_int[(*it).first] = thrust::host_vector<int_type>();
                    a->d_columns_int[(*it).first] = thrust::device_vector<int_type>();
                }
                else if((*it).second.col_type == 1) {
                    a->type[(*it).first] = 1;
                    a->decimal[(*it).first] = 0;
                    //a->h_columns_float[(*it).first] = thrust::host_vector<float_type, pinned_allocator<float_type> >();
                    a->h_columns_float[(*it).first] = thrust::host_vector<float_type>();
                    a->d_columns_float[(*it).first] = thrust::device_vector<float_type>();
                }
                else if ((*it).second.col_type == 3) {
                    a->type[(*it).first] = 1;
                    a->decimal[(*it).first] = 1;
                    //a->h_columns_float[(*it).first] = thrust::host_vector<float_type, pinned_allocator<float_type> >();
                    a->h_columns_float[(*it).first] = thrust::host_vector<float_type>();
                    a->d_columns_float[(*it).first] = thrust::device_vector<float_type>();
                }
                else {
                    a->type[(*it).first] = 2;
                    a->decimal[(*it).first] = 0;
                    a->h_columns_char[(*it).first] = nullptr;
                    a->d_columns_char[(*it).first] = nullptr;
                    a->char_size[(*it).first] = (*it).second.col_length;
                };
                a->columnNames.push_back((*it).first);
            }
        };

        allocColumns(a, op_vx);
        a->resize(a->maxRecs);
        a->prm_d.resize(a->maxRecs);
        size_t cc = a->mRecCount;
        size_t tmp;

        void* d;
        CUDA_SAFE_CALL(cudaMalloc((void **) &d, a->maxRecs*float_size));
        unsigned int new_seg_count = 0;
        char map_check;

        for(unsigned int i = 0; i < a->segCount; i++) {

            map_check = zone_map_check(op_type,op_value,op_nums, op_nums_f, op_nums_precision, a, i);
            if(verbose)
                cout << "MAP CHECK segment " << i << " " << map_check <<  endl;
            if(map_check != 'N') {

                cnt = 0;
                copyColumns(a, op_vx, i, cnt);
                tmp = a->mRecCount;

                if(a->mRecCount) {
                    bool* res = filter(op_type,op_value,op_nums, op_nums_f, op_nums_precision, a, i);
                    thrust::device_ptr<bool> bp((bool*)res);
                    thrust::copy_if(thrust::make_counting_iterator((unsigned int)0), thrust::make_counting_iterator((unsigned int)a->mRecCount),
                                    bp, a->prm_d.begin(), thrust::logical_not<bool>());

                    a->mRecCount = thrust::count(bp, bp + (unsigned int)a->mRecCount, 0);
                    cudaFree(res);

//					cout << "Remained recs count " << a->mRecCount << endl;
                    if(a->mRecCount > maxRecs)
                        maxRecs = a->mRecCount;

                    if (a->mRecCount) {

                        totalRemoved = totalRemoved + (tmp - a->mRecCount);
                        if (a->mRecCount == tmp) { //none deleted
                            if(new_seg_count != i) {
                                for (auto it=data_dict[a->load_file_name].begin() ; it != data_dict[a->load_file_name].end(); ++it ) {
                                    auto colname = (*it).first;
                                    str_old = a->load_file_name + "." + colname + "." + to_string(i);
                                    str = a->load_file_name + "." + colname + "." + to_string(new_seg_count);
                                    remove(str.c_str());
                                    rename(str_old.c_str(), str.c_str());
                                };
                            };
                            new_seg_count++;

                        }
                        else { //some deleted
                            //cout << "writing segment " << new_seg_count << endl;

                            map<string, col_data> s = data_dict[a->load_file_name];
                            for ( map<string, col_data>::iterator it=s.begin() ; it != s.end(); ++it ) {
                                string colname = (*it).first;
                                str = a->load_file_name + "." + colname + "." + to_string(new_seg_count);

                                if(a->type[colname] == 0) {
                                    thrust::device_ptr<int_type> d_col((int_type*)d);
                                    thrust::gather(a->prm_d.begin(), a->prm_d.begin() + a->mRecCount, a->d_columns_int[colname].begin(), d_col);
                                    pfor_compress( d, a->mRecCount*int_size, str, a->h_columns_int[colname], 0);
                                }
                                else if(a->type[colname] == 1) {
                                    thrust::device_ptr<float_type> d_col((float_type*)d);
                                    if(a->decimal[colname]) {
                                        thrust::gather(a->prm_d.begin(), a->prm_d.begin() + a->mRecCount, a->d_columns_float[colname].begin(), d_col);
                                        thrust::device_ptr<long long int> d_col_dec((long long int*)d);
                                        thrust::transform(d_col,d_col+a->mRecCount, d_col_dec, float_to_long());
                                        pfor_compress( d, a->mRecCount*float_size, str, a->h_columns_float[colname], 1);
                                    }
                                    else {
                                        thrust::gather(a->prm_d.begin(), a->prm_d.begin() + a->mRecCount, a->d_columns_float[colname].begin(), d_col);
                                        thrust::copy(d_col, d_col + a->mRecCount, a->h_columns_float[colname].begin());
                                        fstream binary_file(str.c_str(),ios::out|ios::binary);
                                        binary_file.write((char *)&a->mRecCount, 4);
                                        binary_file.write((char *)(a->h_columns_float[colname].data()),a->mRecCount*float_size);
                                        unsigned int comp_type = 3;
                                        binary_file.write((char *)&comp_type, 4);
                                        binary_file.close();

                                    };
                                }
                                else {
                                    thrust::device_ptr<int_type> d_col((int_type*)d);
                                    thrust::gather(a->prm_d.begin(), a->prm_d.begin() + a->mRecCount, a->d_columns_int[colname].begin(), d_col);
                                    pfor_compress( d, a->mRecCount*int_size, str + ".hash", a->h_columns_int[colname], 0);
                                };
                            };
                            new_seg_count++;
                        };
                    }
                    else {
                        totalRemoved = totalRemoved + tmp;
                    };
                }
            }
            else {
                if(new_seg_count != i) {
                    for(unsigned int z = 0; z < a->columnNames.size(); z++) {
                        str_old = a->load_file_name + "." + a->columnNames[z] + "." + to_string(i);
                        str = a->load_file_name + "." + a->columnNames[z] + "." + to_string(new_seg_count);
                        remove(str.c_str());
                        rename(str_old.c_str(), str.c_str());
                    };
                };
                new_seg_count++;
                maxRecs	= a->maxRecs;
            };
        };

        if (new_seg_count < a->segCount) {
            for(unsigned int i = new_seg_count; i < a->segCount; i++) {
                //cout << "delete segment " << i << endl;
                for(unsigned int z = 0; z < a->columnNames.size(); z++) {
                    str = a->load_file_name + "." + a->columnNames[z];
                    str += "." + to_string(i);
                    remove(str.c_str());
                };
            };
        };

        for(unsigned int i = new_seg_count; i < a->segCount; i++) {
            a->reWriteHeader(a->load_file_name, a->columnNames[i], new_seg_count, a->totalRecs-totalRemoved, maxRecs);
        };


        a->mRecCount = cc;
        a->prm_d.resize(0);
        a->segCount = new_seg_count;
        a->deAllocOnDevice();
        cudaFree(d);
    };


};


void save_col_data(map<string, map<string, col_data> >& data_dict, string file_name)
{
    size_t str_len;
    fstream binary_file(file_name.c_str(),ios::out|ios::binary|ios::trunc);
    size_t len = data_dict.size();
    binary_file.write((char *)&len, 8);
    for (auto it=data_dict.begin() ; it != data_dict.end(); ++it ) {
        str_len = (*it).first.size();
        binary_file.write((char *)&str_len, 8);
        binary_file.write((char *)(*it).first.data(), str_len);
        map<string, col_data> s = (*it).second;
        size_t len1 = s.size();
        binary_file.write((char *)&len1, 8);

        for (auto sit=s.begin() ; sit != s.end(); ++sit ) {
            str_len = (*sit).first.size();
            binary_file.write((char *)&str_len, 8);
            binary_file.write((char *)(*sit).first.data(), str_len);
            binary_file.write((char *)&(*sit).second.col_type, 4);
            binary_file.write((char *)&(*sit).second.col_length, 4);
        };
    };
    binary_file.close();
}

void load_col_data(map<string, map<string, col_data> >& data_dict, string file_name)
{
    size_t str_len, recs, len1;
    string str1, str2;
    char buffer[4000];
    unsigned int col_type, col_length;
    fstream binary_file;
    binary_file.open(file_name.c_str(),ios::in|ios::binary);
    if(binary_file.is_open()) {
        binary_file.read((char*)&recs, 8);
        for(unsigned int i = 0; i < recs; i++) {
            binary_file.read((char*)&str_len, 8);
            binary_file.read(buffer, str_len);
            str1.assign(buffer, str_len);
            binary_file.read((char*)&len1, 8);

            for(unsigned int j = 0; j < len1; j++) {
                binary_file.read((char*)&str_len, 8);
                binary_file.read(buffer, str_len);
                str2.assign(buffer, str_len);
                binary_file.read((char*)&col_type, 4);
                binary_file.read((char*)&col_length, 4);
                data_dict[str1][str2].col_type = col_type;
                data_dict[str1][str2].col_length = col_length;
                //cout << "data DICT " << str1 << " " << str2 << " " << col_type << " " << col_length << endl;
            };
        };
        binary_file.close();
    }
    else {
        cout << "Couldn't open data dictionary" << endl;
    };
}

bool var_exists(CudaSet* a, string name) {

    if(std::find(a->columnNames.begin(), a->columnNames.end(), name) !=  a->columnNames.end())
        return 1;
    else

        return 0;
}

int file_exist (const char *filename)
{
    std::ifstream infile(filename);
    return infile.good();
}

bool check_bitmap_file_exist(CudaSet* left, CudaSet* right)
{
    queue<string> cols(right->fil_value);
    bool bitmaps_exist = 1;

    if(cols.size() == 0) {
        bitmaps_exist = 0;
    };
    while(cols.size() ) {
        if (std::find(right->columnNames.begin(), right->columnNames.end(), cols.front()) != right->columnNames.end()) {
            string fname = left->load_file_name + "."  + right->load_file_name + "." + cols.front() + ".0";
            if( !file_exist(fname.c_str())) {
                bitmaps_exist = 0;
            };
        };
        cols.pop();
    };
    return bitmaps_exist;
}

bool check_bitmaps_exist(CudaSet* left, CudaSet* right)
{
    //check if there are join bitmap indexes
    queue<string> cols(right->fil_value);
    bool bitmaps_exist = 1;

    if(cols.size() == 0) {
        bitmaps_exist = 1;
        return 1;
    };
    while(cols.size() ) {
        if (std::find(right->columnNames.begin(), right->columnNames.end(), cols.front()) != right->columnNames.end()) {
            string fname = left->load_file_name + "."  + right->load_file_name + "." + cols.front() + ".0";
            if( !file_exist(fname.c_str())) {
                bitmaps_exist = 0;
            };
        };
        cols.pop();
    };
    if(bitmaps_exist) {
        while(!right->fil_nums.empty() ) {
            left->fil_nums.push(right->fil_nums.front());
            right->fil_nums.pop();
        };
        while(!right->fil_nums_f.empty() ) {
            left->fil_nums_f.push(right->fil_nums_f.front());
            right->fil_nums_f.pop();
        };
        while(!right->fil_value.empty() ) {
            if (std::find(right->columnNames.begin(), right->columnNames.end(), right->fil_value.front()) != right->columnNames.end()) {
                string fname = left->load_file_name + "."  + right->load_file_name + "." + right->fil_value.front();
                left->fil_value.push(fname);
            }
            else
                left->fil_value.push(right->fil_value.front());
            right->fil_value.pop();
        };
        bool add_and = 1;
        if(left->fil_type.empty())
            add_and = 0;
        while(!right->fil_type.empty() ) {
            left->fil_type.push(right->fil_type.front());
            right->fil_type.pop();
        };
        if(add_and) {
            left->fil_type.push("AND");
        };
        return 1;
    }
    else {
        return 0;
    };
}


void check_sort(const string str, const char* rtable, const char* rid)
{
    CudaSet* right = varNames.find(rtable)->second;
    fstream binary_file(str.c_str(),ios::out|ios::binary|ios::app);
    binary_file.write((char *)&right->sort_check, 1);
    binary_file.close();
}

void update_char_permutation(CudaSet* a, string colname, unsigned int* raw_ptr, string ord, void* temp, bool host)
{
    auto s = a->string_map[colname];
    auto pos = s.find_first_of(".");
    auto len = data_dict[s.substr(0, pos)][s.substr(pos+1)].col_length;

    a->h_columns_char[colname] = new char[a->mRecCount*len];
    memset(a->h_columns_char[colname], 0, a->mRecCount*len);

	thrust::device_ptr<unsigned int> perm(raw_ptr);
	thrust::device_ptr<int_type> temp_int((int_type*)temp);
	thrust::gather(perm, perm+a->mRecCount, a->d_columns_int[colname].begin(), temp_int);

	//for(int z = 0 ; z < a->mRecCount; z++) {
	//cout << "Init vals " << a->d_columns_int[colname][z] << " " << perm[z] << " " << temp_int[z] << endl;
	//};

	//cout << "sz " << a->h_columns_int[colname].size() << " " << a->d_columns_int[colname].size() <<  " " << len << endl;
	cudaMemcpy(thrust::raw_pointer_cast(a->h_columns_int[colname].data()), temp, 8*a->mRecCount, cudaMemcpyDeviceToHost);

    FILE *f;
    f = fopen(a->string_map[colname].c_str(), "rb");

    for(int z = 0 ; z < a->mRecCount; z++) {
        fseek(f, a->h_columns_int[colname][z] * len, SEEK_SET);
        fread(a->h_columns_char[colname] + z*len, 1, len, f);
    };
    fclose(f);

    if(!host) {
        void *d;
        cudaMalloc((void **) &d, a->mRecCount*len);
        a->d_columns_char[colname] = (char*)d;

        cudaMemcpy(a->d_columns_char[colname], a->h_columns_char[colname], len*a->mRecCount, cudaMemcpyHostToDevice);

	    if (ord.compare("DESC") == 0 )
			str_sort(a->d_columns_char[colname], a->mRecCount, raw_ptr, 1, len);
		else
			str_sort(a->d_columns_char[colname], a->mRecCount, raw_ptr, 0, len);

        cudaFree(d);
    }
    else {
	    if (ord.compare("DESC") == 0 )
			str_sort_host(a->h_columns_char[colname], a->mRecCount, raw_ptr, 1, len);
		else
			str_sort_host(a->h_columns_char[colname], a->mRecCount, raw_ptr, 0, len);
    };
}



size_t getTotalSystemMemory()
{
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    return pages * page_size;
}
*/
