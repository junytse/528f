#pragma once
#pragma warning(disable:4996)

#include <string>
#include <fstream>
#include <iostream>
#include <sstream>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <csignal>
#include <ctime>
#include <cstring>
#include <climits>
#include <cfloat>
#include <random>
#include <numeric>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <thread>
#include <chrono>
#include <mutex>
#include <vector>
#include <mmintrin.h>

const float DATAVER = 1.3f;
const float MODELVER = 1.3f;

const bool EN_SHOW_SCHED = false;
const bool EN_SHOW_GRID = false;

enum FileType { DATA, MODEL };

struct Clock {
    clock_t begin, end;
    void tic();
    float toc();
};

void Clock::tic() {
    begin = clock();
}

float Clock::toc() {
    end = clock();
    return (float)(end - begin) / CLOCKS_PER_SEC;
}

static unsigned long long seed = 1;
double drand48(void) {
    const unsigned long long a = 0x5DEECE66DLL, c = 0xB16, m = 0x100000000LL;
    seed = (a * seed + c) & 0xFFFFFFFFFFFFLL;
    unsigned int x = unsigned(seed >> 16);
    return((double)x / (double)m);
}

void srand48(unsigned int i) {
    seed = (((long long int) i) << 16) | rand();
}

void exit_file_error(char *path) {
    fprintf(stderr, "\nError: Invalid file name %s.\n", path);
    exit(1);
}

void exit_file_ver(float current_ver, float file_ver) {
    fprintf(stderr, "\nError: Inconsistent file version.\n");
    fprintf(stderr, "current version:%.2f    file version:%.2f\n", current_ver, file_ver);
    exit(1);
}

template<typename _TYPE, std::size_t _LENGTH>
class ArrayIndex {
public:
    _TYPE id[_LENGTH];
    ArrayIndex() {}
    ArrayIndex(const _TYPE(&lhs)[_LENGTH]) {
        memcpy(this, lhs, sizeof(ArrayIndex));
    }
    ArrayIndex(const ArrayIndex &lhs) {
        memcpy(this, &lhs, sizeof(ArrayIndex));
    }
    ArrayIndex& operator = (ArrayIndex &lhs) {
        memcpy(this, &lhs, sizeof(ArrayIndex));
        return *this;
    }
    bool operator == (const ArrayIndex &lhs) const {
        for(int i = 0; i < _LENGTH - 1; ++i)if(id[i] != lhs.id[i])return false;
        return id[_LENGTH - 1] == lhs.id[_LENGTH - 1];
    }
};

template<typename _TYPE>
struct Arrayhash : std::unary_function<_TYPE, std::size_t> {
    std::size_t operator()(const _TYPE &lhs) const {
        const std::size_t SEED = static_cast<size_t>(0xc70f6907UL);
        return _Hash_bytes(&lhs, sizeof(_TYPE), SEED);
    }
};

template<typename _TYPE>
struct Arrayequal : std::binary_function<_TYPE, _TYPE, bool> {
    bool operator()(const _TYPE &lhs, const _TYPE &rhs) const {
        return lhs == rhs;
    }
};

//int main() {
//    unordered_map<ArrayIndex<int, 2>, string, myhash<ArrayIndex<int, 2>>, myequal<ArrayIndex<int, 2>>> map;
//    ArrayIndex<int, 2> a;
//    a.id[0] = 1;
//    a.id[1] = 2;
//    map[a] = "Hello";
//    a.id[1] = 3;
//    map[a] = "world";
//    for(auto it = map.begin(); it != map.end(); ++it) {
//        std::cout << it->first.id[0] << " " << it->first.id[1] << " " << it->second << endl;
//    }
//}

inline std::size_t
unaligned_load(const char* p) {
    std::size_t result;
    memcpy(&result, p, sizeof(result));
    return result;
}

// Loads n bytes, where 1 <= n < 8.
inline std::size_t
load_bytes(const char* p, int n) {
    std::size_t result = 0;
    --n;
    do
        result = (result << 8) + static_cast<unsigned char>(p[n]);
    while(--n >= 0);
    return result;
}

inline std::size_t
shift_mix(std::size_t v) {
    return v ^ (v >> 47);
}

// Implementation of Murmur hash for 64-bit size_t.
size_t
_Hash_bytes(const void* ptr, std::size_t len, std::size_t seed) {
    static const std::size_t mul = (((size_t)0xc6a4a793UL) << 32UL)
                                   + (size_t)0x5bd1e995UL;
    const char* const buf = static_cast<const char*>(ptr);

    // Remove the bytes not divisible by the sizeof(size_t).  This
    // allows the main loop to process the data as 64-bit integers.
    const int len_aligned = len & ~0x7;
    const char* const end = buf + len_aligned;
    std::size_t hash = seed ^ (len * mul);
    for(const char* p = buf; p != end; p += 8) {
        const std::size_t data = shift_mix(unaligned_load(p) * mul) * mul;
        hash ^= data;
        hash *= mul;
    }
    if((len & 0x7) != 0) {
        const std::size_t data = load_bytes(end, len & 0x7);
        hash ^= data;
        hash *= mul;
    }
    hash = shift_mix(hash) * mul;
    hash = shift_mix(hash);
    return hash;
}