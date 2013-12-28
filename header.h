#pragma once
#pragma warning(disable:4996)

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
#include <thread>
#include <chrono>
#include <mutex>
#include <vector>
#include <mmintrin.h>

const float DATAVER = 1;
const float MODELVER = 1;

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