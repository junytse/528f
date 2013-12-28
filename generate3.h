#pragma once
#include <unordered_map>
#include <unordered_set>
#include <fstream>
#include <ctime>
#include <random>
using namespace std;

void generate3() {
    ofstream fout("data.txt");
    if (!fout)return;
    unordered_map<int, float> m01, m12, m02;
    unordered_set<long long> rate;
    srand(unsigned(time_t(NULL)));
    float r01, r12, r02;
    long long a0, a1, a2;
    for (int i = 0, r; i < 100000; ++i) {
        a0 = rand() % 1000;
        a1 = rand() % 1000;
        a2 = rand() % 1000;
        if (rate.find(a0 * 1000000 + a1 * 1000 + a2) == rate.end()) {
            r = 1 + rand() % 4;
            rate.insert(a0 * 1000000 + a1 * 1000 + a2);
        } else {
            --i;
            continue;
        }
        if (m01.find( a0 * 1000 + a1) == m01.end()) {
            r01 = float(rand()) / RAND_MAX;
            m01[a0 * 1000 + a1] = r01;
        } else {
            r01 = m01[a0 * 1000 + a1];
        }
        if (m12.find(a1 * 1000 + a2) == m12.end()) {
            r12 = float(rand()) / RAND_MAX;
            m12[a1 * 1000 + a2] = r12;
        } else {
            r12 = m12[a1 * 1000 + a2];
        }
        if (m02.find(a0 * 1000 + a2) == m02.end()) {
            r02 = float(rand()) / RAND_MAX;
            m02[a0 * 1000 + a2] = r02;
        } else {
            r02 = m02[a0 * 1000 + a2];
        }
        fout << a0 << " " << a1 << " " << a2 << " " << r << " " << r01 << " " << r12 << " " << r02 << endl;
    }
    fout.close();
}