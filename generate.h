#include "header.h"
using namespace std;

template<int nr_id>
void generate(int argc, char **argv)    // generate sample ratings and similarities
{
    stringstream filename;

    cout << "Reading configs...";
    filename.str("");
    filename << argv[2] << ".cfg";
    ifstream fin(filename.str());
    if(!fin)return;
    int nr_rs, nr_s[nr_id];
    int nr_sim[(1 + nr_id)*nr_id / 2];
    float ratelimit;
    fin >> nr_rs;
    fin >> ratelimit;
    for(int i = 0; i < nr_id; ++i)fin >> nr_s[i];
    for(int i = 0, ij = 0; i < nr_id; ++i)
    {
        for(int j = 0; j <= i; ++j, ++ij)
        {
            fin >> nr_sim[ij];
        }
    }
    fin.close();
    cout << "Done" << endl;

    cout << "Generating ratings...";
    filename.str("");
    filename << argv[3] << ".tsv";
    ofstream fout(filename.str());
    if(!fout)return;
    std::unordered_set<ArrayIndex<int, nr_id>,  Arrayhash<ArrayIndex<int, nr_id>>, Arrayequal<ArrayIndex<int, nr_id>>> rateDic;
    ArrayIndex<int, nr_id> record;
    srand(unsigned(time_t(NULL)));
    srand48(unsigned(time_t(NULL)));
    for(int i = 0; i < nr_rs;)
    {
        for(int j = 0; j < nr_id; ++j)record.id[j] = rand() % nr_s[j];
        auto it = rateDic.find(record);
        if(it == rateDic.end())
        {
            ++i;
            rateDic.insert(record);
            for(int j = 0; j < nr_id; ++j)
            {
                fout << record.id[j] << "\t";
            }
            fout << ratelimit * (float)drand48() << endl;
        }
    }
    fout.close();
    cout << "Done" << endl;

    cout << "Generating similarities...";
    for(int i = 0, ij = 0; i < nr_id; ++i)
    {
        for(int j = 0; j <= i; ++j, ++ij)
        {
            if(nr_sim[ij] <= 0)continue;
            filename.str("");
            filename << argv[4] << "_" << i << "_" << j << ".tsv";
            fout.open(filename.str());
            if(!fout)return;
            std::unordered_set<ArrayIndex<int, 2>, Arrayhash<ArrayIndex<int, 2>>, Arrayequal<ArrayIndex<int, 2>>> simDic;
            ArrayIndex<int, 2> sim;
            for(int k = 0; k < nr_sim[ij];)
            {
                sim.id[0] = rand() % nr_s[i];
                sim.id[1] = rand() % nr_s[j];
                auto it = simDic.find(sim);
                if(it == simDic.end())
                {
                    ++k;
                    simDic.insert(sim);
                    fout << sim.id[0] << "\t" << sim.id[1] << "\t" << (float)drand48() << endl;
                }
            }
            fout.close();
        }
    }
    cout << "Done" << endl;
}