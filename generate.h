// created by Zhenhua Xie
// Last modifiy: 2014/2/20
#include "header.h"
using namespace std;

// generate sample ratings and similarities
template<int nr_id>
void generate(int argc, char **argv)    // generate (config file) (output rating file) (output similarity prefix)
{

    // format of config file:
    // 1. number of record
    // 2. limit of rating
    // 3. number of ids of each dimension
    // 4. number of similarities of each dimension

    cout << "Reading configs...";
    ifstream fin(argv[2]);
    if(!fin)
    {
        cout << "Failed" << endl;
        return;
    }
    int nr_rs, nr_s[nr_id];
    int nr_sim[(1 + nr_id)*nr_id / 2];
    //float ratelimit;
    int ratelimit;
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
    ofstream fout(argv[3]);
    if(!fout)
    {
        cout << "Failed" << endl;
        return;
    }
    std::unordered_set<ArrayIndex<int, nr_id>, Arrayhash<ArrayIndex<int, nr_id>>, Arrayequal<ArrayIndex<int, nr_id>>> rateDic;
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
            fout << rand() % ratelimit << endl;
            //fout << ratelimit * (float)drand48() << endl;
        }
    }
    fout.close();
    cout << "Done" << endl;

    cout << "Generating similarities...";
    stringstream filename;
    for(int i = 0, ij = 0; i < nr_id; ++i)
    {
        for(int j = 0; j <= i; ++j, ++ij)
        {
            if(nr_sim[ij] <= 0)continue;
            filename.str("");
            filename << argv[4] << "_" << ij << ".tsv";
            fout.open(filename.str());
            if(!fout)
            {
                cout << "Failed" << endl;
                return;
            }
            std::unordered_set<ArrayIndex<int, 2>, Arrayhash<ArrayIndex<int, 2>>, Arrayequal<ArrayIndex<int, 2>>> simDic;
            ArrayIndex<int, 2> sim;
            for(int k = 0; k < nr_sim[ij];)
            {
                sim.id[0] = rand() % nr_s[i];
                sim.id[1] = rand() % nr_s[j];
                if(i == j)
                {
                    if(sim.id[0] > sim.id[1])
                    {
                        int temp = sim.id[0];
                        sim.id[0] = sim.id[1];
                        sim.id[1] = temp;
                    }
                    else if(sim.id[0] == sim.id[1])continue;
                }
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