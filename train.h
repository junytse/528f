#include "header.h"
#include "model.h"

template<int nr_id>
struct Monitor {
    int iter,		        // 迭代次数
        *nr_tr_srs[nr_id];	// 各维各元素的记录条数
    float tr_time;
    bool en_show_tr_rmse,   // 是否显示rmse
         en_show_obj;       // 是否显示obj
    Matrix<nr_id> *Va;      // validation matrix
    Model<nr_id> *model;    // pointer to model
    Monitor();
    void print_header();    // output list header
    void show(float iter_time, double loss, float tr_rmse);     // output list
    void scan_tr(const Matrix<nr_id> &Tr);							// 计算nr_tr_srs
    double calc_reg();      // calculate ||PQ[i]|| * lambda
    ~Monitor();
};

template<int nr_id>
Monitor<nr_id>::Monitor() : iter(0), tr_time(0.0), Va(NULL), model(NULL) {
    for(int i = 0; i < nr_id; ++i)nr_tr_srs[i] = NULL;
}

template<int nr_id>
void Monitor<nr_id>::print_header() {
    char output[1024];
    sprintf(output, "%4s", "iter");
    sprintf(output + strlen(output), " %10s", "time");
    if(en_show_tr_rmse) sprintf(output + strlen(output), " %10s", "tr_rmse");
    if(Va) sprintf(output + strlen(output), " %10s", "va_rmse");
    if(en_show_obj) sprintf(output + strlen(output), " %13s %13s %13s", "loss", "reg", "obj");
    printf("%s\n", output);
}

template<int nr_id>
void Monitor<nr_id>::show(float iter_time, double loss, float tr_rmse) {
    char output[1024];
    tr_time += iter_time;
    sprintf(output, "%-4d %10.2f", iter++, tr_time);
    if(en_show_tr_rmse) sprintf(output + strlen(output), " %10.3f", tr_rmse);
    if(Va) sprintf(output + strlen(output), " %10.3f", calc_rmse(model, Va));
    if(en_show_obj) {
        double reg = calc_reg();
        sprintf(output + strlen(output), " %13.3e %13.3e %13.3e", loss, reg, loss + reg);   // obj = loss + reg
    }
    printf("%s\n", output);
    fflush(stdout);
}

template<int nr_id>
void Monitor<nr_id>::scan_tr(const Matrix<nr_id> &Tr) {
    for(int i = 0; i < nr_id; ++i) {
        nr_tr_srs[i] = new int[Tr.nr_s[i]];
        memset(nr_tr_srs[i], 0, Tr.nr_s[i] * sizeof(int));
    }
    for(int i = 0; i < Tr.nr_rs; ++i) {
        for(int j = 0; j < nr_id; ++j) {
            ++nr_tr_srs[j][Tr.M[i].id[j]];
        }
    }
}

template<int nr_id>
double Monitor<nr_id>::calc_reg() {
    double result = 0, reg;
    for(int i = 0; i < nr_id; ++i) {
        reg = 0;
        for(int j = 0; j < model->nr_s[i]; ++j)
            reg += nr_tr_srs[i][j] * std::inner_product(&model->PQ[i][j * model->dim_off], &model->PQ[i][j * model->dim_off] + model->dim, &model->PQ[i][j * model->dim_off], 0.0);
        result += reg * model->l[i];
    }
    return result;
}

template<int nr_id>
Monitor<nr_id>::~Monitor() {
    for(int i = 0; i < nr_id; ++i) {
        if(NULL != nr_tr_srs[i]) {
            delete[] nr_tr_srs[i];
            nr_tr_srs[i] = NULL;
        }
    }
}


template<int nr_id>
struct TrainOption {
    char *tr_path, *va_path, *model_path;
    TrainOption(int argc, char **argv, Model<nr_id> *model, Monitor<nr_id> *monitor);
    static void exit_train();
    ~TrainOption();
};

template<int nr_id>
TrainOption<nr_id>::TrainOption(int argc, char **argv, Model<nr_id> *model, Monitor<nr_id> *monitor) : va_path(NULL), tr_path(NULL), model_path(NULL) {
    model->dim = 40;	// 预设值
    model->nr_thrs = 4;
    model->iter = 40;
    model->gamma = 0.001f;
    model->avg = 0.0f;
    model->en_rand_shuffle = true;
    model->en_avg = true;
    monitor->en_show_tr_rmse = true;
    monitor->en_show_obj = true;
    for(int i = 0; i < nr_id; ++i) {
        model->nr_gbs[i] = 2 * (model->nr_thrs);
        model->l[i] = 1;
        model->lb[i] = 1;
        model->en_b[i] = true;
    }
    int i;
    for(i = 2; i < argc; i++) {
        if(argv[i][0] != '-') break;
        if(i + 1 >= argc) exit_train();
        if(!strcmp(argv[i], "-k")) {
            model->dim = atoi(argv[++i]);
            if(model->dim <= 0) {
                fprintf(stderr, "dimensions should > 0\n");
                exit(1);
            }
        } else if(!strcmp(argv[i], "-t")) {
            model->iter = atoi(argv[++i]);
            if(model->iter <= 0) {
                fprintf(stderr, "iterations should > 0\n");
                exit(1);
            }
        } else if(!strcmp(argv[i], "-s")) {
            model->nr_thrs = atoi(argv[++i]);
            if(model->nr_thrs <= 0) {
                fprintf(stderr, "number of threads should > 0\n");
                exit(1);
            }
        } else if(!strcmp(argv[i], "-g")) {
            model->gamma = (float)atof(argv[++i]);
            if(model->gamma <= 0) {
                fprintf(stderr, "learning rate should > 0\n");
                exit(1);
            }
        } else if(!strcmp(argv[i], "-v")) va_path = argv[++i];
        else if(!strcmp(argv[i], "-blk")) {
            model->nr_gbs[0] = atoi(strtok(argv[++i], "x"));
            if(model->nr_gbs[0] <= 0) {
                fprintf(stderr, "number of blocks should > 0\n");
                exit(1);
            }
            for(int j = 1; j < nr_id; ++j) {
                model->nr_gbs[j] = atoi(strtok(NULL, "x"));
                if(model->nr_gbs[j] <= 0) {
                    fprintf(stderr, "number of blocks should > 0\n");
                    exit(1);
                }
            }
        } else if(!strcmp(argv[i], "--rand-shuffle")) model->en_rand_shuffle = true;
        else if(!strcmp(argv[i], "--no-rand-shuffle")) model->en_rand_shuffle = false;
        else if(!strcmp(argv[i], "--tr-rmse")) monitor->en_show_tr_rmse = true;
        else if(!strcmp(argv[i], "--no-tr-rmse")) monitor->en_show_tr_rmse = false;
        else if(!strcmp(argv[i], "--obj")) monitor->en_show_obj = true;
        else if(!strcmp(argv[i], "--no-obj")) monitor->en_show_obj = false;
        else if(!strcmp(argv[i], "--use-avg")) model->en_avg = true;
        else if(!strcmp(argv[i], "--no-use-avg")) model->en_avg = false;
        //else if(!strcmp(argv[i], "--user-bias")) model->en_ub = true;
        //else if(!strcmp(argv[i], "--no-user-bias")) model->en_ub = false;
        //else if(!strcmp(argv[i], "--item-bias")) model->en_ib = true;
        //else if(!strcmp(argv[i], "--no-item-bias")) model->en_ib = false;
        //else if(!strcmp(argv[i], "-ub")) {
        //    float lub = atof(argv[++i]);
        //    if(lub < 0) model->en_ub = false;
        //    else {
        //        model->en_ub = true;
        //        model->lub = lub;
        //    }
        //} else if(!strcmp(argv[i], "-ib")) {
        //    float lib = atof(argv[++i]);
        //    if(lib < 0) model->en_ib = false;
        //    else {
        //        model->en_ib = true;
        //        model->lib = lib;
        //    }
        //} else if(!strcmp(argv[i], "-p")) {
        //    model->lp = atof(argv[++i]);
        //    if(model->lp < 0) {
        //        fprintf(stderr, "cost should >= 0\n");
        //        exit(1);
        //    }
        //} else if(!strcmp(argv[i], "-q")) {
        //    model->lq = atof(argv[++i]);
        //    if(model->lq < 0) {
        //        fprintf(stderr, "cost should >= 0\n");
        //        exit(1);
        //    }
        else {
            fprintf(stderr, "Invalid option: %s\n", argv[i]);
            exit_train();
        }
    }
    if(i >= argc) exit_train();
    tr_path = argv[i++];
    if(i < argc) {
        model_path = new char[strlen(argv[i]) + 1];
        sprintf(model_path, "%s", argv[i]);
    } else {
        char *p = strrchr(argv[i - 1], '/');
        if(p == NULL)
            p = argv[i - 1];
        else
            ++p;
        model_path = new char[strlen(p) + 7];
        sprintf(model_path, "%s.model", p);
    }
    if(va_path) {
        FILE *f = fopen(va_path, "rb");    //Check if validation set exist.
        if(!f) exit_file_error(va_path);
        fclose(f);
    }
}

template<int nr_id>
void TrainOption<nr_id>::exit_train() {
    printf(
        "usage: libmf train [options] binary_train_file model\n"
        "\n"
        "options:\n"
        "-k <dimensions>: set the number of dimensions (default 40)\n"
        "-t <iterations>: set the number of iterations (default 40)\n"
        "-s <number of threads>: set the number of threads (default 4)\n"
        "-p <cost>: set the regularization cost for P (default 1)\n"
        "-q <cost>: set the regularization cost for Q (default 1)\n"
        "-ub <cost>: set the regularization cost for user bias (default 1), set <0 to disable\n"
        "-ib <cost>: set the regularization cost for item bias (default 1), set <0 to disable\n"
        "-g <gamma>: set the learning rate for parallel SGD (default 0.001)\n"
        "-v <path>: set the path to validation set\n"
        "-blk <blocks>: set the number of blocks for parallel SGD (default 2s x 2s)\n"
        "    For example, if you want 3x4 blocks, then use '-blk 3x4'\n"
        "--rand-shuffle --no-rand-shuffle: enable / disable random suffle (default disabled)\n"
        "    This options should be used when the data is imbalanced.\n"
        "--tr-rmse --no-tr-rmse: enable / disable show rmse on training data (default disabled)\n"
        "    This option shows the estimated RMSE on training data. It also slows down the training procedure.\n"
        "--obj --no-obj: enable / disable show objective value (default disabled)\n"
        "    This option shows the estimated objective value on training data. It also slows down the training procedure.\n"
        "--use-avg --no-use-avg: enable / disable using training data average (default enabled)\n"
    );
    exit(1);
}

template<int nr_id>
TrainOption<nr_id>::~TrainOption() {
    delete[] model_path;
}

template<int nr_id>
struct GridMatrix {
    int nr_gbs[nr_id],  // number of block for each dimension
        nr_gbs_a;           // total blocks
    long nr_rs;         // number of ratings
    Matrix<nr_id> **GMS;// grid matrix
    GridMatrix(const Matrix<nr_id> &R, int **map, int *nr_gbs, int nr_thrs);
    static void sort_ratings(Matrix<nr_id> *R, std::mutex *mtx, int *nr_thrs);
    ~GridMatrix();
};

template<int nr_id>
GridMatrix<nr_id>::GridMatrix(const Matrix<nr_id> &R, int **map, int *_nr_gbs, int nr_thrs) {
    printf("Griding...");
    fflush(stdout);
    Clock clock;
    clock.tic();

    // assign block numbers and total block numbers
    nr_gbs_a = 1;
    for(int i = 0; i < nr_id; ++i) {
        nr_gbs[i] = _nr_gbs[i];
        nr_gbs_a *= nr_gbs[i];
    }

    GMS = new Matrix<nr_id>*[nr_gbs_a];
    nr_rs = R.nr_rs;
    std::mutex mtx;
    int seg[nr_id];
    for(int i = 0; i < nr_id; ++i) {
        seg[i] = (int)ceil(double(R.nr_s[i]) / nr_gbs[i]);
    }

    // r_map to collect record number for each blocks
    int *r_map = new int[nr_gbs_a];
    memset(r_map, 0, nr_gbs_a * sizeof(int));
    for(int i = 0, idx = 0; i < R.nr_rs; ++i, idx = 0) {
        for(int j = 0; j < nr_id; ++j) {
            idx *= nr_gbs[j];
            idx += (map[j] ? map[j][R.M[i].id[j]] : R.M[i].id[j]) / seg[j];
        }
        ++r_map[idx];
    }

    // create gird matrix and clear r_map
    for(int i = 0; i < nr_gbs_a; ++i) {
        GMS[i] = new Matrix<nr_id>(r_map[i], R.nr_s, 0);	// GMS[i] 第i块的Matrix
        r_map[i] = 0;									    // r_map: 各块中的元素个数
    }

    // assign value to grid matrix
    int new_id[nr_id];  // store mapped node
    for(int i = 0, idx = 0; i < R.nr_rs; ++i, idx = 0) {
        for(int j = 0; j < nr_id; ++j) {
            idx *= nr_gbs[j];
            new_id[j] = map[j] ? map[j][R.M[i].id[j]] : R.M[i].id[j];
            idx += new_id[j] / seg[j];
        }
        GMS[idx]->M[r_map[idx]] = R.M[i];  // deep copy
        memcpy(GMS[idx]->M[r_map[idx]].id, new_id, nr_id * sizeof(int));
        //for(int j = 0; j < nr_id; ++j) {
        //GMS[idx]->M[r_map[idx]].id[j] = new_id[j];
        //}
        ++r_map[idx];
    }
    delete[] r_map;

    // sort ratings for each grid if it is shuffled
    if(map[0]) {
        int nr_alive_thrs = 0;
        for(int i = 0; i < nr_gbs_a; ++i) {
            while(nr_alive_thrs >= nr_thrs) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }
            {
                std::lock_guard<std::mutex> lock(mtx);
                nr_alive_thrs++;
            }
            std::thread worker = std::thread(GridMatrix::sort_ratings, GMS[i], &mtx, &nr_alive_thrs);
            worker.detach();
        }
        while(nr_alive_thrs != 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
    }
    printf("done. %.2f\n", clock.toc());
    fflush(stdout);
    //if(EN_SHOW_GRID) {    // cannot show grid in multidimension
    //printf("\n");
    //for(int mx = 0; mx < nr_gubs; mx++) {
    //for(int nx = 0; nx < nr_gibs; nx++) printf("%7ld ", GMS[mx * nr_gibs + nx]->nr_rs);
    //printf("\n");
    //}
    //printf("\n");
    //}
}

// multithread
template<int nr_id>
void GridMatrix<nr_id>::sort_ratings(Matrix<nr_id> *M, std::mutex *mtx, int *nr_thrs) {
    M->sort();
    std::lock_guard<std::mutex> lock(*mtx);
    (*nr_thrs)--;
}

template<int nr_id>
GridMatrix<nr_id>::~GridMatrix() {
    if(NULL != GMS) {
        for(int i = 0; i < nr_gbs_a; ++i)delete GMS[i];
        delete[] GMS;
        GMS = NULL;
    }
}

template<int nr_id>
class Scheduler {
    int *nr_jts,            // update times for each job
        *order,             // mapper of job id
        nr_gbs[nr_id],
        nr_gbs_a,
        nr_thrs,            // number of thread
        total_jobs,         // number of jobs finished
        nr_paused_thrs;     // number of paused threads
    bool *blocked[nr_id],   // blocked flag for blocks
         paused,            // if threads are paused
         terminated;        // if all threads are terminated
    double *losses;         // losses for jobs
    std::mutex mtx;
    bool all_paused();      // return if all threads are paused
public:
    Scheduler(int *nr_gbs, int nr_thrs);
    int get_job();  // get an available job by id
    void put_job(int jid, double loss); // put back job and corresponding loss
    double get_loss();                  // sum up losses and return
    int get_total_jobs();               // return total_job
    void pause_sgd();                   // set paused = true and wait all thread to be paused
    void pause();                       //
    void resume();                      // shuffle the jobs and set paused = false
    void terminate();                   // set terminated = true
    bool is_terminated();               // return terminated
    //void show();
    ~Scheduler();
};

template<int nr_id>
Scheduler<nr_id>::Scheduler(int *_nr_gbs, int _nr_thrs) : nr_gbs_a(1), nr_jts(NULL), order(NULL), losses(NULL), nr_thrs(_nr_thrs), total_jobs(0), nr_paused_thrs(0), paused(false), terminated(false) {
    //blocked = new bool*[nr_id];
    //nr_gbs = new int[nr_id];
    for(int i = 0; i < nr_id; ++i) {
        nr_gbs[i] = _nr_gbs[i];
        blocked[i] = new bool[nr_gbs[i]];
        for(int j = 0; j < nr_gbs[i]; ++j)blocked[i][j] = false;
        nr_gbs_a *= nr_gbs[i];
    }
    nr_jts = new int[nr_gbs_a];
    order = new int[nr_gbs_a];
    losses = new double[nr_gbs_a];
    for(int i = 0; i < nr_gbs_a; ++i) {
        nr_jts[i] = 0;
        losses[i] = 0;
        order[i] = i;
    }
}

template<int nr_id>
int Scheduler<nr_id>::get_job() {
    // find available jid
    int jid = -1, ts = INT_MAX;
    while(true) {
        {
            std::lock_guard<std::mutex> lock(mtx);
            bool blocked_flag;
            for(int mx = 0; mx < nr_gbs_a; mx++) {
                int nx = order[mx];
                blocked_flag = false;
                for(int i = nr_id - 1; i >= 0; --i) {
                    if(blocked[i][nx % nr_gbs[i]]) {
                        blocked_flag = true;
                        break;
                    }
                    nx /= nr_gbs[i];
                }
                if(blocked_flag)continue;
                if(nr_jts[nx] < ts) ts = nr_jts[nx], jid = nx;
            }
        }
        if(jid != -1) break;
        pause();
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    // lock up current jid and return
    {
        std::lock_guard<std::mutex> lock(mtx);
        for(int i = nr_id - 1, jid1 = jid; i >= 0; --i) {
            blocked[i][jid1 % nr_gbs[i]] = true;
            jid1 /= nr_gbs[i];
        }
        ++nr_jts[jid];
    }
    return jid;
}

template<int nr_id>
void Scheduler<nr_id>::put_job(int jid, double loss) {
    std::lock_guard<std::mutex> lock(mtx);
    for(int i = nr_id - 1, jid1 = jid; i >= 0; --i) {
        blocked[i][jid1 % nr_gbs[i]] = false;
        jid1 /= nr_gbs[i];
    }
    losses[jid] = loss;
    total_jobs++;
}

template<int nr_id>
double Scheduler<nr_id>::get_loss() {
    double loss = 0;
    for(int ix = 0; ix < nr_gbs_a; ix++) loss += losses[ix];
    return loss;
}

template<int nr_id>
int Scheduler<nr_id>::get_total_jobs() {
    return total_jobs;
}

template<int nr_id>
void Scheduler<nr_id>::pause_sgd() {
    {
        std::lock_guard<std::mutex> lock(mtx);
        paused = true;
    }
    while(!all_paused()) std::this_thread::sleep_for(std::chrono::milliseconds(1));
}

template<int nr_id>
void Scheduler<nr_id>::pause() {
    {
        std::lock_guard<std::mutex> lock(mtx);
        if(!paused) return;
    } {
        std::lock_guard<std::mutex> lock(mtx);
        ++nr_paused_thrs;
    }

    // wait until paused = false
    while(paused) std::this_thread::sleep_for(std::chrono::milliseconds(1));

    {
        std::lock_guard<std::mutex> lock(mtx);
        --nr_paused_thrs;
    }
}

template<int nr_id>
bool Scheduler<nr_id>::all_paused() {
    std::lock_guard<std::mutex> lock(mtx);
    return (nr_paused_thrs == nr_thrs);
}

template<int nr_id>
void Scheduler<nr_id>::resume() {
    std::lock_guard<std::mutex> lock(mtx);
    std::random_shuffle(order, order + nr_gbs_a);
    paused = false;
}

template<int nr_id>
void Scheduler<nr_id>::terminate() {
    terminated = true;
}

template<int nr_id>
bool Scheduler<nr_id>::is_terminated() {
    return terminated;
}

//void Scheduler::show() {
//for(int mx = 0; mx < nr_gubs; mx++) {
//for(int nx = 0; nx < nr_gibs; nx++) printf("%3d ", nr_jts[mx * nr_gibs + nx]);
//printf("\n");
//}
//printf("\n");
//fflush(stdout);
//}

template<int nr_id>
Scheduler<nr_id>::~Scheduler() {
    delete[] nr_jts;
    delete[] order;
    delete[] losses;
    for(int i = 0; i < nr_id; ++i)delete[] blocked[i];
}

template<int nr_id>
void sgd(GridMatrix<nr_id> *TrG, Model<nr_id> *model, Scheduler<nr_id> *scheduler, int tid) {
    // copy from model: PQ, B, dim, en_b
    // copy from TrG: nr_rs, rn,
    float **PQ = new float*[nr_id];
    float **B = new float*[nr_id];
    for(int i = 0; i < nr_id; ++i) {
        PQ[i] = model->PQ[i];
        B[i] = model->B[i];
    }
    const int dim = model->dim_off;
    Node<nr_id> *r = NULL, *rn = NULL;
    float **pq = new float*[nr_id];
    float **b = new float*[nr_id];
    bool *en_b = new bool[nr_id];
    for(int i = 0; i < nr_id; ++i) {
        pq[i] = NULL;
        b[i] = NULL;
        en_b[i] = model->en_b[i];
    }
    int dx, jid;
    long mx, nr_rs;
    double loss;
    __m128 XMMgl[nr_id];
    for(int i = 0; i < nr_id; ++i) {
        XMMgl[i] = _mm_load1_ps(&model->gl[i]);
    }
    __m128 XMMg = _mm_load1_ps(&model->gamma);
    __m128 XMMavg = _mm_load1_ps(&model->avg);
    __m128d XMMl = _mm_setzero_pd();
    while(true) {
        jid = scheduler->get_job();
        rn = TrG->GMS[jid]->M;
        nr_rs = TrG->GMS[jid]->nr_rs;
        XMMl = _mm_setzero_pd();
        for(mx = 0; mx < nr_rs; mx++) {						// 每次更新一个分数记录: PQ[id[0~(nr_id-1)]], loss
            r = rn;
            rn++;
            __m128 XMMr = _mm_load1_ps(&r->rate), XMMge = _mm_setzero_ps();
            for(int i = 0; i < nr_id; ++i) {
                pq[i] = PQ[i] + r->id[i] * dim;
                b[i] = B[i] + r->id[i];
            }
            _mm_prefetch((const char *)(r), _MM_HINT_T0);
            for(int i = 0; i < nr_id; ++i) {
                _mm_prefetch((const char *)(pq[i]), _MM_HINT_T0);
            }
            if(mx + 7 < nr_rs) {
                _mm_prefetch((const char *)(r + 7), _MM_HINT_T1);
                for(int i = 0; i < nr_id; ++i) {
                    _mm_prefetch((const char *)(PQ[i] + (r + 7)->id[i] * dim), _MM_HINT_T1);
                }
                if(mx + 15 < nr_rs) {
                    _mm_prefetch((const char *)(r + 15), _MM_HINT_T2);
                    for(int i = 0; i < nr_id; ++i) {
                        _mm_prefetch((const char *)(PQ[i] + (r + 15)->id[i] * dim), _MM_HINT_T2);
                    }
                }
            }

            // TODO: ???????????l += sum((Qij - pq[i]^2 - b[i] - b[j] - avgx)^2)
            // TODO: ???????????l += sum((pq[i]^2) + sum(b[i]^2)

            for(int i = 0; i < nr_id; ++i) {					// ge += sum(pqi * pqj)
                for(int j = i + 1; j < nr_id; ++j) {
                    for(dx = 0; dx < dim - 7; dx += 8) {
                        __m128 XMMp0 = _mm_loadu_ps(pq[i] + dx);
                        __m128 XMMq0 = _mm_loadu_ps(pq[j] + dx);
                        __m128 XMMp1 = _mm_loadu_ps(pq[i] + dx + 4);
                        __m128 XMMq1 = _mm_loadu_ps(pq[j] + dx + 4);
                        XMMp0 = _mm_mul_ps(XMMp0, XMMq0);
                        XMMp1 = _mm_mul_ps(XMMp1, XMMq1);
                        XMMge = _mm_add_ps(XMMge, _mm_add_ps(XMMp0, XMMp1));
                    }
                    for(; dx < dim; dx += 4) {
                        __m128 XMMp0 = _mm_loadu_ps(pq[i] + dx);
                        __m128 XMMq0 = _mm_loadu_ps(pq[j] + dx);
                        XMMge = _mm_add_ps(XMMge, _mm_mul_ps(XMMp0, XMMq0));
                    }
                }
            }
            XMMge = _mm_hadd_ps(XMMge, XMMge);
            XMMge = _mm_hadd_ps(XMMge, XMMge);

            for(int i = 0; i < nr_id; ++i) {					// ge += sum(bias)
                if(en_b[i])XMMge = _mm_add_ps(XMMge, _mm_load1_ps(b[i]));
            }

            XMMge = _mm_sub_ps(XMMr, _mm_add_ps(XMMge, XMMavg));// ge = r - (ge + avg1)
            XMMl = _mm_add_pd(XMMl, _mm_cvtps_pd(_mm_mul_ps(XMMge, XMMge))); // l += (double)(ge^2)

            XMMge = _mm_mul_ps(XMMge, XMMg); // ge *= gamma

            for(dx = 0; dx < dim - 7; dx += 8) {				// 更新：pqi = ge * (sum(pq) - pqi) + gli * pqi
                __m128 XMMsum0 = _mm_setzero_ps();
                __m128 XMMsum1 = _mm_setzero_ps();
                for(int i = 0; i < nr_id; ++i) {
                    XMMsum0 = _mm_add_ps(XMMsum0, _mm_loadu_ps(pq[i] + dx));
                    XMMsum1 = _mm_add_ps(XMMsum1, _mm_loadu_ps(pq[i] + dx + 4));
                }
                for(int i = 0; i < nr_id; ++i) {
                    __m128 XMMpq0 = _mm_loadu_ps(pq[i] + dx);
                    __m128 XMMpq1 = _mm_loadu_ps(pq[i] + dx + 4);
                    XMMpq0 = _mm_add_ps(_mm_mul_ps(XMMge, _mm_sub_ps(XMMsum0, XMMpq0)), _mm_mul_ps(XMMgl[i], XMMpq0));
                    XMMpq1 = _mm_add_ps(_mm_mul_ps(XMMge, _mm_sub_ps(XMMsum1, XMMpq1)), _mm_mul_ps(XMMgl[i], XMMpq1));
                    _mm_storeu_ps(pq[i] + dx, XMMpq0);
                    _mm_storeu_ps(pq[i] + dx + 4, XMMpq1);
                }
            }
            for(; dx < dim; dx += 4) {
                __m128 XMMsum0 = _mm_setzero_ps();
                for(int i = 0; i < nr_id; ++i) {
                    XMMsum0 = _mm_add_ps(XMMsum0, _mm_loadu_ps(pq[i] + dx));
                }
                for(int i = 0; i < nr_id; ++i) {
                    __m128 XMMpq0 = _mm_loadu_ps(pq[i] + dx);
                    XMMpq0 = _mm_add_ps(_mm_mul_ps(XMMge, _mm_sub_ps(XMMsum0, XMMpq0)), _mm_mul_ps(XMMgl[i], XMMpq0));
                    _mm_storeu_ps(pq[i] + dx, XMMpq0);
                }
            }

            float ge[4];
            //float ge;
            _mm_storeu_ps(ge, XMMge);
            for(int i = 0; i < nr_id; ++i) {                    // 更新：b[i] = gl[i] * b[i] + ge;
                if(en_b[i])(*b[i]) = model->gl[i] * (*b[i]) + ge[0];
            }
        }
        _mm_store_sd(&loss, XMMl);
        scheduler->put_job(jid, loss);
        scheduler->pause();
        if(scheduler->is_terminated()) break;
    }
    delete[] PQ;
    delete[] B;
    delete[] pq;
    delete[] b;
    delete[] en_b;
}

template<int nr_id>
void gsgd(GridMatrix<nr_id> *TrG, Model<nr_id> *model, Monitor<nr_id> *monitor) {
    printf("SGD Starts!\n");
    fflush(stdout);

    int iter = 1;
    Scheduler<nr_id> *scheduler = new Scheduler<nr_id>(model->nr_gbs, model->nr_thrs);
    std::vector<std::thread> threads;
    Clock clock;
    clock.tic();

    for(int tx = 0; tx < model->nr_thrs; tx++)
        threads.push_back(std::thread(sgd<nr_id>, TrG, model, scheduler, tx));

    monitor->print_header();

    // output loss for each iteration
    while(iter <= model->iter) {
        if(scheduler->get_total_jobs() >= iter * TrG->nr_gbs_a) {
            scheduler->pause_sgd();
            float iter_time = clock.toc();
            double loss = scheduler->get_loss();
            //if (EN_SHOW_SCHED) scheduler->show();
            monitor->show(iter_time, loss, (float)sqrt(loss / TrG->nr_rs)); // rmse = sqrt(loss / nr_rs)
            iter++;
            clock.tic();
            scheduler->resume();
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    scheduler->terminate();

    printf("Waiting for all threads terminate...");
    fflush(stdout);
    clock.tic();
    for(auto it = threads.begin(); it != threads.end(); it++) it->join();
    delete scheduler;
    printf("done. %.2f\n", clock.toc());
    fflush(stdout);
}

template<int nr_id>
void train(int argc, char **argv) {
    // get options from argv
    Model<nr_id> *model = new Model<nr_id>;
    Monitor<nr_id> *monitor = new Monitor<nr_id>;
    TrainOption<nr_id> *option = new TrainOption<nr_id>(argc, argv, model, monitor);

    // create model from train matrix
    Matrix<nr_id> Tr(option->tr_path);
    model->initialize(Tr);
    if(model->en_rand_shuffle) model->gen_rand_map();

    // create validation matrix
    Matrix<nr_id> *Va = NULL;
    if(option->va_path) {
        if(model->en_rand_shuffle)
            Va = new Matrix<nr_id>(option->va_path, model->map_f);
        else Va = new Matrix<nr_id>(option->va_path);
    }
    if(Va) {
        for(int i = 0; i < nr_id; ++i) {
            if(Va->nr_s[i] > Tr.nr_s[i]) {
                fprintf(stderr, "Validation set out of range.\n");
                exit(1);
            }
        }
    }

    // shuffle the model
    if(model->en_rand_shuffle) model->shuffle();

    // configulate monitor
    monitor->model = model;
    monitor->Va = Va;
    monitor->scan_tr(Tr);

    // create grid matrix from original matrix
    GridMatrix<nr_id> *TrG;
    TrG = new GridMatrix<nr_id>(Tr, model->map_f, model->nr_gbs, model->nr_thrs);

    // preform gsgd
    gsgd(TrG, model, monitor);

    // inverse shuffle
    if(model->en_rand_shuffle) model->inv_shuffle();

    // output model
    model->write(option->model_path);

    delete model;
    delete monitor;
    delete option;
    delete TrG;
    if(NULL != Va)delete Va;
}
