#pragma once

// Rating record (ids, rating), have no pointer
template<int nr_id>
struct Node {
    int id[nr_id];
    float rate;
    Node() {
    }
    Node(const Node<nr_id>& lhs): rate(lhs.rate) {
        for(int i = 0; i < nr_id; ++i)id[i] = lhs.id[i];
    }
    Node& operator = (Node& lhs) {
        for(int i = 0; i < nr_id; ++i)id[i] = lhs.id[i];
        rate = lhs.rate;
        return *this;
    }
};

template<int nr_id>
struct Matrix {
    int nr_s[nr_id];    // Number of each dimension
    int nr_rs;          // Number of records
    float avg;          // avg
    Node<nr_id> *M;     // Records
    Matrix();
    Matrix(int nr_rs, const int *nr_s, float avg);
    Matrix(char *path);
    Matrix(char *path, int **map);
    void read_meta(FILE *f);    // read meta from file
    void read(char *path);      // read data from file
    void write(char *path);     // write to file
    void sort();                // sort the records
    static bool compare(Node<nr_id> lhs, Node<nr_id> rhs);  // compare function for sort
    ~Matrix();
};

template<int nr_id>
struct Similarity {
    std::vector<Node<2>> M[(1 + nr_id)*nr_id / 2];
    bool en_sim[(1 + nr_id)*nr_id / 2];
    Similarity();
    Similarity(char *path, int **map = NULL);
    void read(char *path, int **map);
    void generate(int, int);
};

template<int nr_id>
struct Model {
    int nr_s[nr_id],                            // Number of each dimension
        dim,                                    // latent factor
        dim_off,                                // latent factor £¨4µÄ±¶Êý£©
        nr_thrs,                                // Number of threads
        iter,                                   // Number of iterations
        nr_gbs[nr_id],                          // Number of blocks for each dimension
        *map_f[nr_id],                          // Mapper
        *map_b[nr_id];                          // Reverse mapper
    float *P[nr_id],                           // latent matrix: nr_id * (nr_s[nr_id] * dim_off)
          *B[nr_id],                            // bias: nr_id * nr_s[nr_id]
          l[nr_id],                             // lambda for P
          lb[nr_id],                            // lambda for bias
          gl[nr_id],                            // gamma * lambda for P
          glb[nr_id],                           // gamma * lambda for bias
          gamma,                                // penalty
          avg,                                  // average rating
          lambda2[(1 + nr_id) * nr_id / 2],
          lambda3[nr_id],
          lambda4[(1 + nr_id) * nr_id / 2],
          lambda5[nr_id];
    bool en_rand_shuffle,                       // random shuffle
         en_avg,                                // average rating
         en_b[nr_id],                           // bias
         en_sim[(1 + nr_id)*nr_id / 2];         // similarity
    Model();
    Model(char *path);
    void initialize(const Matrix<nr_id> &Tr);   // initialize
    void read_meta(FILE *f);                    // read meta from file
    void read(char *path);                      // read data from file
    void write(char *path);                     // write to file
    void gen_rand_map();                        // generate shuffle map
    void shuffle();                             // shuffle the data
    void inv_shuffle();                         // inverse shuffle the data
    ~Model();
};

template<int nr_id>
Matrix<nr_id>::Matrix(): M(NULL) { /*, nr_rs(0), avg(0)*/
}

template<int nr_id>
Matrix<nr_id>::Matrix(int _nr_rs, const int *_nr_s, float _avg) : /*M(NULL),*/ avg(_avg), nr_rs(_nr_rs) {
    memcpy(nr_s, _nr_s, sizeof(int) * nr_id);
    M = new Node<nr_id>[nr_rs];
}

template<int nr_id>
Matrix<nr_id>::Matrix(char *path) {
    read(path);
}

template<int nr_id>
Matrix<nr_id>::Matrix(char *path, int **map) {
    read(path);
    for(int i = 0; i < nr_rs; ++i) {
        for(int j = 0; j < nr_id; ++j) {
            M[i].id[j] = map[j][M[i].id[j]];
        }
    }
}

template<int nr_id>
void Matrix<nr_id>::read_meta(FILE *f) {
    // check file type
    FileType type;
    fread(&type, sizeof(FileType), 1, f);
    if(DATA != type) {
        fprintf(stderr, "Error: It is not a data file.\n");
        exit(1);
    }
    // check version
    float ver;
    fread(&ver, sizeof(float), 1, f);
    if(DATAVER != ver) exit_file_ver(DATAVER, ver);

    // meta: nr_rs, avg, nr_s
    fread(&nr_rs, sizeof(int), 1, f);       //nr_rs
    fread(&avg, sizeof(float), 1, f);       //avg
    fread(nr_s, sizeof(int), nr_id, f);     //nr_s
}

template<int nr_id>
void Matrix<nr_id>::read(char *path) {
    printf("Reading from %s...", path);
    fflush(stdout);
    Clock clock;
    clock.tic();

    FILE *f = fopen(path, "rb");
    if(!f)exit_file_error(path);

    read_meta(f);

    M = new Node<nr_id>[nr_rs];
    fread(M, sizeof(Node<nr_id>), nr_rs, f);

    fclose(f);

    printf("done. %.2f\n", clock.toc());
    fflush(stdout);
}

template<int nr_id>
void Matrix<nr_id>::write(char *path) {
    printf("Writing %s... ", path);
    fflush(stdout);
    Clock clock;
    clock.tic();

    FILE *f = fopen(path, "wb");
    if(!f) exit_file_error(path);

    float ver = DATAVER;
    FileType type = DATA;
    fwrite(&type, sizeof(FileType), 1, f);
    fwrite(&ver, sizeof(float), 1, f);
    fwrite(&nr_rs, sizeof(int), 1, f);
    fwrite(&avg, sizeof(float), 1, f);
    fwrite(nr_s, sizeof(int), nr_id, f);
    fwrite(M, sizeof(Node<nr_id>), nr_rs, f);

    fclose(f);

    printf("done. %.2fs\n", clock.toc());
    fflush(stdout);
}

template<int nr_id>
void Matrix<nr_id>::sort() {
    std::sort(M, M + nr_rs, Matrix<nr_id>::compare);
}

// sort by ids
template<int nr_id>
bool Matrix<nr_id>::compare(Node<nr_id> lhs, Node<nr_id> rhs) {
    for(int i = 0; i < nr_id - 1; ++i) {
        if(lhs.id[i] != rhs.id[i])return lhs.id[i] < rhs.id[i];
    }
    return lhs.id[nr_id - 1] < rhs.id[nr_id - 1];
}

template<int nr_id>
Matrix<nr_id>::~Matrix() {
    if(NULL != M) {
        delete[]M;
        M = NULL;
    }
}

template<int nr_id>
Similarity<nr_id>::Similarity() {}

template<int nr_id>
Similarity<nr_id>::Similarity(char *path, int **map = NULL) {
    read(path, map);
}

template<int nr_id>
void Similarity<nr_id>::read(char *path, int **map) {
    std::stringstream filename;
    //ArrayIndex<int, 2> key;
    Node<2> node;
    int ka, kb;
    float val;
    for(int i = 0, ij = 0; i < nr_id; ++i) {
        for(int j = 0; j <= i; ++j, ++ij) {
            filename.str("");
            filename << path << "_" << ij;
            std::ifstream fin(filename.str());
            if(!fin) {
                en_sim[ij] = false;
                printf("Similarity file %d not exists", ij);
                continue;
            }
            en_sim[ij] = true;
            while(fin >> ka >> kb >> val) {
                if(map) {
                    ka = map[i][ka];
                    kb = map[j][kb];
                }
                if(ka < kb || i != j) {
                    node.id[0] = ka;
                    node.id[1] = kb;
                } else {
                    node.id[0] = kb;
                    node.id[1] = ka;
                }
                node.rate = val;
                M[ij].push_back(node);
                //M[ij][key] = val;
            }
            fin.close();
        }
    }
}

template<int nr_id>
void Similarity<nr_id>::generate(int recordNum, int termNum) {
    srand(unsigned(time(NULL)));
    //ArrayIndex<int, 2> key;
    Node<2> node;
    int ka, kb;
    for(int i = 0, ij = 0; i < nr_id; ++i) {
        for(int j = 0; j <= i; ++j, ++ij) {
            en_sim[ij] = true;
            for(int c = 0; c < recordNum; ++c) {
                ka = rand() % termNum;
                kb = rand() % termNum;
                if(ka < kb || i != j) {
                    node.id[0] = ka;
                    node.id[1] = kb;
                } else {
                    node.id[0] = kb;
                    node.id[1] = ka;
                }
                if(i != j || ka != kb)node.rate = float(rand()) / RAND_MAX;
                else node.rate = 1.0f;
                M[ij].push_back(node);
                //M[ij][key] = float(rand()) / RAND_MAX;
            }
        }
    }
}

template<int nr_id>
Model<nr_id>::Model(): en_rand_shuffle(false), en_avg(false), gamma(0.0F), avg(0.0F), dim(0), dim_off(0), nr_thrs(0), iter(0) {
    //en_b = new bool[nr_id];
    //l = new float[nr_id];
    //lb = new float[nr_id];
    //gl = new float[nr_id];
    //glb = new float[nr_id];
    //nr_s = new int[nr_id];
    //nr_gbs = new int[nr_id];
    //map_f = new int *[nr_id];
    //map_b = new int *[nr_id];
    //P = new float *[nr_id];
    //B = new float *[nr_id];
    for(int i = 0; i < nr_id; ++i) {
        en_b[i] = false;
        l[i] = 0.0F;
        lb[i] = 0.0F;
        gl[i] = 0.0F;
        glb[i] = 0.0F;
        nr_s[i] = 0;
        nr_gbs[i] = 0;
        map_f[i] = NULL;
        map_b[i] = NULL;
        P[i] = NULL;
        B[i] = NULL;
    }
}

template<int nr_id>
Model<nr_id>::Model(char *path): en_rand_shuffle(false), en_avg(false), gamma(0.0F), avg(0.0F), dim(0), dim_off(0), nr_thrs(0), iter(0) {
    for(int i = 0; i < nr_id; ++i) {
        en_b[i] = false;
        l[i] = 0.0F;
        lb[i] = 0.0F;
        gl[i] = 0.0F;
        glb[i] = 0.0F;
        nr_s[i] = 0;
        nr_gbs[i] = 0;
        map_f[i] = NULL;
        map_b[i] = NULL;
        P[i] = NULL;
        B[i] = NULL;
    }
    read(path);
}

template<int nr_id>
void Model<nr_id>::initialize(const Matrix<nr_id> &Tr) {
    printf("Initializing model...");
    fflush(stdout);
    Clock clock;
    clock.tic();

    dim_off = dim % 4 ? (dim / 4) * 4 + 4 : dim;
    avg = en_avg ? Tr.avg : 0.0f;
    memcpy(nr_s, Tr.nr_s, nr_id * sizeof(int));
    srand48(unsigned(time(NULL)));
    for(int i = 0; i < nr_id; ++i) {
        //srand48(0L);
        // assign gl = 1 - gamma * l
        gl[i] = 1 - gamma * l[i];
        glb[i] = 1 - gamma * lb[i];

        // assign P
        P[i] = new float[nr_s[i] * dim_off];
        float *pq = P[i];
        for(int j = 0; j < nr_s[i]; ++j) {
            for(int k = 0; k < dim; ++k)
                *(pq++) = float(0.1 * drand48());
            for(int k = dim; k < dim_off; ++k)
                *(pq++) = 0.0f;
        }

        //assign b
        if(en_b[i]) {
            B[i] = new float[nr_s[i]];
            for(int j = 0; j < nr_s[i]; ++j)B[i][j] = 0;
        }

        // assign lambda3&5
        lambda3[i] = 0.01f;
        lambda5[i] = 0.01f;
    }

    // assign lambda2&4
    for(int i = 0, ij = 0; i < nr_id; ++i) {
        for(int j = 0; j <= i; ++j, ++ij) {
            lambda2[ij] = 0.1f;
            lambda4[ij] = 0.1f;
        }
    }

    printf("done. %.2f\n", clock.toc());
    fflush(stdout);
}

template<int nr_id>
void Model<nr_id>::read_meta(FILE *f) {
    FileType type;
    fread(&type, sizeof(FileType), 1, f);
    if(MODEL != type) {
        fprintf(stderr, "Error: It is not a model file.\n");
        exit(1);
    }
    float ver;
    fread(&ver, sizeof(float), 1, f);
    if(MODELVER != ver) exit_file_ver(MODELVER, ver);

    fread(&dim, sizeof(int), 1, f);
    fread(&dim_off, sizeof(int), 1, f);
    fread(&nr_thrs, sizeof(int), 1, f);
    fread(&iter, sizeof(int), 1, f);
    fread(&gamma, sizeof(float), 1, f);
    fread(&avg, sizeof(float), 1, f);
    fread(&en_rand_shuffle, sizeof(bool), 1, f);
    fread(&en_avg, sizeof(bool), 1, f);
    fread(nr_s, sizeof(int), nr_id, f);
    fread(nr_gbs, sizeof(int), nr_id, f);
    fread(l, sizeof(float), nr_id, f);
    fread(lb, sizeof(float), nr_id, f);
    fread(gl, sizeof(float), nr_id, f);
    fread(glb, sizeof(float), nr_id, f);
    fread(en_b, sizeof(bool), nr_id, f);
}

template<int nr_id>
void Model<nr_id>::read(char *path) {
    printf("Reading model...");
    fflush(stdout);
    Clock clock;
    clock.tic();

    FILE *f = fopen(path, "rb");
    if(!f) exit_file_error(path);

    read_meta(f);

    for(int i = 0; i < nr_id; ++i) {
        P[i] = new float[nr_s[i] * dim_off];
        fread(P[i], sizeof(float), nr_s[i] * dim_off, f);
        if(en_b[i]) {
            B[i] = new float[nr_s[i]];
            fread(B[i], sizeof(float), nr_s[i], f);
        }
    }
    if(en_rand_shuffle) {
        for(int i = 0; i < nr_id; ++i) {
            map_f[i] = new int[nr_s[i]];
            map_b[i] = new int[nr_s[i]];
            fread(map_f[i], sizeof(int), nr_s[i], f);
            fread(map_b[i], sizeof(int), nr_s[i], f);
        }
    }

    fclose(f);
    printf("done. %.2f\n", clock.toc());
    fflush(stdout);
}

template<int nr_id>
void Model<nr_id>::write(char *path) {
    printf("Writing model...");
    fflush(stdout);
    Clock clock;
    clock.tic();

    FILE *f = fopen(path, "wb");
    if(!f) exit_file_error(path);
    float ver = (float)MODELVER;
    FileType file_type = MODEL;

    fwrite(&file_type, sizeof(FileType), 1, f);
    fwrite(&ver, sizeof(float), 1, f);
    fwrite(&dim, sizeof(int), 1, f);
    fwrite(&dim_off, sizeof(int), 1, f);
    fwrite(&nr_thrs, sizeof(int), 1, f);
    fwrite(&iter, sizeof(int), 1, f);
    fwrite(&gamma, sizeof(float), 1, f);
    fwrite(&avg, sizeof(float), 1, f);
    fwrite(&en_rand_shuffle, sizeof(bool), 1, f);
    fwrite(&en_avg, sizeof(bool), 1, f);
    fwrite(nr_s, sizeof(int), nr_id, f);
    fwrite(nr_gbs, sizeof(int), nr_id, f);
    fwrite(l, sizeof(float), nr_id, f);
    fwrite(lb, sizeof(float), nr_id, f);
    fwrite(gl, sizeof(float), nr_id, f);
    fwrite(glb, sizeof(float), nr_id, f);
    fwrite(en_b, sizeof(bool), nr_id, f);
    for(int i = 0; i < nr_id; ++i) {
        fwrite(P[i], sizeof(float), nr_s[i] * dim_off, f);
        if(en_b[i]) fwrite(B[i], sizeof(float), nr_s[i], f);
    }
    if(en_rand_shuffle) {
        for(int i = 0; i < nr_id; ++i) {
            fwrite(map_f[i], sizeof(int), nr_s[i], f);
            fwrite(map_b[i], sizeof(int), nr_s[i], f);
        }
    }

    fclose(f);
    printf("done. %.2f\n", clock.toc());
    fflush(stdout);
}

template<int nr_id>
void Model<nr_id>::gen_rand_map() {
    for(int i = 0; i < nr_id; ++i) {
        map_f[i] = new int[nr_s[i]];
        map_b[i] = new int[nr_s[i]];
        for(int j = 0; j < nr_s[i]; ++j) map_f[i][j] = j;
        std::random_shuffle(map_f[i], map_f[i] + nr_s[i]);
        for(int j = 0; j < nr_s[i]; ++j)map_b[i][map_f[i][j]] = j;
    }
}

template<int nr_id>
void Model<nr_id>::shuffle() {
    for(int i = 0; i < nr_id; ++i) {
        float *_P = new float[nr_s[i] * dim_off];
        for(int j = 0; j < nr_s[i]; ++j) {
            std::copy(&P[i][j * dim_off], &P[i][j * dim_off + dim_off], &_P[map_f[i][j] * dim_off]);
        }
        swap(_P, P[i]);
        delete[] _P;
        _P = NULL;
        if(en_b[i]) {
            float *_B = new float[nr_s[i]];
            for(int j = 0; j < nr_s[i]; ++j)_B[map_f[i][j]] = B[i][j];
            swap(_B, B[i]);
            delete _B;
            _B = NULL;
        }
    }
}

template<int nr_id>
void Model<nr_id>::inv_shuffle() {
    for(int i = 0; i < nr_id; ++i) {
        float *_P = new float[nr_s[i] * dim_off];
        for(int j = 0; j < nr_s[i]; ++j) {
            std::copy(&P[i][j * dim_off], &P[i][j * dim_off + dim_off], &_P[map_b[i][j] * dim_off]);
        }
        swap(_P, P[i]);
        delete[] _P;
        _P = NULL;
        if(en_b[i]) {
            float *_B = new float[nr_s[i]];
            for(int j = 0; j < nr_s[i]; ++j)_B[map_b[i][j]] = B[i][j];
            swap(_B, B[i]);
            delete _B;
            _B = NULL;
        }
    }
}

template<int nr_id>
Model<nr_id>::~Model() {
    for(int i = 0; i < nr_id; ++i) {
        if(NULL != map_f[i]) {
            delete[] map_f[i];
            map_f[i] = NULL;
        }
        if(NULL != map_b[i]) {
            delete[] map_b[i];
            map_b[i] = NULL;
        }
        if(NULL != P[i]) {
            delete[] P[i];
            P[i] = NULL;
        }
        if(en_b[i] && NULL != B[i]) {
            delete[] B[i];
            B[i] = NULL;
        }
    }
}

template<int nr_id>
float calc_rate(Model<nr_id> *model, Node<nr_id> *r) {
    float rate = model->avg;
    for(int i = 0; i < nr_id; ++i) {
        for(int j = i + 1; j < nr_id; ++j) {
            rate += std::inner_product(
                        &model->P[i][r->id[i] * model->dim_off],
                        &model->P[i][r->id[i] * model->dim_off] + model->dim,
                        &model->P[j][r->id[j] * model->dim_off], 0.0F);
        }
        if(model->en_b[i])rate += model->B[i][r->id[i]];
    }
    return rate;
}

template<int nr_id>
float calc_rmse(Model<nr_id> *model, Matrix<nr_id> *R) {
    double loss = 0;
    float e;
    for(int i = 0; i < R->nr_rs; i++) {
        e = R->M[i].rate - calc_rate(model, &R->M[i]);
        loss += e * e;
    }
    return (float)sqrt(loss / R->nr_rs);
}
