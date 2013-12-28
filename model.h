#pragma once
template<int nr_id>
struct Node {
    int *id;
    float rate;
    Node() {
        id = new int[nr_id];
    }
    Node(const Node<nr_id>& lhs) {
        id = new int[nr_id];
        for(int i = 0; i < nr_id; ++i)id[i] = lhs.id[i];
        rate = lhs.rate;
    }
    Node& operator = (Node& lhs) {
        for (int i = 0; i < nr_id; ++i)id[i] = lhs.id[i];
        rate = lhs.rate;
        return *this;
    }
    ~Node() {
        delete[]id;
    }
};

template<int nr_id>
struct Matrix {
    int *nr_s;  // Number of each dimension
    int nr_rs;  // Number of records
    float avg;
    Node<nr_id> *M;    // Records
    Matrix();
    Matrix(int nr_rs, int *nr_s, float avg);
    Matrix(char *path);
    Matrix(char *path, int **map);
    void read_meta(FILE *f);
    void read(char *path);
    void write(char *path);
    void sort();
    static bool compare(Node<nr_id> lhs, Node<nr_id> rhs);
    ~Matrix();
};

template<int nr_id>
struct Model {
    int *nr_s,      // 各维id数量
        dim,        // latent factor
        dim_off,    // latent factor （4的倍数）
        nr_thrs,    // 线程数
        iter,       // 迭代次数
        *nr_gbs,    // 各维分块数
        **map_f,   // 乱序映射
        **map_b;   // 乱序逆映射
    float **PQ,     // latent matrix: nr_id * (nr_s[nr_id] * dim_off)
          **B,      // bias: nr_id * nr_s[nr_id]
          *l,         //
          *lb,
          *gl,
          *glb,
          gamma,      // penalty
          avg;        // average rating
    bool en_rand_shuffle,   // 是否随机乱序
         en_avg,            // 是否使用 average rating
         *en_b;             // 是否使用 bias
    Model();
    Model(char *path);
    void initialize(Matrix<nr_id> *Tr);
    void read_meta(FILE *f);
    void read(char *path);
    void write(char *path);
    void gen_rand_map();
    void shuffle();
    void inv_shuffle();
    ~Model();
};

template<int nr_id>
Matrix<nr_id>::Matrix() : M(NULL), nr_rs(0), avg(0) {
    nr_s = new int[nr_id];
}

template<int nr_id>
Matrix<nr_id>::Matrix(int nr_rs1, int *nr_s1, float avg1) : M(NULL), avg(avg1), nr_rs(nr_rs1) {
    nr_s = new int[nr_id];
    memcpy(nr_s, nr_s1, sizeof(int)*nr_id);
    M = new Node<nr_id>[nr_rs];
}

template<int nr_id>
Matrix<nr_id>::Matrix(char *path) {
    nr_s = new int[nr_id];
    read(path);
}

template<int nr_id>
Matrix<nr_id>::Matrix(char *path, int **map) {
    nr_s = new int[nr_id];
    read(path);
    for(int i = 0; i < nr_rs; ++i) {
        for(int j = 0; j < nr_id; ++j) {
            M[i].id[j] = map[j][M[i].id[j]];
        }
    }
}

template<int nr_id>
void Matrix<nr_id>::read_meta(FILE *f) {
    FileType type;
    float ver;
    fread(&type, sizeof(FileType), 1, f);
    if(DATA != type) {
        fprintf(stderr, "Error: It is not a data file.\n");
        exit(1);
    }
    fread(&ver, sizeof(float), 1, f);
    if(DATAVER != ver) {
        exit_file_ver(DATAVER, ver);
    }
    int *nr_s1 = this->nr_s;
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
    for(int i = 0; i < nr_rs; ++i) {
        fread(M[i].id, sizeof(int), nr_id, f);
        fread(&(M[i].rate), sizeof(float), 1, f);
    }
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
    fwrite(&nr_rs, sizeof(int), 1, f);		//nr_rs
    fwrite(&avg, sizeof(float), 1, f);		//avg
    fwrite(nr_s, sizeof(int), nr_id, f);    //nr_s
    for(int i = 0; i < nr_rs; ++i) {
        fwrite(M[i].id, sizeof(int), nr_id, f);
        fwrite(&(M[i].rate), sizeof(float), 1, f);
    }
    fclose(f);
    printf("done. %.2fs\n", clock.toc());
    fflush(stdout);
}

template<int nr_id>
void Matrix<nr_id>::sort() {
    std::sort(M, M + nr_rs, Matrix<nr_id>::compare);
}

template<int nr_id>
bool Matrix<nr_id>::compare(Node<nr_id> lhs, Node<nr_id> rhs) {
    for(int i = 0; i < nr_id - 1; ++i) {
        if(lhs.id[i] != rhs.id[i])return lhs.id[i] < rhs.id[i];
    }
    return lhs.id[nr_id - 1] < rhs.id[nr_id - 1];
}

template<int nr_id>
Matrix<nr_id>::~Matrix() {
    if(NULL != nr_s) {
        delete[]nr_s;
        nr_s = NULL;
    }
    if(NULL != M) {
        delete[]M;
        M = NULL;
    }
}

template<int nr_id>
Model<nr_id>::Model() : en_rand_shuffle(false), en_avg(false), gamma(0.0F), avg(0.0F), dim(0), dim_off(0), nr_thrs(0), iter(0) {
    en_b = new bool[nr_id];
    l = new float[nr_id];
    lb = new float[nr_id];
    gl = new float[nr_id];
    glb = new float[nr_id];
    nr_s = new int[nr_id];
    nr_gbs = new int[nr_id];
    map_f = new int *[nr_id];
    map_b = new int *[nr_id];
    PQ = new float *[nr_id];
    B = new float *[nr_id];
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
        PQ[i] = NULL;
        B[i] = NULL;
    }
}

template<int nr_id>
Model<nr_id>::Model(char *path) : en_rand_shuffle(false), en_avg(false), gamma(0.0F), avg(0.0F), dim(0), dim_off(0), nr_thrs(0), iter(0) {
    en_b = new bool[nr_id];
    l = new float[nr_id];
    lb = new float[nr_id];
    gl = new float[nr_id];
    glb = new float[nr_id];
    nr_s = new int[nr_id];
    nr_gbs = new int[nr_id];
    map_f = new int *[nr_id];
    map_b = new int *[nr_id];
    PQ = new float *[nr_id];
    B = new float *[nr_id];
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
        PQ[i] = NULL;
        B[i] = NULL;
    }
    read(path);
}

template<int nr_id>
void Model<nr_id>::initialize(Matrix<nr_id> *Tr) {
    printf("Initializing model...");
    fflush(stdout);
    Clock clock;
    clock.tic();
    dim_off = dim % 4 ? (dim / 4) * 4 + 4 : dim;
    avg = en_avg ? Tr->avg : 0.0f;
    for(int i = 0; i < nr_id; ++i) {
        nr_s[i] = Tr->nr_s[i];
        gl[i] = 1 - gamma * l[i];	// gl = 1 - gamma * l
        glb[i] = 1 - gamma * lb[i];
        srand48(0L);
        PQ[i] = new float[nr_s[i] * dim_off];
        float *pq = PQ[i];
        for(int j = 0; j < nr_s[i]; ++j) {
            for(int k = 0; k < dim; ++k)*(pq++) = float(0.1 * drand48());
            for(int k = dim; k < dim_off; ++k)*(pq++) = 0.0f;
        }
        if(en_b[i]) {
            B[i] = new float[nr_s[i]];
            for(int j = 0; j < nr_s[i]; ++j)B[i][j] = 0;
        }
    }
    printf("done. %.2f\n", clock.toc());
    fflush(stdout);
}

template<int nr_id>
void Model<nr_id>::read_meta(FILE *f) {
    FileType type;
    float ver;
    fread(&type, sizeof(FileType), 1, f);
    if(MODEL != type) {
        fprintf(stderr, "Error: It is not a model file.\n");
        exit(1);
    }
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
        PQ[i] = new float[nr_s[i] * dim_off];
        fread(PQ[i], sizeof(float), nr_s[i] * dim_off, f);
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
        fwrite(PQ[i], sizeof(float), nr_s[i] * dim_off, f);
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
        float *PQ1 = new float[nr_s[i] * dim_off];
        for(int j = 0; j < nr_s[i]; ++j) {
            std::copy(&PQ[i][j * dim_off], &PQ[i][j * dim_off + dim_off], &PQ1[map_f[i][j] * dim_off]);
        }
        delete[] PQ[i];
        PQ[i] = PQ1;
        if(en_b[i]) {
            float *B1 = new float[nr_s[i]];
            for(int j = 0; j < nr_s[i]; ++j)B1[map_f[i][j]] = B[i][j];
            delete B[i];
            B[i] = B1;
        }
    }
}

template<int nr_id>
void Model<nr_id>::inv_shuffle() {
    for(int i = 0; i < nr_id; ++i) {
        float *PQ1 = new float[nr_s[i] * dim_off];
        for(int j = 0; j < nr_s[i]; ++j) {
            std::copy(&PQ[i][j * dim_off], &PQ[i][j * dim_off + dim_off], &PQ1[map_b[i][j] * dim_off]);
        }
        delete[] PQ[i];
        PQ[i] = PQ1;
        if(en_b[i]) {
            float *B1 = new float[nr_s[i]];
            for(int j = 0; j < nr_s[i]; ++j)B1[map_b[i][j]] = B[i][j];
            delete[]B[i];
            B[i] = B1;
        }
    }
}

template<int nr_id>
Model<nr_id>::~Model() {
    if(NULL != map_f) {
        for(int i = 0; i < nr_id; ++i) {
            if(NULL != map_f[i]) {
                delete[] map_f[i];
            }
        }
        delete map_f;
        map_f = NULL;
    }
    if(NULL != map_b) {
        for(int i = 0; i < nr_id; ++i) {
            if(NULL != map_b[i]) {
                delete[] map_b[i];
            }
        }
        delete map_b;
        map_b = NULL;
    }
    if(NULL != PQ) {
        for(int i = 0; i < nr_id; ++i) {
            if(NULL != PQ[i]) {
                delete[] PQ[i];
            }
        }
        delete[] PQ;
        PQ = NULL;
    }
    if(NULL != B) {
        for(int i = 0; i < nr_id; ++i) {
            if(en_b[i] && NULL != B[i]) {
                delete[] B[i];
            }
        }
        delete[] B;
        B = NULL;
    }
    if(NULL != nr_s) {
        delete[] nr_s;
        nr_s = NULL;
    }
    if(NULL != nr_gbs) {
        delete[] nr_gbs;
        nr_gbs = NULL;
    }
    if(NULL != l) {
        delete[] l;
        l = NULL;
    }
    if(NULL != lb) {
        delete[] lb;
        lb = NULL;
    }
    if(NULL != gl) {
        delete[] gl;
        gl = NULL;
    }
    if(NULL != glb) {
        delete[] glb;
        glb = NULL;
    }
    if(NULL != en_b) {
        delete[] en_b;
        en_b = NULL;
    }
}

template<int nr_id>
float calc_rate(Model<nr_id> *model, Node<nr_id> *r) {
    float rate = model->avg;
    for(int i = 0; i < nr_id; ++i) {
        for(int j = i + 1; j < nr_id; ++j) {
            rate += std::inner_product(
                        &model->PQ[i][r->id[i] * model->dim_off],
                        &model->PQ[i][r->id[i] * model->dim_off] + model->dim,
                        &model->PQ[j][r->id[j] * model->dim_off], 0.0F);
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
