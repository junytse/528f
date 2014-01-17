#include "header.h"
#include "model.h"

struct ConvertOption {
    char *src, *dst;
    ConvertOption(int argc, char **argv);
    static void exit_convert();
    ~ConvertOption();
};

ConvertOption::ConvertOption(int argc, char **argv) {
    if(argc != 3 && argc != 4) exit_convert();
    src = argv[2];
    if(argc == 4) {
        dst = new char[strlen(argv[3]) + 1];
        sprintf(dst, "%s", argv[3]);
    } else {
        char *p = strrchr(argv[2], '/');
        if(p == NULL) p = argv[2];
        else p++;
        dst = new char[strlen(p) + 5];
        sprintf(dst, "%s.bin", p);
    }
}

void ConvertOption::exit_convert() {
    printf(
        "usage: libmf convert text_file binary_file\n"
        "\n"
        "Convert a text file to a binary file\n"
    );
    exit(1);
}

ConvertOption::~ConvertOption() {
    delete[] dst;
}

template<int nr_id>
void convert(char *src_path, char *dst_path) {
    printf("Converting %s... ", src_path);
    fflush(stdout);
    Clock clock;
    clock.tic();
    FILE *f = fopen(src_path, "r");
    if(!f) exit_file_error(src_path);
    int *nr_s, nr_rs;
    nr_s = new int[nr_id];
    double sum = 0;
    std::vector<Node<nr_id>> rs;
    Node<nr_id> r;
    while(fscanf(f, "%d", &r.id[0]) != EOF) {
        if(r.id[0] + 1 > nr_s[0])nr_s[0] = r.id[0] + 1;
        for(int i = 1; i < nr_id; ++i) {
            fscanf(f, "%d", &r.id[i]);
            if(r.id[i] + 1 > nr_s[i])nr_s[i] = r.id[i] + 1;
        }
        fscanf(f, "%f", &r.rate);
        sum += r.rate;
        rs.push_back(r);
    }
    nr_rs = (int)rs.size();
    fclose(f);
    Matrix<nr_id> R(nr_rs, nr_s, (float)sum / nr_rs);
    delete[]nr_s;
    {
        int i = 0;
        for(auto it = rs.begin(); it != rs.end(); ++it, ++i) R.M[i] = (*it);
    }
    printf("done. %.2fs\n", clock.toc());
    fflush(stdout);
    R.write(dst_path);
}

template<int nr_id>
void convert(int argc, char **argv) {
    printf("%d\n", nr_id);
    ConvertOption option(argc, argv);
    convert<nr_id>(option.src, option.dst);
}
