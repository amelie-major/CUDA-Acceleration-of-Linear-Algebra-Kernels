#include "benchmarks.hpp"
#include <cstdio>

void append_csv(const std::string& path, int rank,
                const std::string& mode, const std::string& kernel,
                long long N, int M, int Nmat, int K,
                double ms_gpu, double gflops, double gbs)
{
    FILE* f = std::fopen(path.c_str(), "a");
    if (!f) return;
    std::fprintf(f,
        "rank,%d,mode,%s,kernel,%s,N,%lld,M,%d,Nmat,%d,K,%d,ms_gpu,%.6f,GFLOPs,%.3f,GBs,%.3f\n",
        rank, mode.c_str(), kernel.c_str(), N, M, Nmat, K, ms_gpu, gflops, gbs);
    std::fclose(f);
}
