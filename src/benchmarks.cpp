#include "benchmarks.hpp"
#include <cstdio>
// Appends a line of benchmark results to a CSV file at the given path.

void append_csv(const std::string& path, int rank,
                const std::string& mode, const std::string& kernel,
                long long N, int M, int Nmat, int K,
                double ms_gpu, double gflops, double gbs,
                double ms_comm, double ms_total)
{
    FILE* f = std::fopen(path.c_str(), "a");
    if (!f) return;
    std::fprintf(f,
        "rank,%d,mode,%s,kernel,%s,N,%lld,M,%d,Nmat,%d,K,%d,ms_gpu,%.6f,GFLOPs,%.3f,GBs,%.3f,ms_comm,%.6f,ms_total,%.6f\n",
        rank, mode.c_str(), kernel.c_str(), N, M, Nmat, K, ms_gpu, gflops, gbs, ms_comm, ms_total);
    std::fclose(f);
}
