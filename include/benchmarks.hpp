#pragma once
#include <string>
void append_csv(const std::string& path, int rank,
                const std::string& mode, const std::string& kernel,
                long long N, int M, int Nmat, int K,
                double ms_gpu, double gflops, double gbs,
                double ms_comm=0.0, double ms_total=0.0);
