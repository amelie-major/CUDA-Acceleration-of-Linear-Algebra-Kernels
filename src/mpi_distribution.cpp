#include "mpi_utils.hpp"
#include <vector>

void build_counts_displs_rows(int M, int cols, int size, std::vector<int>& counts, std::vector<int>& displs) {
    counts.resize(size);
    displs.resize(size);
    int offset_rows = 0;
    for (int r=0; r<size; ++r) {
        auto d = dist_rows(M, 0, 0, r, size);
        counts[r] = d.M_local * cols;
        displs[r] = offset_rows * cols;
        offset_rows += d.M_local;
    }
}
