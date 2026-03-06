#include <cmath>
#include <algorithm>

void cpu_axpy(int n, float alpha, const float* x, float* y) {
    for (int i=0;i<n;++i) y[i] = alpha * x[i] + y[i];
}
void cpu_vadd(int n, const float* x, const float* y, float* z) {
    for (int i=0;i<n;++i) z[i] = x[i] + y[i];
}
void cpu_vcopy(int n, const float* x, float* z) {
    for (int i=0;i<n;++i) z[i] = x[i];
}

float cpu_reduce_sum(int n, const float* x) {
    double s = 0.0;
    for (int i=0;i<n;++i) s += x[i];
    return (float)s;
}
void cpu_gemm(int M, int N, int K, const float* A, const float* B, float* C) {
    for (int i=0;i<M;++i) {
        for (int j=0;j<N;++j) {
            double acc = 0.0;
            for (int k=0;k<K;++k) acc += (double)A[i*K+k] * (double)B[k*N+j];
            C[i*N+j] = (float)acc;
        }
    }
}
float max_abs_diff(int n, const float* a, const float* b) {
    float m = 0.0f;
    for (int i=0;i<n;++i) m = std::max(m, std::fabs(a[i]-b[i]));
    return m;
}
