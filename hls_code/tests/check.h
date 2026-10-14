#ifndef CHECK_H
#define CHECK_H
#include <cmath>
#include <cstdio>
// Compare a hardware array against a float reference: MAE, max error, cosine similarity.
template <typename T>
static bool check(const char* name, const T* hw, const float* ref, int n, double min_cos, double max_mae) {
    double sa = 0, mx = 0, d = 0, a = 0, b = 0;
    for (int i = 0; i < n; ++i) {
        const double h = (double)hw[i], r = ref[i], e = std::fabs(h - r);
        sa += e; mx = e > mx ? e : mx; d += h * r; a += h * h; b += r * r;
    }
    const double mae = sa / n, cos = d / std::sqrt(a * b + 1e-30);
    const bool ok = cos >= min_cos && mae <= max_mae;
    std::printf("  %-18s n=%5d  MAE=%.5f  max=%.4f  cos=%.5f  %s\n", name, n, mae, mx, cos, ok ? "PASS" : "FAIL");
    return ok;
}
#endif
