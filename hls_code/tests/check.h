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

// Two references: `fx` = numpy fixed-point simulation of the same formats (written by the export), `ref` = float
// model. PASS/FAIL tests the hardware arithmetic against `fx` (cos >= 0.999, MAE <= fx_mae); the float columns
// show what quantisation costs and are not part of the criterion (a model trained without QAT can saturate).
template <typename T>
static bool check_fx(const char* name, const T* hw, const float* fx, const float* ref, int n, double fx_mae = 0.01) {
    double sf = 0, mf = 0, df = 0, af = 0, bf = 0, sr = 0, dr = 0, br = 0;
    for (int i = 0; i < n; ++i) {
        const double h = (double)hw[i], f = fx[i], r = ref[i];
        sf += std::fabs(h - f); mf = std::fabs(h - f) > mf ? std::fabs(h - f) : mf; df += h * f; af += h * h; bf += f * f;
        sr += std::fabs(h - r); dr += h * r; br += r * r;
    }
    const double cf = df / std::sqrt(af * bf + 1e-30), cr = dr / std::sqrt(af * br + 1e-30);
    const bool ok = cf >= 0.999 && sf / n <= fx_mae;
    std::printf("  %-18s n=%5d  vs fixed: MAE=%.5f max=%.4f cos=%.5f %s | vs float: MAE=%.4f cos=%.4f\n",
                name, n, sf / n, mf, cf, ok ? "PASS" : "FAIL", sr / n, cr);
    return ok;
}
#endif
