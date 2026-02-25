#include "gpu_efit/gpu_efit.h"
#include "gpu_efit/efit_kernels.h"
#include "ray_tracing/ray_tracing.h"
#include "common/plasma_profiles.h"
#include "common/timer.h"
#include "common/hip_check.h"
#include "distributed/rfm_transport.h"

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>

using namespace rocm_rtpc;

struct GpuTimer {
    hipEvent_t start_evt, stop_evt;
    GpuTimer() {
        HIP_CHECK(hipEventCreate(&start_evt));
        HIP_CHECK(hipEventCreate(&stop_evt));
    }
    ~GpuTimer() { hipEventDestroy(start_evt); hipEventDestroy(stop_evt); }
    void start(hipStream_t stream = nullptr) { HIP_CHECK(hipEventRecord(start_evt, stream)); }
    void stop(hipStream_t stream = nullptr)  { HIP_CHECK(hipEventRecord(stop_evt, stream)); }
    float elapsed_ms() {
        HIP_CHECK(hipEventSynchronize(stop_evt));
        float ms = 0.0f;
        HIP_CHECK(hipEventElapsedTime(&ms, start_evt, stop_evt));
        return ms;
    }
};

struct Stats { double min_ms, max_ms, mean_ms, median_ms, stddev_ms; int n; };

Stats compute_stats(std::vector<double>& samples) {
    Stats s{};
    s.n = (int)samples.size();
    if (s.n == 0) return s;
    std::sort(samples.begin(), samples.end());
    s.min_ms = samples.front();
    s.max_ms = samples.back();
    s.median_ms = (s.n % 2 == 0) ? (samples[s.n/2 - 1] + samples[s.n/2]) / 2.0 : samples[s.n/2];
    double sum = std::accumulate(samples.begin(), samples.end(), 0.0);
    s.mean_ms = sum / s.n;
    double sq_sum = 0.0;
    for (auto v : samples) sq_sum += (v - s.mean_ms) * (v - s.mean_ms);
    s.stddev_ms = (s.n > 1) ? std::sqrt(sq_sum / (s.n - 1)) : 0.0;
    return s;
}

void print_stats(const char* label, const Stats& s) {
    printf("  %-28s  %8.3f  %8.3f  %8.3f  %8.3f  %8.3f  (%d runs)\n",
           label, s.min_ms, s.median_ms, s.mean_ms, s.max_ms, s.stddev_ms, s.n);
}

struct DesignTarget { const char* name; int grid; int beams; double target_ms; double cpu_baseline_ms; };

static const DesignTarget efit_targets[] = {
    {"EFIT 129x129 10iter",  129, 0,   1.50,   24.0},
    {"EFIT 257x257 10iter",  257, 0,  21.00,  170.0},
    {"EFIT 513x513 10iter",  513, 0,  80.00, 1200.0},
};
static const DesignTarget rt_targets[] = {
    {"RayTrace  1 beam",   129,  1, 0.50, 20.0},
    {"RayTrace  4 beams",  129,  4, 0.50, 80.0},
    {"RayTrace  8 beams",  129,  8, 0.50,160.0},
    {"RayTrace 12 beams",  129, 12, 0.50,240.0},
};
static const DesignTarget pipeline_target = {
    "E2E Pipeline 129x129 4beam", 129, 4, 1.40, 25.0
};

float* make_J_plasma(int grid) {
    int M = grid - 2;
    int N_inner = M * M;
    auto* h_J = new float[N_inner];
    for (int i = 0; i < N_inner; i++) {
        int ix = i % M, iy = i / M;
        float x = (float)(ix + 1) / (grid - 1) - 0.5f;
        float y = (float)(iy + 1) / (grid - 1) - 0.5f;
        float r2 = x * x + y * y;
        h_J[i] = (r2 < 0.2f) ? 1.0f - r2 / 0.2f : 0.0f;
    }
    return h_J;
}

// ═══════════════════════════════════════════════════════════════════
// BENCHMARK 1: GPU-EFIT (Anderson+NN, all rocBLAS)
// ═══════════════════════════════════════════════════════════════════
void bench_efit(int grid, int warmup, int repeats) {
    printf("\n  [EFIT %d×%d, Anderson+NN, rocBLAS]\n", grid, grid);

    auto* h_J = make_J_plasma(grid);
    GpuEfit efit(grid);
    efit.initialize();

    for (int w = 0; w < warmup; w++) {
        EquilibriumData eq{};
        efit.reconstruct(h_J, eq, 10);
        PlasmaProfileGenerator::free_profiles(eq);
    }

    std::vector<double> times;
    Timer timer;
    for (int r = 0; r < repeats; r++) {
        EquilibriumData eq{};
        HIP_CHECK(hipDeviceSynchronize());
        timer.start();
        efit.reconstruct(h_J, eq, 10);
        HIP_CHECK(hipDeviceSynchronize());
        times.push_back(timer.elapsed_ms());
        PlasmaProfileGenerator::free_profiles(eq);
    }

    Stats s = compute_stats(times);
    print_stats("reconstruct()", s);
    delete[] h_J;
}

// ═══════════════════════════════════════════════════════════════════
// BENCHMARK 2: Individual EFIT kernels (hipEvent timing)
// Only kernels still in the production path are benchmarked.
// ═══════════════════════════════════════════════════════════════════
void bench_efit_kernels(int grid, int repeats) {
    printf("\n  [EFIT Kernel-level %d×%d]\n", grid, grid);

    int M = grid - 2;
    size_t mm = (size_t)M * M;

    float *d_A, *d_B, *d_C;
    float *d_a_coeff, *d_m_coeff;
    HIP_CHECK(hipMalloc(&d_A, mm * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_B, mm * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_C, mm * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_a_coeff, M * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_m_coeff, mm * sizeof(float)));

    auto* h_buf = new float[mm];
    for (size_t i = 0; i < mm; i++) h_buf[i] = 0.01f * (i % 100);
    HIP_CHECK(hipMemcpy(d_A, h_buf, mm * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B, h_buf, mm * sizeof(float), hipMemcpyHostToDevice));
    for (int i = 0; i < M; i++) h_buf[i] = 1.0f / (i + 1.0f);
    HIP_CHECK(hipMemcpy(d_a_coeff, h_buf, M * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_m_coeff, d_A, mm * sizeof(float), hipMemcpyDeviceToDevice));

    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));
    GpuTimer gtimer;

    // tridiag_solve_kernel (the only custom kernel in the hot path)
    { int threads = ((M + 31) / 32) * 32;
      size_t smem = 2 * M * sizeof(float);
      std::vector<double> times;
      for (int r = 0; r < repeats + 3; r++) {
          gtimer.start(stream);
          tridiag_solve_kernel<<<M, threads, smem, stream>>>(d_a_coeff, d_m_coeff, d_B, d_C, M);
          gtimer.stop(stream);
          if (r >= 3) times.push_back(gtimer.elapsed_ms());
      }
      print_stats("tridiag_solve_kernel", compute_stats(times)); }

    // convergence_kernel (uses M*M elements, matching d_A/d_B allocation)
    { int total = M * M, threads = 256, blocks = (total + threads - 1) / threads;
      float* d_max; HIP_CHECK(hipMalloc(&d_max, sizeof(float)));
      std::vector<double> times;
      for (int r = 0; r < repeats + 3; r++) {
          HIP_CHECK(hipMemsetAsync(d_max, 0, sizeof(float), stream));
          gtimer.start(stream);
          convergence_kernel<<<blocks, threads, threads * sizeof(float), stream>>>(d_A, d_B, d_max, total);
          gtimer.stop(stream);
          if (r >= 3) times.push_back(gtimer.elapsed_ms());
      }
      print_stats("convergence_kernel", compute_stats(times));
      hipFree(d_max); }

    // profiles_from_psi_kernel (allocate grid*grid buffers; d_A is M*M so use separate psi buffer)
    { int nn = grid * grid;
      float *d_psi_buf, *d_ne, *d_Te, *d_Bphi;
      HIP_CHECK(hipMalloc(&d_psi_buf, nn * sizeof(float)));
      HIP_CHECK(hipMemcpy(d_psi_buf, h_buf, std::min(mm, (size_t)nn) * sizeof(float), hipMemcpyHostToDevice));
      HIP_CHECK(hipMalloc(&d_ne, nn * sizeof(float)));
      HIP_CHECK(hipMalloc(&d_Te, nn * sizeof(float)));
      HIP_CHECK(hipMalloc(&d_Bphi, nn * sizeof(float)));
      dim3 blk(16, 16), grd((grid + 15) / 16, (grid + 15) / 16);
      std::vector<double> times;
      for (int r = 0; r < repeats + 3; r++) {
          gtimer.start(stream);
          profiles_from_psi_kernel<<<grd, blk, 0, stream>>>(
              d_psi_buf, d_ne, d_Te, d_Bphi, grid, grid, 1.25f, 0.01f, -0.6f, 0.01f,
              1.85f, 2.0f, 6.0e19f, 5.0e3f, 0.0f, 1.0f);
          gtimer.stop(stream);
          if (r >= 3) times.push_back(gtimer.elapsed_ms());
      }
      print_stats("profiles_from_psi_kernel", compute_stats(times));
      hipFree(d_psi_buf); hipFree(d_ne); hipFree(d_Te); hipFree(d_Bphi); }

    hipStreamDestroy(stream);
    hipFree(d_A); hipFree(d_B); hipFree(d_C);
    hipFree(d_a_coeff); hipFree(d_m_coeff);
    delete[] h_buf;
}

// ═══════════════════════════════════════════════════════════════════
// BENCHMARK 3: Ray Tracing
// ═══════════════════════════════════════════════════════════════════
void bench_ray_tracing(int grid, int num_beams, int warmup, int repeats) {
    printf("\n  [Ray Tracing %d beams, grid %d×%d]\n", num_beams, grid, grid);

    PlasmaProfileGenerator gen; EquilibriumData eq{}; gen.generate(eq, grid, grid);
    GpuRayTracing rt; rt.upload_equilibrium(eq);

    ECRHTarget target{}; target.num_beams = num_beams;
    for (int b = 0; b < num_beams; b++) {
        target.rho_target[b] = 0.3f + 0.4f * b / (num_beams > 1 ? num_beams - 1 : 1);
    }
    BeamResult results[MAX_BEAMS];

    for (int w = 0; w < warmup; w++) rt.compute_optimal_angles(target, results);

    std::vector<double> times; Timer timer;
    for (int r = 0; r < repeats; r++) {
        HIP_CHECK(hipDeviceSynchronize()); timer.start();
        rt.compute_optimal_angles(target, results);
        HIP_CHECK(hipDeviceSynchronize()); times.push_back(timer.elapsed_ms());
    }
    Stats s = compute_stats(times);
    print_stats("compute_optimal_angles()", s);
    printf("    → per beam: %.3f ms\n", s.median_ms / num_beams);
    PlasmaProfileGenerator::free_profiles(eq);
}

// ═══════════════════════════════════════════════════════════════════
// BENCHMARK 4: End-to-end pipeline
// ═══════════════════════════════════════════════════════════════════
void bench_pipeline(int grid, int beams, int warmup, int repeats) {
    printf("\n  [Pipeline E2E: grid %d, %d beams]\n", grid, beams);

    auto* h_J = make_J_plasma(grid);
    GpuEfit efit(grid); efit.initialize();
    GpuRayTracing rt;

    ECRHTarget target{}; target.num_beams = beams;
    for (int b = 0; b < beams; b++) {
        target.rho_target[b] = 0.3f + 0.15f * b;
    }
    BeamResult results[MAX_BEAMS];

    for (int w = 0; w < warmup; w++) {
        EquilibriumData eq{};
        efit.reconstruct(h_J, eq, 10);
        rt.upload_equilibrium(eq); rt.compute_optimal_angles(target, results);
        PlasmaProfileGenerator::free_profiles(eq);
    }

    std::vector<double> t_efit, t_xfer, t_rt, t_total;
    Timer timer, seg_timer;
    for (int r = 0; r < repeats; r++) {
        EquilibriumData eq_efit{}, eq_rt{};
        HIP_CHECK(hipDeviceSynchronize()); timer.start();

        seg_timer.start();
        efit.reconstruct(h_J, eq_efit, 10);
        HIP_CHECK(hipDeviceSynchronize());
        t_efit.push_back(seg_timer.elapsed_ms());

        seg_timer.start();
        RfmTransport::local_transfer(eq_efit, eq_rt);
        t_xfer.push_back(seg_timer.elapsed_ms());

        seg_timer.start();
        rt.upload_equilibrium(eq_rt); rt.compute_optimal_angles(target, results);
        HIP_CHECK(hipDeviceSynchronize());
        t_rt.push_back(seg_timer.elapsed_ms());

        t_total.push_back(timer.elapsed_ms());
        PlasmaProfileGenerator::free_profiles(eq_efit);
        PlasmaProfileGenerator::free_profiles(eq_rt);
    }

    print_stats("EFIT phase", compute_stats(t_efit));
    print_stats("RFM transfer phase", compute_stats(t_xfer));
    print_stats("Ray tracing phase", compute_stats(t_rt));
    print_stats("Pipeline total", compute_stats(t_total));
    delete[] h_J;
}

// ═══════════════════════════════════════════════════════════════════
// BENCHMARK 5: Memory bandwidth
// ═══════════════════════════════════════════════════════════════════
void bench_memory_bandwidth() {
    printf("\n  [Memory Bandwidth]\n");
    hipStream_t stream; HIP_CHECK(hipStreamCreate(&stream));
    GpuTimer gtimer;

    for (size_t size_mb : {1, 4, 16, 64, 128, 256}) {
        size_t bytes = size_mb * 1024 * 1024;
        float *d_src, *d_dst; float* h_src = new float[bytes / sizeof(float)];
        HIP_CHECK(hipMalloc(&d_src, bytes)); HIP_CHECK(hipMalloc(&d_dst, bytes));

        std::vector<double> times;
        for (int r = 0; r < 13; r++) {
            gtimer.start(stream); HIP_CHECK(hipMemcpyAsync(d_src, h_src, bytes, hipMemcpyHostToDevice, stream));
            gtimer.stop(stream); if (r >= 3) times.push_back(gtimer.elapsed_ms());
        }
        Stats s = compute_stats(times); double bw_h2d = bytes / (s.median_ms * 1e-3) / 1e9;

        times.clear();
        for (int r = 0; r < 13; r++) {
            gtimer.start(stream); HIP_CHECK(hipMemcpyAsync(d_dst, d_src, bytes, hipMemcpyDeviceToDevice, stream));
            gtimer.stop(stream); if (r >= 3) times.push_back(gtimer.elapsed_ms());
        }
        Stats s2 = compute_stats(times); double bw_d2d = bytes / (s2.median_ms * 1e-3) / 1e9;

        times.clear();
        for (int r = 0; r < 13; r++) {
            gtimer.start(stream); HIP_CHECK(hipMemcpyAsync(h_src, d_src, bytes, hipMemcpyDeviceToHost, stream));
            gtimer.stop(stream); if (r >= 3) times.push_back(gtimer.elapsed_ms());
        }
        Stats s3 = compute_stats(times); double bw_d2h = bytes / (s3.median_ms * 1e-3) / 1e9;

        printf("  %4zu MB  H2D: %7.1f GB/s (%6.3f ms)  D2D: %7.1f GB/s (%6.3f ms)"
               "  D2H: %7.1f GB/s (%6.3f ms)\n",
               size_mb, bw_h2d, s.median_ms, bw_d2d, s2.median_ms, bw_d2h, s3.median_ms);
        hipFree(d_src); hipFree(d_dst); delete[] h_src;
    }
    hipStreamDestroy(stream);
}

// ═══════════════════════════════════════════════════════════════════
// BENCHMARK 6: VRAM usage
// ═══════════════════════════════════════════════════════════════════
void bench_vram_usage() {
    printf("\n  [VRAM Usage Analysis]\n");
    for (int grid : {129, 257, 513}) {
        size_t free_before, total; HIP_CHECK(hipMemGetInfo(&free_before, &total));
        {
            GpuEfit efit(grid); efit.initialize();
            size_t free_after_efit; HIP_CHECK(hipMemGetInfo(&free_after_efit, &total));
            double efit_mb = (double)(free_before - free_after_efit) / (1024 * 1024);

            PlasmaProfileGenerator gen; EquilibriumData eq{}; gen.generate(eq, grid, grid);
            GpuRayTracing rt(grid); rt.upload_equilibrium(eq);
            size_t free_after_rt; HIP_CHECK(hipMemGetInfo(&free_after_rt, &total));
            double rt_mb = (double)(free_after_efit - free_after_rt) / (1024 * 1024);
            double total_mb = (double)(free_before - free_after_rt) / (1024 * 1024);

            printf("  Grid %3d×%-3d  EFIT: %8.1f MB  RT: %8.1f MB  Total: %8.1f MB"
                   "  (VRAM free: %.0f / %.0f MB)\n",
                   grid, grid, efit_mb, rt_mb, total_mb,
                   free_after_rt / (1024.0 * 1024.0), total / (1024.0 * 1024.0));
            PlasmaProfileGenerator::free_profiles(eq);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// MAIN
// ═══════════════════════════════════════════════════════════════════
int main(int argc, char** argv) {
    int warmup = 3, repeats = 20;
    bool run_all = true;
    bool run_efit = false, run_rt = false, run_pipe = false;
    bool run_kernel = false, run_mem = false, run_vram = false;
    bool run_target = false;

    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) warmup = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--repeats") == 0 && i + 1 < argc) repeats = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--efit") == 0)     { run_efit = true; run_all = false; }
        else if (std::strcmp(argv[i], "--rt") == 0)       { run_rt = true; run_all = false; }
        else if (std::strcmp(argv[i], "--pipeline") == 0) { run_pipe = true; run_all = false; }
        else if (std::strcmp(argv[i], "--kernel") == 0)   { run_kernel = true; run_all = false; }
        else if (std::strcmp(argv[i], "--mem") == 0)      { run_mem = true; run_all = false; }
        else if (std::strcmp(argv[i], "--vram") == 0)     { run_vram = true; run_all = false; }
        else if (std::strcmp(argv[i], "--target") == 0)   { run_target = true; run_all = false; }
    }

    printf("╔═══════════════════════════════════════════════════════════════════╗\n");
    printf("║   ROCM-RTPC Performance Benchmark                               ║\n");
    printf("╠═══════════════════════════════════════════════════════════════════╣\n");
    printf("║   Warmup: %d    Repeats: %d                                      ║\n", warmup, repeats);
    printf("╚═══════════════════════════════════════════════════════════════════╝\n");

    hipDeviceProp_t prop; HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    printf("\n  GPU:             %s\n", prop.name);
    printf("  Compute Units:   %d\n", prop.multiProcessorCount);
    printf("  Clock:           %d MHz\n", prop.clockRate / 1000);
    printf("  VRAM:            %.1f GB\n", prop.totalGlobalMem / 1e9);
    printf("  GCN Arch:        %s\n", prop.gcnArchName);
    printf("\n  Stats format:    [label]  min  median  mean  max  stddev  (N)\n");

    if (run_all || run_efit) {
        printf("\n══════════════════════════════════════════════════════════════════\n");
        printf("  SECTION 1: GPU-EFIT Reconstruction (Anderson+NN, rocBLAS)\n");
        printf("══════════════════════════════════════════════════════════════════\n");
        for (int grid : {129, 257, 513}) bench_efit(grid, warmup, repeats);
    }

    if (run_all || run_kernel) {
        printf("\n══════════════════════════════════════════════════════════════════\n");
        printf("  SECTION 2: EFIT Kernel-Level Profiling\n");
        printf("══════════════════════════════════════════════════════════════════\n");
        for (int grid : {129, 257, 513}) bench_efit_kernels(grid, repeats);
    }

    if (run_all || run_rt) {
        printf("\n══════════════════════════════════════════════════════════════════\n");
        printf("  SECTION 3: GPU Ray Tracing\n");
        printf("══════════════════════════════════════════════════════════════════\n");
        for (int beams : {1, 4, 8, 12}) bench_ray_tracing(129, beams, warmup, repeats);
    }

    if (run_all || run_pipe) {
        printf("\n══════════════════════════════════════════════════════════════════\n");
        printf("  SECTION 4: End-to-End Pipeline\n");
        printf("══════════════════════════════════════════════════════════════════\n");
        bench_pipeline(129, 4, warmup, repeats);
    }

    if (run_all || run_mem) {
        printf("\n══════════════════════════════════════════════════════════════════\n");
        printf("  SECTION 5: Memory Bandwidth\n");
        printf("══════════════════════════════════════════════════════════════════\n");
        bench_memory_bandwidth();
    }

    if (run_all || run_vram) {
        printf("\n══════════════════════════════════════════════════════════════════\n");
        printf("  SECTION 6: VRAM Usage\n");
        printf("══════════════════════════════════════════════════════════════════\n");
        bench_vram_usage();
    }

    if (run_all || run_target) {
        printf("\n══════════════════════════════════════════════════════════════════\n");
        printf("  SECTION 7: Design Target Comparison\n");
        printf("══════════════════════════════════════════════════════════════════\n");
        printf("\n  Running final comparison tests (median of %d runs)...\n\n", repeats);
        printf("  ┌────────────────────────────────┬──────────┬──────────┬──────────┬────────┬────────┐\n");
        printf("  │ Test Case                       │ Measured │ Target   │ CPU Base │ vs Tgt │vs CPU  │\n");
        printf("  ├────────────────────────────────┼──────────┼──────────┼──────────┼────────┼────────┤\n");

        for (auto& t : efit_targets) {
            auto* h_J = make_J_plasma(t.grid);
            GpuEfit efit(t.grid); efit.initialize();
            for (int w = 0; w < warmup; w++) { EquilibriumData eq{}; efit.reconstruct(h_J, eq, 10); PlasmaProfileGenerator::free_profiles(eq); }
            std::vector<double> times; Timer timer;
            for (int r = 0; r < repeats; r++) {
                EquilibriumData eq{}; HIP_CHECK(hipDeviceSynchronize()); timer.start();
                efit.reconstruct(h_J, eq, 10); HIP_CHECK(hipDeviceSynchronize());
                times.push_back(timer.elapsed_ms()); PlasmaProfileGenerator::free_profiles(eq);
            }
            Stats s = compute_stats(times);
            printf("  │ %-30s │ %8.3f │ %8.2f │ %8.1f │%s│ %5.1fx │\n",
                   t.name, s.median_ms, t.target_ms, t.cpu_baseline_ms,
                   (s.median_ms <= t.target_ms) ? " PASS " : " FAIL ", t.cpu_baseline_ms / s.median_ms);
            delete[] h_J;
        }

        for (auto& t : rt_targets) {
            PlasmaProfileGenerator gen; EquilibriumData eq{}; gen.generate(eq, t.grid, t.grid);
            GpuRayTracing rt; rt.upload_equilibrium(eq);
            ECRHTarget target{}; target.num_beams = t.beams;
            for (int b = 0; b < t.beams; b++) { target.rho_target[b] = 0.3f + 0.4f * b / (t.beams > 1 ? t.beams - 1 : 1); }
            BeamResult results[MAX_BEAMS];
            for (int w = 0; w < warmup; w++) rt.compute_optimal_angles(target, results);
            std::vector<double> times; Timer timer;
            for (int r = 0; r < repeats; r++) {
                HIP_CHECK(hipDeviceSynchronize()); timer.start();
                rt.compute_optimal_angles(target, results); HIP_CHECK(hipDeviceSynchronize());
                times.push_back(timer.elapsed_ms());
            }
            Stats s = compute_stats(times);
            printf("  │ %-30s │ %8.3f │ %8.2f │ %8.1f │%s│ %5.1fx │\n",
                   t.name, s.median_ms, t.target_ms, t.cpu_baseline_ms,
                   (s.median_ms <= t.target_ms) ? " PASS " : " FAIL ", t.cpu_baseline_ms / s.median_ms);
            PlasmaProfileGenerator::free_profiles(eq);
        }

        { auto& t = pipeline_target;
          auto* h_J = make_J_plasma(t.grid);
          GpuEfit efit(t.grid); efit.initialize(); GpuRayTracing grt;
          ECRHTarget target{}; target.num_beams = t.beams;
          for (int b = 0; b < t.beams; b++) { target.rho_target[b] = 0.3f + 0.15f * b; }
          BeamResult results[MAX_BEAMS];
          for (int w = 0; w < warmup; w++) {
              EquilibriumData eq{}, eq2{}; efit.reconstruct(h_J, eq, 10);
              RfmTransport::local_transfer(eq, eq2); grt.upload_equilibrium(eq2);
              grt.compute_optimal_angles(target, results);
              PlasmaProfileGenerator::free_profiles(eq); PlasmaProfileGenerator::free_profiles(eq2);
          }
          std::vector<double> times; Timer timer;
          for (int r = 0; r < repeats; r++) {
              EquilibriumData eq{}, eq2{}; HIP_CHECK(hipDeviceSynchronize()); timer.start();
              efit.reconstruct(h_J, eq, 10); RfmTransport::local_transfer(eq, eq2);
              grt.upload_equilibrium(eq2); grt.compute_optimal_angles(target, results);
              HIP_CHECK(hipDeviceSynchronize()); times.push_back(timer.elapsed_ms());
              PlasmaProfileGenerator::free_profiles(eq); PlasmaProfileGenerator::free_profiles(eq2);
          }
          Stats s = compute_stats(times);
          printf("  │ %-30s │ %8.3f │ %8.2f │ %8.1f │%s│ %5.1fx │\n",
                 t.name, s.median_ms, t.target_ms, t.cpu_baseline_ms,
                 (s.median_ms <= t.target_ms) ? " PASS " : " FAIL ", t.cpu_baseline_ms / s.median_ms);
          delete[] h_J;
        }

        printf("  └────────────────────────────────┴──────────┴──────────┴──────────┴────────┴────────┘\n");
        printf("\n  Legend: PASS = measured ≤ target,  FAIL = measured > target\n");
    }

    printf("\nBenchmark complete.\n");
    return 0;
}
