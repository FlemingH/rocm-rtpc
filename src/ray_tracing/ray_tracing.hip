#include "ray_tracing/ray_tracing.h"
#include "ray_tracing/rt_kernels.h"
#include "common/hip_check.h"
#include <cstdio>
#include <cmath>
#include <algorithm>

namespace rocm_rtpc {

GpuRayTracing::GpuRayTracing(int max_grid_n) {
    HIP_CHECK(hipStreamCreate(&stream_));

    int max_angles = MAX_ANGLES_COARSE;
    int total_slots = MAX_BEAMS * max_angles;

    HIP_CHECK(hipMalloc(&d_rho_dep_,     total_slots * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_eta_cd_,      total_slots * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_theta_grid_,  max_angles * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_phi_grid_,    max_angles * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_opt_theta_,  MAX_BEAMS * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_opt_phi_,    MAX_BEAMS * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_opt_rho_,    MAX_BEAMS * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_rho_targets_, MAX_BEAMS * sizeof(float)));

    // Pre-allocate equilibrium buffers for the maximum grid size
    eq_buf_elems_ = (size_t)max_grid_n * max_grid_n;
    HIP_CHECK(hipMalloc(&d_psi_,  eq_buf_elems_ * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_ne_,   eq_buf_elems_ * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_Te_,   eq_buf_elems_ * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_Bphi_, eq_buf_elems_ * sizeof(float)));

    // Pinned host buffers for async D2H readback (eliminates mid-pipeline sync)
    HIP_CHECK(hipHostMalloc(&h_opt_theta_, MAX_BEAMS * sizeof(float), hipHostMallocDefault));
    HIP_CHECK(hipHostMalloc(&h_opt_phi_,   MAX_BEAMS * sizeof(float), hipHostMallocDefault));
    HIP_CHECK(hipHostMalloc(&h_opt_rho_,   MAX_BEAMS * sizeof(float), hipHostMallocDefault));

    eq_nr_ = eq_nz_ = 0;
}

GpuRayTracing::~GpuRayTracing() {
    hipFree(d_rho_dep_);   hipFree(d_eta_cd_);
    hipFree(d_theta_grid_);hipFree(d_phi_grid_);
    hipFree(d_opt_theta_); hipFree(d_opt_phi_);
    hipFree(d_opt_rho_);   hipFree(d_rho_targets_);
    hipFree(d_psi_);  hipFree(d_ne_);
    hipFree(d_Te_);   hipFree(d_Bphi_);
    hipHostFree(h_opt_theta_);
    hipHostFree(h_opt_phi_);
    hipHostFree(h_opt_rho_);
    hipStreamDestroy(stream_);
}

void GpuRayTracing::upload_equilibrium(const EquilibriumData& eq) {
    eq_nr_ = eq.nr;  eq_nz_ = eq.nz;
    eq_R_min_ = eq.R_min;  eq_R_max_ = eq.R_max;
    eq_Z_min_ = eq.Z_min;  eq_Z_max_ = eq.Z_max;
    psi_axis_ = eq.psi_axis;  psi_bnd_ = eq.psi_boundary;

    size_t n = (size_t)eq.nr * eq.nz;
    if (n > eq_buf_elems_) {
        fprintf(stderr, "GpuRayTracing: grid %dx%d exceeds pre-allocated buffer\n",
                eq.nr, eq.nz);
        abort();
    }

    size_t bytes = n * sizeof(float);
    HIP_CHECK(hipMemcpyAsync(d_psi_,  eq.psi,  bytes, hipMemcpyHostToDevice, stream_));
    HIP_CHECK(hipMemcpyAsync(d_ne_,   eq.ne,   bytes, hipMemcpyHostToDevice, stream_));
    HIP_CHECK(hipMemcpyAsync(d_Te_,   eq.Te,   bytes, hipMemcpyHostToDevice, stream_));
    HIP_CHECK(hipMemcpyAsync(d_Bphi_, eq.Bphi, bytes, hipMemcpyHostToDevice, stream_));
    HIP_CHECK(hipStreamSynchronize(stream_));
}

void GpuRayTracing::generate_angle_grid_gpu(
    float theta_min, float theta_max,
    float phi_min, float phi_max,
    int n_theta, int n_phi)
{
    int n_total = n_theta * n_phi;
    float dtheta = (n_theta > 1) ? (theta_max - theta_min) / (n_theta - 1) : 0.0f;
    float dphi   = (n_phi > 1)   ? (phi_max - phi_min)     / (n_phi - 1)   : 0.0f;

    int threads = 256;
    int blocks = (n_total + threads - 1) / threads;
    generate_angle_grid_kernel<<<blocks, threads, 0, stream_>>>(
        d_theta_grid_, d_phi_grid_,
        theta_min, dtheta, phi_min, dphi,
        n_theta, n_phi);
}

void GpuRayTracing::multibeam_search(
    int num_beams, int n_angles,
    float theta_lo, float theta_hi,
    float phi_lo, float phi_hi,
    int n_theta, int n_phi)
{
    float dR = (eq_R_max_ - eq_R_min_) / (eq_nr_ - 1);
    float dZ = (eq_Z_max_ - eq_Z_min_) / (eq_nz_ - 1);

    generate_angle_grid_gpu(theta_lo, theta_hi, phi_lo, phi_hi, n_theta, n_phi);

    int threads = 256;
    dim3 grid_dim((n_angles + threads - 1) / threads, num_beams);

    ray_trace_multibeam_kernel<<<grid_dim, threads, 0, stream_>>>(
        d_psi_, d_ne_, d_Te_, d_Bphi_,
        eq_nr_, eq_nz_,
        eq_R_min_, dR, eq_Z_min_, dZ,
        psi_axis_, psi_bnd_,
        d_theta_grid_, d_phi_grid_, n_angles,
        LAUNCHER_R, LAUNCHER_Z, FREQ_GHZ,
        d_rho_dep_, d_eta_cd_, ODE_STEPS,
        num_beams);

    int opt_threads = 256;
    size_t opt_smem = opt_threads * (sizeof(float) + sizeof(int));

    angle_optimize_multibeam_kernel<<<num_beams, opt_threads, opt_smem, stream_>>>(
        d_rho_dep_, d_eta_cd_, d_theta_grid_, d_phi_grid_,
        n_angles, d_rho_targets_,
        d_opt_theta_, d_opt_phi_, d_opt_rho_,
        num_beams);
}

void GpuRayTracing::compute_optimal_angles(
    const ECRHTarget& target, BeamResult* results)
{
    int num_beams = target.num_beams;

    HIP_CHECK(hipMemcpyAsync(d_rho_targets_, target.rho_target,
                              num_beams * sizeof(float),
                              hipMemcpyHostToDevice, stream_));

    // Stage 1: Coarse search
    int n_theta_c = 10, n_phi_c = 10;
    int n_coarse = n_theta_c * n_phi_c;

    multibeam_search(num_beams, n_coarse,
                     THETA_MIN, THETA_MAX, PHI_MIN, PHI_MAX,
                     n_theta_c, n_phi_c);

    // Async readback of coarse results via pinned memory
    HIP_CHECK(hipMemcpyAsync(h_opt_theta_, d_opt_theta_,
                              num_beams * sizeof(float),
                              hipMemcpyDeviceToHost, stream_));
    HIP_CHECK(hipMemcpyAsync(h_opt_phi_, d_opt_phi_,
                              num_beams * sizeof(float),
                              hipMemcpyDeviceToHost, stream_));
    HIP_CHECK(hipStreamSynchronize(stream_));

    // Stage 2: Fine search around coarse optimal
    float dtheta = (THETA_MAX - THETA_MIN) / (n_theta_c - 1);
    float dphi   = (PHI_MAX - PHI_MIN) / (n_phi_c - 1);

    float t_lo = h_opt_theta_[0] - dtheta, t_hi = h_opt_theta_[0] + dtheta;
    float p_lo = h_opt_phi_[0] - dphi,     p_hi = h_opt_phi_[0] + dphi;
    for (int b = 1; b < num_beams; b++) {
        t_lo = std::min(t_lo, h_opt_theta_[b] - dtheta);
        t_hi = std::max(t_hi, h_opt_theta_[b] + dtheta);
        p_lo = std::min(p_lo, h_opt_phi_[b] - dphi);
        p_hi = std::max(p_hi, h_opt_phi_[b] + dphi);
    }

    int n_theta_f = 10, n_phi_f = 10;
    int n_fine = n_theta_f * n_phi_f;

    multibeam_search(num_beams, n_fine,
                     t_lo, t_hi, p_lo, p_hi,
                     n_theta_f, n_phi_f);

    // Final async readback
    HIP_CHECK(hipMemcpyAsync(h_opt_theta_, d_opt_theta_,
                              num_beams * sizeof(float),
                              hipMemcpyDeviceToHost, stream_));
    HIP_CHECK(hipMemcpyAsync(h_opt_phi_, d_opt_phi_,
                              num_beams * sizeof(float),
                              hipMemcpyDeviceToHost, stream_));
    HIP_CHECK(hipMemcpyAsync(h_opt_rho_, d_opt_rho_,
                              num_beams * sizeof(float),
                              hipMemcpyDeviceToHost, stream_));
    HIP_CHECK(hipStreamSynchronize(stream_));

    for (int b = 0; b < num_beams; b++) {
        results[b].theta_opt = h_opt_theta_[b];
        results[b].phi_opt   = h_opt_phi_[b];
        results[b].rho_dep   = h_opt_rho_[b];
    }
}

}  // namespace rocm_rtpc
