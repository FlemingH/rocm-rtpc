#pragma once

#include "common/types.h"
#include <hip/hip_runtime.h>

namespace rocm_rtpc {

// GPU Ray Tracing: Hamilton ray equation solver for ECRH launcher optimization.
// All buffers pre-allocated; no per-call hipMalloc/hipFree in the hot path.
class GpuRayTracing {
public:
    explicit GpuRayTracing(int max_grid_n = 257);
    ~GpuRayTracing();

    void upload_equilibrium(const EquilibriumData& eq);

    void compute_optimal_angles(const ECRHTarget& target,
                                BeamResult* results);

private:
    // Pre-allocated equilibrium buffers (sized for max_grid_n × max_grid_n)
    float* d_psi_;
    float* d_ne_;
    float* d_Te_;
    float* d_Bphi_;
    size_t eq_buf_elems_;

    int    eq_nr_, eq_nz_;
    float  eq_R_min_, eq_R_max_, eq_Z_min_, eq_Z_max_;
    float  psi_axis_, psi_bnd_;

    // Ray tracing workspace
    float* d_rho_dep_;
    float* d_eta_cd_;
    float* d_theta_grid_;
    float* d_phi_grid_;
    float* d_opt_theta_;
    float* d_opt_phi_;
    float* d_opt_rho_;
    float* d_rho_targets_;

    // Pinned host buffers for async readback
    float* h_opt_theta_;
    float* h_opt_phi_;
    float* h_opt_rho_;

    hipStream_t stream_;

    static constexpr float LAUNCHER_R  = 2.2f;
    static constexpr float LAUNCHER_Z  = 0.8f;
    static constexpr float THETA_MIN   = -0.6f;
    static constexpr float THETA_MAX   =  0.3f;
    static constexpr float PHI_MIN     = -0.3f;
    static constexpr float PHI_MAX     =  0.3f;

    void generate_angle_grid_gpu(float theta_min, float theta_max,
                                 float phi_min, float phi_max,
                                 int n_theta, int n_phi);

    void multibeam_search(int num_beams, int n_angles,
                          float theta_lo, float theta_hi,
                          float phi_lo, float phi_hi,
                          int n_theta, int n_phi);
};

}  // namespace rocm_rtpc
