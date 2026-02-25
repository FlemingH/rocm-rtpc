#pragma once

#include "common/types.h"
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

namespace rocm_rtpc {

// GPU-EFIT: Grad-Shafranov solver (P-EFIT 5-step + Anderson(3) + NN warm-start).
// All hot-path linear algebra delegated to rocBLAS (SGEMM, SGEAM, SGEMV, SDOT).
// Minimum grid: 129×129 (industry standard for real-time plasma control).
class GpuEfit {
public:
    explicit GpuEfit(int grid_size = 129);
    ~GpuEfit();

    void initialize();

    void reconstruct(const float* h_J_plasma, EquilibriumData& eq_out,
                     int max_iterations = 10, float tol = 1e-4f);

private:
    int N_;
    int M_;

    float* d_psi_;
    float* d_psi_new_;
    float* d_psi_rhs_;
    float* d_Q_;          // Q is symmetric (sin basis), so Q == Q^T
    float* d_J_plasma_;
    float* d_psi_bnd_;
    float* d_work1_;
    float* d_work2_;
    float* d_a_coeff_;
    float* d_m_coeff_;

    float* d_ne_;
    float* d_Te_;
    float* d_Bphi_;

    float* d_conv_max_;
    float* h_conv_pinned_;

    // Green matrix: FP32 column-major [N_bnd × N_inner] for rocblas_sgemv
    float* d_G_col_;

    // rocBLAS device-side scalar constants
    float* d_alpha_one_;
    float* d_beta_zero_;

    // Anderson acceleration
    float* d_anderson_Gx_[ANDERSON_DEPTH + 1];
    float* d_anderson_res_[ANDERSON_DEPTH + 1];
    float* d_dot_batch_;
    float* h_dot_pinned_;

    float* h_psi_pinned_;
    float* h_ne_pinned_;
    float* h_Te_pinned_;
    float* h_Bphi_pinned_;

    hipStream_t stream_;
    rocblas_handle rocblas_handle_;

    int N_bnd_;
    int N_inner_;

    void gs_step1_eigen_decomp(float* d_out, const float* d_in);
    void gs_step2_transpose(float* d_out, const float* d_in);
    void gs_step3_tridiag_solve(float* d_out, const float* d_rhs);
    void gs_step5_inv_eigen(float* d_out, const float* d_in);

    void compute_green_boundary(float* d_psi_bnd, const float* d_J);
    void compute_profiles_from_psi();

    void  launch_convergence_async();
    float read_convergence_result();

    void precompute_eigen_matrix();
    void precompute_green_matrix();
    void precompute_tridiag_coefficients();

    void nn_initialize_psi();
    void scatter_to_grid(float* d_NxN, const float* d_MxM);
    void reconstruct_anderson(int max_iterations, float tol);
    int  anderson_solve_ls(int iter, int depth, float* alpha);
};

}  // namespace rocm_rtpc
