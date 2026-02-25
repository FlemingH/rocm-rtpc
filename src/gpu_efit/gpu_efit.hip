#include "gpu_efit/gpu_efit.h"
#include "gpu_efit/efit_kernels.h"
#include "common/hip_check.h"
#include <rocblas/rocblas.h>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <algorithm>

#define ROCBLAS_CHECK(call) do {                                       \
    rocblas_status _s = (call);                                       \
    if (_s != rocblas_status_success) {                                \
        fprintf(stderr, "rocBLAS error %d at %s:%d\n", _s,           \
                __FILE__, __LINE__);                                  \
        abort();                                                      \
    }                                                                 \
} while(0)

namespace rocm_rtpc {

GpuEfit::GpuEfit(int grid_size) : N_(grid_size), M_(grid_size - 2) {
    if (N_ < 129) {
        fprintf(stderr, "GpuEfit: minimum grid size is 129 (got %d)\n", N_);
        abort();
    }

    N_bnd_   = 4 * (N_ - 1);
    N_inner_ = M_ * M_;

    HIP_CHECK(hipStreamCreate(&stream_));

    ROCBLAS_CHECK(rocblas_create_handle(&rocblas_handle_));
    ROCBLAS_CHECK(rocblas_set_stream(rocblas_handle_, stream_));
    ROCBLAS_CHECK(rocblas_set_pointer_mode(rocblas_handle_, rocblas_pointer_mode_device));

    size_t nn = (size_t)N_ * N_;
    size_t mm = (size_t)M_ * M_;

    HIP_CHECK(hipMalloc(&d_psi_,      nn * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_psi_new_,  nn * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_psi_rhs_,  mm * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_Q_,        mm * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_J_plasma_, N_inner_ * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_psi_bnd_,  N_bnd_ * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_work1_,    mm * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_work2_,    mm * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_a_coeff_,  M_ * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_m_coeff_,  mm * sizeof(float)));

    HIP_CHECK(hipMalloc(&d_ne_,   nn * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_Te_,   nn * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_Bphi_, nn * sizeof(float)));

    HIP_CHECK(hipMalloc(&d_conv_max_, sizeof(float)));
    HIP_CHECK(hipHostMalloc(&h_conv_pinned_, sizeof(float), hipHostMallocDefault));

    HIP_CHECK(hipMalloc(&d_G_col_, (size_t)N_bnd_ * N_inner_ * sizeof(float)));

    HIP_CHECK(hipMalloc(&d_alpha_one_, sizeof(float)));
    HIP_CHECK(hipMalloc(&d_beta_zero_, sizeof(float)));
    float one = 1.0f, zero = 0.0f;
    HIP_CHECK(hipMemcpy(d_alpha_one_, &one, sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_beta_zero_, &zero, sizeof(float), hipMemcpyHostToDevice));

    for (int i = 0; i <= ANDERSON_DEPTH; i++) {
        HIP_CHECK(hipMalloc(&d_anderson_Gx_[i],  nn * sizeof(float)));
        HIP_CHECK(hipMalloc(&d_anderson_res_[i], nn * sizeof(float)));
    }
    HIP_CHECK(hipMalloc(&d_dot_batch_, 16 * sizeof(float)));
    HIP_CHECK(hipHostMalloc(&h_dot_pinned_, 16 * sizeof(float), hipHostMallocDefault));

    HIP_CHECK(hipHostMalloc(&h_psi_pinned_,  nn * sizeof(float), hipHostMallocDefault));
    HIP_CHECK(hipHostMalloc(&h_ne_pinned_,   nn * sizeof(float), hipHostMallocDefault));
    HIP_CHECK(hipHostMalloc(&h_Te_pinned_,   nn * sizeof(float), hipHostMallocDefault));
    HIP_CHECK(hipHostMalloc(&h_Bphi_pinned_, nn * sizeof(float), hipHostMallocDefault));
}

GpuEfit::~GpuEfit() {
    hipFree(d_psi_);     hipFree(d_psi_new_);  hipFree(d_psi_rhs_);
    hipFree(d_Q_);
    hipFree(d_J_plasma_);hipFree(d_psi_bnd_);
    hipFree(d_work1_);   hipFree(d_work2_);
    hipFree(d_a_coeff_); hipFree(d_m_coeff_);
    hipFree(d_ne_);      hipFree(d_Te_);        hipFree(d_Bphi_);
    hipFree(d_conv_max_);
    hipHostFree(h_conv_pinned_);
    hipFree(d_G_col_);
    hipFree(d_alpha_one_); hipFree(d_beta_zero_);
    for (int i = 0; i <= ANDERSON_DEPTH; i++) {
        hipFree(d_anderson_Gx_[i]);
        hipFree(d_anderson_res_[i]);
    }
    hipFree(d_dot_batch_);
    hipHostFree(h_dot_pinned_);
    hipHostFree(h_psi_pinned_);
    hipHostFree(h_ne_pinned_);
    hipHostFree(h_Te_pinned_);
    hipHostFree(h_Bphi_pinned_);
    rocblas_destroy_handle(rocblas_handle_);
    hipStreamDestroy(stream_);
}

void GpuEfit::initialize() {
    precompute_eigen_matrix();
    precompute_green_matrix();
    precompute_tridiag_coefficients();
    HIP_CHECK(hipMemsetAsync(d_psi_, 0, (size_t)N_ * N_ * sizeof(float), stream_));

    // Warm up rocBLAS JIT for all kernel shapes used in the hot path
    ROCBLAS_CHECK(rocblas_sgemm(rocblas_handle_,
        rocblas_operation_none, rocblas_operation_none,
        M_, M_, M_,
        d_alpha_one_, d_work1_, M_, d_Q_, M_,
        d_beta_zero_, d_work2_, M_));
    ROCBLAS_CHECK(rocblas_sgeam(rocblas_handle_,
        rocblas_operation_transpose, rocblas_operation_none,
        M_, M_,
        d_alpha_one_, d_work1_, M_,
        d_beta_zero_, d_work1_, M_,
        d_work2_, M_));
    ROCBLAS_CHECK(rocblas_sgemv(rocblas_handle_,
        rocblas_operation_none,
        N_bnd_, N_inner_,
        d_alpha_one_, d_G_col_, N_bnd_,
        d_J_plasma_, 1,
        d_beta_zero_, d_psi_bnd_, 1));

    HIP_CHECK(hipStreamSynchronize(stream_));
}

// ─── Precomputation ──────────────────────────────────────────

void GpuEfit::precompute_eigen_matrix() {
    size_t mm = (size_t)M_ * M_;

    // Q is symmetric (sin basis), so row-major = col-major
    auto* h_Q = new float[mm];
    float scale = std::sqrt(2.0f / (M_ + 1));
    for (int i = 0; i < M_; i++) {
        for (int j = 0; j < M_; j++) {
            h_Q[i * M_ + j] = scale *
                std::sin(PI * (i + 1) * (j + 1) / (M_ + 1));
        }
    }
    HIP_CHECK(hipMemcpy(d_Q_, h_Q, mm * sizeof(float), hipMemcpyHostToDevice));

    delete[] h_Q;
}

void GpuEfit::precompute_green_matrix() {
    size_t size = (size_t)N_bnd_ * N_inner_;
    auto* h_G = new float[size];

    float h = 1.0f / (N_ - 1);
    float reg = h * 0.5f;

    for (int i = 0; i < N_bnd_; i++) {
        float xb, yb;
        int side = i / (N_ - 1);
        int pos  = i % (N_ - 1);
        float t  = pos * h;
        switch (side) {
            case 0: xb = t;   yb = 0.0f; break;
            case 1: xb = 1.0f; yb = t;   break;
            case 2: xb = 1.0f - t; yb = 1.0f; break;
            default: xb = 0.0f; yb = 1.0f - t; break;
        }

        for (int j = 0; j < N_inner_; j++) {
            int ix = j % M_;
            int iy = j / M_;
            float xi = (ix + 1) * h;
            float yi = (iy + 1) * h;
            float dx = xb - xi;
            float dy = yb - yi;
            float dist = std::sqrt(dx * dx + dy * dy + reg * reg);
            h_G[(size_t)i * N_inner_ + j] = -1.0f / (2.0f * PI * dist);
        }
    }

    // Column-major for rocblas_sgemv: G_col[j * N_bnd + i] = G_row[i * N_inner + j]
    auto* h_G_col = new float[size];
    for (int i = 0; i < N_bnd_; i++) {
        for (int j = 0; j < N_inner_; j++) {
            h_G_col[(size_t)j * N_bnd_ + i] = h_G[(size_t)i * N_inner_ + j];
        }
    }
    HIP_CHECK(hipMemcpy(d_G_col_, h_G_col, size * sizeof(float),
                         hipMemcpyHostToDevice));

    printf("  Green matrix %d×%d: FP32 col-major %.1f MB (rocBLAS sgemv)\n",
           N_bnd_, N_inner_,
           size * sizeof(float) / (1024.0 * 1024.0));

    delete[] h_G;
    delete[] h_G_col;
}

void GpuEfit::precompute_tridiag_coefficients() {
    auto* h_a = new float[M_];
    auto* h_m = new float[(size_t)M_ * M_];
    std::memset(h_m, 0, (size_t)M_ * M_ * sizeof(float));

    float h2 = 1.0f / ((N_ - 1.0f) * (N_ - 1.0f));
    for (int k = 0; k < M_; k++) {
        float lambda = -2.0f + 2.0f * std::cos(PI * (k + 1) / (M_ + 1));
        float diag = lambda / h2;
        h_a[k] = (std::fabs(diag) > 1e-12f) ? 1.0f / diag : 0.0f;
    }

    for (int i = 1; i < M_; i++) {
        h_m[i * M_ + (i - 1)] = -1.0f / h2 * h_a[i];
    }

    HIP_CHECK(hipMemcpy(d_a_coeff_, h_a, M_ * sizeof(float),
                         hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_m_coeff_, h_m, (size_t)M_ * M_ * sizeof(float),
                         hipMemcpyHostToDevice));

    delete[] h_a;
    delete[] h_m;
}

// ─── G-S Solver Steps (all rocBLAS) ──────────────────────────

void GpuEfit::gs_step1_eigen_decomp(float* d_out, const float* d_in) {
    ROCBLAS_CHECK(rocblas_sgemm(rocblas_handle_,
        rocblas_operation_none, rocblas_operation_none,
        M_, M_, M_,
        d_alpha_one_, d_in, M_, d_Q_, M_,
        d_beta_zero_, d_out, M_));
}

void GpuEfit::gs_step2_transpose(float* d_out, const float* d_in) {
    ROCBLAS_CHECK(rocblas_sgeam(rocblas_handle_,
        rocblas_operation_transpose, rocblas_operation_none,
        M_, M_,
        d_alpha_one_, d_in, M_,
        d_beta_zero_, d_in, M_,
        d_out, M_));
}

void GpuEfit::gs_step3_tridiag_solve(float* d_out, const float* d_rhs) {
    int threads = ((M_ + 31) / 32) * 32;
    size_t smem = 2 * M_ * sizeof(float);
    tridiag_solve_kernel<<<M_, threads, smem, stream_>>>(
        d_a_coeff_, d_m_coeff_, d_rhs, d_out, M_);
}

void GpuEfit::gs_step5_inv_eigen(float* d_out, const float* d_in) {
    ROCBLAS_CHECK(rocblas_sgemm(rocblas_handle_,
        rocblas_operation_none, rocblas_operation_none,
        M_, M_, M_,
        d_alpha_one_, d_in, M_, d_Q_, M_,
        d_beta_zero_, d_out, M_));
}

// ─── Green Boundary (rocBLAS sgemv) ─────────────────────────

void GpuEfit::compute_green_boundary(float* d_psi_bnd, const float* d_J) {
    ROCBLAS_CHECK(rocblas_sgemv(rocblas_handle_,
        rocblas_operation_none,
        N_bnd_, N_inner_,
        d_alpha_one_,
        d_G_col_, N_bnd_,
        d_J, 1,
        d_beta_zero_,
        d_psi_bnd, 1));
}

void GpuEfit::compute_profiles_from_psi() {
    float R0 = 1.85f, B0 = 2.0f, a = 0.5f;
    float R_min = R0 - 1.2f * a, R_max = R0 + 1.2f * a;
    float Z_min = -1.2f * a,     Z_max = 1.2f * a;
    float dR = (R_max - R_min) / (N_ - 1);
    float dZ = (Z_max - Z_min) / (N_ - 1);
    float ne0 = 6.0e19f, Te0 = 5.0e3f;

    dim3 block(16, 16);
    dim3 grid((N_ + 15) / 16, (N_ + 15) / 16);
    profiles_from_psi_kernel<<<grid, block, 0, stream_>>>(
        d_psi_, d_ne_, d_Te_, d_Bphi_,
        N_, N_, R_min, dR, Z_min, dZ,
        R0, B0, ne0, Te0, 0.0f, 1.0f);
}

// ─── Convergence ─────────────────────────────────────────────

void GpuEfit::launch_convergence_async() {
    int total = N_ * N_;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    HIP_CHECK(hipMemsetAsync(d_conv_max_, 0, sizeof(float), stream_));
    convergence_kernel<<<blocks, threads, threads * sizeof(float), stream_>>>(
        d_psi_new_, d_psi_, d_conv_max_, total);
    HIP_CHECK(hipMemcpyAsync(h_conv_pinned_, d_conv_max_, sizeof(float),
                              hipMemcpyDeviceToHost, stream_));
}

float GpuEfit::read_convergence_result() {
    HIP_CHECK(hipStreamSynchronize(stream_));
    return *h_conv_pinned_;
}

// ─── Scatter ─────────────────────────────────────────────────

void GpuEfit::scatter_to_grid(float* d_NxN, const float* d_MxM) {
    HIP_CHECK(hipMemsetAsync(d_NxN, 0, (size_t)N_ * N_ * sizeof(float), stream_));
    dim3 block(16, 16);
    dim3 grid((M_ + 15) / 16, (M_ + 15) / 16);
    scatter_interior_kernel<<<grid, block, 0, stream_>>>(d_NxN, d_MxM, M_, N_);
}

// ─── NN Initialization ───────────────────────────────────────

void GpuEfit::nn_initialize_psi() {
    HIP_CHECK(hipMemcpyAsync(d_psi_rhs_, d_J_plasma_,
                              (size_t)M_ * M_ * sizeof(float),
                              hipMemcpyDeviceToDevice, stream_));

    gs_step1_eigen_decomp(d_work1_, d_psi_rhs_);
    gs_step2_transpose(d_work2_, d_work1_);
    gs_step3_tridiag_solve(d_work1_, d_work2_);
    gs_step2_transpose(d_work2_, d_work1_);
    gs_step5_inv_eigen(d_work1_, d_work2_);

    scatter_to_grid(d_psi_, d_work1_);
}

// ─── Anderson LS Solver (rocBLAS sdot) ───────────────────────

int GpuEfit::anderson_solve_ls(int iter, int depth, float* alpha) {
    int m = std::min(depth, iter);
    if (m == 0) return 0;

    size_t nn = (size_t)N_ * N_;
    int cur_slot = iter % (ANDERSON_DEPTH + 1);

    int slots[4];
    slots[0] = cur_slot;
    for (int i = 1; i <= m; i++) {
        slots[i] = (iter - i + ANDERSON_DEPTH + 1) % (ANDERSON_DEPTH + 1);
    }

    int n_dots = 0;
    int pair_a[16], pair_b[16];
    for (int a = 0; a <= m; a++) {
        for (int b = a; b <= m; b++) {
            pair_a[n_dots] = a;
            pair_b[n_dots] = b;
            n_dots++;
        }
    }

    for (int d = 0; d < n_dots; d++) {
        ROCBLAS_CHECK(rocblas_sdot(rocblas_handle_,
            (rocblas_int)nn,
            d_anderson_res_[slots[pair_a[d]]], 1,
            d_anderson_res_[slots[pair_b[d]]], 1,
            d_dot_batch_ + d));
    }

    HIP_CHECK(hipMemcpyAsync(h_dot_pinned_, d_dot_batch_,
                              n_dots * sizeof(float),
                              hipMemcpyDeviceToHost, stream_));
    HIP_CHECK(hipStreamSynchronize(stream_));

    float dots[16] = {};
    for (int d = 0; d < n_dots; d++) {
        int a = pair_a[d], b = pair_b[d];
        dots[a * (m + 1) + b] = h_dot_pinned_[d];
        dots[b * (m + 1) + a] = h_dot_pinned_[d];
    }

    float H[9], b_vec[3];
    for (int i = 0; i < m; i++) {
        b_vec[i] = dots[0] - dots[i + 1];
        for (int j = 0; j < m; j++) {
            H[i * m + j] = dots[0]
                          - dots[i + 1]
                          - dots[j + 1]
                          + dots[(i + 1) * (m + 1) + (j + 1)];
        }
    }

    float gamma[3] = {};
    float A[9], rhs[3];
    std::memcpy(A, H, m * m * sizeof(float));
    std::memcpy(rhs, b_vec, m * sizeof(float));

    for (int col = 0; col < m; col++) {
        int pivot = col;
        float max_val = std::fabs(A[col * m + col]);
        for (int row = col + 1; row < m; row++) {
            if (std::fabs(A[row * m + col]) > max_val) {
                max_val = std::fabs(A[row * m + col]);
                pivot = row;
            }
        }
        if (max_val < 1e-12f) {
            return anderson_solve_ls(iter, m - 1, alpha);
        }
        if (pivot != col) {
            for (int k = 0; k < m; k++) std::swap(A[col * m + k], A[pivot * m + k]);
            std::swap(rhs[col], rhs[pivot]);
        }
        for (int row = col + 1; row < m; row++) {
            float factor = A[row * m + col] / A[col * m + col];
            for (int k = col; k < m; k++) A[row * m + k] -= factor * A[col * m + k];
            rhs[row] -= factor * rhs[col];
        }
    }
    for (int row = m - 1; row >= 0; row--) {
        gamma[row] = rhs[row];
        for (int k = row + 1; k < m; k++) gamma[row] -= A[row * m + k] * gamma[k];
        gamma[row] /= A[row * m + row];
    }

    for (int i = 0; i < m; i++) alpha[i] = gamma[i];
    return m;
}

// ─── Anderson-Accelerated Reconstruction ─────────────────────

void GpuEfit::reconstruct_anderson(int max_iterations, float tol) {
    size_t nn = (size_t)N_ * N_;
    int threads = 256;
    bool conv_pending = false;

    for (int iter = 0; iter < max_iterations; iter++) {
        if (conv_pending) {
            float max_diff = read_convergence_result();
            if (max_diff < tol && tol > 0.0f) {
                break;
            }
        }

        int cur_slot = iter % (ANDERSON_DEPTH + 1);

        compute_green_boundary(d_psi_bnd_, d_J_plasma_);

        HIP_CHECK(hipMemcpyAsync(d_psi_rhs_, d_J_plasma_,
                                  (size_t)M_ * M_ * sizeof(float),
                                  hipMemcpyDeviceToDevice, stream_));

        gs_step1_eigen_decomp(d_work1_, d_psi_rhs_);
        gs_step2_transpose(d_work2_, d_work1_);
        gs_step3_tridiag_solve(d_work1_, d_work2_);
        gs_step2_transpose(d_work2_, d_work1_);
        gs_step5_inv_eigen(d_work1_, d_work2_);

        scatter_to_grid(d_anderson_Gx_[cur_slot], d_work1_);

        int blocks_r = (nn + threads - 1) / threads;
        anderson_residual_kernel<<<blocks_r, threads, 0, stream_>>>(
            d_anderson_Gx_[cur_slot], d_psi_, d_anderson_res_[cur_slot], nn);

        int depth = std::min(iter, ANDERSON_DEPTH);
        if (depth == 0) {
            HIP_CHECK(hipMemcpyAsync(d_psi_new_, d_anderson_Gx_[cur_slot],
                                      nn * sizeof(float),
                                      hipMemcpyDeviceToDevice, stream_));
        } else {
            float alpha_arr[3] = {};
            int actual_depth = anderson_solve_ls(iter, depth, alpha_arr);

            if (actual_depth == 0) {
                HIP_CHECK(hipMemcpyAsync(d_psi_new_, d_anderson_Gx_[cur_slot],
                                          nn * sizeof(float),
                                          hipMemcpyDeviceToDevice, stream_));
            } else {
                float* g_hist[3] = {nullptr, nullptr, nullptr};
                for (int i = 0; i < actual_depth; i++) {
                    int hist_slot = (iter - 1 - i + ANDERSON_DEPTH + 1) % (ANDERSON_DEPTH + 1);
                    g_hist[i] = d_anderson_Gx_[hist_slot];
                }

                anderson_mix_kernel<<<blocks_r, threads, 0, stream_>>>(
                    d_psi_new_,
                    d_anderson_Gx_[cur_slot],
                    g_hist[0] ? g_hist[0] : d_anderson_Gx_[cur_slot],
                    g_hist[1] ? g_hist[1] : d_anderson_Gx_[cur_slot],
                    g_hist[2] ? g_hist[2] : d_anderson_Gx_[cur_slot],
                    alpha_arr[0], alpha_arr[1], alpha_arr[2],
                    nn, actual_depth);
            }
        }

        launch_convergence_async();
        conv_pending = true;

        std::swap(d_psi_, d_psi_new_);
    }
}

// ─── Main Reconstruction ─────────────────────────────────────

void GpuEfit::reconstruct(const float* h_J_plasma, EquilibriumData& eq_out,
                           int max_iterations, float tol) {
    HIP_CHECK(hipMemcpyAsync(d_J_plasma_, h_J_plasma,
                              N_inner_ * sizeof(float),
                              hipMemcpyHostToDevice, stream_));

    nn_initialize_psi();
    reconstruct_anderson(max_iterations, tol);

    compute_profiles_from_psi();

    size_t nn = (size_t)N_ * N_;
    size_t bytes = nn * sizeof(float);
    HIP_CHECK(hipMemcpyAsync(h_psi_pinned_,  d_psi_,  bytes, hipMemcpyDeviceToHost, stream_));
    HIP_CHECK(hipMemcpyAsync(h_ne_pinned_,   d_ne_,   bytes, hipMemcpyDeviceToHost, stream_));
    HIP_CHECK(hipMemcpyAsync(h_Te_pinned_,   d_Te_,   bytes, hipMemcpyDeviceToHost, stream_));
    HIP_CHECK(hipMemcpyAsync(h_Bphi_pinned_, d_Bphi_, bytes, hipMemcpyDeviceToHost, stream_));
    HIP_CHECK(hipStreamSynchronize(stream_));

    eq_out.nr = N_;
    eq_out.nz = N_;
    eq_out.R_min = 1.85f - 1.2f * 0.5f;
    eq_out.R_max = 1.85f + 1.2f * 0.5f;
    eq_out.Z_min = -1.2f * 0.5f;
    eq_out.Z_max =  1.2f * 0.5f;
    eq_out.psi_axis = 0.0f;
    eq_out.psi_boundary = 1.0f;
    eq_out.R_axis = 1.85f;
    eq_out.Z_axis = 0.0f;

    eq_out.psi  = new float[nn];
    eq_out.ne   = new float[nn];
    eq_out.Te   = new float[nn];
    eq_out.Bphi = new float[nn];

    std::memcpy(eq_out.psi,  h_psi_pinned_,  bytes);
    std::memcpy(eq_out.ne,   h_ne_pinned_,   bytes);
    std::memcpy(eq_out.Te,   h_Te_pinned_,   bytes);
    std::memcpy(eq_out.Bphi, h_Bphi_pinned_, bytes);
}

}  // namespace rocm_rtpc
