#include "gpu_efit/efit_kernels.h"

namespace rocm_rtpc {

// ═══════════════════════════════════════════════════════════════
// Tridiagonal solver (parallel prefix-sum)
// Each block solves one of M independent tridiagonal systems.
// ═══════════════════════════════════════════════════════════════
__global__ void tridiag_solve_kernel(
    const float* __restrict__ a_coeff,
    const float* __restrict__ m_coeff,
    const float* __restrict__ rhs,
    float* __restrict__ x,
    int M)
{
    int j = blockIdx.x;
    if (j >= M) return;

    extern __shared__ float sdata[];
    float* s_d = sdata;
    float* s_x = sdata + M;

    int tid = threadIdx.x;

    if (tid < M) {
        s_d[tid] = rhs[tid * M + j];
    }
    __syncthreads();

    for (int stride = 1; stride < M; stride <<= 1) {
        float val = 0.0f;
        if (tid >= stride && tid < M) {
            val = m_coeff[tid * M + (tid - stride)] * s_d[tid - stride];
        }
        __syncthreads();
        if (tid >= stride && tid < M) {
            s_d[tid] += val;
        }
        __syncthreads();
    }

    if (tid == M - 1) {
        s_x[tid] = s_d[tid] * a_coeff[tid];
    }
    __syncthreads();

    for (int stride = M >> 1; stride >= 1; stride >>= 1) {
        float val = 0.0f;
        if (tid < M - stride) {
            val = a_coeff[tid] * (s_d[tid] - s_x[tid + stride]);
        }
        __syncthreads();
        if (tid < M - stride) {
            s_x[tid] = val;
        }
        __syncthreads();
    }

    if (tid < M) {
        x[tid * M + j] = s_x[tid];
    }
}

// ═══════════════════════════════════════════════════════════════
// Anderson residual: r[i] = g[i] - x[i]
// ═══════════════════════════════════════════════════════════════
__global__ void anderson_residual_kernel(
    const float* __restrict__ g_x,
    const float* __restrict__ x,
    float* __restrict__ residual,
    int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        residual[i] = g_x[i] - x[i];
    }
}

// ═══════════════════════════════════════════════════════════════
// Anderson mixing
// ═══════════════════════════════════════════════════════════════
__global__ void anderson_mix_kernel(
    float* __restrict__ x_new,
    const float* __restrict__ g_current,
    const float* __restrict__ g_hist0,
    const float* __restrict__ g_hist1,
    const float* __restrict__ g_hist2,
    float alpha0, float alpha1, float alpha2,
    int N, int depth)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float val = (1.0f - alpha0 - alpha1 - alpha2) * g_current[i];
    if (depth >= 1) val += alpha0 * g_hist0[i];
    if (depth >= 2) val += alpha1 * g_hist1[i];
    if (depth >= 3) val += alpha2 * g_hist2[i];
    x_new[i] = val;
}

// ═══════════════════════════════════════════════════════════════
// Scatter M×M interior into N×N grid at [1..M, 1..M]
// ═══════════════════════════════════════════════════════════════
__global__ void scatter_interior_kernel(
    float* __restrict__ dst_NxN,
    const float* __restrict__ src_MxM,
    int M, int N)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= M || iy >= M) return;
    dst_NxN[(iy + 1) * N + (ix + 1)] = src_MxM[iy * M + ix];
}

// ═══════════════════════════════════════════════════════════════
// Convergence: max |ψ_new - ψ_old|
// ═══════════════════════════════════════════════════════════════
__global__ void convergence_kernel(
    const float* __restrict__ psi_new,
    const float* __restrict__ psi_old,
    float* __restrict__ max_diff,
    int N)
{
    extern __shared__ float smax[];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    float local_max = 0.0f;
    if (gid < N) {
        local_max = fabsf(psi_new[gid] - psi_old[gid]);
    }
    smax[tid] = local_max;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && smax[tid + s] > smax[tid]) {
            smax[tid] = smax[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicMax((int*)max_diff, __float_as_int(smax[0]));
    }
}

// ═══════════════════════════════════════════════════════════════
// Plasma profiles from ψ
// ═══════════════════════════════════════════════════════════════
__global__ void profiles_from_psi_kernel(
    const float* __restrict__ psi,
    float* __restrict__ ne,
    float* __restrict__ Te,
    float* __restrict__ Bphi,
    int nr, int nz,
    float R_min, float dR, float Z_min, float dZ,
    float R0, float B0, float ne0, float Te0,
    float psi_axis, float psi_bnd)
{
    (void)Z_min; (void)dZ;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nr || j >= nz) return;

    int idx = j * nr + i;
    float R = R_min + i * dR;

    float psi_range = psi_bnd - psi_axis;
    float psi_norm = (psi_range != 0.0f) ?
        (psi[idx] - psi_axis) / psi_range : 0.0f;
    psi_norm = fmaxf(0.0f, fminf(psi_norm, 1.0f));
    float rho = sqrtf(psi_norm);

    float val = 1.0f - rho * rho;
    ne[idx] = ne0 * (val > 0.0f ? sqrtf(val) : 0.0f);
    Te[idx] = Te0 * (val > 0.0f ? val * val : 0.0f);
    Bphi[idx] = (R > 0.1f) ? B0 * R0 / R : B0;
}

}  // namespace rocm_rtpc
