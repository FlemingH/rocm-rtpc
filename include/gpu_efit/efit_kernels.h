#pragma once

#include <hip/hip_runtime.h>
#include "common/types.h"

namespace rocm_rtpc {

// Tridiagonal solver (no rocBLAS equivalent for batched prefix-sum)
__global__ void tridiag_solve_kernel(
    const float* __restrict__ a_coeff,
    const float* __restrict__ m_coeff,
    const float* __restrict__ rhs,
    float* __restrict__ x,
    int M);

// Anderson acceleration
__global__ void anderson_residual_kernel(
    const float* __restrict__ g_x,
    const float* __restrict__ x,
    float* __restrict__ residual,
    int N);

__global__ void anderson_mix_kernel(
    float* __restrict__ x_new,
    const float* __restrict__ g_current,
    const float* __restrict__ g_hist0,
    const float* __restrict__ g_hist1,
    const float* __restrict__ g_hist2,
    float alpha0, float alpha1, float alpha2,
    int N, int depth);

// Scatter M×M interior into N×N grid at [1..M, 1..M]
__global__ void scatter_interior_kernel(
    float* __restrict__ dst_NxN,
    const float* __restrict__ src_MxM,
    int M, int N);

// Convergence: max |ψ_new - ψ_old|
__global__ void convergence_kernel(
    const float* __restrict__ psi_new,
    const float* __restrict__ psi_old,
    float* __restrict__ max_diff,
    int N);

// Plasma profiles from ψ
__global__ void profiles_from_psi_kernel(
    const float* __restrict__ psi,
    float* __restrict__ ne,
    float* __restrict__ Te,
    float* __restrict__ Bphi,
    int nr, int nz,
    float R_min, float dR, float Z_min, float dZ,
    float R0, float B0, float ne0, float Te0,
    float psi_axis, float psi_bnd);

}  // namespace rocm_rtpc
