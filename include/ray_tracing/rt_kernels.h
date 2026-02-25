#pragma once

#include <hip/hip_runtime.h>
#include "common/types.h"

namespace rocm_rtpc {

// Generate angle grid on GPU (eliminates host alloc + H2D copy)
__global__ void generate_angle_grid_kernel(
    float* __restrict__ theta_grid,
    float* __restrict__ phi_grid,
    float theta_min, float dtheta,
    float phi_min, float dphi,
    int n_theta, int n_phi);

// Multi-beam ray tracing: blockIdx.y = beam, threadIdx.x + blockIdx.x = angle
__global__ void ray_trace_multibeam_kernel(
    const float* __restrict__ psi,
    const float* __restrict__ ne,
    const float* __restrict__ Te,
    const float* __restrict__ Bphi,
    int nr, int nz,
    float R_min, float dR, float Z_min, float dZ,
    float psi_axis, float psi_bnd,
    const float* __restrict__ theta_grid,
    const float* __restrict__ phi_grid,
    int n_angles,
    float launcher_R, float launcher_Z,
    float freq_ghz,
    float* __restrict__ rho_dep,
    float* __restrict__ eta_cd,
    int ode_steps,
    int num_beams);

// Multi-beam angle optimization: one block per beam
__global__ void angle_optimize_multibeam_kernel(
    const float* __restrict__ rho_dep,
    const float* __restrict__ eta_cd,
    const float* __restrict__ theta_grid,
    const float* __restrict__ phi_grid,
    int n_angles,
    const float* __restrict__ rho_targets,
    float* __restrict__ opt_theta,
    float* __restrict__ opt_phi,
    float* __restrict__ opt_rho,
    int num_beams);

}  // namespace rocm_rtpc
