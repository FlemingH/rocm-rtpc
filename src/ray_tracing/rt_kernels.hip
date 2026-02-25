#include "ray_tracing/rt_kernels.h"
#include <float.h>

namespace rocm_rtpc {

// ═══════════════════════════════════════════════════════════════
// Generate angle grid directly on GPU
// ═══════════════════════════════════════════════════════════════
__global__ void generate_angle_grid_kernel(
    float* __restrict__ theta_grid,
    float* __restrict__ phi_grid,
    float theta_min, float dtheta,
    float phi_min, float dphi,
    int n_theta, int n_phi)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n_total = n_theta * n_phi;
    if (idx >= n_total) return;

    int it = idx / n_phi;
    int ip = idx % n_phi;
    theta_grid[idx] = theta_min + it * dtheta;
    phi_grid[idx]   = phi_min   + ip * dphi;
}

// ═══════════════════════════════════════════════════════════════
// Bilinear interpolation on 2D grid
// ═══════════════════════════════════════════════════════════════
__device__ float interp2d(const float* __restrict__ field,
                          int nr, int nz,
                          float R_min, float dR,
                          float Z_min, float dZ,
                          float R, float Z) {
    float fi = (R - R_min) / dR;
    float fj = (Z - Z_min) / dZ;
    int i0 = max(0, min((int)fi, nr - 2));
    int j0 = max(0, min((int)fj, nz - 2));
    float u = fi - i0;
    float v = fj - j0;

    float v00 = field[j0       * nr + i0];
    float v10 = field[j0       * nr + i0 + 1];
    float v01 = field[(j0 + 1) * nr + i0];
    float v11 = field[(j0 + 1) * nr + i0 + 1];

    return (1 - u) * (1 - v) * v00 + u * (1 - v) * v10 +
           (1 - u) * v       * v01 + u * v       * v11;
}

// ═══════════════════════════════════════════════════════════════
// Normalized flux coordinate ρ from ψ
// ═══════════════════════════════════════════════════════════════
__device__ float psi_to_rho(float psi_val, float psi_axis, float psi_bnd) {
    float psi_range = psi_bnd - psi_axis;
    if (fabsf(psi_range) < 1e-10f) return 0.0f;
    float psi_norm = (psi_val - psi_axis) / psi_range;
    psi_norm = fmaxf(0.0f, fminf(psi_norm, 2.0f));
    return sqrtf(psi_norm);
}

// Absorption coefficient (simplified relativistic)
__device__ float absorption_coeff(float ne, float Te, float B,
                                   float freq_ghz) {
    float omega_ce = B * 1.759e11f;
    float omega = freq_ghz * 2.0f * 3.14159265f * 1e9f;
    float delta = (omega - omega_ce) / omega_ce;
    float Te_norm = Te / 511.0e3f;

    float alpha = ne * 1e-20f * Te_norm *
                  expf(-0.5f * delta * delta / (Te_norm + 0.01f));
    return fmaxf(alpha, 0.0f);
}

// ═══════════════════════════════════════════════════════════════
// Multi-Beam Ray Tracing Kernel
// Optimized: refraction gradient uses only ne (4 interp2d instead of 12)
// ═══════════════════════════════════════════════════════════════
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
    float* __restrict__ rho_dep_out,
    float* __restrict__ eta_cd_out,
    int ode_steps,
    int num_beams)
{
    int angle_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int beam_idx  = blockIdx.y;
    if (angle_idx >= n_angles || beam_idx >= num_beams) return;

    int flat_idx = beam_idx * n_angles + angle_idx;

    float theta = theta_grid[angle_idx];
    float phi   = phi_grid[angle_idx];

    float R = launcher_R;
    float Z = launcher_Z;

    float kR = -cosf(theta) * cosf(phi);
    float kZ = -sinf(theta);

    float ds = 0.001f;
    float power_remaining = 1.0f;
    float total_absorption = 0.0f;
    float rho_at_max_absorption = 1.0f;
    float max_absorption_rate = 0.0f;
    float total_cd = 0.0f;

    float R_min_domain = R_min + dR;
    float R_max_domain = R_min + (nr - 2) * dR;
    float Z_min_domain = Z_min + dZ;
    float Z_max_domain = Z_min + (nz - 2) * dZ;

    // Precompute omega constants outside the loop
    float omega = freq_ghz * 2.0f * 3.14159265f * 1e9f;
    float inv_omega2 = 1.0f / (omega * omega);
    float pe2_factor = 3.18e-12f;

    for (int step = 0; step < ode_steps; step++) {
        if (R < R_min_domain || R > R_max_domain ||
            Z < Z_min_domain || Z > Z_max_domain)
            break;

        float ne_local = interp2d(ne, nr, nz, R_min, dR, Z_min, dZ, R, Z);

        if (ne_local < 1e15f) {
            R += kR * ds;
            Z += kZ * ds;
            continue;
        }

        float Te_local = interp2d(Te, nr, nz, R_min, dR, Z_min, dZ, R, Z);
        float B_local  = interp2d(Bphi, nr, nz, R_min, dR, Z_min, dZ, R, Z);

        float n2 = 1.0f - ne_local * pe2_factor * inv_omega2;

        float alpha = absorption_coeff(ne_local, Te_local, B_local, freq_ghz);
        float dP = alpha * ds;
        power_remaining -= dP;
        total_absorption += dP;

        if (alpha > max_absorption_rate) {
            max_absorption_rate = alpha;
            float psi_local = interp2d(psi, nr, nz, R_min, dR, Z_min, dZ, R, Z);
            rho_at_max_absorption = psi_to_rho(psi_local, psi_axis, psi_bnd);
        }

        total_cd += alpha * Te_local * 1e-3f * ds;

        // Refraction: only ne matters for O-mode n² gradient (4 interp2d, not 12)
        float eps = dR * 0.5f;
        float ne_Rp = interp2d(ne, nr, nz, R_min, dR, Z_min, dZ, R + eps, Z);
        float ne_Rm = interp2d(ne, nr, nz, R_min, dR, Z_min, dZ, R - eps, Z);
        float ne_Zp = interp2d(ne, nr, nz, R_min, dR, Z_min, dZ, R, Z + eps);
        float ne_Zm = interp2d(ne, nr, nz, R_min, dR, Z_min, dZ, R, Z - eps);

        float dn2_dR = -(ne_Rp - ne_Rm) * pe2_factor * inv_omega2 / (2.0f * eps);
        float dn2_dZ = -(ne_Zp - ne_Zm) * pe2_factor * inv_omega2 / (2.0f * eps);

        float n_mag = sqrtf(fmaxf(n2, 0.01f));
        kR += 0.5f * dn2_dR / n_mag * ds;
        kZ += 0.5f * dn2_dZ / n_mag * ds;

        float k_mag = sqrtf(kR * kR + kZ * kZ);
        if (k_mag > 1e-6f) { kR /= k_mag; kZ /= k_mag; }

        R += kR * n_mag * ds;
        Z += kZ * n_mag * ds;

        if (power_remaining < 0.05f) break;
    }

    rho_dep_out[flat_idx] = rho_at_max_absorption;
    eta_cd_out[flat_idx] = (total_absorption > 1e-6f) ?
        total_cd / total_absorption : 0.0f;
}

// ═══════════════════════════════════════════════════════════════
// Multi-Beam Angle Optimization
// ═══════════════════════════════════════════════════════════════
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
    int num_beams)
{
    extern __shared__ float smem[];
    float* s_cost = smem;
    int*   s_idx  = (int*)(smem + blockDim.x);

    int beam = blockIdx.x;
    if (beam >= num_beams) return;

    int tid = threadIdx.x;
    float rho_target = rho_targets[beam];

    float my_cost = FLT_MAX;
    int   my_idx  = -1;

    int base = beam * n_angles;
    for (int i = tid; i < n_angles; i += blockDim.x) {
        float drho = fabsf(rho_dep[base + i] - rho_target);
        float cost = drho - 0.01f * eta_cd[base + i];
        if (cost < my_cost) {
            my_cost = cost;
            my_idx  = i;
        }
    }

    s_cost[tid] = my_cost;
    s_idx[tid]  = my_idx;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && s_cost[tid + s] < s_cost[tid]) {
            s_cost[tid] = s_cost[tid + s];
            s_idx[tid]  = s_idx[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0 && s_idx[0] >= 0) {
        opt_theta[beam] = theta_grid[s_idx[0]];
        opt_phi[beam]   = phi_grid[s_idx[0]];
        opt_rho[beam]   = rho_dep[base + s_idx[0]];
    }
}

}  // namespace rocm_rtpc
