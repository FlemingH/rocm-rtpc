#pragma once

#include <cmath>

namespace rocm_rtpc {

struct EquilibriumData {
    int nr;
    int nz;
    float R_min, R_max;
    float Z_min, Z_max;

    float* psi;
    float* ne;
    float* Te;
    float* Bphi;

    float psi_axis;
    float psi_boundary;
    float R_axis, Z_axis;
};

struct BeamResult {
    float theta_opt;
    float phi_opt;
    float rho_dep;
};

struct ECRHTarget {
    int num_beams;
    float rho_target[12];
};

struct TimingInfo {
    double efit_ms;
    double transfer_ms;
    double raytrace_ms;
    double total_ms;
};

constexpr int    MAX_BEAMS          = 12;
constexpr int    MAX_ANGLES_COARSE  = 100;
constexpr int    ODE_STEPS          = 10000;
constexpr float  FREQ_GHZ           = 140.0f;
constexpr float  PI                 = 3.14159265358979323846f;
constexpr int    ANDERSON_DEPTH = 3;

}  // namespace rocm_rtpc
