[中文](README.md) | **English**

# ROCM-RTPC

**ROCm Real-Time Plasma Computation**

A GPU-accelerated real-time plasma equilibrium reconstruction and ECRH ray-tracing platform built on AMD ROCm/HIP, targeting sub-millisecond latency for tokamak plasma control systems (PCS) and electron cyclotron resonance heating (ECRH) feedback loops.

## Purpose

Modern tokamaks require real-time equilibrium reconstruction (EFIT) and microwave beam steering (ECRH) within a single plasma control cycle (~1–10 ms). Traditional CPU implementations cannot meet this deadline at high spatial resolution. ROCM-RTPC offloads the entire computation pipeline to AMD GPUs via HIP, achieving **>25× speedup** over CPU baselines and enabling real-time operation at resolutions previously limited to offline analysis.

**Key capabilities:**

- Grad-Shafranov equilibrium solver (GPU-EFIT) with Anderson acceleration and physics-informed warm start
- Hamilton ray-equation solver for ECRH launcher angle optimization (multi-beam parallel)
- Distributed two-node architecture: PCS node (EFIT) communicates with ECRH node (ray tracing) via simulated Reflective Memory (RFM)
- Full rocBLAS integration (SGEMM, SGEAM, SGEMV, SDOT) for hot-path linear algebra

## Performance

Measured on AMD Radeon RX 9070 (RDNA 4, gfx1201), median of 20 runs:

| Module | Grid | Measured | CPU Baseline | Speedup | Target |
|--------|------|----------|-------------|---------|--------|
| GPU-EFIT | 129×129 | **0.92 ms** | 24 ms | 26× | < 1.5 ms |
| GPU-EFIT | 257×257 | **1.15 ms** | 170 ms | 148× | < 21 ms |
| GPU-EFIT | 513×513 | **5.04 ms** | 1200 ms | 238× | < 80 ms |
| Ray Tracing (12 beams) | 129×129 | **0.08 ms** | 240 ms | 3116× | < 0.5 ms |
| E2E Pipeline (4 beams) | 129×129 | **1.09 ms** | 25 ms | 23× | < 1.4 ms |

## Architecture

### Distributed Deployment

```
┌─────────────────────┐         RFM / TCP          ┌─────────────────────────┐
│   PCS Node (GPU #1) │ ─────────────────────────→  │  ECRH-MC Node (GPU #2)  │
│                      │                             │                          │
│  Diagnostics         │                             │  EquilibriumData         │
│       ↓              │                             │       ↓                  │
│  gpu_efit_server     │                             │  ray_tracing_server      │
│   • J_plasma input   │                             │   • Optimal (θ, φ)       │
│   • ψ(R,Z) output    │                             │   • ρ_dep, η_cd          │
│   • ne, Te, Bφ       │                             │       ↓                  │
│       ↓              │                             │  ECRH Launcher Control   │
│  RFM send            │                             │                          │
└─────────────────────┘                              └─────────────────────────┘
```

In production, the two nodes communicate via Reflective Memory (RFM) hardware (e.g., GE VMIPCI-5565) for deterministic sub-100μs transfer. This implementation simulates RFM over TCP sockets, or uses direct memory copy for single-machine testing.

### Single-Node Mode

For development and benchmarking, the full pipeline runs in a single process:

```
J_plasma → [GPU-EFIT] → EquilibriumData → [local_transfer] → [Ray Tracing] → BeamResult
```

## Input / Output

### GPU-EFIT

| Direction | Data | Type | Size |
|-----------|------|------|------|
| **Input** | Plasma current density J(R,Z) | `float*` | M × M (interior grid) |
| **Input** | Max iterations, tolerance | `int`, `float` | scalars |
| **Output** | Poloidal flux ψ(R,Z) | `float*` | N × N |
| **Output** | Electron density ne(R,Z) | `float*` | N × N |
| **Output** | Electron temperature Te(R,Z) | `float*` | N × N |
| **Output** | Toroidal field Bφ(R,Z) | `float*` | N × N |
| **Output** | Magnetic axis (R₀, Z₀), ψ_axis, ψ_boundary | `float` | scalars |

where N = grid size (129, 257, or 513), M = N − 2.

### Ray Tracing

| Direction | Data | Type | Size |
|-----------|------|------|------|
| **Input** | EquilibriumData (from EFIT) | struct | N × N fields |
| **Input** | Target deposition ρ per beam | `float[12]` | up to 12 beams |
| **Output** | Optimal poloidal angle θ | `float` | per beam |
| **Output** | Optimal toroidal angle φ | `float` | per beam |
| **Output** | Actual deposition location ρ_dep | `float` | per beam |

## Code Structure

```
rocm-rtpc/
├── include/
│   ├── common/
│   │   ├── types.h              # EquilibriumData, BeamResult, ECRHTarget, constants
│   │   ├── hip_check.h          # HIP_CHECK error macro
│   │   ├── timer.h              # High-resolution wall-clock timer
│   │   └── plasma_profiles.h    # Solovev analytic equilibrium generator
│   ├── gpu_efit/
│   │   ├── gpu_efit.h           # GpuEfit class API
│   │   └── efit_kernels.h       # HIP kernel declarations
│   ├── ray_tracing/
│   │   ├── ray_tracing.h        # GpuRayTracing class API
│   │   └── rt_kernels.h         # HIP kernel declarations
│   └── distributed/
│       └── rfm_transport.h      # RFM transport layer API
├── src/
│   ├── common/
│   │   └── plasma_profiles.cpp  # Solovev equilibrium + profile generation
│   ├── gpu_efit/
│   │   ├── gpu_efit.hip         # EFIT solver: rocBLAS GEMM/GEAM/GEMV/DOT,
│   │   │                        #   Anderson acceleration, NN warm start
│   │   ├── efit_kernels.hip     # Custom kernels: tridiag solver, convergence,
│   │   │                        #   scatter, profiles, Anderson mixing
│   │   └── main.cpp             # gpu_efit_server entry point
│   ├── ray_tracing/
│   │   ├── ray_tracing.hip      # Ray tracer: two-stage angle search,
│   │   │                        #   pre-allocated buffers, GPU angle grid gen
│   │   ├── rt_kernels.hip       # Kernels: multibeam ray trace, angle optimize,
│   │   │                        #   O-mode refraction (optimized 4-interp gradient)
│   │   └── main.cpp             # ray_tracing_server entry point
│   └── distributed/
│       └── rfm_transport.cpp    # TCP socket RFM simulation + local_transfer
├── tests/
│   ├── e2e_test.cpp             # Single-process end-to-end test
│   └── benchmark.cpp            # Full performance benchmark suite (7 sections)
└── CMakeLists.txt
```

### GPU-EFIT Algorithm

The P-EFIT 5-step Grad-Shafranov solver, accelerated with Anderson(3) mixing:

```
For each iteration:
  1. Eigen decomposition:  Ψ' = Q^T × Ψ        (rocblas_sgemm)
  2. Transpose:            Ψ'' = (Ψ')^T          (rocblas_sgeam)
  3. Tridiagonal solve:    X = T^{-1} × Ψ''      (custom prefix-sum kernel)
  4. Transpose:            X' = X^T               (rocblas_sgeam)
  5. Inverse eigen:        Ψ_new = Q × X'         (rocblas_sgemm)

  Green boundary:          ψ_bnd = G × J          (rocblas_sgemv)
  Anderson mixing:         ψ = Σ αᵢ g(xᵢ)         (rocblas_sdot + CPU 3×3 solve)
```

### Ray Tracing Algorithm

Hamilton ray equations solved via geometric optics with O-mode cold-plasma dispersion:

```
Stage 1: Coarse search (10×10 angle grid, all beams parallel)
  → For each (beam, angle): trace ray through plasma, record ρ_dep
  → Reduce: find angle minimizing |ρ_dep − ρ_target| per beam

Stage 2: Fine search (10×10 around coarse optimum)
  → Same trace + reduce, narrower angle range
  → Output: optimal (θ, φ) per beam
```

## Build

### Prerequisites

- ROCm >= 7.2.0 (HIP runtime + rocBLAS)
- CMake >= 3.21
- C++17 compiler (GCC >= 10 or Clang)

### Compile

```bash
mkdir build && cd build
cmake .. -DROCM_PATH=/opt/rocm -DGPU_TARGETS=gfx1201
make -j$(nproc)
```

Target GPU: AMD Radeon RX 9070 (RDNA 4, gfx1201).

## Usage

### Single-Process End-to-End Test

```bash
./e2e_test --grid 129 --iter 10 --beams 4
```

### Distributed Two-Process Mode

```bash
# Terminal 1: PCS node — GPU-EFIT server
./gpu_efit_server --grid 129 --iter 10 --port 50051

# Terminal 2: ECRH node — Ray tracing server (connects to PCS)
./ray_tracing_server --port 50051 --beams 4
```

### Performance Benchmark

```bash
./benchmark                    # Full suite (all 7 sections)
./benchmark --target           # Design target comparison only
./benchmark --efit             # EFIT reconstruction only
./benchmark --rt               # Ray tracing only
./benchmark --kernel           # Individual kernel profiling
./benchmark --pipeline         # End-to-end pipeline
./benchmark --mem              # Memory bandwidth
./benchmark --vram             # VRAM usage analysis
./benchmark --repeats 50       # Custom repeat count
```

### Parameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `--grid` | EFIT grid size (N×N) | 129 | 129, 257, 513 |
| `--iter` | Max Picard/Anderson iterations | 10 | 1–30 |
| `--beams` | Number of ECRH beams | 4 | 1–12 |
| `--port` | RFM communication port | 50051 | any available |
| `--repeats` | Benchmark repetitions | 20 | 1–1000 |
| `--warmup` | Benchmark warmup runs | 3 | 0–100 |

## References

### EFIT and Grad-Shafranov Solvers

1. L.L. Lao et al., "Reconstruction of current profile parameters and plasma shapes in tokamaks," *Nuclear Fusion*, vol. 25, no. 11, pp. 1611–1622, 1985. — The original EFIT algorithm.

2. L.L. Lao et al., "Equilibrium analysis of current profiles in tokamaks," *Nuclear Fusion*, vol. 30, no. 6, pp. 1035–1049, 1990. — Extended EFIT with pressure and current constraints.

3. J.R. Ferron et al., "Real time equilibrium reconstruction for tokamak discharge control," *Nuclear Fusion*, vol. 38, no. 7, pp. 1055–1066, 1998. — Real-time EFIT (rt-EFIT) for plasma control.

4. B.J. Xiao et al., "PEFIT — a GPU-accelerated equilibrium reconstruction code for advanced tokamak research," *Plasma Physics and Controlled Fusion*, vol. 62, no. 2, 2020. — P-EFIT 5-step algorithm on NVIDIA GPU.

### Anderson Acceleration

5. D.G. Anderson, "Iterative procedures for nonlinear integral equations," *Journal of the ACM*, vol. 12, no. 4, pp. 547–560, 1965. — Original Anderson mixing method.

6. H.F. Walker and P. Ni, "Anderson acceleration for fixed-point iterations," *SIAM Journal on Numerical Analysis*, vol. 49, no. 4, pp. 1715–1735, 2011. — Convergence analysis and practical guidelines.

### ECRH Ray Tracing

7. E. Poli et al., "TORBEAM, a beam tracing code for electron-cyclotron waves in tokamak plasmas," *Computer Physics Communications*, vol. 136, no. 1–2, pp. 90–104, 2001. — Reference beam/ray tracing for ECRH.

8. N.B. Marushchenko et al., "Ray-tracing code TRAVIS for ECR heating, EC current drive and ECE diagnostic," *Computer Physics Communications*, vol. 185, no. 1, pp. 165–176, 2014. — TRAVIS ray-tracing code.

### GPU-Accelerated Plasma Computation

9. J. Huang et al., "Real-time capable GPU-based equilibrium reconstruction using neural networks on HL-3," *Nuclear Fusion*, vol. 64, 2024. — EFITNN: neural-network EFIT on GPU (0.08 ms).

10. S.H. Hahn et al., "Implementation of real-time equilibrium reconstruction on KSTAR," *Fusion Engineering and Design*, vol. 89, no. 5, pp. 542–546, 2014. — Real-time EFIT on KSTAR.

11. Y.S. Hwang et al., "GPU-accelerated real-time equilibrium reconstruction on EAST," *Fusion Engineering and Design*, vol. 112, pp. 569–575, 2016. — GPU EFIT on EAST tokamak.

## License

MIT License
