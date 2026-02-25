# ROCM-RTPC

**ROCm 实时等离子体计算平台**

基于 AMD ROCm/HIP 的 GPU 加速实时等离子体平衡重建与 ECRH 射线追踪平台，面向托卡马克等离子体控制系统 (PCS) 和电子回旋共振加热 (ECRH) 反馈回路，目标延迟低于毫秒级。

## 项目目的

现代托卡马克装置要求在单个等离子体控制周期（约 1–10 ms）内完成平衡重建 (EFIT) 和微波束导向 (ECRH) 计算。传统 CPU 实现在高空间分辨率下无法满足实时性要求。ROCM-RTPC 将整个计算流水线卸载到 AMD GPU，通过 HIP 编程实现 **>25 倍加速**，使得此前仅限于离线分析的高分辨率计算能够实时运行。

**核心能力：**

- Grad-Shafranov 平衡求解器 (GPU-EFIT)：Anderson 加速 + 物理先验暖启动
- Hamilton 射线方程求解器：ECRH 发射角度优化（多束并行）
- 分布式双节点架构：PCS 节点 (EFIT) 通过模拟反射内存 (RFM) 与 ECRH 节点（射线追踪）通信
- 全面集成 rocBLAS 库 (SGEMM, SGEAM, SGEMV, SDOT) 处理热路径线性代数运算

## 性能指标

测试平台：AMD Radeon RX 9070 (RDNA 4, gfx1201)，20 次运行取中位数：

| 模块 | 网格 | 实测延迟 | CPU 基线 | 加速比 | 设计目标 |
|------|------|---------|---------|--------|---------|
| GPU-EFIT | 129×129 | **0.92 ms** | 24 ms | 26× | < 1.5 ms |
| GPU-EFIT | 257×257 | **1.15 ms** | 170 ms | 148× | < 21 ms |
| GPU-EFIT | 513×513 | **5.04 ms** | 1200 ms | 238× | < 80 ms |
| 射线追踪 (12 束) | 129×129 | **0.08 ms** | 240 ms | 3116× | < 0.5 ms |
| 端到端流水线 (4 束) | 129×129 | **1.09 ms** | 25 ms | 23× | < 1.4 ms |

## 系统架构

### 分布式部署

```
┌─────────────────────┐         RFM / TCP          ┌─────────────────────────┐
│  PCS 节点 (GPU #1)   │ ─────────────────────────→  │  ECRH-MC 节点 (GPU #2)  │
│                      │                             │                          │
│  诊断系统输入         │                             │  接收平衡数据             │
│       ↓              │                             │       ↓                  │
│  gpu_efit_server     │                             │  ray_tracing_server      │
│   • 输入: J_plasma   │                             │   • 输出: 最优角度 (θ, φ) │
│   • 输出: ψ(R,Z)     │                             │   • 输出: ρ_dep, η_cd    │
│   • 输出: ne, Te, Bφ │                             │       ↓                  │
│       ↓              │                             │  ECRH 发射器控制          │
│  RFM 发送            │                             │                          │
└─────────────────────┘                              └─────────────────────────┘
```

在生产环境中，两个节点通过反射内存 (RFM) 硬件（如 GE VMIPCI-5565）通信，确保确定性的亚 100μs 传输延迟。本实现通过 TCP Socket 模拟 RFM 通信，也支持单机内存直接拷贝模式用于开发测试。

### 单节点模式

用于开发和性能测试，全流水线在单进程内运行：

```
J_plasma → [GPU-EFIT] → EquilibriumData → [local_transfer] → [射线追踪] → BeamResult
```

## 输入 / 输出

### GPU-EFIT

| 方向 | 数据 | 类型 | 尺寸 |
|------|------|------|------|
| **输入** | 等离子体电流密度 J(R,Z) | `float*` | M × M（内部网格） |
| **输入** | 最大迭代次数、收敛容差 | `int`, `float` | 标量 |
| **输出** | 极向磁通 ψ(R,Z) | `float*` | N × N |
| **输出** | 电子密度 ne(R,Z) | `float*` | N × N |
| **输出** | 电子温度 Te(R,Z) | `float*` | N × N |
| **输出** | 环向磁场 Bφ(R,Z) | `float*` | N × N |
| **输出** | 磁轴位置 (R₀, Z₀)、ψ_axis、ψ_boundary | `float` | 标量 |

其中 N = 网格尺寸 (129, 257 或 513)，M = N − 2。

### 射线追踪

| 方向 | 数据 | 类型 | 尺寸 |
|------|------|------|------|
| **输入** | EquilibriumData（来自 EFIT） | 结构体 | N × N 场数据 |
| **输入** | 各束目标沉积位置 ρ_target | `float[12]` | 最多 12 束 |
| **输出** | 最优极向角 θ | `float` | 每束 |
| **输出** | 最优环向角 φ | `float` | 每束 |
| **输出** | 实际沉积位置 ρ_dep | `float` | 每束 |

## 代码架构

```
rocm-rtpc/
├── include/
│   ├── common/
│   │   ├── types.h              # 公共数据结构：EquilibriumData, BeamResult, ECRHTarget, 常量
│   │   ├── hip_check.h          # HIP_CHECK 错误检查宏
│   │   ├── timer.h              # 高精度壁钟计时器
│   │   └── plasma_profiles.h    # Solovev 解析平衡生成器
│   ├── gpu_efit/
│   │   ├── gpu_efit.h           # GpuEfit 类 API
│   │   └── efit_kernels.h       # HIP kernel 声明
│   ├── ray_tracing/
│   │   ├── ray_tracing.h        # GpuRayTracing 类 API
│   │   └── rt_kernels.h         # HIP kernel 声明
│   └── distributed/
│       └── rfm_transport.h      # RFM 传输层 API
├── src/
│   ├── common/
│   │   └── plasma_profiles.cpp  # Solovev 平衡 + 剖面生成
│   ├── gpu_efit/
│   │   ├── gpu_efit.hip         # EFIT 求解器：rocBLAS GEMM/GEAM/GEMV/DOT，
│   │   │                        #   Anderson 加速，NN 暖启动
│   │   ├── efit_kernels.hip     # 自定义 kernel：三对角求解器、收敛检测、
│   │   │                        #   scatter、剖面计算、Anderson 混合
│   │   └── main.cpp             # gpu_efit_server 入口
│   ├── ray_tracing/
│   │   ├── ray_tracing.hip      # 射线追踪器：两阶段角度搜索、
│   │   │                        #   预分配缓冲区、GPU 端角度网格生成
│   │   ├── rt_kernels.hip       # kernel：多束射线追踪、角度优化、
│   │   │                        #   O 模折射（优化 4 次插值梯度）
│   │   └── main.cpp             # ray_tracing_server 入口
│   └── distributed/
│       └── rfm_transport.cpp    # TCP Socket 模拟 RFM + 本地直接传输
├── tests/
│   ├── e2e_test.cpp             # 单进程端到端测试
│   └── benchmark.cpp            # 完整性能测试套件（7 个测试段）
└── CMakeLists.txt
```

### GPU-EFIT 算法

P-EFIT 五步 Grad-Shafranov 求解器，结合 Anderson(3) 加速混合：

```
每次迭代：
  1. 特征分解:    Ψ' = Q^T × Ψ        (rocblas_sgemm)
  2. 矩阵转置:    Ψ'' = (Ψ')^T          (rocblas_sgeam)
  3. 三对角求解:   X = T^{-1} × Ψ''      (自定义前缀和 kernel)
  4. 矩阵转置:    X' = X^T               (rocblas_sgeam)
  5. 逆特征重构:   Ψ_new = Q × X'         (rocblas_sgemm)

  Green 边界条件:  ψ_bnd = G × J          (rocblas_sgemv)
  Anderson 混合:   ψ = Σ αᵢ g(xᵢ)         (rocblas_sdot + CPU 3×3 求解)
```

### 射线追踪算法

基于几何光学和 O 模冷等离子体色散关系求解 Hamilton 射线方程：

```
阶段 1：粗搜索（10×10 角度网格，所有束并行）
  → 对每个 (束, 角度) 组合：追踪射线穿过等离子体，记录沉积位置 ρ_dep
  → 归约：找到使 |ρ_dep − ρ_target| 最小的角度（每束独立）

阶段 2：精搜索（10×10，围绕粗搜索最优值）
  → 相同追踪 + 归约流程，角度范围缩窄
  → 输出：每束最优角度 (θ, φ)
```

## 编译

### 依赖项

- ROCm >= 7.2.0（含 HIP 运行时 + rocBLAS）
- CMake >= 3.21
- C++17 编译器（GCC >= 10 或 Clang）

### 编译步骤

```bash
mkdir build && cd build
cmake .. -DROCM_PATH=/opt/rocm -DGPU_TARGETS=gfx1201
make -j$(nproc)
```

目标 GPU：AMD Radeon RX 9070 (RDNA 4, gfx1201)。

## 使用方式

### 单进程端到端测试

```bash
./e2e_test --grid 129 --iter 10 --beams 4
```

### 分布式双进程模式

```bash
# 终端 1：PCS 节点 — GPU-EFIT 服务器
./gpu_efit_server --grid 129 --iter 10 --port 50051

# 终端 2：ECRH 节点 — 射线追踪服务器（连接到 PCS）
./ray_tracing_server --port 50051 --beams 4
```

### 性能测试

```bash
./benchmark                    # 完整测试套件（全部 7 个段）
./benchmark --target           # 仅设计目标对比
./benchmark --efit             # 仅 EFIT 重建
./benchmark --rt               # 仅射线追踪
./benchmark --kernel           # 单个 kernel 性能分析
./benchmark --pipeline         # 端到端流水线
./benchmark --mem              # 内存带宽测试
./benchmark --vram             # 显存占用分析
./benchmark --repeats 50       # 自定义重复次数
```

### 参数说明

| 参数 | 说明 | 默认值 | 取值范围 |
|------|------|--------|---------|
| `--grid` | EFIT 网格尺寸 (N×N) | 129 | 129, 257, 513 |
| `--iter` | 最大 Picard/Anderson 迭代次数 | 10 | 1–30 |
| `--beams` | ECRH 束数 | 4 | 1–12 |
| `--port` | RFM 通信端口 | 50051 | 任意可用端口 |
| `--repeats` | Benchmark 重复次数 | 20 | 1–1000 |
| `--warmup` | Benchmark 预热次数 | 3 | 0–100 |

## 参考文献

### EFIT 与 Grad-Shafranov 求解器

1. L.L. Lao 等, "Reconstruction of current profile parameters and plasma shapes in tokamaks," *Nuclear Fusion*, vol. 25, no. 11, pp. 1611–1622, 1985. — EFIT 原始算法。

2. L.L. Lao 等, "Equilibrium analysis of current profiles in tokamaks," *Nuclear Fusion*, vol. 30, no. 6, pp. 1035–1049, 1990. — 含压力和电流约束的扩展 EFIT。

3. J.R. Ferron 等, "Real time equilibrium reconstruction for tokamak discharge control," *Nuclear Fusion*, vol. 38, no. 7, pp. 1055–1066, 1998. — 实时 EFIT (rt-EFIT) 用于等离子体控制。

4. B.J. Xiao 等, "PEFIT — a GPU-accelerated equilibrium reconstruction code for advanced tokamak research," *Plasma Physics and Controlled Fusion*, vol. 62, no. 2, 2020. — NVIDIA GPU 上的 P-EFIT 五步算法。

### Anderson 加速

5. D.G. Anderson, "Iterative procedures for nonlinear integral equations," *Journal of the ACM*, vol. 12, no. 4, pp. 547–560, 1965. — Anderson 混合方法原始论文。

6. H.F. Walker and P. Ni, "Anderson acceleration for fixed-point iterations," *SIAM Journal on Numerical Analysis*, vol. 49, no. 4, pp. 1715–1735, 2011. — 收敛性分析与实践指南。

### ECRH 射线追踪

7. E. Poli 等, "TORBEAM, a beam tracing code for electron-cyclotron waves in tokamak plasmas," *Computer Physics Communications*, vol. 136, no. 1–2, pp. 90–104, 2001. — ECRH 束/射线追踪参考实现。

8. N.B. Marushchenko 等, "Ray-tracing code TRAVIS for ECR heating, EC current drive and ECE diagnostic," *Computer Physics Communications*, vol. 185, no. 1, pp. 165–176, 2014. — TRAVIS 射线追踪代码。

### GPU 加速等离子体计算

9. J. Huang 等, "Real-time capable GPU-based equilibrium reconstruction using neural networks on HL-3," *Nuclear Fusion*, vol. 64, 2024. — EFITNN：基于神经网络的 GPU EFIT (0.08 ms)。

10. S.H. Hahn 等, "Implementation of real-time equilibrium reconstruction on KSTAR," *Fusion Engineering and Design*, vol. 89, no. 5, pp. 542–546, 2014. — KSTAR 上的实时 EFIT。

11. Y.S. Hwang 等, "GPU-accelerated real-time equilibrium reconstruction on EAST," *Fusion Engineering and Design*, vol. 112, pp. 569–575, 2016. — EAST 托卡马克上的 GPU EFIT。

## 许可证

MIT License
