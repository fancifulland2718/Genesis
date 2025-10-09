# MPMSolver 求解器文档

## 概述
MPMSolver (Material Point Method Solver) 是一个物质点法求解器，用于模拟连续体材料（如弹性体、塑性体、液体、雪等）。它结合了拉格朗日和欧拉视角，在粒子上存储物质属性，在网格上求解动量守恒。

**涉及的类**: `MPMSolver`
**涉及的辅助函数**: `signmax()`, `backward_svd()`

## 功能模块划分

### 1. 初始化模块 (Initialization)
**功能**: 设置MPM求解器的网格、粒子场和边界条件

#### 涉及的函数:
- **`__init__(self, scene, sim, options)`**
  - 功能: 构造函数，初始化MPM求解器
  - 参数:
    - `scene`: 场景对象
    - `sim`: 模拟器对象
    - `options`: MPMOptions配置对象
  - 关键配置:
    - `grid_density`: 网格密度
    - `particle_size`: 粒子尺寸
    - `upper_bound/lower_bound`: 模拟区域边界
    - `enable_CPIC`: 是否启用Contact-aware Particle-In-Cell
  - 计算参数:
    - `dx`: 网格间距 = 1.0 / grid_density
    - `inv_dx`: 网格间距的倒数
    - `grid_res`: 网格分辨率
    - `particle_volume`: 粒子体积（带缩放）

- **`setup_boundary(self)`**
  - 功能: 设置边界条件
  - 操作: 创建CubeBoundary，添加安全padding（3*dx）

- **`init_particle_fields(self)`**
  - 功能: 初始化粒子场结构
  - 字段包括:
    - 动态状态: pos, vel, C (仿射速度场), F (变形梯度), F_tmp, U/S/V (SVD分解), Jp (体积压缩比), actu (激活)
    - 静态信息: mass, rho (密度), E (杨氏模量), nu (泊松比), material_idx, free, muscle_direction

- **`init_grid_fields(self)`**
  - 功能: 初始化网格场结构
  - 字段包括: vel_in, vel_out, mass, force

- **`init_vvert_fields(self)`**
  - 功能: 初始化可视化顶点场
  - 用于渲染网格生成

- **`init_ckpt(self)`**
  - 功能: 初始化检查点字典

### 2. 实体与材料管理模块 (Entity and Material Management)
**功能**: 管理MPM实体和材料类型

#### 涉及的函数:
- **`add_entity(self, idx, material, morph, surface)`**
  - 功能: 向求解器添加MPM实体
  - 操作:
    1. 添加材料（如果未注册）
    2. 创建MPMEntity对象
    3. 设置粒子和顶点的起始索引

- **`add_material(self, material)`**
  - 功能: 注册材料更新方法
  - 操作:
    - 检查材料是否已注册
    - 注册`update_F_S_Jp`和`update_stress`方法
    - 分配材料索引

- **`is_active(self)`**
  - 功能: 检查求解器是否活跃
  - 返回: 是否有粒子 (n_particles > 0)

### 3. MPM核心计算模块 (MPM Core Computation)
**功能**: 实现MPM算法的核心步骤

#### 涉及的函数:
- **`compute_F_tmp(self, f)`** (Taichi Kernel)
  - 功能: 计算临时变形梯度
  - 公式: `F_tmp = (I + dt * C) @ F`
  - 说明: C是仿射速度场

- **`svd(self, f)`** (Taichi Kernel)
  - 功能: 对F_tmp进行奇异值分解
  - 操作: `U, S, V = SVD(F_tmp)`
  - 用途: 用于材料模型的应力计算

- **`svd_grad(self, f)`** (Taichi Kernel)
  - 功能: 计算SVD的梯度
  - 操作: 调用`backward_svd()`进行反向传播

- **`p2g(self, f)`** (Taichi Kernel - 核心函数)
  - 功能: Particle-to-Grid转移
  - 步骤:
    1. 更新F、S、Jp（调用材料的update_F_S_Jp）
    2. 计算应力（调用材料的update_stress）
    3. 将粒子动量和质量投影到网格
    4. 使用三次B样条插值权重
    5. 如果启用CPIC，处理接触感知分离
  - 关键公式:
    - `stress = -dt * p_vol * 4 * inv_dx^2 * material_stress`
    - `affine = stress + mass * C`

- **`g2p(self, f)`** (Taichi Kernel - 核心函数)
  - 功能: Grid-to-Particle转移
  - 步骤:
    1. 从网格采样速度到粒子
    2. 更新粒子速度和位置
    3. 更新仿射速度场C
    4. 如果启用CPIC，处理刚体耦合
  - 关键公式:
    - `new_vel = Σ(weight * grid_vel)`
    - `new_C = 4 * inv_dx * Σ(weight * grid_vel ⊗ dpos)`
    - `new_pos = pos + dt * new_vel`

### 4. 时间步进模块 (Time Stepping)
**功能**: 管理模拟的时间步进流程

#### 涉及的函数:
- **`process_input(self, in_backward=False)`**
  - 功能: 处理实体输入
  - 操作: 为每个实体调用process_input

- **`process_input_grad(self)`**
  - 功能: 处理输入梯度
  - 操作: 逆序为每个实体调用process_input_grad

- **`substep_pre_coupling(self, f)`**
  - 功能: 耦合前的子步骤
  - 步骤:
    1. reset_grid_and_grad(f) - 重置网格
    2. compute_F_tmp(f) - 计算临时变形梯度
    3. svd(f) - SVD分解
    4. p2g(f) - 粒子到网格转移

- **`substep_pre_coupling_grad(self, f)`**
  - 功能: 耦合前子步骤的梯度
  - 步骤（逆序）:
    1. p2g.grad(f)
    2. svd_grad(f)
    3. compute_F_tmp.grad(f)

- **`substep_post_coupling(self, f)`**
  - 功能: 耦合后的子步骤
  - 操作: g2p(f) - 网格到粒子转移

- **`substep_post_coupling_grad(self, f)`**
  - 功能: 耦合后子步骤的梯度
  - 操作: g2p.grad(f)

### 5. 梯度管理模块 (Gradient Management)
**功能**: 管理可微分模拟的梯度

#### 涉及的函数:
- **`reset_grad(self)`**
  - 功能: 重置梯度场
  - 操作: 清空particles和grid的梯度

- **`reset_grad_till_frame(self, f)`** (Taichi Kernel)
  - 功能: 重置到指定帧的梯度

- **`collect_output_grads(self)`**
  - 功能: 收集输出梯度
  - 操作: 为每个实体收集输出梯度

- **`add_grad_from_state(self, state)`**
  - 功能: 从状态对象添加梯度
  - 操作: 调用_kernel添加pos、vel、C、F、Jp的梯度

- **`add_grad_from_*`系列函数**
  - `add_grad_from_pos()`: 添加位置梯度
  - `add_grad_from_vel()`: 添加速度梯度
  - `add_grad_from_C()`: 添加仿射速度场梯度
  - `add_grad_from_F()`: 添加变形梯度梯度
  - `add_grad_from_Jp()`: 添加体积压缩比梯度

### 6. 状态管理模块 (State Management)
**功能**: 管理MPM求解器的状态保存、加载和查询

#### 涉及的函数:
- **`get_state(self, f)`**
  - 功能: 获取指定帧的状态
  - 返回: MPMSolverState对象
  - 包含: pos, vel, C, F, Jp, active

- **`set_state(self, f, state, envs_idx=None)`**
  - 功能: 设置指定帧的状态
  - 操作: 调用_kernel_set_state设置状态字段

- **`save_ckpt(self, ckpt_name)`**
  - 功能: 保存检查点
  - 操作: 保存particles, particles_info, vverts_info到检查点字典

- **`load_ckpt(self, ckpt_name)`**
  - 功能: 加载检查点
  - 操作: 从检查点字典恢复状态

### 7. 渲染支持模块 (Rendering Support)
**功能**: 更新用于渲染的场数据

#### 涉及的函数:
- **`update_render_fields(self)`**
  - 功能: 更新渲染字段
  - 操作: 调用_kernel_update_render_fields

### 8. 粒子操作模块 (Particle Operations)
**功能**: 提供粒子级别的操作接口

#### 涉及的内核函数:
- `_kernel_add_particles()`: 添加粒子
- `_kernel_set_particles_pos()`: 设置粒子位置
- `_kernel_get_particles_pos()`: 获取粒子位置
- `_kernel_set_particles_vel()`: 设置粒子速度
- `_kernel_get_particles_vel()`: 获取粒子速度
- `_kernel_set_particles_active()`: 设置粒子激活状态
- `_kernel_get_particles_active()`: 获取粒子激活状态
- `_kernel_set_particles_actu()`: 设置粒子激活值（用于肌肉）
- `_kernel_get_particles_actu()`: 获取粒子激活值
- `_kernel_set_particles_muscle_group()`: 设置肌肉组
- `_kernel_set_particles_muscle_direction()`: 设置肌肉方向
- `_kernel_set_particles_free()`: 设置粒子自由状态
- `_kernel_get_mass()`: 获取粒子质量

### 9. 属性访问模块 (Property Access)
**功能**: 提供只读属性访问

#### 涉及的属性:
- 粒子数量: `n_particles`, `n_vverts`, `n_vfaces`
- 网格参数: `grid_density`, `dx`, `inv_dx`, `grid_res`, `grid_offset`
- 粒子参数: `particle_size`, `particle_radius`, `particle_volume`, `particle_volume_real`, `particle_volume_scale`
- 边界: `upper_bound`, `lower_bound`, `upper_bound_cell`, `lower_bound_cell`
- 其他: `leaf_block_size`, `use_sparse_grid`, `enable_CPIC`, `is_built`

## 辅助函数

### signmax(a, eps)
- **功能**: 带符号的最大值函数
- **公式**: `sign(a) * max(|a|, eps)`
- **用途**: 防止除零错误

### backward_svd(grad_U, grad_S, grad_V, U, S, V)
- **功能**: SVD的反向传播
- **参数**: 
  - grad_U/S/V: U/S/V的梯度
  - U/S/V: 正向传播的SVD结果
- **返回**: F_tmp的梯度
- **算法**: 基于PyTorch的SVD梯度实现

## 主要功能管线

### MPM时间步进管线 (MPM Time Stepping Pipeline)
```
1. process_input() - 处理实体输入
   ↓
2. substep_pre_coupling(f) - 耦合前计算
   ├─ reset_grid_and_grad(f) - 重置网格和梯度
   ├─ compute_F_tmp(f) - 计算临时变形梯度: F_tmp = (I + dt*C) @ F
   ├─ svd(f) - 奇异值分解: U, S, V = SVD(F_tmp)
   └─ p2g(f) - 粒子到网格转移
      ├─ 更新F, S, Jp (材料模型)
      ├─ 计算应力
      └─ 投影动量到网格
   ↓
3. [耦合处理 - 外部力、边界条件、与刚体耦合等]
   ↓
4. substep_post_coupling(f) - 耦合后计算
   └─ g2p(f) - 网格到粒子转移
      ├─ 采样网格速度
      ├─ 更新粒子速度和位置
      └─ 更新仿射速度场C
```

### MPM反向传播管线 (MPM Backward Propagation Pipeline)
```
1. add_grad_from_state(state) - 从查询状态添加梯度
   ↓
2. collect_output_grads() - 收集输出梯度
   ↓
3. substep_post_coupling_grad(f) - 耦合后梯度（逆序）
   └─ g2p.grad(f)
   ↓
4. [耦合梯度处理]
   ↓
5. substep_pre_coupling_grad(f) - 耦合前梯度（逆序）
   ├─ p2g.grad(f)
   ├─ svd_grad(f) - SVD梯度
   └─ compute_F_tmp.grad(f)
   ↓
6. process_input_grad() - 处理输入梯度
```

### 材料模型集成管线 (Material Model Integration Pipeline)
```
1. add_material(material) - 注册材料
   ├─ 分配材料索引
   ├─ 注册update_F_S_Jp方法
   └─ 注册update_stress方法
   ↓
2. p2g阶段调用材料方法
   ├─ update_F_S_Jp(J, F_tmp, U, S, V, Jp)
   │  └─ 返回: F_new, S_new, Jp_new
   └─ update_stress(U, S, V, F_tmp, F_new, J, Jp, actu, m_dir)
      └─ 返回: stress (3x3矩阵)
```

## 设计特点
1. **混合视角**: 结合拉格朗日（粒子）和欧拉（网格）视角
2. **SVD分解**: 使用奇异值分解处理变形梯度，支持多种材料模型
3. **可微分**: 完整支持梯度反向传播，可用于优化和控制
4. **CPIC支持**: 可选的接触感知粒子投影，改善刚体耦合
5. **材料扩展性**: 灵活的材料接口，支持自定义材料模型
6. **数值稳定性**: 使用体积缩放因子避免数值不稳定
7. **批处理**: 支持多环境并行模拟
