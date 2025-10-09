# PBDSolver 求解器文档

## 概述
PBDSolver (Position Based Dynamics Solver) 是一个基于位置的动力学求解器，用于模拟布料、弹性体、液体和粒子。使用约束投影方法而非力积分，具有无条件稳定性和快速收敛特点。

**涉及的类**: `PBDSolver`
**涉及的枚举**: `MATERIAL` (材料类型)

## 功能模块划分

### 1. 初始化模块 (Initialization)
**功能**: 设置PBD求解器的基本配置和场结构

#### 材料类型枚举:
- **`MATERIAL`** (IntEnum)
  - `CLOTH = 0`: 布料
  - `ELASTIC = 1`: 弹性体
  - `LIQUID = 2`: 液体
  - `PARTICLE = 3`: 非物理粒子

#### 涉及的函数:
- **`__init__(self, scene, sim, options)`**
  - 功能: 构造函数，初始化PBD求解器
  - 关键配置:
    - `upper_bound/lower_bound`: 模拟区域边界
    - `particle_size`: 粒子尺寸
    - 求解器迭代次数:
      - `max_stretch_solver_iterations`: 拉伸约束迭代次数
      - `max_bending_solver_iterations`: 弯曲约束迭代次数
      - `max_volume_solver_iterations`: 体积约束迭代次数
      - `max_density_solver_iterations`: 密度约束迭代次数
      - `max_viscosity_solver_iterations`: 粘度约束迭代次数
  - 核函数参数:
    - `dist_scale`: 距离缩放 = particle_radius / 0.4
    - `h`: 核函数半径 = 1.0
    - `poly6_Coe`: Poly6核系数 = 315/(64*π)
    - `spiky_Coe`: Spiky核系数 = -45/π
  - 空间哈希: SpatialHasher用于邻居搜索

- **`setup_boundary(self)`**
  - 功能: 设置边界条件
  - 操作: 创建CubeBoundary

- **`init_vvert_fields(self)`**
  - 功能: 初始化可视化顶点场

- **`init_particle_fields(self)`**
  - 功能: 初始化粒子场结构
  - 静态信息:
    - mass: 质量
    - pos_rest: 静止位置
    - rho_rest: 静止密度
    - material_type: 材料类型
    - mu_s/mu_k: 静摩擦/动摩擦系数
    - air_resistance: 空气阻力
    - density_relaxation: 密度松弛
    - viscosity_relaxation: 粘度松弛
  - 动态状态:
    - free: 是否自由
    - pos/ipos: 位置/初始位置
    - dpos: 位置增量
    - vel: 速度
    - lambda_: λ (XPBD参数)

- **`init_edge_fields(self)`**
  - 功能: 初始化边场结构
  - 用于拉伸约束

- **`init_elem_fields(self)`**
  - 功能: 初始化单元场结构
  - 用于体积约束

- **`init_ckpt(self)`**
  - 功能: 初始化检查点字典

### 2. 实体管理模块 (Entity Management)
**功能**: 管理PBD实体

#### 涉及的函数:
- **`add_entity(self, idx, material, morph, surface)`**
  - 功能: 添加PBD实体
  - 支持:
    - PBD2DEntity: 2D布料/弹性体
    - PBD3DEntity: 3D弹性体
    - PBDParticleEntity: 液体粒子
    - PBDFreeParticleEntity: 非物理粒子

- **`is_active(self)`**
  - 功能: 检查求解器是否活跃
  - 返回: _n_particles > 0

### 3. 核函数模块 (Kernel Functions)
**功能**: 实现SPH风格的核函数

#### 涉及的函数:
- **`poly6(self, dist)`** (Taichi函数)
  - 功能: Poly6核函数（向量输入）
  - 公式: W = 315/(64πh^9) * (h^2 - r^2)^3, if 0 < r < h
  - 用途: 密度和粘度计算

- **`poly6_scalar(self, dist)`** (Taichi函数)
  - 功能: Poly6核函数（标量输入）

- **`spiky(self, dist)`** (Taichi函数)
  - 功能: Spiky核函数梯度
  - 公式: ∇W = -45/(πh^6) * (h-r)^2 / r * r_vec
  - 用途: 压力力和涡量计算

- **`S_Corr(self, dist)`** (Taichi函数)
  - 功能: 表面张力修正
  - 公式: s_corr = -k * (W(r) / W(Δq))^4

### 4. 外力应用模块 (External Force Application)
**功能**: 应用外力和预测位置

#### 涉及的函数:
- **`_kernel_store_initial_pos(self, f)`** (Taichi Kernel)
  - 功能: 存储初始位置
  - 操作: ipos = pos

- **`_kernel_apply_external_force(self, f, t)`** (Taichi Kernel)
  - 功能: 应用外力并预测位置
  - 步骤:
    1. 应用重力: vel += g * dt
    2. 应用力场: vel += acc_field * dt
    3. 应用空气阻力（布料）: vel -= air_resistance * vel^2 * dt / mass
    4. 预测位置: pos += vel * dt

### 5. 拓扑约束模块 (Topology Constraints)
**功能**: 处理基于拓扑结构的约束（不需要空间哈希）

#### 涉及的函数:
- **`_kernel_solve_stretch(self, f)`** (Taichi Kernel)
  - 功能: 解决拉伸约束（边长度保持）
  - 约束: C = |p1 - p2| - len_rest
  - XPBD公式: dp = -C / (w1 + w2 + α) * n * relaxation
  - 迭代: max_stretch_solver_iterations次

- **`_kernel_solve_bending(self, f)`** (Taichi Kernel)
  - 功能: 解决弯曲约束（二面角保持）
  - 约束: C = arccos(n1·n2) - π (使布料平坦)
  - 基于: Muller et al. "Position Based Dynamics" (2007)
  - 迭代: max_bending_solver_iterations次

- **`_kernel_solve_volume(self, f)`** (Taichi Kernel)
  - 功能: 解决体积约束（四面体体积保持）
  - 约束: C = vol - vol_rest
  - 梯度: ∇C_i = (p_j - p_k) × (p_l - p_k) / 6
  - XPBD公式: dp = -C / (sum(w_i * |∇C_i|^2) + α) * ∇C * relaxation
  - 迭代: max_volume_solver_iterations次

### 6. 空间约束模块 (Spatial Constraints)
**功能**: 处理需要邻居搜索的约束

#### 涉及的函数:
- **`_kernel_reorder_particles(self, f)`** (Taichi Kernel)
  - 功能: 根据空间哈希重排序粒子
  - 操作: 将粒子按哈希网格排序，加速邻居搜索

- **`_kernel_solve_density(self, f)`** (Taichi Kernel)
  - 功能: 解决密度约束（不可压缩性）
  - 约束: C_i = ρ_i / ρ_0 - 1
  - 算法: 基于XSPH和PBF (Position Based Fluids)
  - 步骤:
    1. 计算λ (拉格朗日乘子)
    2. 计算位置修正: dp = sum_j((λ_i + λ_j + s_corr) * ∇W(r_ij))
    3. 应用松弛: dp *= density_relaxation
  - 迭代: max_density_solver_iterations次

- **`_kernel_solve_viscosity(self, f)`** (Taichi Kernel)
  - 功能: 解决粘度约束（XSPH）
  - 算法: XSPH方法
  - 公式: dp = sum_j(W(r_ij) * v_ij) * viscosity_relaxation
  - 用途: 使邻近粒子速度趋于一致
  - 迭代: max_viscosity_solver_iterations次

### 7. 碰撞处理模块 (Collision Handling)
**功能**: 处理粒子间和边界碰撞

#### 涉及的函数:
- **`_func_solve_collision(self, i, j, i_b)`** (Taichi函数)
  - 功能: 解决两个粒子间的碰撞
  - 步骤:
    1. 检查距离: dist < 2*radius
    2. 计算碰撞法线: n = (pos_i - pos_j).normalized()
    3. 分离: dp = 0.5 * (2*radius - dist) * n
    4. 处理摩擦

- **`_kernel_solve_collision(self, f)`** (Taichi Kernel)
  - 功能: 解决所有粒子间碰撞
  - 操作: 使用空间哈希查找邻居并处理碰撞

- **`_kernel_solve_boundary_collision(self, f)`** (Taichi Kernel)
  - 功能: 解决边界碰撞
  - 操作: 检查并修正超出边界的粒子

### 8. 速度更新模块 (Velocity Update)
**功能**: 从位置变化计算速度

#### 涉及的函数:
- **`_kernel_compute_velocity(self, f)`** (Taichi Kernel)
  - 功能: 计算有效速度
  - 公式: vel = (pos - ipos) / dt
  - 说明: 从位置修正隐式计算速度

- **`_kernel_copy_from_reordered(self, f)`** (Taichi Kernel)
  - 功能: 从重排序数组复制回原数组

### 9. 时间步进模块 (Time Stepping)
**功能**: 管理时间步进流程

#### 涉及的函数:
- **`process_input(self, in_backward=False)`**
  - 功能: 处理实体输入

- **`process_input_grad(self)`**
  - 功能: 处理输入梯度（空实现）

- **`substep_pre_coupling(self, f)`**
  - 功能: 耦合前的子步骤
  - 步骤:
    1. _kernel_store_initial_pos(f) - 存储初始位置
    2. _kernel_apply_external_force(f, t) - 应用外力，预测位置
    3. 拓扑约束（并行）:
       - _kernel_solve_stretch(f) - 拉伸约束
       - _kernel_solve_bending(f) - 弯曲约束
       - _kernel_solve_volume(f) - 体积约束
    4. _kernel_reorder_particles(f) - 空间哈希重排序
    5. 空间约束（需要邻居）:
       - _kernel_solve_density(f) - 密度约束
       - _kernel_solve_viscosity(f) - 粘度约束
    6. _kernel_solve_collision(f) - 粒子间碰撞
    7. _kernel_compute_velocity(f) - 计算速度

- **`substep_pre_coupling_grad(self, f)`**
  - 功能: 耦合前梯度（空实现）

- **`substep_post_coupling(self, f)`**
  - 功能: 耦合后的子步骤
  - 步骤:
    1. _kernel_copy_from_reordered(f) - 复制回原数组
    2. _kernel_solve_boundary_collision(f) - 边界碰撞

- **`substep_post_coupling_grad(self, f)`**
  - 功能: 耦合后梯度（空实现）

### 10. 梯度管理模块 (Gradient Management)
**功能**: 梯度管理（不支持可微分）

#### 涉及的函数:
- **`reset_grad(self)`**
  - 功能: 重置梯度（空实现）

- **`collect_output_grads(self)`**
  - 功能: 收集输出梯度（空实现）

- **`add_grad_from_state(self, state)`**
  - 功能: 从状态添加梯度（空实现）

### 11. 状态管理模块 (State Management)
**功能**: 管理状态的保存、加载和查询

#### 涉及的函数:
- **`get_state(self, f)`**
  - 功能: 获取状态
  - 返回: PBDSolverState对象(pos, vel, active)

- **`set_state(self, f, state, envs_idx=None)`**
  - 功能: 设置状态

- **`save_ckpt(self, ckpt_name)`**
  - 功能: 保存检查点（空实现）

- **`load_ckpt(self, ckpt_name)`**
  - 功能: 加载检查点（空实现）

### 12. 渲染支持模块 (Rendering Support)
**功能**: 更新用于渲染的场数据

#### 涉及的函数:
- **`update_render_fields(self)`**
  - 功能: 更新渲染字段

### 13. 粒子操作模块 (Particle Operations)
**功能**: 提供粒子级别的操作接口

#### 涉及的内核函数:
- `_kernel_set_particles_pos()`: 设置粒子位置
- `_kernel_get_particles_pos()`: 获取粒子位置
- `_kernel_set_particles_vel()`: 设置粒子速度
- `_kernel_get_particles_vel()`: 获取粒子速度
- `_kernel_set_particles_active()`: 设置粒子激活状态
- `_kernel_get_particles_active()`: 获取粒子激活状态
- `_kernel_fix_particles()`: 固定粒子
- `_kernel_release_particle()`: 释放粒子
- `_kernel_get_mass()`: 获取粒子质量
- `set_animate_particles_by_link()`: 通过连杆动画化粒子

### 14. 属性访问模块 (Property Access)
**功能**: 提供只读属性访问

#### 涉及的属性:
- `n_particles`: 粒子总数
- `n_fluid_particles`: 流体粒子数
- `n_edges`: 边数量
- `n_inner_edges`: 内边数量
- `n_elems`: 单元数量
- `n_vverts/n_vfaces`: 可视化顶点/面数量
- `particle_size/particle_radius`: 粒子尺寸/半径
- `hash_grid_res/hash_grid_cell_size`: 哈希网格参数
- `upper_bound/lower_bound`: 边界

## 主要功能管线

### PBD时间步进管线 (PBD Time Stepping Pipeline)
```
1. substep_pre_coupling(f) - 主要计算
   ├─ A. 预测阶段 (Prediction)
   │  ├─ _kernel_store_initial_pos(f) - 存储初始位置
   │  └─ _kernel_apply_external_force(f, t) - 应用外力
   │     ├─ vel += gravity * dt
   │     ├─ vel += force_field * dt
   │     ├─ vel -= air_resistance (布料)
   │     └─ pos += vel * dt (预测位置)
   │  
   ├─ B. 约束投影阶段 (Constraint Projection)
   │  ├─ 拓扑约束 (并行，不需要邻居查找):
   │  │  ├─ _kernel_solve_stretch(f) - 拉伸约束
   │  │  │  └─ 迭代保持边长度
   │  │  ├─ _kernel_solve_bending(f) - 弯曲约束
   │  │  │  └─ 迭代保持二面角
   │  │  └─ _kernel_solve_volume(f) - 体积约束
   │  │     └─ 迭代保持四面体体积
   │  │  
   │  ├─ _kernel_reorder_particles(f) - 空间哈希重排序
   │  │  
   │  └─ 空间约束 (需要邻居查找):
   │     ├─ _kernel_solve_density(f) - 密度约束
   │     │  └─ 迭代使密度接近rest_density (不可压缩)
   │     └─ _kernel_solve_viscosity(f) - 粘度约束
   │        └─ 迭代使邻近粒子速度一致 (XSPH)
   │  
   ├─ C. 碰撞处理 (Collision)
   │  └─ _kernel_solve_collision(f) - 粒子间碰撞
   │     └─ 分离重叠粒子，处理摩擦
   │  
   └─ D. 速度更新 (Velocity Update)
      └─ _kernel_compute_velocity(f)
         └─ vel = (pos - ipos) / dt
   
2. substep_post_coupling(f) - 后处理
   ├─ _kernel_copy_from_reordered(f) - 复制回原数组
   └─ _kernel_solve_boundary_collision(f) - 边界碰撞
```

### 约束求解详细流程 (Constraint Solving Detail)

#### 拉伸约束 (Distance/Stretch Constraint):
```
约束: C = |p1 - p2| - L0

1. 计算约束值: C = |p1 - p2| - len_rest
2. 计算权重: w1 = free1/m1, w2 = free2/m2
3. XPBD修正:
   α = compliance / dt^2
   dp = -C / (w1 + w2 + α) * n * relaxation
4. 应用修正:
   p1 += dp * w1
   p2 -= dp * w2
```

#### 密度约束 (Density Constraint):
```
约束: C_i = ρ_i/ρ_0 - 1

1. 计算密度: ρ_i = sum_j(m_j * W(r_ij))

2. 计算λ:
   λ_i = -C_i / (sum_j |∇C_i|^2 + ε)
   其中 ∇C_i = sum_j ∇W(r_ij)

3. 计算位置修正:
   dp_i = (1/ρ_0) * sum_j((λ_i + λ_j + s_corr) * ∇W(r_ij))
   
4. 应用松弛:
   dp_i *= density_relaxation

5. 更新位置: p_i += dp_i
```

#### 粘度约束 (Viscosity Constraint - XSPH):
```
XSPH方法:

1. 计算速度差异:
   v_ij = (p_j - ipos_j) - (p_i - ipos_i)

2. 加权求和:
   dp_i = sum_j(W(r_ij) * v_ij) * viscosity_relaxation

3. 更新位置: p_i += dp_i
```

## 设计特点
1. **基于位置**: 直接操作位置而非力，无条件稳定
2. **约束投影**: 使用约束投影方法，迭代满足约束
3. **XPBD**: 扩展的PBD，使用柔度(compliance)参数控制约束强度
4. **多材料**: 支持布料、弹性体、液体、粒子
5. **分层约束**: 
   - 拓扑约束（边、弯曲、体积）- 不需要邻居查找
   - 空间约束（密度、粘度）- 需要邻居查找
6. **空间哈希**: 高效的邻居查找
7. **不可压缩**: 使用密度约束实现液体不可压缩性
8. **XSPH粘度**: 平滑速度场
9. **摩擦处理**: 支持静摩擦和动摩擦
10. **不支持可微分**: 当前实现不支持梯度反向传播

## 算法基础
基于：
- Muller et al. "Position Based Dynamics" (2007)
- Macklin & Muller "Position Based Fluids" (2013)
- Macklin et al. "XPBD: Position-Based Simulation of Compliant Constrained Dynamics" (2016)

## 应用场景
- 布料模拟（衣服、旗帜）
- 软体模拟（橡胶、肌肉）
- 液体模拟（水、泥浆）
- 粒子效果（非物理粒子）

## 优势
- 无条件稳定
- 快速迭代收敛
- 易于控制和调节
- 支持多种材料类型
- 高效的实时模拟

## 限制
- 不支持可微分模拟
- 近似方法，不精确求解
- 迭代次数影响精度和性能
