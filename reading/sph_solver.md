# SPHSolver 求解器文档

## 概述
SPHSolver (Smoothed Particle Hydrodynamics Solver) 是一个光滑粒子流体动力学求解器，用于模拟液体。支持WCSPH(Weakly Compressible SPH)和DFSPH(Divergence-Free SPH)两种压力求解方法。

**涉及的类**: `SPHSolver`

## 功能模块划分

### 1. 初始化模块 (Initialization)
**功能**: 设置SPH求解器的基本配置和场结构

#### 涉及的函数:
- **`__init__(self, scene, sim, options)`**
  - 功能: 构造函数，初始化SPH求解器
  - 关键配置:
    - `particle_size`: 粒子尺寸
    - `support_radius`: 支持半径（核函数作用范围）
    - `pressure_solver`: 压力求解器类型（"WCSPH"或"DFSPH"）
    - DFSPH参数:
      - `max_divergence_error`: 最大散度误差
      - `max_density_error_percent`: 最大密度误差百分比
      - `max_divergence_solver_iterations`: 散度求解最大迭代次数
      - `max_density_solver_iterations`: 密度求解最大迭代次数
    - `upper_bound/lower_bound`: 模拟区域边界
  - 计算参数:
    - `particle_volume`: 粒子体积 = 0.8 * particle_size^3

- **`setup_boundary(self)`**
  - 功能: 设置边界条件
  - 操作: 创建CubeBoundary

- **`init_particle_fields(self)`**
  - 功能: 初始化粒子场结构
  - 动态状态字段:
    - pos: 位置
    - vel: 速度
    - acc: 加速度
    - rho: 密度
    - p: 压力
    - dfsph_factor: DFSPH因子
    - drho: 密度导数
  - 静态信息字段:
    - rho (rest): 静止密度
    - mass: 质量
    - stiffness: 刚度系数
    - exponent: 指数
    - mu: 粘度
    - gamma: 表面张力

- **`init_ckpt(self)`**
  - 功能: 初始化检查点字典

### 2. 实体管理模块 (Entity Management)
**功能**: 管理SPH实体

#### 涉及的函数:
- **`add_entity(self, idx, material, morph, surface)`**
  - 功能: 添加SPH实体
  - 操作: 创建SPHEntity对象

- **`is_active(self)`**
  - 功能: 检查求解器是否活跃
  - 返回: n_particles > 0

### 3. 空间哈希模块 (Spatial Hashing)
**功能**: 使用空间哈希加速邻居搜索

#### 涉及的函数:
- **`_kernel_reorder_particles(self, f)`** (Taichi Kernel)
  - 功能: 根据空间哈希重新排序粒子
  - 操作: 将粒子按哈希网格单元排序，便于邻居查找

- **`_kernel_copy_from_reordered(self, f)`** (Taichi Kernel)
  - 功能: 从重排序的数组复制回原始数组

### 4. 核函数模块 (Kernel Functions)
**功能**: 实现SPH核函数及其导数

#### 涉及的函数:
- **`cubic_kernel(self, r_norm)`** (Taichi函数)
  - 功能: 计算三次样条核函数
  - 参数: r_norm - 归一化距离
  - 返回: 核函数值
  - 公式:
    ```
    q = r_norm / h
    if 0 <= q < 1:
        W = k * (1 - 1.5*q^2 + 0.75*q^3)
    elif 1 <= q < 2:
        W = k * 0.25 * (2-q)^3
    else:
        W = 0
    ```

- **`cubic_kernel_derivative(self, r)`** (Taichi函数)
  - 功能: 计算核函数的梯度
  - 参数: r - 距离向量
  - 返回: 核函数梯度向量

### 5. WCSPH压力求解模块 (WCSPH Pressure Solver)
**功能**: 实现弱可压缩SPH的压力计算

#### 涉及的函数:
- **`_task_compute_rho(self, i, j, ret, i_b)`** (Taichi任务)
  - 功能: 计算粒子i与邻居j的密度贡献
  - 公式: rho += m_j * W(r_ij)

- **`_kernel_compute_rho(self, f)`** (Taichi Kernel)
  - 功能: 计算所有粒子的密度
  - 操作: 对每个粒子累积邻居的密度贡献
  - 压力计算: p = stiffness * ((rho/rho_0)^exponent - 1)

### 6. 力计算模块 (Force Computation)
**功能**: 计算SPH粒子受到的各种力

#### 涉及的函数:
- **`_task_compute_non_pressure_forces(self, i, j, ret, i_b)`** (Taichi任务)
  - 功能: 计算非压力力（粘性力、表面张力）
  - 粘性力: f_visc = mu * m_j * (v_j - v_i) / rho_j * nabla_W
  - 表面张力: f_surf = -gamma * nabla_W

- **`_kernel_compute_non_pressure_forces(self, f, t)`** (Taichi Kernel)
  - 功能: 计算所有粒子的非压力力
  - 包括: 粘性力、表面张力、重力

- **`_task_compute_pressure_forces(self, i, j, ret, i_b)`** (Taichi任务)
  - 功能: 计算压力力
  - 公式: f_press = -m_j * (p_i/rho_i^2 + p_j/rho_j^2) * nabla_W

- **`_kernel_compute_pressure_forces(self, f)`** (Taichi Kernel)
  - 功能: 计算所有粒子的压力力

### 7. 时间积分模块 (Time Integration)
**功能**: 更新粒子速度和位置

#### 涉及的函数:
- **`_kernel_advect_velocity(self, f)`** (Taichi Kernel)
  - 功能: 更新粒子速度
  - 公式: vel += dt * acc

- **`_kernel_advect_position(self, f)`** (Taichi Kernel)
  - 功能: 更新粒子位置
  - 公式: pos += dt * vel
  - 处理边界碰撞

### 8. DFSPH压力求解模块 (DFSPH Pressure Solver)
**功能**: 实现无散度SPH的压力投影

#### 涉及的函数:
- **`_task_compute_DFSPH_factor(self, i, j, ret, i_b)`** (Taichi任务)
  - 功能: 计算DFSPH因子
  - 公式: factor = -1 / (sum_j (m_j * nabla_W)^2 / rho_j^2)

- **`_kernel_compute_DFSPH_factor(self, f)`** (Taichi Kernel)
  - 功能: 计算所有粒子的DFSPH因子

- **`_task_compute_density_time_derivative(self, i, j, ret, i_b)`** (Taichi任务)
  - 功能: 计算密度时间导数
  - 公式: drho = sum_j (m_j * (v_i - v_j) · nabla_W)

- **`_kernel_compute_density_time_derivative(self)`** (Taichi Kernel)
  - 功能: 计算所有粒子的密度时间导数

- **`_task_divergence_solver_iteration(self, i, j, ret, i_b)`** (Taichi任务)
  - 功能: 散度求解器的一次迭代
  - 计算压力修正和速度修正

- **`_kernel_divergence_solver_iteration(self)`** (Taichi Kernel)
  - 功能: 执行散度求解器迭代

- **`_divergence_solver_iteration(self)`**
  - 功能: 完整的散度求解器迭代
  - 步骤:
    1. 计算密度导数
    2. 迭代更新压力和速度直到散度误差小于阈值

- **`_divergence_solve(self, f)`**
  - 功能: 完整的散度求解
  - 操作:
    1. 计算DFSPH因子
    2. 执行散度求解器迭代

- **`_kernel_predict_velocity(self, f)`** (Taichi Kernel)
  - 功能: 预测速度
  - 操作: 添加非压力力的影响

- **`_task_compute_density_star(self, i, j, ret, i_b)`** (Taichi任务)
  - 功能: 计算预测密度

- **`_kernel_compute_density_star(self)`** (Taichi Kernel)
  - 功能: 计算所有粒子的预测密度

- **`density_solve_iteration_task(self, i, j, ret, i_b)`** (Taichi任务)
  - 功能: 密度求解器的一次迭代任务

- **`_kernel_density_solve_iteration(self)`** (Taichi Kernel)
  - 功能: 执行密度求解器迭代

- **`_density_solve_iteration(self)`**
  - 功能: 完整的密度求解器迭代

- **`_kernel_multiply_time_step(self, field, time_step)`** (Taichi Kernel)
  - 功能: 将场乘以时间步长

- **`_density_solve(self, f)`**
  - 功能: 完整的密度求解
  - 步骤:
    1. 预测速度
    2. 计算预测密度
    3. 迭代求解压力和速度直到密度误差小于阈值

### 9. 时间步进模块 (Time Stepping)
**功能**: 管理时间步进流程

#### 涉及的函数:
- **`process_input(self, in_backward=False)`**
  - 功能: 处理实体输入

- **`process_input_grad(self)`**
  - 功能: 处理输入梯度

- **`substep_pre_coupling(self, f)`**
  - 功能: 耦合前的子步骤
  - WCSPH流程:
    1. _kernel_reorder_particles(f) - 重排序粒子
    2. _kernel_compute_rho(f) - 计算密度和压力
    3. _kernel_compute_non_pressure_forces(f, t) - 计算非压力力
    4. _kernel_compute_pressure_forces(f) - 计算压力力
    5. _kernel_copy_from_reordered(f) - 复制回原始数组
  - DFSPH流程:
    1. _kernel_reorder_particles(f) - 重排序粒子
    2. _kernel_compute_non_pressure_forces(f, t) - 计算非压力力
    3. _divergence_solve(f) - 散度求解
    4. _kernel_copy_from_reordered(f) - 复制回原始数组

- **`substep_pre_coupling_grad(self, f)`**
  - 功能: 耦合前梯度

- **`substep_post_coupling(self, f)`**
  - 功能: 耦合后的子步骤
  - WCSPH: _kernel_advect_velocity(f) + _kernel_advect_position(f)
  - DFSPH: _density_solve(f) + _kernel_advect_velocity(f) + _kernel_advect_position(f)

- **`substep_post_coupling_grad(self, f)`**
  - 功能: 耦合后梯度

### 10. 状态管理模块 (State Management)
**功能**: 管理状态的保存、加载和查询

#### 涉及的函数:
- **`get_state(self, f)`**
  - 功能: 获取状态
  - 返回: SPHSolverState对象(pos, vel, active)

- **`set_state(self, f, state, envs_idx=None)`**
  - 功能: 设置状态

- **`save_ckpt(self, ckpt_name)`**
  - 功能: 保存检查点

- **`load_ckpt(self, ckpt_name)`**
  - 功能: 加载检查点

### 11. 梯度管理模块 (Gradient Management)
**功能**: 管理可微分模拟的梯度

#### 涉及的函数:
- **`reset_grad(self)`**
  - 功能: 重置梯度

- **`collect_output_grads(self)`**
  - 功能: 收集输出梯度

- **`add_grad_from_state(self, state)`**
  - 功能: 从状态添加梯度

### 12. 渲染支持模块 (Rendering Support)
**功能**: 更新用于渲染的场数据

#### 涉及的函数:
- **`update_render_fields(self)`**
  - 功能: 更新渲染字段
  - 操作: 调用_kernel_update_render_fields

### 13. 粒子操作模块 (Particle Operations)
**功能**: 提供粒子级别的操作接口

#### 涉及的内核函数:
- `_kernel_add_particles()`: 添加粒子
- `_kernel_set_particles_pos()`: 设置粒子位置
- `_kernel_get_particles_pos()`: 获取粒子位置
- `_kernel_set_particles_vel()`: 设置粒子速度
- `_kernel_get_particles_vel()`: 获取粒子速度
- `_kernel_set_particles_active()`: 设置粒子激活状态
- `_kernel_get_particles_active()`: 获取粒子激活状态
- `_kernel_get_mass()`: 获取粒子质量

### 14. 属性访问模块 (Property Access)
**功能**: 提供只读属性访问

#### 涉及的属性:
- `n_particles`: 粒子数量
- `particle_size`: 粒子尺寸
- `support_radius`: 支持半径
- `upper_bound/lower_bound`: 边界

## 主要功能管线

### WCSPH时间步进管线 (WCSPH Time Stepping Pipeline)
```
1. process_input() - 处理输入
   ↓
2. substep_pre_coupling(f) - 耦合前计算
   ├─ _kernel_reorder_particles(f) - 空间哈希重排序
   ├─ _kernel_compute_rho(f) - 计算密度
   │  ├─ 累积邻居密度贡献: rho = sum_j(m_j * W(r_ij))
   │  └─ 计算压力: p = k * ((rho/rho_0)^gamma - 1)
   ├─ _kernel_compute_non_pressure_forces(f, t) - 计算非压力力
   │  ├─ 粘性力
   │  ├─ 表面张力
   │  └─ 重力
   ├─ _kernel_compute_pressure_forces(f) - 计算压力力
   └─ _kernel_copy_from_reordered(f) - 复制回原数组
   ↓
3. [耦合处理]
   ↓
4. substep_post_coupling(f) - 耦合后计算
   ├─ _kernel_advect_velocity(f) - 更新速度: vel += dt * acc
   └─ _kernel_advect_position(f) - 更新位置: pos += dt * vel
```

### DFSPH时间步进管线 (DFSPH Time Stepping Pipeline)
```
1. process_input() - 处理输入
   ↓
2. substep_pre_coupling(f) - 耦合前计算
   ├─ _kernel_reorder_particles(f) - 空间哈希重排序
   ├─ _kernel_compute_non_pressure_forces(f, t) - 计算非压力力
   ├─ _divergence_solve(f) - 散度自由求解
   │  ├─ _kernel_compute_DFSPH_factor(f) - 计算DFSPH因子
   │  └─ 迭代直到散度误差 < threshold:
   │     ├─ _kernel_compute_density_time_derivative() - 计算drho/dt
   │     └─ _kernel_divergence_solver_iteration() - 更新压力和速度
   └─ _kernel_copy_from_reordered(f) - 复制回原数组
   ↓
3. [耦合处理]
   ↓
4. substep_post_coupling(f) - 耦合后计算
   ├─ _density_solve(f) - 密度修正
   │  ├─ _kernel_predict_velocity(f) - 预测速度
   │  ├─ _kernel_compute_density_star() - 计算预测密度
   │  └─ 迭代直到密度误差 < threshold:
   │     └─ _kernel_density_solve_iteration() - 更新压力和速度
   ├─ _kernel_advect_velocity(f) - 更新速度
   └─ _kernel_advect_position(f) - 更新位置
```

### DFSPH散度求解管线 (DFSPH Divergence Solve Pipeline)
```
1. _kernel_compute_DFSPH_factor(f) - 计算alpha_i
   └─ alpha_i = -1 / sum_j((m_j * nabla_W / rho_j)^2)
   ↓
2. 迭代 (直到散度误差 < max_divergence_error):
   ├─ _kernel_compute_density_time_derivative() - 计算drho/dt
   │  └─ drho_i = sum_j(m_j * (v_i - v_j) · nabla_W)
   ├─ _kernel_divergence_solver_iteration() - 更新
   │  ├─ kappa_i = drho_i / (dt * rho_0 * alpha_i)
   │  └─ v_i -= dt * alpha_i * sum_j(m_j * kappa_j / rho_j * nabla_W)
   └─ 检查: max|drho| < max_divergence_error
```

### DFSPH密度求解管线 (DFSPH Density Solve Pipeline)
```
1. _kernel_predict_velocity(f) - 预测速度
   └─ v_i = v_i + dt * a_i (非压力加速度)
   ↓
2. _kernel_compute_density_star() - 计算预测密度
   └─ rho*_i = sum_j(m_j * W(r_i + dt*v_i, r_j + dt*v_j))
   ↓
3. 迭代 (直到密度误差 < max_density_error_percent):
   ├─ _kernel_density_solve_iteration() - 更新
   │  ├─ kappa_i = (rho*_i - rho_0) / (dt^2 * alpha_i)
   │  └─ v_i -= dt * alpha_i * sum_j(m_j * kappa_j / rho_j * nabla_W)
   └─ 检查: max|rho* - rho_0| / rho_0 < max_density_error_percent
```

## 设计特点
1. **双求解器**: 支持WCSPH和DFSPH两种压力求解方法
   - WCSPH: 简单快速，但稍有可压缩性
   - DFSPH: 严格不可压缩，但计算量较大
2. **空间哈希**: 使用空间哈希加速邻居搜索
3. **核函数**: 使用三次样条核函数
4. **可微分**: 支持梯度反向传播（仅WCSPH）
5. **力模型**:
   - 压力力（保持不可压缩性）
   - 粘性力（能量耗散）
   - 表面张力（表面效应）
   - 重力（外力）
6. **DFSPH优势**:
   - 无散度约束: ∇·v = 0
   - 常密度约束: rho = rho_0
   - 更稳定的模拟结果
7. **批处理**: 支持多环境并行模拟
