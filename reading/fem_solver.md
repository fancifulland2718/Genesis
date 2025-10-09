# FEMSolver 求解器文档

## 概述
FEMSolver (Finite Element Method Solver) 是一个有限元法求解器，用于模拟可变形固体，特别是四面体网格。支持显式和隐式时间积分，以及多种材料模型。

**涉及的类**: `FEMSolver`

## 功能模块划分

### 1. 初始化模块 (Initialization)
**功能**: 设置FEM求解器的基本配置、场结构和边界条件

#### 涉及的函数:
- **`__init__(self, scene, sim, options)`**
  - 功能: 构造函数，初始化FEM求解器
  - 关键配置:
    - `floor_height`: 地板高度
    - `damping`: 阻尼系数
    - `use_implicit_solver`: 是否使用隐式求解器
    - `n_newton_iterations`: 牛顿迭代次数
    - `newton_dx_threshold`: 牛顿法阈值
    - `n_pcg_iterations`: PCG(预条件共轭梯度)迭代次数
    - `pcg_threshold`: PCG收敛阈值
    - `n_linesearch_iterations`: 线搜索迭代次数
    - `enable_vertex_constraints`: 是否启用顶点约束
  - 其他:
    - `vol_scale`: 体积缩放因子(1e4)，提高数值稳定性

- **`setup_boundary(self)`**
  - 功能: 设置边界条件
  - 操作: 创建FloorBoundary

- **`init_batch_fields(self)`**
  - 功能: 初始化批次字段
  - 字段: batch_active, batch_pcg_active, batch_linesearch_active, pcg_state

- **`init_element_fields(self)`**
  - 功能: 初始化单元场结构
  - 字段包括:
    - `elements_v`: 顶点状态(pos, vel)
    - `elements_v_info`: 顶点信息(mass, mass_over_dt2)
    - `elements_v_energy`: 能量相关(force, inertia)
    - `elements_i`: 单元信息(mu, lam, rho, mass, V, B, el2v等)
    - `elements_el`: 单元状态(actu)
    - `elements_el_energy`: 能量和梯度
    - `elements_el_hessian`: Hessian矩阵(12x12)

- **`init_surface_fields(self)`**
  - 功能: 初始化表面场结构
  - 用于渲染和接触

- **`init_constraints(self)`**
  - 功能: 初始化顶点约束
  - 字段: is_constrained, target_pos, link_idx

### 2. 实体与材料管理模块 (Entity and Material Management)
**功能**: 管理FEM实体和材料

#### 涉及的函数:
- **`add_entity(self, idx, material, morph, surface)`**
  - 功能: 添加FEM实体
  - 操作:
    1. 注册材料方法(update_stress, compute_energy_gradient_hessian, compute_energy)
    2. 创建FEMEntity对象

- **`is_active(self)`**
  - 功能: 检查求解器是否活跃
  - 返回: n_elements_max > 0

### 3. 显式时间积分模块 (Explicit Time Integration)
**功能**: 实现显式时间积分方案

#### 涉及的函数:
- **`init_pos_and_vel(self, f)`** (Taichi Kernel)
  - 功能: 初始化位置和速度
  - 操作: pos[f+1] = pos[f], vel[f+1] = vel[f]

- **`compute_vel(self, f)`** (Taichi Kernel)
  - 功能: 计算内力导致的速度变化
  - 步骤:
    1. 对每个单元计算变形梯度F = D @ B
    2. 计算应力(调用材料的update_stress)
    3. 计算力 H = -V * stress @ B^T
    4. 更新速度 vel += dt * force / mass

- **`apply_uniform_force(self, f)`** (Taichi Kernel)
  - 功能: 应用外力(重力)和阻尼
  - 操作:
    1. 应用阻尼: vel *= exp(-dt * damping)
    2. 添加重力: vel += dt * gravity

- **`compute_pos(self, f)`** (Taichi Kernel)
  - 功能: 从速度更新位置
  - 公式: pos[f+1] = pos[f] + dt * vel[f+1]

### 4. 隐式时间积分模块 (Implicit Time Integration)
**功能**: 实现隐式时间积分方案（牛顿法+PCG+线搜索）

#### 涉及的函数:
- **`precompute_material_data(self, f)`** (Taichi Kernel)
  - 功能: 预计算材料数据
  - 操作: 调用材料的pre_compute方法

- **`init_pos_and_inertia(self, f)`** (Taichi Kernel)
  - 功能: 初始化位置和惯性项
  - 公式: inertia = pos + vel*dt + gravity*dt^2
  - 处理顶点约束

- **`_compute_ele_J_F(self, f, i_e, i_b)`** (Taichi函数)
  - 功能: 计算单元的J(行列式)和F(变形梯度)
  - 返回: J, F

- **`compute_ele_hessian_gradient(self, f)`** (Taichi Kernel)
  - 功能: 计算单元能量、梯度和Hessian
  - 操作: 调用材料的compute_energy_gradient_hessian

- **`_func_compute_ele_energy(self, f)`** (Taichi函数)
  - 功能: 计算单元能量
  - 用于线搜索

- **`accumulate_vertex_force_preconditioner(self, f)`** (Taichi Kernel)
  - 功能: 累积顶点力和预条件子
  - 步骤:
    1. 计算单元对顶点的贡献
    2. 累积梯度得到力
    3. 累积Hessian对角块得到预条件子

#### PCG求解器子模块:
- **`init_pcg_solve(self)`** (Taichi Kernel)
  - 功能: 初始化PCG求解
  - 操作: 设置初始残差r、预条件z、搜索方向p

- **`compute_Ap(self)`** (Taichi Kernel)
  - 功能: 计算Hessian矩阵与向量p的乘积
  - 公式: Ap = (M/dt^2 + alpha_damping*M/dt + H) @ p

- **`one_pcg_iter(self)`** (Taichi Kernel)
  - 功能: 执行一次PCG迭代
  - 步骤:
    1. 计算Ap
    2. 计算alpha = rTz / pTAp
    3. 更新x, r, z
    4. 检查收敛
    5. 更新beta和p

- **`pcg_solve(self)`**
  - 功能: 完整的PCG求解
  - 操作: 调用init_pcg_solve和多次one_pcg_iter

#### 线搜索子模块:
- **`init_linesearch(self, f)`** (Taichi Kernel)
  - 功能: 初始化线搜索
  - 操作: 计算初始能量、搜索方向的梯度投影m

- **`one_linesearch_iter(self, f)`** (Taichi Kernel)
  - 功能: 执行一次线搜索迭代
  - 步骤:
    1. 更新位置: pos = pos_prev + step_size * dx
    2. 计算新能量
    3. 检查Armijo条件
    4. 如果不满足，缩小步长

- **`linesearch(self, f)`**
  - 功能: 完整的线搜索
  - 算法: Backtracking线搜索

- **`skip_linesearch(self, f)`** (Taichi Kernel)
  - 功能: 跳过线搜索，直接应用牛顿步
  - 用于n_linesearch_iterations=0的情况

#### 隐式求解主函数:
- **`batch_solve(self, f)`**
  - 功能: 批次隐式求解
  - 步骤（对每次牛顿迭代）:
    1. compute_ele_hessian_gradient(f) - 计算能量、梯度、Hessian
    2. accumulate_vertex_force_preconditioner(f) - 累积到顶点
    3. pcg_solve() - PCG求解线性系统
    4. linesearch(f) - 线搜索确定步长

- **`setup_pos_vel(self, f)`** (Taichi Kernel)
  - 功能: 从位置更新速度
  - 公式: vel = (pos[f+1] - pos[f]) / dt

### 5. 时间步进模块 (Time Stepping)
**功能**: 管理时间步进流程

#### 涉及的函数:
- **`process_input(self, in_backward=False)`**
  - 功能: 处理实体输入

- **`process_input_grad(self)`**
  - 功能: 处理输入梯度

- **`substep_pre_coupling(self, f)`**
  - 功能: 耦合前的子步骤
  - 隐式求解器:
    1. precompute_material_data(f)
    2. init_pos_and_inertia(f)
    3. batch_solve(f)
    4. setup_pos_vel(f)
  - 显式求解器:
    1. init_pos_and_vel(f)
    2. compute_vel(f)
    3. apply_uniform_force(f)
    4. compute_pos(f)

- **`substep_pre_coupling_grad(self, f)`**
  - 功能: 耦合前梯度（逆序）

- **`substep_post_coupling(self, f)`**
  - 功能: 耦合后的子步骤
  - 操作: 应用硬约束和软约束

- **`substep_post_coupling_grad(self, f)`**
  - 功能: 耦合后梯度

### 6. 约束处理模块 (Constraint Handling)
**功能**: 处理顶点约束（固定或目标位置）

#### 涉及的函数:
- **`apply_hard_constraints(self, f)`** (Taichi Kernel)
  - 功能: 应用硬约束
  - 操作: 强制约束顶点的位置

- **`apply_soft_constraints(self, f)`** (Taichi Kernel)
  - 功能: 应用软约束
  - 操作: 计算约束力

- **`_kernel_set_vertex_constraints()`** (Taichi Kernel)
  - 功能: 设置顶点约束

- **`_kernel_update_constraint_targets()`** (Taichi Kernel)
  - 功能: 更新约束目标位置

- **`_kernel_update_linked_vertex_constraints()`** (Taichi Kernel)
  - 功能: 更新与连杆链接的顶点约束

- **`_kernel_remove_specific_constraints()`** (Taichi Kernel)
  - 功能: 移除特定约束

### 7. 状态管理模块 (State Management)
**功能**: 管理状态的保存、加载和查询

#### 涉及的函数:
- **`get_state(self, f)`**
  - 功能: 获取状态
  - 返回: FEMSolverState对象(pos, vel)

- **`get_state_render(self, f)`**
  - 功能: 获取渲染状态

- **`set_state(self, f, state, envs_idx=None)`**
  - 功能: 设置状态

- **`save_ckpt(self, ckpt_name)`**
  - 功能: 保存检查点

- **`load_ckpt(self, ckpt_name)`**
  - 功能: 加载检查点

- **`get_forces(self)`**
  - 功能: 获取顶点力

### 8. 梯度管理模块 (Gradient Management)
**功能**: 管理可微分模拟的梯度

#### 涉及的函数:
- **`reset_grad(self)`**
  - 功能: 重置梯度

- **`reset_grad_till_frame(self, f)`** (Taichi Kernel)
  - 功能: 重置到指定帧的梯度

- **`collect_output_grads(self)`**
  - 功能: 收集输出梯度

- **`add_grad_from_state(self, state)`**
  - 功能: 从状态添加梯度

- **`_kernel_add_grad_from_pos(self, f, pos_grad)`** (Taichi Kernel)
  - 功能: 添加位置梯度

- **`_kernel_add_grad_from_vel(self, f, vel_grad)`** (Taichi Kernel)
  - 功能: 添加速度梯度

### 9. 粒子/单元操作模块 (Element Operations)
**功能**: 提供单元和顶点级别的操作接口

#### 涉及的内核函数:
- `_kernel_add_elements()`: 添加单元
- `_kernel_set_elements_pos()`: 设置顶点位置
- `_kernel_set_elements_vel()`: 设置顶点速度
- `_kernel_set_elements_actu()`: 设置单元激活值
- `_kernel_set_active()`: 设置激活状态
- `_kernel_set_muscle_group()`: 设置肌肉组
- `_kernel_set_muscle_direction()`: 设置肌肉方向
- `_kernel_get_el2v()`: 获取单元到顶点映射
- `_kernel_get_state()`: 获取状态

### 10. 属性访问模块 (Property Access)
**功能**: 提供只读属性访问

#### 涉及的属性:
- `floor_height`: 地板高度
- `damping`: 阻尼系数
- `n_vertices`: 顶点数量
- `n_elements`: 单元数量
- `n_surfaces`: 表面数量
- `n_vertices_max`: 最大顶点数
- `n_elements_max`: 最大单元数
- `vol_scale`: 体积缩放因子
- `n_surface_vertices`: 表面顶点数
- `n_surface_elements`: 表面单元数

## 主要功能管线

### 显式时间积分管线 (Explicit Time Integration Pipeline)
```
1. init_pos_and_vel(f) - 初始化
   ↓
2. compute_vel(f) - 计算内力导致的速度变化
   ├─ 计算变形梯度 F = D @ B
   ├─ 调用材料模型计算应力
   └─ 更新速度 vel += dt * force / mass
   ↓
3. apply_uniform_force(f) - 应用外力和阻尼
   ├─ 阻尼: vel *= exp(-dt * damping)
   └─ 重力: vel += dt * gravity
   ↓
4. compute_pos(f) - 更新位置
   └─ pos = pos + dt * vel
```

### 隐式时间积分管线 (Implicit Time Integration Pipeline)
```
1. precompute_material_data(f) - 材料预计算
   ↓
2. init_pos_and_inertia(f) - 初始化惯性项
   └─ inertia = pos + vel*dt + gravity*dt^2
   ↓
3. batch_solve(f) - 牛顿法求解
   └─ 对每次牛顿迭代:
      ├─ compute_ele_hessian_gradient(f) - 计算能量、梯度、Hessian
      ├─ accumulate_vertex_force_preconditioner(f) - 累积到顶点
      ├─ pcg_solve() - PCG求解: (M/dt^2 + H) dx = -force
      │  └─ 多次PCG迭代直到收敛
      └─ linesearch(f) - 线搜索确定步长
         └─ Backtracking直到满足Armijo条件
   ↓
4. setup_pos_vel(f) - 从位置计算速度
   └─ vel = (pos_new - pos_old) / dt
```

### PCG求解管线 (PCG Solver Pipeline)
```
1. init_pcg_solve() - 初始化
   ├─ x = 0
   ├─ r = b (残差 = 力)
   ├─ z = M^-1 @ r (预条件)
   └─ p = z (搜索方向)
   ↓
2. 迭代 (直到收敛或达到最大迭代次数):
   ├─ compute_Ap() - 计算 Ap
   ├─ alpha = rTz / pTAp
   ├─ x += alpha * p
   ├─ r -= alpha * Ap
   ├─ z = M^-1 @ r
   ├─ 检查收敛: ||r|| < threshold
   ├─ beta = rTz_new / rTz_old
   └─ p = z + beta * p
```

### 线搜索管线 (Line Search Pipeline)
```
1. init_linesearch(f) - 初始化
   ├─ 计算当前能量 E_0
   ├─ 计算方向导数 m = -dx^T @ force
   └─ 初始步长 step_size = 1.0
   ↓
2. 迭代 (直到满足Armijo条件):
   ├─ 更新位置: pos = pos_prev + step_size * dx
   ├─ 计算新能量 E
   ├─ 检查: E <= E_0 + c * step_size * m
   ├─ 如果不满足: step_size *= tau (缩小步长)
   └─ 否则: 接受步长
```

## 设计特点
1. **双模式**: 同时支持显式和隐式时间积分
2. **隐式求解**: 使用牛顿法+PCG+线搜索实现鲁棒的隐式求解
3. **可微分**: 完整支持梯度反向传播
4. **材料扩展性**: 灵活的材料接口
5. **约束支持**: 支持硬约束和软约束
6. **优化技术**: 
   - 预条件共轭梯度(PCG)加速线性系统求解
   - 线搜索保证能量递减
   - Hessian缓存优化（对于Hessian不变的材料）
7. **数值稳定性**: 体积缩放因子、阻尼项
