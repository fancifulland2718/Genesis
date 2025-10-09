# RigidSolver 求解器文档

## 概述
RigidSolver是一个刚体动力学求解器，用于模拟铰接刚体系统（关节机器人、角色等）。支持碰撞检测、约束求解、关节限制、接触岛和休眠等高级特性。

**涉及的类**: `RigidSolver`, `StaticRigidSimConfig`
**涉及的辅助类**: `ConstraintSolver`, `ConstraintSolverIsland`, `Collider`, `SDF`

## 功能模块划分

### 1. 初始化模块 (Initialization)
**功能**: 设置刚体求解器的基本配置和数据结构

#### 涉及的类:
- **`StaticRigidSimConfig`**
  - 功能: 存储静态模拟配置
  - 属性: 各种求解器参数（iterations, tolerance, 阻尼等）

#### 涉及的函数:
- **`__init__(self, scene, sim, options)`**
  - 功能: 构造函数，初始化刚体求解器
  - 关键配置:
    - `enable_collision`: 是否启用碰撞检测
    - `enable_multi_contact`: 是否启用多接触
    - `enable_mujoco_compatibility`: MuJoCo兼容模式
    - `enable_joint_limit`: 是否启用关节限制
    - `enable_self_collision`: 是否启用自碰撞
    - `enable_adjacent_collision`: 是否启用相邻碰撞
    - `disable_constraint`: 是否禁用约束
    - `max_collision_pairs`: 最大碰撞对数
    - `integrator`: 积分器类型
    - `box_box_detection`: 盒子碰撞检测方法
    - `use_contact_island`: 是否使用接触岛
    - `use_hibernation`: 是否使用休眠

- **`_init_mass_mat(self)`**
  - 功能: 初始化质量矩阵
  - 操作: 计算每个实体的最大自由度数

- **`_init_dof_fields(self)`**
  - 功能: 初始化自由度场
  - 字段: pos, vel, qf (控制力), kp, kv, force_range, stiffness, armature, damping, frictionloss, limit等

- **`_init_link_fields(self)`**
  - 功能: 初始化连杆场
  - 字段: pos, quat, vel, ang, inertia, mass, com等

- **`_init_geom_fields(self)`**
  - 功能: 初始化几何体场
  - 字段: pos, quat, friction_ratio, needs_coup, sol_params等

- **`_init_entity_fields(self)`**
  - 功能: 初始化实体场
  - 包括实体全局索引、自由度偏移等

- **`_init_equality_fields(self)`**
  - 功能: 初始化等式约束场
  - 用于weld、distance等约束

- **`_init_collider(self)`**
  - 功能: 初始化碰撞检测器
  - 操作: 创建Collider对象

- **`_init_constraint_solver(self)`**
  - 功能: 初始化约束求解器
  - 选择: ConstraintSolver或ConstraintSolverIsland（如果使用接触岛）

### 2. 实体管理模块 (Entity Management)
**功能**: 管理刚体实体

#### 涉及的函数:
- **`add_entity(self, idx, material, morph, surface, visualize_contact)`**
  - 功能: 添加刚体实体
  - 支持: RigidEntity, DroneEntity, AvatarEntity
  - 返回: 创建的Entity对象

- **`is_active(self)`**
  - 功能: 检查求解器是否活跃
  - 返回: n_links > 0

### 3. 正向动力学模块 (Forward Dynamics)
**功能**: 计算加速度和力

#### 涉及的函数:
- **`_func_forward_dynamics(self)`**
  - 功能: 正向动力学计算
  - 操作: 调用kernel_forward_dynamics
  - 算法: 递归牛顿-欧拉算法(RNEA)或复合刚体算法(CRB)

- **`_func_update_acc(self)`**
  - 功能: 更新加速度
  - 操作: 从力计算加速度

### 4. 正向运动学模块 (Forward Kinematics)
**功能**: 从关节角度计算连杆位姿

#### 涉及的函数:
- **`_func_forward_kinematics_entity(self, i_e, envs_idx)`**
  - 功能: 为特定实体计算正向运动学
  - 操作: 调用kernel_forward_kinematics_entity
  - 算法: 从基座开始递归计算每个连杆的位姿

- **`_func_integrate_dq_entity(self, dq, i_e, i_b, respect_joint_limit)`**
  - 功能: 积分关节速度得到关节位置
  - 参数: dq - 关节速度增量
  - 操作: 考虑关节限制（如果启用）

- **`_func_update_geoms(self, envs_idx, force_update_fixed_geoms=False)`**
  - 功能: 更新几何体位姿
  - 操作: 根据连杆位姿更新几何体

### 5. 约束求解模块 (Constraint Solving)
**功能**: 求解约束力

#### 涉及的函数:
- **`_func_constraint_force(self)`**
  - 功能: 计算约束力
  - 步骤:
    1. _func_constraint_clear() - 清除约束
    2. add_equality_constraints() - 添加等式约束
    3. collider.detection() - 碰撞检测
    4. add_collision_constraints() - 添加碰撞约束
    5. add_joint_limit_constraints() - 添加关节限制约束
    6. add_frictionloss_constraints() - 添加摩擦损失约束
    7. resolve() - 求解约束

- **`_func_constraint_clear(self)`**
  - 功能: 清除约束计数器

### 6. 碰撞检测模块 (Collision Detection)
**功能**: 检测几何体间的碰撞

#### 涉及的函数:
- **`_kernel_detect_collision(self)`**
  - 功能: 执行碰撞检测
  - 操作: 
    1. collider.clear() - 清除碰撞
    2. collider.detection() - 检测碰撞

- **`detect_collision(self, env_idx=0)`**
  - 功能: 获取碰撞对
  - 返回: 碰撞几何体对的数组

### 7. 时间步进模块 (Time Stepping)
**功能**: 管理模拟的时间步进

#### 涉及的函数:
- **`substep(self)`**
  - 功能: 执行一个子步骤
  - 步骤:
    1. kernel_step_1() - 第一阶段
       - 正向运动学
       - 更新几何体
       - 碰撞检测（如果启用接触岛）
    2. _func_constraint_force() - 约束力计算（如果不使用SAP耦合）
       - 碰撞检测
       - 添加约束
       - 求解约束
    3. kernel_step_2() - 第二阶段
       - 正向动力学
       - 积分
       - 更新状态

- **`substep_pre_coupling(self, f)`**
  - 功能: 耦合前的子步骤
  - 操作: 空实现（刚体求解器不需要）

- **`substep_post_coupling(self, f)`**
  - 功能: 耦合后的子步骤
  - 操作: 空实现

- **`process_input(self, in_backward=False)`**
  - 功能: 处理输入
  - 操作: 空实现

- **`process_input_grad(self)`**
  - 功能: 处理输入梯度
  - 操作: 空实现

### 8. 状态管理模块 (State Management)
**功能**: 管理刚体系统的状态

#### 涉及的函数:
- **`get_state(self, f)`**
  - 功能: 获取状态
  - 返回: RigidSolverState对象
  - 包含: qpos, dofs_vel, links_pos, links_quat, links_vel, links_ang

- **`set_state(self, f, state, envs_idx=None)`**
  - 功能: 设置状态
  - 操作: 设置关节位置、速度、连杆位姿等
  - 后处理: 正向运动学、更新几何体、碰撞检测、约束清除

- **`save_ckpt(self, ckpt_name)`**
  - 功能: 保存检查点
  - 操作: 空实现

- **`load_ckpt(self, ckpt_name)`**
  - 功能: 加载检查点
  - 操作: 空实现

### 9. 控制接口模块 (Control Interface)
**功能**: 提供丰富的控制接口

#### 位置和姿态设置:
- **`set_links_pos(pos, links_idx, envs_idx, skip_forward, unsafe)`**
  - 功能: 设置连杆位置

- **`set_base_links_pos(pos, envs_idx, entities_idx, skip_forward, unsafe)`**
  - 功能: 设置基座连杆位置

- **`set_links_quat(quat, links_idx, envs_idx, skip_forward, unsafe)`**
  - 功能: 设置连杆四元数

- **`set_base_links_quat(quat, envs_idx, entities_idx, skip_forward, unsafe)`**
  - 功能: 设置基座连杆四元数

#### 质量和惯性设置:
- **`set_links_mass_shift(mass, links_idx, envs_idx, unsafe)`**
  - 功能: 设置连杆质量偏移

- **`set_links_COM_shift(com, links_idx, envs_idx, unsafe)`**
  - 功能: 设置连杆质心偏移

- **`set_links_inertial_mass(mass, links_idx, envs_idx, unsafe)`**
  - 功能: 设置连杆惯性质量

#### 摩擦和求解器参数:
- **`set_geoms_friction_ratio(friction_ratio, geoms_idx, envs_idx, unsafe)`**
  - 功能: 设置几何体摩擦比率

- **`set_global_sol_params(sol_params, unsafe)`**
  - 功能: 设置全局求解器参数

- **`set_sol_params(sol_params, geoms_idx, envs_idx, joints_idx, eqs_idx, unsafe)`**
  - 功能: 设置特定约束的求解器参数
  - 参数: (timeconst, dampratio, dmin, dmax, width, mid, power)

- **`get_sol_params(geoms_idx, envs_idx, joints_idx, eqs_idx, unsafe)`**
  - 功能: 获取求解器参数

#### 关节参数设置:
- **`set_dofs_kp(kp, dofs_idx, envs_idx, unsafe)`**
  - 功能: 设置PD控制器的比例增益

- **`set_dofs_kv(kv, dofs_idx, envs_idx, unsafe)`**
  - 功能: 设置PD控制器的微分增益

- **`set_dofs_force_range(lower, upper, dofs_idx, envs_idx, unsafe)`**
  - 功能: 设置关节力范围

- **`set_dofs_stiffness(stiffness, dofs_idx, envs_idx, unsafe)`**
  - 功能: 设置关节刚度

- **`set_dofs_armature(armature, dofs_idx, envs_idx, unsafe)`**
  - 功能: 设置关节电枢（rotor inertia）

- **`set_dofs_damping(damping, dofs_idx, envs_idx, unsafe)`**
  - 功能: 设置关节阻尼

- **`set_dofs_frictionloss(frictionloss, dofs_idx, envs_idx, unsafe)`**
  - 功能: 设置关节摩擦损失

- **`set_dofs_limit(lower, upper, dofs_idx, envs_idx, unsafe)`**
  - 功能: 设置关节限制

- **`set_dofs_velocity(velocity, dofs_idx, envs_idx, skip_forward, unsafe)`**
  - 功能: 设置关节速度

- **`set_dofs_position(position, dofs_idx, envs_idx, skip_forward, unsafe)`**
  - 功能: 设置关节位置

#### 控制命令:
- **`control_dofs_force(force, dofs_idx, envs_idx, unsafe)`**
  - 功能: 力控制

- **`control_dofs_velocity(velocity, dofs_idx, envs_idx, unsafe)`**
  - 功能: 速度控制
  - 操作: 设置qf = kv * (target_vel - current_vel)

- **`control_dofs_position(position, dofs_idx, envs_idx, unsafe)`**
  - 功能: 位置控制
  - 操作: PD控制 qf = kp * (target_pos - current_pos) + kv * (0 - current_vel)

#### 外力施加:
- **`apply_links_external_force(force, pos, links_idx, envs_idx, unsafe)`**
  - 功能: 在指定位置施加外力
  - 参数:
    - force: 力向量
    - pos: 施力点位置
    - links_idx: 连杆索引

- **`apply_links_external_torque(torque, links_idx, envs_idx, unsafe)`**
  - 功能: 施加外力矩

### 10. 梯度管理模块 (Gradient Management)
**功能**: 管理梯度（部分支持）

#### 涉及的函数:
- **`reset_grad(self)`**
  - 功能: 重置梯度
  - 操作: 空实现

- **`collect_output_grads(self)`**
  - 功能: 收集输出梯度
  - 操作: 空实现

- **`add_grad_from_state(self, state)`**
  - 功能: 从状态添加梯度
  - 操作: 空实现

### 11. 渲染支持模块 (Rendering Support)
**功能**: 更新渲染所需的几何体变换

#### 涉及的函数:
- **`update_geoms_render_T(self)`**
  - 功能: 更新几何体渲染变换矩阵

- **`update_vgeoms_render_T(self)`**
  - 功能: 更新可视几何体渲染变换矩阵

### 12. 批处理与并行化模块 (Batching and Parallelization)
**功能**: 支持多环境并行模拟

#### 涉及的函数:
- **`_batch_shape(self, shape, first_dim, B)`**
  - 功能: 计算批处理形状

- **`_batch_array(self, arr, first_dim)`**
  - 功能: 批处理数组

- **`_process_dim(self, tensor, envs_idx)`**
  - 功能: 处理张量维度

### 13. 参数验证模块 (Parameter Sanitization)
**功能**: 验证和处理输入参数

#### 涉及的函数:
- **`_sanitize_1D_io_variables(tensor, inputs_idx, input_size, envs_idx, batched, idx_name, skip_allocation, unsafe)`**
  - 功能: 验证1D输入/输出变量
  - 检查: 维度、范围、类型

- **`_sanitize_2D_io_variables(tensor, inputs_idx, input_size, vec_size, envs_idx, batched, idx_name, skip_allocation, unsafe)`**
  - 功能: 验证2D输入/输出变量

- **`_sanitize_sol_params(sol_params, min_timeconst, default_timeconst)`**
  - 功能: 验证求解器参数
  - 检查: timeconst, dampratio, dmin, dmax, width, mid, power的合法性

### 14. 接触岛与休眠模块 (Contact Island and Hibernation)
**功能**: 优化性能的高级特性

#### 概念:
- **接触岛**: 将刚体分组为独立的接触岛，每个岛独立求解约束
- **休眠**: 静止的刚体进入休眠状态，跳过计算

#### 相关字段:
- `n_awake_dofs`: 清醒的自由度数
- `awake_dofs`: 清醒的自由度索引
- `n_awake_links`: 清醒的连杆数
- `awake_links`: 清醒的连杆索引
- `n_awake_entities`: 清醒的实体数
- `awake_entities`: 清醒的实体索引

## 主要功能管线

### 刚体时间步进管线 (Rigid Body Time Stepping Pipeline)
```
substep() - 主循环
├─ kernel_step_1() - 第一阶段
│  ├─ 正向运动学
│  │  └─ 从关节角度计算连杆位姿
│  ├─ 更新几何体位姿
│  └─ (如果使用接触岛) 碰撞检测
│
├─ _func_constraint_force() - 约束力计算（如果不使用SAP耦合）
│  ├─ _func_constraint_clear() - 清除约束
│  ├─ add_equality_constraints() - 添加等式约束
│  ├─ collider.detection() - 碰撞检测
│  ├─ add_collision_constraints() - 添加碰撞约束
│  ├─ add_joint_limit_constraints() - 添加关节限制约束
│  ├─ add_frictionloss_constraints() - 添加摩擦损失约束
│  └─ resolve() - 求解约束（迭代优化）
│
└─ kernel_step_2() - 第二阶段
   ├─ 正向动力学
   │  └─ 从力计算加速度 (RNEA/CRB)
   ├─ 积分
   │  ├─ vel += acc * dt
   │  └─ pos += vel * dt
   └─ 更新状态
```

### 约束求解管线 (Constraint Solving Pipeline)
```
resolve() - 约束求解
└─ 迭代 (直到收敛或达到最大迭代次数):
   ├─ 对每个约束:
   │  ├─ 计算约束雅可比 J
   │  ├─ 计算约束违反 C
   │  ├─ 计算约束力 λ
   │  └─ 应用约束力: qf += J^T * λ
   └─ 检查收敛: ||C|| < tolerance
```

### 正向运动学管线 (Forward Kinematics Pipeline)
```
对每个实体:
├─ 从基座开始
└─ 递归计算每个连杆:
   ├─ 从父连杆获取位姿
   ├─ 应用关节变换
   │  └─ T_child = T_parent * T_joint(q)
   ├─ 计算连杆位置和姿态
   └─ 更新几何体位姿
```

### PD控制管线 (PD Control Pipeline)
```
control_dofs_position(target_pos):
├─ 计算位置误差: e_pos = target_pos - current_pos
├─ 计算速度误差: e_vel = 0 - current_vel
├─ 计算控制力: qf = kp * e_pos + kv * e_vel
├─ 钳位到力范围: qf = clamp(qf, force_range)
└─ 应用控制力
```

## 设计特点
1. **铰接刚体**: 专为关节机器人和铰接系统设计
2. **高效约束求解**: 迭代求解器，支持等式和不等式约束
3. **接触岛**: 将刚体分组为独立岛屿，提高求解效率
4. **休眠机制**: 静止刚体进入休眠，显著减少计算
5. **MuJoCo兼容**: 支持MuJoCo风格的接触模型
6. **丰富控制接口**: 位置、速度、力控制，PD控制器
7. **批处理**: 支持多环境并行模拟
8. **碰撞检测**: 支持多种几何体类型和碰撞算法
9. **关节限制**: 自动处理关节限制约束
10. **摩擦模型**: 支持库仑摩擦和摩擦损失

## 算法基础
- **正向动力学**: 递归牛顿-欧拉算法(RNEA)或复合刚体算法(CRB)
- **约束求解**: 投影Gauss-Seidel或其他迭代求解器
- **碰撞检测**: GJK, MPR, SAT等算法
- **接触模型**: MuJoCo风格的软接触或硬约束

## 应用场景
- 机器人仿真（机械臂、腿足机器人）
- 人形角色动画
- 物体操作
- 接触丰富的任务

## 优势
- 高效的铰接刚体模拟
- 稳定的约束求解
- 丰富的控制选项
- 良好的扩展性（接触岛、休眠）
- 多环境并行

## 限制
- 梯度支持有限（不完全可微）
- 主要针对铰接系统，非自由刚体的支持较弱
