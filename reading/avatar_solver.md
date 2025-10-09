# AvatarSolver 求解器文档

## 概述
AvatarSolver是RigidSolver的派生类，维护运动学树但不考虑实际物理。主要用于角色动画和运动学控制。

**涉及的类**: `AvatarSolver` (继承自 `RigidSolver`)

## 功能模块划分

### 1. 初始化模块 (Initialization)
**功能**: 设置Avatar求解器的基本配置和约束

#### 涉及的函数:
- **`__init__(self, scene, sim, options)`**
  - 功能: 构造函数，初始化Avatar求解器
  - 参数:
    - `scene`: 场景对象
    - `sim`: 模拟器对象
    - `options`: 配置选项
  - 操作: 
    - 调用Solver基类的构造函数（跳过RigidSolver）
    - 设置碰撞检测选项（enable_collision, enable_self_collision, enable_adjacent_collision）
    - 设置最大碰撞对数量
    - 存储配置选项

- **`_init_mass_mat(self)`**
  - 功能: 初始化质量矩阵
  - 操作: 计算所有实体中的最大自由度数

- **`_init_invweight(self)`**
  - 功能: 初始化逆权重
  - 操作: 空实现（Avatar不需要物理约束求解）

- **`_init_constraint_solver(self)`**
  - 功能: 初始化约束求解器
  - 操作: 将constraint_solver设为None（Avatar不需要约束求解）

### 2. 运动学更新模块 (Kinematic Update)
**功能**: 更新角色的运动学状态

#### 涉及的函数:
- **`update_body(self)`**
  - 功能: 更新身体状态
  - 操作:
    - 调用`_kernel_forward_kinematics()`进行正向运动学
    - 调用`_kernel_update_geoms()`更新几何体

- **`substep(self)`**
  - 功能: 执行子步骤
  - 操作: 调用`_kernel_step()`

- **`_kernel_step(self)`** (Taichi Kernel)
  - 功能: 运动学步进的核心内核
  - 操作流程:
    1. 调用`_func_integrate()`进行积分
    2. 对每个环境批次执行：
       - `_func_forward_kinematics()` - 正向运动学
       - `_func_update_geoms()` - 更新几何体
    3. 如果启用碰撞检测，调用`_func_detect_collision()`

- **`_kernel_forward_kinematics_links_geoms(self, envs_idx)`** (Taichi Kernel)
  - 功能: 为指定环境更新连杆和几何体的正向运动学
  - 参数: `envs_idx` - 环境索引数组
  - 操作:
    - 对每个指定环境执行正向运动学计算
    - 更新几何体位置和姿态

### 3. 碰撞检测模块 (Collision Detection)
**功能**: 检测Avatar之间或Avatar与环境的碰撞

#### 涉及的函数:
- **`_func_detect_collision(self)`** (Taichi函数)
  - 功能: 检测碰撞
  - 操作:
    - 清空碰撞检测器
    - 执行碰撞检测

### 4. 状态管理模块 (State Management)
**功能**: 管理Avatar的状态获取

#### 涉及的函数:
- **`get_state(self, f)`**
  - 功能: 获取指定帧的Avatar状态
  - 参数: `f` - 帧索引
  - 返回: AvatarSolverState对象或None
  - 操作:
    - 如果求解器活跃，创建AvatarSolverState对象
    - 调用`_kernel_get_state()`获取状态数据：
      - qpos: 关节位置
      - dofs_vel: 自由度速度
      - links_pos: 连杆位置
      - links_quat: 连杆四元数
      - links_vel: 连杆速度
      - links_ang: 连杆角速度

### 5. 调试输出模块 (Debug Output)
**功能**: 输出碰撞接触数据用于调试

#### 涉及的函数:
- **`print_contact_data(self)`**
  - 功能: 打印接触数据
  - 操作: 调用碰撞检测器的print_contact_data方法

## 主要功能管线

### 运动学更新管线 (Kinematic Update Pipeline)
```
1. _func_integrate() - 从控制输入积分得到关节位置/速度
   ↓
2. _func_forward_kinematics() - 正向运动学计算
   ├─ 从关节空间计算到笛卡尔空间
   ├─ 更新连杆位置和姿态
   └─ 计算连杆速度和角速度
   ↓
3. _func_update_geoms() - 更新几何体
   ├─ 根据连杆变换更新几何体位置
   └─ 更新碰撞检测所需的几何信息
   ↓
4. _func_detect_collision() - 碰撞检测（如果启用）
   └─ 检测Avatar之间或与环境的碰撞
```

### 状态查询管线 (State Query Pipeline)
```
1. get_state(f) - 请求获取状态
   ↓
2. _kernel_get_state() - Taichi内核提取状态
   ├─ qpos: 关节位置
   ├─ dofs_vel: 关节速度
   ├─ links_pos: 连杆位置
   ├─ links_quat: 连杆四元数
   ├─ links_vel: 连杆线速度
   └─ links_ang: 连杆角速度
   ↓
3. 返回AvatarSolverState对象
```

## 与RigidSolver的主要区别
1. **无物理约束**: Avatar不进行物理约束求解，不计算接触力
2. **纯运动学**: 仅执行运动学计算，不考虑动力学
3. **简化质量**: 不需要完整的质量矩阵和逆权重计算
4. **控制驱动**: 直接从控制输入更新状态，不经过力/加速度计算

## 设计特点
1. **继承复用**: 继承RigidSolver的大部分功能，选择性覆盖物理相关部分
2. **运动学优先**: 专注于运动学树的维护和更新
3. **轻量级**: 去除了约束求解和动力学计算的开销
4. **碰撞支持**: 保留碰撞检测功能用于交互查询
