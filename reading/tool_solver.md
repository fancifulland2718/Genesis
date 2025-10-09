# ToolSolver 求解器文档

## 概述
ToolSolver是一个临时解决方案，用于实现刚体到柔体的单向可微耦合。该类将在RigidSolver添加可微性后被移除。

**涉及的类**: `ToolSolver`

## 功能模块划分

### 1. 初始化模块 (Initialization)
**功能**: 负责求解器的初始化配置和边界设置

#### 涉及的函数:
- **`__init__(self, scene, sim, options)`**
  - 功能: 构造函数，初始化求解器
  - 参数:
    - `scene`: 场景对象
    - `sim`: 模拟器对象
    - `options`: 配置选项
  - 操作: 设置地板高度，调用边界设置
  
- **`build(self)`**
  - 功能: 构建求解器
  - 操作: 调用父类build方法，为每个实体调用build
  
- **`setup_boundary(self)`**
  - 功能: 设置边界条件
  - 操作: 创建FloorBoundary对象，使用floor_height参数

- **`add_entity(self, idx, material, morph, surface)`**
  - 功能: 向求解器添加实体
  - 参数:
    - `idx`: 实体索引
    - `material`: 材料属性
    - `morph`: 形态信息
    - `surface`: 表面属性
  - 返回: 创建的ToolEntity对象

### 2. 状态管理模块 (State Management)
**功能**: 管理求解器和实体的状态获取、设置和检查点保存/加载

#### 涉及的函数:
- **`get_state(self, f)`**
  - 功能: 获取指定帧的状态
  - 参数: `f` - 帧索引
  - 返回: ToolSolverState对象或None
  - 操作: 如果求解器活跃，收集所有实体的状态
  
- **`set_state(self, f, state, envs_idx=None)`**
  - 功能: 设置指定帧的状态
  - 参数:
    - `f`: 帧索引
    - `state`: 状态对象
    - `envs_idx`: 环境索引（可选）
  - 操作: 为每个实体设置状态
  
- **`save_ckpt(self, ckpt_name)`**
  - 功能: 保存检查点
  - 参数: `ckpt_name` - 检查点名称
  - 操作: 为每个实体保存检查点
  
- **`load_ckpt(self, ckpt_name)`**
  - 功能: 加载检查点
  - 参数: `ckpt_name` - 检查点名称
  - 操作: 为每个实体加载检查点

- **`is_active(self)`**
  - 功能: 检查求解器是否活跃
  - 返回: 布尔值，根据实体数量判断

### 3. 梯度计算模块 (Gradient Computation)
**功能**: 处理反向传播和梯度计算

#### 涉及的函数:
- **`reset_grad(self)`**
  - 功能: 重置梯度
  - 操作: 为每个实体重置梯度
  
- **`add_grad_from_state(self, state)`**
  - 功能: 从状态添加梯度
  - 操作: 空实现，因为梯度已经在tool_entity中缓存
  
- **`collect_output_grads(self)`**
  - 功能: 收集输出梯度
  - 操作: 如果求解器活跃，为每个实体收集输出梯度

### 4. 时间步进模块 (Time Stepping)
**功能**: 处理模拟的时间步进，包括前向和反向传播

#### 涉及的函数:
- **`process_input(self, in_backward=False)`**
  - 功能: 处理输入
  - 参数: `in_backward` - 是否在反向传播中
  - 操作: 为每个实体处理输入
  
- **`process_input_grad(self)`**
  - 功能: 处理输入的梯度
  - 操作: 逆序为每个实体处理输入梯度
  
- **`substep_pre_coupling(self, f)`**
  - 功能: 耦合前的子步骤
  - 参数: `f` - 帧索引
  - 操作: 为每个实体执行耦合前子步骤
  
- **`substep_pre_coupling_grad(self, f)`**
  - 功能: 耦合前子步骤的梯度
  - 参数: `f` - 帧索引
  - 操作: 逆序为每个实体执行耦合前子步骤梯度
  
- **`substep_post_coupling(self, f)`**
  - 功能: 耦合后的子步骤
  - 参数: `f` - 帧索引
  - 操作: 为每个实体执行耦合后子步骤
  
- **`substep_post_coupling_grad(self, f)`**
  - 功能: 耦合后子步骤的梯度
  - 参数: `f` - 帧索引
  - 操作: 逆序为每个实体执行耦合后子步骤梯度

### 5. 碰撞处理模块 (Collision Handling)
**功能**: 处理与PBD系统的碰撞

#### 涉及的函数:
- **`pbd_collide(self, f, pos_world, thickness, dt)`** (Taichi函数)
  - 功能: 处理PBD碰撞
  - 参数:
    - `f`: 帧索引
    - `pos_world`: 世界坐标系中的位置
    - `thickness`: 厚度
    - `dt`: 时间步长
  - 返回: 更新后的世界坐标位置
  - 操作: 依次为每个实体处理碰撞

## 主要功能管线

### 时间步进管线 (Time Stepping Pipeline)
```
1. process_input() - 处理输入
   ↓
2. substep_pre_coupling() - 耦合前计算
   ↓
3. [外部耦合处理]
   ↓
4. substep_post_coupling() - 耦合后计算
   ↓
5. get_state() - 获取状态（如果需要）
```

### 反向传播管线 (Backward Propagation Pipeline)
```
1. add_grad_from_state() - 从查询的状态添加梯度
   ↓
2. collect_output_grads() - 收集输出梯度
   ↓
3. substep_post_coupling_grad() - 耦合后梯度（逆序）
   ↓
4. [外部耦合梯度处理]
   ↓
5. substep_pre_coupling_grad() - 耦合前梯度（逆序）
   ↓
6. process_input_grad() - 处理输入梯度（逆序）
```

### 检查点管理管线 (Checkpoint Management Pipeline)
```
保存: save_ckpt(name) → 为所有实体保存状态
加载: load_ckpt(name) → 为所有实体恢复状态
```

## 设计特点
1. **实体驱动**: 所有操作都委托给底层的ToolEntity对象
2. **梯度逆序**: 反向传播时按逆序处理实体，确保正确的梯度流
3. **临时性质**: 这是一个临时解决方案，将被完善的RigidSolver替代
4. **简单边界**: 仅支持地板边界条件
