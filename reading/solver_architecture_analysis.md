# Genesis 求解器架构深度分析

## 1. 概述

本文档深入分析 Genesis 物理引擎中 `genesis/engine/solver` 模块的架构设计、核心依赖、代码风格和设计模式，并详细说明其如何通过统一接口支持多种仿真场景（刚体、软体、流体、刚软耦合等）。

## 2. 核心依赖关系

### 2.1 技术栈依赖

```
核心依赖:
├── Taichi (gstaichi) - GPU/CPU 并行计算框架，所有求解器的计算引擎
├── NumPy - 数值计算和数组操作
├── PyTorch - 可微分计算和梯度传播
├── IGL (libigl) - 几何处理（网格、四面体等）
└── Genesis 内部模块
    ├── genesis.utils.geom - 几何工具（变换、SDF等）
    ├── genesis.utils.array_class - 数组类封装
    ├── genesis.utils.sdf_decomp - SDF分解计算
    └── genesis.engine.entities - 实体管理
```

### 2.2 模块间依赖图

```
Scene
  └── Simulator (协调器)
       ├── Solver 实例列表
       │    ├── ToolSolver
       │    ├── RigidSolver
       │    ├── AvatarSolver
       │    ├── MPMSolver
       │    ├── SPHSolver
       │    ├── PBDSolver
       │    ├── FEMSolver
       │    └── SFSolver
       └── Coupler (耦合器)
            ├── LegacyCoupler
            └── SAPCoupler
```

## 3. 架构设计

### 3.1 分层架构

Genesis 采用典型的**分层架构**设计：

```
┌─────────────────────────────────────────┐
│         Scene (场景层)                   │
│  - 实体管理                               │
│  - 可视化配置                             │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│      Simulator (仿真管理层)              │
│  - 时间步进                               │
│  - 求解器协调                             │
│  - 梯度管理                               │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│    Coupler (耦合层)                      │
│  - 跨求解器碰撞检测                       │
│  - 力交换                                 │
│  - 约束装配                               │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│    Solver (求解器层)                     │
│  - 物理计算                               │
│  - 状态更新                               │
│  - 实体管理                               │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│    Entity (实体层)                       │
│  - 几何数据                               │
│  - 材质参数                               │
│  - 状态数据                               │
└─────────────────────────────────────────┘
```

### 3.2 核心设计模式

#### 3.2.1 **模板方法模式 (Template Method Pattern)**

`Solver` 基类定义了求解器的标准流程，子类实现具体细节：

```python
# 基类定义接口
class Solver:
    def substep_pre_coupling(self, f):
        """子步进前处理 - 由子类实现"""
        pass
    
    def substep_post_coupling(self, f):
        """子步进后处理 - 由子类实现"""
        pass
    
    def process_input(self, in_backward=False):
        """输入处理 - 由子类实现"""
        pass

# 子类实现
class MPMSolver(Solver):
    def substep_pre_coupling(self, f):
        # P2G: 粒子到网格
        self.p2g(f)
    
    def substep_post_coupling(self, f):
        # G2P: 网格到粒子
        self.g2p(f)

class RigidSolver(Solver):
    def substep_pre_coupling(self, f):
        # 刚体动力学计算
        self.substep()
```

**优点：**
- 统一的求解器接口
- 灵活的算法实现
- 便于添加新求解器

#### 3.2.2 **策略模式 (Strategy Pattern)**

通过 Options 配置选择不同的求解策略：

```python
# 不同的耦合策略
if isinstance(coupler_options, SAPCouplerOptions):
    self._coupler = SAPCoupler(self, coupler_options)
elif isinstance(coupler_options, LegacyCouplerOptions):
    self._coupler = LegacyCoupler(self, coupler_options)
```

#### 3.2.3 **工厂模式 (Factory Pattern)**

`Simulator._add_entity` 根据材质类型创建对应实体：

```python
def _add_entity(self, morph, material, surface, visualize_contact=False):
    if isinstance(material, gs.materials.Tool):
        entity = self.tool_solver.add_entity(...)
    elif isinstance(material, gs.materials.Rigid):
        entity = self.rigid_solver.add_entity(...)
    elif isinstance(material, gs.materials.MPM.Base):
        entity = self.mpm_solver.add_entity(...)
    # ... 其他材质类型
```

**优点：**
- 自动路由到正确的求解器
- 用户无需关心内部实现
- 材质驱动的设计

#### 3.2.4 **观察者模式 (Observer Pattern)**

状态查询与梯度追踪：

```python
def get_state(self):
    state = SimState(...)
    # 存储所有查询的状态以追踪梯度流
    self._queried_states.append(state)
    return state
```

#### 3.2.5 **命令模式 (Command Pattern)**

输入处理和控制：

```python
def process_input(self, in_backward=False):
    """将外部命令转换为内部状态更新"""
    for solver in self._active_solvers:
        solver.process_input(in_backward=in_backward)
```

## 4. 统一接口机制

### 4.1 Solver 基类接口

所有求解器必须实现的核心接口：

```python
class Solver:
    # ===== 生命周期管理 =====
    def build(self):
        """构建求解器，初始化数据结构"""
    
    # ===== 时间步进接口 =====
    def substep_pre_coupling(self, f):
        """耦合前的子步计算"""
    
    def substep_post_coupling(self, f):
        """耦合后的子步计算"""
    
    def process_input(self, in_backward=False):
        """处理外部输入（控制、力等）"""
    
    # ===== 梯度接口（可微分求解器）=====
    def substep_pre_coupling_grad(self, f):
        """前向耦合的反向传播"""
    
    def substep_post_coupling_grad(self, f):
        """后向耦合的反向传播"""
    
    def process_input_grad(self):
        """输入的反向传播"""
    
    # ===== 状态管理接口 =====
    def get_state(self, f):
        """获取当前状态"""
    
    def set_state(self, f, state, envs_idx=None):
        """设置状态"""
    
    def reset_grad(self):
        """重置梯度"""
    
    # ===== 检查点接口 =====
    def save_ckpt(self, ckpt_name):
        """保存检查点"""
    
    def load_ckpt(self, ckpt_name):
        """加载检查点"""
    
    # ===== 工具接口 =====
    def is_active(self):
        """求解器是否激活"""
        return self.n_entities > 0
```

### 4.2 统一的时间步进流程

所有求解器遵循统一的时间步进协议：

```python
def step(self):
    """完整的仿真步"""
    self.process_input()  # 1. 处理输入
    
    for _ in range(self._substeps):  # 2. 子步循环
        self.substep(self.cur_substep_local)
    
    self._sensor_manager.step()  # 3. 传感器更新

def substep(self, f):
    """单个子步"""
    self._coupler.preprocess(f)         # 1. 耦合预处理
    self.substep_pre_coupling(f)        # 2. 各求解器独立计算
    self._coupler.couple(f)             # 3. 耦合力计算
    self.substep_post_coupling(f)       # 4. 耦合后更新
```

**关键设计点：**
1. **前后耦合分离**：求解器计算分为耦合前和耦合后两个阶段
2. **统一调度**：Simulator 统一调度所有求解器
3. **灵活耦合**：Coupler 负责跨求解器交互

## 5. 耦合机制详解

### 5.1 Coupler 的作用

Coupler 是实现多物理耦合的核心组件：

```
Solver A (刚体)        Coupler         Solver B (MPM软体)
    │                    │                    │
    ├─ substep_pre ────→ │                    │
    │                    │ ←─── substep_pre ─┤
    │                    │                    │
    │                    │ 碰撞检测            │
    │                    │ 力计算              │
    │                    │ 约束求解            │
    │                    │                    │
    │ ←──── couple ─────┤                    │
    │                    ├───── couple ──────→│
    │                    │                    │
    ├─ substep_post ───→ │                    │
    │                    │ ←─── substep_post ┤
```

### 5.2 LegacyCoupler 实现

**设计理念：**
- 基于 SDF（有向距离场）的碰撞检测
- 单向或双向力传递
- 支持刚体-MPM、刚体-SPH、刚体-PBD、刚体-FEM 等多种耦合

**核心方法：**

```python
class LegacyCoupler:
    def couple(self, f):
        if self._rigid_mpm:
            self._couple_rigid_mpm(f)
        if self._rigid_sph:
            self._couple_rigid_sph(f)
        if self._rigid_pbd:
            self._couple_rigid_pbd(f)
        if self._rigid_fem:
            self._couple_rigid_fem(f)
        # ... 其他耦合
```

**耦合原理（以刚体-MPM为例）：**

```python
@ti.func
def _func_collide_with_rigid_geom(self, pos_world, vel, mass, geom_idx, batch_idx):
    # 1. 计算粒子到刚体的有向距离
    signed_dist = sdf_func_world(pos_world, geom_idx, batch_idx)
    
    # 2. 计算影响因子（指数衰减）
    influence = ti.min(ti.exp(-signed_dist / coup_softness), 1)
    
    if influence > 0.1:
        # 3. 获取刚体表面法向量
        normal_rigid = sdf_func_normal_world(...)
        
        # 4. 计算刚体在碰撞点的速度
        vel_rigid = self.rigid_solver._func_vel_at_point(pos_world, link_idx, i_b)
        
        # 5. 相对速度
        rvel = vel - vel_rigid
        rvel_normal = rvel.dot(normal_rigid)
        
        if rvel_normal < 0:  # 碰撞
            # 6. 施加摩擦和恢复系数
            rvel_tan = rvel - rvel_normal * normal_rigid
            rvel_new = apply_friction_and_restitution(rvel_tan, rvel_normal)
            
            # 7. 更新粒子速度
            vel = vel_rigid + rvel_new * influence + rvel * (1 - influence)
            
            # 8. 反向施加力到刚体（牛顿第三定律）
            force = -mass * (vel - vel_old) / dt
            self.rigid_solver._func_apply_external_force(pos_world, force, ...)
    
    return vel
```

### 5.3 SAPCoupler 实现

**设计理念：**
- 基于 Drake 的 Semi-Analytic Primal (SAP) 接触求解器
- 使用四面体网格进行精确接触检测
- 支持流体弹性接触（hydroelastic contact）
- 高精度、隐式求解

**关键特性：**
1. **Hydroelastic Contact**：基于压力场的接触模型
2. **BVH 加速**：使用 LBVH（线性 BVH）加速碰撞检测
3. **隐式求解**：通过迭代求解约束优化问题
4. **高精度**：要求 64 位精度

**核心流程：**

```python
class SAPCoupler:
    def couple(self, f):
        # 1. BVH 碰撞检测
        self._detect_contacts_bvh()
        
        # 2. 装配约束矩阵
        self._assemble_constraints()
        
        # 3. SAP 迭代求解
        for iter in range(self._n_sap_iterations):
            # 3a. PCG 求解线性系统
            self._pcg_solve()
            
            # 3b. Line search 优化
            self._line_search()
            
            # 3c. 检查收敛
            if converged:
                break
        
        # 4. 应用约束力到各求解器
        self._apply_constraint_forces()
```

## 6. 不同仿真场景的统一接口应用

### 6.1 纯刚体仿真

```python
scene = gs.Scene(
    sim_options=gs.options.SimOptions(dt=0.01, substeps=10),
    rigid_options=gs.options.RigidOptions(
        enable_collision=True,
        enable_joint_limit=True,
    ),
)

# 添加刚体实体
robot = scene.add_entity(
    material=gs.materials.Rigid(),
    morph=gs.morphs.URDF(file="robot.urdf"),
)

scene.build()
```

**调用链：**
```
Scene.add_entity()
  → Simulator._add_entity()
    → RigidSolver.add_entity()
      → 创建 RigidEntity

Simulator.step()
  → RigidSolver.substep_pre_coupling()
    → kernel_step_1(): 动力学计算
    → constraint_solver.resolve(): 约束求解
    → kernel_step_2(): 积分更新
```

### 6.2 刚软耦合（Rigid-MPM）

```python
scene = gs.Scene(
    sim_options=gs.options.SimOptions(dt=0.01),
    coupler_options=gs.options.LegacyCouplerOptions(
        rigid_mpm=True,  # 启用刚体-MPM耦合
    ),
    rigid_options=gs.options.RigidOptions(...),
    mpm_options=gs.options.MPMOptions(...),
)

# 添加刚体
rigid_obj = scene.add_entity(
    material=gs.materials.Rigid(),
    morph=gs.morphs.Mesh(file="tool.obj"),
)

# 添加软体（MPM）
soft_obj = scene.add_entity(
    material=gs.materials.MPM.Elastic(),
    morph=gs.morphs.Mesh(file="bunny.obj"),
)

scene.build()
```

**调用链：**
```
Simulator.step()
  → Simulator.substep(f)
    → Coupler.preprocess(f)
    
    → RigidSolver.substep_pre_coupling(f)
      → kernel_step_1(): 刚体动力学
    
    → MPMSolver.substep_pre_coupling(f)
      → p2g(f): 粒子到网格
      → grid 更新
    
    → Coupler.couple(f)
      → _couple_rigid_mpm(f)
        → 对每个 MPM 粒子:
          ├─ 计算与刚体的距离和法向量
          ├─ 更新粒子速度（考虑碰撞和摩擦）
          └─ 施加反作用力到刚体
    
    → MPMSolver.substep_post_coupling(f)
      → g2p(f): 网格到粒子
      → 更新 F, Jp 等
    
    → RigidSolver.substep_post_coupling(f)
      → (无操作或后处理)
```

### 6.3 刚体-FEM 耦合（高精度）

```python
scene = gs.Scene(
    sim_options=gs.options.SimOptions(dt=0.01),
    coupler_options=gs.options.SAPCouplerOptions(
        enable_rigid_fem_contact=True,
        rigid_floor_contact_type="tet",
    ),
    rigid_options=gs.options.RigidOptions(...),
    fem_options=gs.options.FEMOptions(
        use_implicit_solver=True,  # SAP 要求隐式求解器
    ),
)

# 添加刚体和 FEM 实体
# ...

scene.build()
```

**调用链：**
```
Simulator.substep(f)
  → SAPCoupler.preprocess(f)
    → BVH 构建
  
  → RigidSolver.substep_pre_coupling(f)
  → FEMSolver.substep_pre_coupling(f)
    → 牛顿迭代 + PCG 求解
  
  → SAPCoupler.couple(f)
    → detect_contacts_bvh()
      ├─ Rigid-FEM 接触检测
      ├─ FEM-Floor 接触检测
      └─ Rigid-Floor 接触检测
    
    → assemble_constraints()
      └─ 构建约束矩阵
    
    → sap_iterate()
      └─ 迭代求解约束优化问题
    
    → apply_constraint_forces()
      ├─ 更新刚体速度
      └─ 更新 FEM 顶点速度
  
  → FEMSolver.substep_post_coupling(f)
  → RigidSolver.substep_post_coupling(f)
```

### 6.4 多物理耦合（Hybrid）

```python
# 混合刚体+软体实体
hybrid_entity = scene.add_entity(
    material=gs.materials.Hybrid(),
    morph=gs.morphs.Mesh(file="complex_object.obj"),
)

# 系统自动处理:
# 1. 刚体部分 → RigidSolver
# 2. 软体部分 → MPMSolver/FEMSolver
# 3. Coupler 处理内部耦合
```

## 7. 代码风格与规范

### 7.1 命名约定

```python
# 数据结构
self._entities       # 实体列表
self._options        # 配置选项
self._B              # batch 大小

# 状态与信息
links_state         # 动态状态
links_info          # 静态信息
*_global_info       # 全局共享数据
*_cache_key         # 编译缓存键

# Taichi 函数
@ti.kernel
def _kernel_xxx():   # GPU/CPU 并行核函数
    pass

@ti.func
def _func_xxx():     # 内联函数（可在 kernel 中调用）
    pass

# 私有方法
def _private_method():  # 下划线前缀表示私有
    pass
```

### 7.2 数据布局

采用 **SoA (Structure of Arrays)** 布局以优化 GPU 性能：

```python
# ✓ 好的设计（SoA）
struct_particle_state = ti.types.struct(
    pos=gs.ti_vec3,
    vel=gs.ti_vec3,
    mass=gs.ti_float,
)
particles = struct_particle_state.field(
    shape=(n_particles,),
    layout=ti.Layout.SOA  # 每个字段独立存储
)

# ✗ 避免（AoS）
# 每个粒子的数据连续存储（GPU不友好）
```

### 7.3 批处理设计

支持多环境并行仿真：

```python
# 数据形状: (batch_size, n_entities, ...)
def _batch_shape(self, shape=None, first_dim=False):
    if first_dim:
        return (B,) + shape
    else:
        return shape + (B,)

# 批处理循环
@ti.kernel
def compute():
    for i_b in range(self._B):  # 批次循环
        for i_e in range(n_entities):  # 实体循环
            # 计算
```

### 7.4 可微分设计

```python
# 1. 启用梯度
particles = struct_particle_state.field(
    shape=...,
    needs_grad=True  # 支持自动微分
)

# 2. 前向和反向函数成对出现
def substep_pre_coupling(self, f):
    self.p2g(f)

def substep_pre_coupling_grad(self, f):
    self.p2g.grad(f)  # 反向传播

# 3. 梯度累积
def add_grad_from_state(self, state):
    """从输出状态累积梯度"""
    self.particles.grad[f] += state.grad
```

## 8. 设计亮点

### 8.1 分离式求解器（Decomposed Solver）

以 `RigidSolver` 为例，将单体求解器拆分为多个小的 kernel：

```python
# 传统设计：一个大 kernel
@ti.kernel
def substep():
    # 1000+ 行代码
    # 运动学 + 动力学 + 约束 + 积分 ...
    pass

# Genesis 设计：多个小 kernel
@ti.kernel
def kernel_step_1():
    func_forward_kinematics()
    func_compute_mass_matrix()
    func_forward_dynamics()

@ti.kernel  
def kernel_step_2():
    func_integrate()
    func_update_geoms()
```

**优点：**
1. **编译缓存优化**：小 kernel 更容易复用编译结果
2. **灵活组合**：可以根据需求选择性执行
3. **调试友好**：问题定位更精确
4. **性能分析**：可以单独测量每个阶段

### 8.2 接触岛与休眠机制

`RigidSolver` 实现了高效的接触岛（Contact Island）和休眠（Hibernation）机制：

```python
class RigidSolver:
    def __init__(...):
        self._use_contact_island = True
        self._use_hibernation = True
    
    def substep(self):
        # 1. 构建接触岛
        self.constraint_solver.contact_island.build_islands()
        
        # 2. 只对活跃岛求解
        for island in active_islands:
            self.solve_island(island)
        
        # 3. 检测可休眠的岛
        for island in islands:
            if is_sleeping(island):
                hibernate_island(island)
```

**优点：**
- 大规模场景性能提升 10-100 倍
- 静止物体不浪费计算资源
- 自动唤醒机制

### 8.3 统一的状态管理

所有求解器遵循统一的状态管理接口：

```python
# 1. 获取状态
state = scene.get_state()
# 返回 SimState，包含所有求解器的状态

# 2. 重置到某状态
scene.reset(state)

# 3. 检查点机制
solver.save_ckpt("ckpt_0")
solver.load_ckpt("ckpt_0")

# 4. 梯度追踪
queried_states = [state1, state2, ...]
# 反向传播时自动追踪
```

### 8.4 零拷贝数据交换

使用 Taichi ndarray 实现零拷贝数据交换：

```python
@ti.kernel
def _kernel_get_state(
    self,
    f: ti.i32,
    pos: ti.types.ndarray(),  # 外部分配的 numpy/torch 数组
):
    for i_p, i_b in ti.ndrange(n_particles, self._B):
        for i in ti.static(range(3)):
            pos[i_b, i_p, i] = self.particles[f, i_p, i_b].pos[i]
```

**优点：**
- GPU 到 CPU 无拷贝
- 支持 PyTorch tensor 直接传递
- 可微分梯度回传

### 8.5 灵活的材质系统

材质驱动的实体分配：

```python
# 用户层：只关心材质
scene.add_entity(
    material=gs.materials.MPM.Elastic(E=1e5, nu=0.3),
    morph=gs.morphs.Mesh(file="bunny.obj"),
)

# 系统层：自动路由
Simulator._add_entity()
  → 检测 material 类型
  → 调用对应 solver.add_entity()
  → 创建对应 Entity
```

**支持的材质类型：**
- `Rigid`: 刚体
- `Avatar`: 角色动画
- `MPM.Elastic/Liquid/Snow/Sand`: MPM 材质
- `SPH.Liquid`: SPH 液体
- `PBD.Cloth/Elastic/Liquid`: PBD 材质
- `FEM.Elastic/NeoHookean`: FEM 材质
- `Hybrid`: 混合刚体+软体

## 9. 代码库中的调用示例

### 9.1 从 Scene 到 Solver

```python
# 文件: genesis/engine/scene.py
class Scene:
    def add_entity(self, material, morph, surface):
        entity = self.sim._add_entity(material, morph, surface)
        return entity
    
    def build(self):
        self.sim.build()
        # → 调用所有 solver.build()
```

### 9.2 Simulator 协调

```python
# 文件: genesis/engine/simulator.py
class Simulator:
    def __init__(self, ...):
        # 创建所有求解器
        self.rigid_solver = RigidSolver(...)
        self.mpm_solver = MPMSolver(...)
        # ...
        
        self._solvers = [
            self.rigid_solver,
            self.mpm_solver,
            # ...
        ]
    
    def step(self):
        if self._rigid_only:
            # 优化路径：仅刚体
            for _ in range(self._substeps):
                self.rigid_solver.substep()
        else:
            # 通用路径：多求解器 + 耦合
            self.process_input()
            for _ in range(self._substeps):
                self.substep(self.cur_substep_local)
```

### 9.3 典型使用场景

#### 场景 1：机器人仿真（纯刚体）

```python
# examples/robot_control.py
import genesis as gs

gs.init()
scene = gs.Scene(
    rigid_options=gs.options.RigidOptions(
        enable_collision=True,
        use_contact_island=True,
        use_hibernation=True,
    ),
)

robot = scene.add_entity(
    material=gs.materials.Rigid(),
    morph=gs.morphs.URDF(file="franka.urdf"),
)

scene.build()

for _ in range(1000):
    robot.set_dofs_kp([100]*7)
    robot.set_dofs_kd([10]*7)
    robot.set_dofs_position(target_pos)
    scene.step()
```

**求解器链路：**
```
Scene.step()
  → Simulator.step()
    → RigidSolver.process_input()
      ← 设置关节目标
    → RigidSolver.substep()
      → kernel_step_1()
        → func_forward_kinematics()
        → func_compute_mass_matrix()
        → func_forward_dynamics()
      → constraint_solver.resolve()
        → 关节驱动力
        → 接触力
        → 关节限位
      → kernel_step_2()
        → func_integrate()
        → func_update_geoms()
```

#### 场景 2：软体机器人（刚软耦合）

```python
# examples/soft_robot.py
scene = gs.Scene(
    coupler_options=gs.options.LegacyCouplerOptions(
        rigid_mpm=True,
    ),
    rigid_options=gs.options.RigidOptions(...),
    mpm_options=gs.options.MPMOptions(...),
)

# 刚性支撑
base = scene.add_entity(
    material=gs.materials.Rigid(),
    morph=gs.morphs.Box(size=(1, 1, 0.1)),
)

# 软体触手
tentacle = scene.add_entity(
    material=gs.materials.MPM.Elastic(E=5e4, nu=0.3),
    morph=gs.morphs.Mesh(file="tentacle.obj"),
)

scene.build()

for _ in range(1000):
    # 刚体基座移动
    base.set_velocity([0.1, 0, 0])
    scene.step()
```

**求解器链路：**
```
Simulator.substep(f)
  → RigidSolver.substep_pre_coupling(f)
    → 计算刚体动力学
    → 更新刚体速度和位置
  
  → MPMSolver.substep_pre_coupling(f)
    → p2g(f): 粒子到网格
    → 网格上计算压力、粘性等
  
  → LegacyCoupler.couple(f)
    → _couple_rigid_mpm(f)
      → 对每个 MPM 粒子:
        ├─ 计算到刚体的 SDF 距离
        ├─ 如果距离 < 阈值:
        │   ├─ 获取刚体表面法向量
        │   ├─ 获取刚体速度
        │   ├─ 计算碰撞响应
        │   ├─ 更新粒子速度
        │   └─ 施加反作用力到刚体
        └─ 返回新速度
  
  → MPMSolver.substep_post_coupling(f)
    → g2p(f): 网格到粒子
    → 更新 F, Jp
    → SVD 分解
    → 更新应力
```

#### 场景 3：精确接触（SAP 耦合）

```python
# examples/precision_contact.py
scene = gs.Scene(
    coupler_options=gs.options.SAPCouplerOptions(
        n_sap_iterations=10,
        enable_rigid_fem_contact=True,
        rigid_floor_contact_type="tet",
    ),
    rigid_options=gs.options.RigidOptions(...),
    fem_options=gs.options.FEMOptions(
        use_implicit_solver=True,
    ),
)

# 刚性抓手
gripper = scene.add_entity(
    material=gs.materials.Rigid(),
    morph=gs.morphs.URDF(file="gripper.urdf"),
)

# FEM 软物体
soft_obj = scene.add_entity(
    material=gs.materials.FEM.NeoHookean(E=1e6, nu=0.45),
    morph=gs.morphs.Mesh(file="sponge.obj"),
)

scene.build()
```

**求解器链路（SAP）：**
```
Simulator.substep(f)
  → SAPCoupler.preprocess(f)
    → 构建/更新 BVH
  
  → RigidSolver.substep_pre_coupling(f)
  → FEMSolver.substep_pre_coupling(f)
    → 隐式时间积分（牛顿法+PCG）
  
  → SAPCoupler.couple(f)
    → detect_contacts_bvh()
      ├─ AABB 相交测试
      ├─ 四面体相交测试
      └─ 生成接触约束列表
    
    → assemble_constraints()
      ├─ 接触约束
      ├─ 摩擦锥约束
      └─ 构建雅可比矩阵
    
    → sap_iterate()
      └─ 迭代优化:
          ├─ solve_pcg(): 求解线性系统
          ├─ line_search(): 步长搜索
          ├─ project_to_cone(): 投影到摩擦锥
          └─ 检查收敛
    
    → apply_constraint_forces()
      ├─ 将约束力转换为广义力
      ├─ 更新刚体速度
      └─ 更新 FEM 顶点速度
  
  → FEMSolver.substep_post_coupling(f)
  → RigidSolver.substep_post_coupling(f)
```

## 10. 总结

### 10.1 架构优势

1. **模块化**：清晰的分层和职责分离
2. **可扩展**：易于添加新求解器和耦合方式
3. **高性能**：GPU 加速 + 接触岛 + 休眠机制
4. **可微分**：支持梯度反向传播
5. **灵活**：统一接口支持多种仿真场景
6. **用户友好**：材质驱动的简洁 API

### 10.2 技术特色

1. **Taichi 深度集成**：充分利用 GPU 并行和自动微分
2. **SoA 数据布局**：优化 GPU 内存访问
3. **分解式设计**：小 kernel 提升编译缓存效率
4. **统一接口**：所有求解器遵循相同的协议
5. **双层耦合器**：Legacy (快速) 和 SAP (精确) 两种选择
6. **零拷贝交换**：高效的 CPU-GPU 数据传输

### 10.3 设计模式总结

- **模板方法模式**：Solver 基类定义标准流程
- **策略模式**：Options 配置选择算法
- **工厂模式**：材质类型自动路由
- **观察者模式**：状态查询和梯度追踪
- **命令模式**：输入处理和控制

### 10.4 代码风格特点

- **命名清晰**：`_state` vs `_info`, `_kernel_` vs `_func_`
- **注释丰富**：中英文混合，关键算法有详细说明
- **类型标注**：使用 Python type hints 和 Taichi types
- **数据驱动**：SoA 布局，批处理优先

### 10.5 适用场景

| 场景 | 推荐求解器 | 耦合器 | 特点 |
|------|-----------|--------|------|
| 机器人控制 | RigidSolver | - | 高效、稳定 |
| 软体机器人 | Rigid + MPM | Legacy | 快速、近似 |
| 精密操作 | Rigid + FEM | SAP | 精确、隐式 |
| 流体交互 | Rigid + SPH/MPM | Legacy | 实时、视觉效果好 |
| 布料模拟 | PBD | - | 无条件稳定 |
| 烟雾效果 | SFSolver | - | 艺术效果 |

---

**文档版本**: v1.0  
**创建日期**: 2024年  
**基于代码**: Genesis 引擎 `genesis/engine/solvers/`  
**作者**: Genesis 架构分析
