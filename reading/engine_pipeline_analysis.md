# Genesis Engine 工作管线分析

## 1. 概述

本文档深入分析 Genesis 物理引擎 `genesis/engine` 模块各组件之间的关系，详细说明整个引擎的工作管线，从用户接口到底层物理计算的完整流程。

**分析范围**：
- `scene.py` - 场景管理（用户接口层）
- `simulator.py` - 仿真协调器（调度层）
- `solvers/` - 求解器模块（计算层）
- `entities/` - 实体模块（数据层）
- `couplers/` - 耦合器（交互层）
- `materials/` - 材料模块（物理模型）
- `boundaries/` - 边界条件
- `states/` - 状态管理
- `mesh.py` - 网格处理
- `bvh.py` - 碰撞加速结构
- `force_fields.py` - 力场系统

---

## 2. 模块层次结构

### 2.1 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                        用户层                                │
│                      Scene (场景)                            │
│  • add_entity()     • add_camera()    • add_force_field()   │
│  • build()          • step()          • reset()             │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                      调度协调层                              │
│                   Simulator (仿真器)                         │
│  • 时间步进管理      • 求解器协调      • 梯度管理           │
│  • 批处理调度        • 状态缓存        • 检查点管理         │
└─────────────────────────────────────────────────────────────┘
                              ↓
        ┌─────────────────────┴─────────────────────┐
        ↓                                            ↓
┌──────────────────┐                      ┌──────────────────┐
│   计算层 (左)     │                      │   交互层 (右)     │
│   8个 Solvers     │ ←───────────────→   │    Coupler       │
│  • ToolSolver    │                      │  • LegacyCoupler │
│  • RigidSolver   │                      │  • SAPCoupler    │
│  • AvatarSolver  │                      └──────────────────┘
│  • MPMSolver     │
│  • SPHSolver     │
│  • PBDSolver     │
│  • FEMSolver     │
│  • SFSolver      │
└──────────────────┘
        ↓
┌─────────────────────────────────────────────────────────────┐
│                        数据层                                │
│              Entities (实体) + Materials (材料)              │
│  • 几何数据         • 物理状态         • 材料参数           │
└─────────────────────────────────────────────────────────────┘
        ↓
┌─────────────────────────────────────────────────────────────┐
│                      工具支持层                              │
│  Mesh (网格)  │  BVH (加速)  │  Boundaries (边界)           │
│  States (状态) │  ForceFields (力场)                        │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 模块依赖关系

```
Scene
  ├── Simulator [1]
  │   ├── Solvers [8] (列表)
  │   │   ├── ToolSolver
  │   │   │   └── ToolEntity [n]
  │   │   ├── RigidSolver
  │   │   │   └── RigidEntity / AvatarEntity / DroneEntity [n]
  │   │   │       ├── RigidLink [m]
  │   │   │       ├── RigidJoint [m]
  │   │   │       └── RigidGeom [m]
  │   │   ├── AvatarSolver
  │   │   │   └── AvatarEntity [n]
  │   │   ├── MPMSolver
  │   │   │   └── MPMEntity [n]
  │   │   ├── SPHSolver
  │   │   │   └── SPHEntity [n]
  │   │   ├── PBDSolver
  │   │   │   └── PBD2D/3D/Particle/FreeParticleEntity [n]
  │   │   ├── FEMSolver
  │   │   │   └── FEMEntity [n]
  │   │   └── SFSolver
  │   │       └── SFParticleEntity [n]
  │   ├── Coupler [1]
  │   │   ├── LegacyCoupler (或)
  │   │   └── SAPCoupler
  │   └── SensorManager [1]
  │       └── Sensors [n]
  ├── Visualizer [1]
  │   ├── Viewer [1]
  │   └── Cameras [n]
  ├── RecorderManager [1]
  │   └── Recorders [n]
  └── ForceFields [n]
```

---

## 3. 核心模块详解

### 3.1 Scene（场景管理器）

**职责**：
- 用户接口的统一入口
- 场景构建和生命周期管理
- 可视化和传感器管理
- 录制和回放

**核心方法**：

```python
class Scene:
    def __init__(self, 
                 sim_options,        # 仿真选项
                 rigid_options,      # 刚体求解器选项
                 mpm_options, ...):  # 其他求解器选项
        # 创建 Simulator
        self._sim = Simulator(self, sim_options, ...)
        # 创建 Visualizer
        self._visualizer = Visualizer(self, ...)
    
    def add_entity(self, morph, material, surface):
        """添加实体（路由到对应求解器）"""
        entity = self._sim._add_entity(morph, material, surface)
        return entity
    
    def build(self):
        """构建场景（两阶段初始化）"""
        self._sim.build()
        self._visualizer.build()
        self._is_built = True
    
    def step(self):
        """时间步进"""
        self._sim.step()
    
    def reset(self, state):
        """重置到指定状态"""
        self._sim.reset(state)
```

**关键特性**：
1. **延迟构建**：`__init__` 阶段只配置，`build()` 阶段才分配内存
2. **材料路由**：根据 `material` 类型自动选择合适的求解器
3. **统一接口**：用户无需直接操作 Simulator 或 Solver

### 3.2 Simulator（仿真协调器）

**职责**：
- 管理所有求解器的生命周期
- 协调时间步进和梯度传播
- 管理实体注册和状态缓存
- 批处理和并行化调度

**核心数据结构**：

```python
class Simulator:
    def __init__(self, scene, options, ...):
        # 时间参数
        self._dt = options.dt                    # 时间步长
        self._substeps = options.substeps        # 子步数
        self._substep_dt = dt / substeps         # 子步时间步长
        
        # 创建所有求解器（无论是否使用）
        self.tool_solver = ToolSolver(...)
        self.rigid_solver = RigidSolver(...)
        self.avatar_solver = AvatarSolver(...)
        self.mpm_solver = MPMSolver(...)
        self.sph_solver = SPHSolver(...)
        self.pbd_solver = PBDSolver(...)
        self.fem_solver = FEMSolver(...)
        self.sf_solver = SFSolver(...)
        
        self._solvers = [tool_solver, rigid_solver, ...]
        self._active_solvers = []  # 运行时填充
        
        # 创建耦合器
        if isinstance(coupler_options, SAPCouplerOptions):
            self._coupler = SAPCoupler(self, coupler_options)
        else:
            self._coupler = LegacyCoupler(self, coupler_options)
    
    def build(self):
        """构建所有活跃的求解器"""
        for solver in self._solvers:
            solver.build()
            if solver.is_active():
                self._active_solvers.append(solver)
        self._coupler.build()
```

**关键方法**：

```python
def step(self, in_backward=False):
    """一个完整的时间步"""
    if self._rigid_only:  # 仅刚体优化路径
        for _ in range(self._substeps):
            self.rigid_solver.substep()
    else:
        self.process_input()  # 处理外部输入
        for _ in range(self._substeps):
            self.substep(self.cur_substep_local)

def substep(self, f):
    """单个子步（核心计算流程）"""
    self._coupler.preprocess(f)          # 耦合预处理
    self.substep_pre_coupling(f)         # 各求解器前向计算
    self._coupler.couple(f)              # 跨求解器耦合
    self.substep_post_coupling(f)        # 各求解器后处理

def substep_pre_coupling(self, f):
    """耦合前计算（并行执行）"""
    for solver in self._active_solvers:
        solver.substep_pre_coupling(f)

def substep_post_coupling(self, f):
    """耦合后计算（并行执行）"""
    for solver in self._active_solvers:
        solver.substep_post_coupling(f)
```

### 3.3 Solver（求解器基类）

**职责**：
- 定义统一的求解器接口
- 管理实体列表
- 提供重力、状态管理等通用功能

**基类接口**：

```python
class Solver(RBC):
    def __init__(self, scene, sim, options):
        self._uid = gs.UID()
        self._scene = scene
        self._sim = sim
        self._dt = options.dt
        self._substep_dt = options.dt / sim.substeps
        self._entities = gs.List()
        self._gravity = None
    
    def build(self):
        """构建求解器（初始化数据结构）"""
        self._B = self._sim._B  # 批次大小
        if self._init_gravity is not None:
            # 初始化重力场
            ...
    
    # ===== 必须实现的接口 =====
    def substep_pre_coupling(self, f):
        """耦合前计算 - 由子类实现"""
        raise NotImplementedError
    
    def substep_post_coupling(self, f):
        """耦合后计算 - 由子类实现"""
        raise NotImplementedError
    
    def process_input(self, in_backward=False):
        """处理外部输入 - 由子类实现"""
        raise NotImplementedError
    
    def get_state(self, f):
        """获取当前状态 - 由子类实现"""
        raise NotImplementedError
    
    def set_state(self, f, state, envs_idx=None):
        """设置状态 - 由子类实现"""
        raise NotImplementedError
    
    # ===== 可选的梯度接口 =====
    def substep_pre_coupling_grad(self, f):
        pass
    
    def substep_post_coupling_grad(self, f):
        pass
    
    def process_input_grad(self):
        pass
    
    # ===== 通用属性 =====
    @property
    def entities(self) -> list[Entity]:
        return self._entities
    
    def is_active(self):
        """是否有实体"""
        return len(self._entities) > 0
```

### 3.4 Coupler（耦合器）

**职责**：
- 处理不同求解器之间的物理交互
- 跨求解器碰撞检测
- 力和约束的传递

**两种实现**：

#### 3.4.1 LegacyCoupler（传统耦合器）

```python
class LegacyCoupler:
    def __init__(self, simulator, options):
        # 引用所有求解器
        self.rigid_solver = simulator.rigid_solver
        self.mpm_solver = simulator.mpm_solver
        self.sph_solver = simulator.sph_solver
        self.pbd_solver = simulator.pbd_solver
        self.fem_solver = simulator.fem_solver
    
    def build(self):
        """根据选项决定启用哪些耦合"""
        self._rigid_mpm = (self.rigid_solver.is_active() and 
                           self.mpm_solver.is_active() and 
                           self.options.rigid_mpm)
        self._rigid_sph = ...
        self._rigid_pbd = ...
        # ... 其他耦合对
    
    def couple(self, f):
        """执行所有启用的耦合"""
        # MPM <-> Rigid
        if self._rigid_mpm and self.mpm_solver.is_active():
            self.mpm_grid_op(f, self.sim.cur_t)
        
        # SPH <-> Rigid
        if self._rigid_sph and self.rigid_solver.is_active():
            self.sph_rigid(f)
        
        # PBD <-> Rigid
        if self._rigid_pbd and self.rigid_solver.is_active():
            self.kernel_pbd_rigid_collide()
        
        # FEM <-> Rigid
        if self.fem_solver.is_active():
            self.fem_surface_force(f)
            self.fem_rigid_link_constraints()
```

**耦合类型**：
1. **MPM ↔ Rigid**：网格操作，处理 MPM 粒子与刚体的碰撞
2. **SPH ↔ Rigid**：粒子-几何碰撞，计算边界力
3. **PBD ↔ Rigid**：位置约束，刚体链接约束
4. **FEM ↔ Rigid**：表面力传递，链接约束
5. **MPM ↔ SPH**：粒子-网格交互
6. **MPM ↔ PBD**：粒子-网格交互

#### 3.4.2 SAPCoupler（扫掠与剪枝耦合器）

```python
class SAPCoupler:
    """基于 BVH 的高效碰撞检测"""
    def __init__(self, simulator, options):
        self.bvh = BVH(...)  # 层次包围盒
    
    def couple(self, f):
        """基于空间数据结构的高效耦合"""
        self.bvh.update()
        self.broad_phase()   # 粗检测
        self.narrow_phase()  # 精检测
        self.resolve()       # 响应
```

### 3.5 Entity（实体）

**职责**：
- 存储几何、材料、状态数据
- 提供实体级别的计算接口
- 管理可视化数据

**基类结构**：

```python
class Entity(RBC):
    def __init__(self, idx, scene, morph, solver, material, surface):
        self._uid = gs.UID()
        self._idx = idx              # 全局实体索引
        self._scene = scene
        self._solver = solver        # 所属求解器
        self._material = material    # 材料属性
        self._morph = morph          # 几何形状
        self._surface = surface      # 表面属性
    
    @property
    def solver(self):
        return self._solver
    
    @property
    def material(self):
        return self._material
```

**具体实现示例（RigidEntity）**：

```python
class RigidEntity(Entity):
    def __init__(self, ...):
        super().__init__(...)
        self._links = []   # 链接列表
        self._joints = []  # 关节列表
        self._geoms = []   # 几何体列表
    
    def _build(self):
        """构建刚体结构"""
        self._build_links()
        self._build_joints()
        self._build_geoms()
```

### 3.6 Material（材料）

**职责**：
- 定义物理属性（密度、弹性模量等）
- 提供本构模型（应力-应变关系）
- 决定实体路由到哪个求解器

**材料层次**：

```
Material (基类)
├── Rigid                    → RigidSolver
├── Avatar                   → AvatarSolver
├── Tool                     → ToolSolver
├── MPM.Base                 → MPMSolver
│   ├── MPM.Elastic
│   ├── MPM.Liquid
│   ├── MPM.Sand
│   ├── MPM.Snow
│   └── MPM.Muscle
├── SPH.Base                 → SPHSolver
│   ├── SPH.Liquid
│   └── SPH.Visco
├── PBD.Base                 → PBDSolver
│   ├── PBD.Cloth
│   ├── PBD.Elastic
│   └── PBD.Liquid
├── FEM.Base                 → FEMSolver
│   ├── FEM.Elastic
│   ├── FEM.NeoHookean
│   └── FEM.Muscle
└── Hybrid                   → Multiple Solvers
```

**材料接口示例（MPM）**：

```python
class MPM_Elastic(MPM.Base):
    def __init__(self, E=1e5, nu=0.2, rho=1000.0):
        self.E = E          # 杨氏模量
        self.nu = nu        # 泊松比
        self.rho = rho      # 密度
        
        # 拉梅参数
        self.mu = E / (2 * (1 + nu))
        self.lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    
    @ti.func
    def update_stress(self, U, S, V, F, Jp):
        """计算 Cauchy 应力"""
        J = gu.ti_determinant(F)
        stress = 2 * self.mu * (F - U @ V.transpose()) @ F.transpose()
        stress += self.lam * ti.log(J) * ti.Matrix.identity(gs.ti_float, 3)
        return stress / J
```

---

## 4. 主要工作管线

### 4.1 初始化管线

```
┌──────────────────────────────────────────────────────────┐
│ 1. 用户创建 Scene                                         │
│    scene = gs.Scene(sim_options, rigid_options, ...)     │
└────────────────────┬─────────────────────────────────────┘
                     ↓
┌──────────────────────────────────────────────────────────┐
│ 2. Scene 创建 Simulator 和 Visualizer                    │
│    self._sim = Simulator(...)                            │
│    self._visualizer = Visualizer(...)                    │
└────────────────────┬─────────────────────────────────────┘
                     ↓
┌──────────────────────────────────────────────────────────┐
│ 3. Simulator 创建所有 Solver 和 Coupler                  │
│    self.rigid_solver = RigidSolver(...)                  │
│    self.mpm_solver = MPMSolver(...)                      │
│    ...                                                    │
│    self._coupler = LegacyCoupler(...)                    │
└────────────────────┬─────────────────────────────────────┘
                     ↓
┌──────────────────────────────────────────────────────────┐
│ 4. 用户添加实体                                           │
│    entity = scene.add_entity(morph, material, surface)   │
└────────────────────┬─────────────────────────────────────┘
                     ↓
┌──────────────────────────────────────────────────────────┐
│ 5. Scene 路由到 Simulator                                │
│    entity = self._sim._add_entity(morph, material, ...)  │
└────────────────────┬─────────────────────────────────────┘
                     ↓
┌──────────────────────────────────────────────────────────┐
│ 6. Simulator 根据 material 类型路由到对应 Solver         │
│    if isinstance(material, gs.materials.Rigid):          │
│        entity = self.rigid_solver.add_entity(...)        │
│    elif isinstance(material, gs.materials.MPM.Base):     │
│        entity = self.mpm_solver.add_entity(...)          │
└────────────────────┬─────────────────────────────────────┘
                     ↓
┌──────────────────────────────────────────────────────────┐
│ 7. Solver 创建 Entity 并添加到列表                       │
│    entity = MPMEntity(...)                               │
│    self._entities.append(entity)                         │
└────────────────────┬─────────────────────────────────────┘
                     ↓
┌──────────────────────────────────────────────────────────┐
│ 8. 用户调用 build()                                       │
│    scene.build()                                         │
└────────────────────┬─────────────────────────────────────┘
                     ↓
┌──────────────────────────────────────────────────────────┐
│ 9. Scene 调用 Simulator.build()                          │
└────────────────────┬─────────────────────────────────────┘
                     ↓
┌──────────────────────────────────────────────────────────┐
│ 10. Simulator 调用所有 Solver.build()                    │
│     for solver in self._solvers:                         │
│         solver.build()                                   │
│         if solver.is_active():                           │
│             self._active_solvers.append(solver)          │
└────────────────────┬─────────────────────────────────────┘
                     ↓
┌──────────────────────────────────────────────────────────┐
│ 11. Solver.build() 分配内存并初始化                       │
│     self.init_particle_fields()  # MPMSolver             │
│     self.init_grid_fields()      # MPMSolver             │
│     entity._add_to_solver()      # 实体初始化            │
└────────────────────┬─────────────────────────────────────┘
                     ↓
┌──────────────────────────────────────────────────────────┐
│ 12. Coupler.build() 初始化耦合数据结构                    │
└────────────────────┬─────────────────────────────────────┘
                     ↓
┌──────────────────────────────────────────────────────────┐
│ 13. 场景构建完成，准备仿真                                │
└──────────────────────────────────────────────────────────┘
```

### 4.2 仿真步进管线

#### 4.2.1 完整步进流程

```
用户调用 scene.step()
        ↓
┌──────────────────────────────────────────────────┐
│ Scene.step()                                     │
│   self._sim.step()                               │
└────────────────────┬─────────────────────────────┘
                     ↓
┌──────────────────────────────────────────────────┐
│ Simulator.step()                                 │
│   if rigid_only:  # 纯刚体优化                    │
│       for _ in range(substeps):                  │
│           rigid_solver.substep()                 │
│   else:                                          │
│       process_input()  # 处理外部输入             │
│       for _ in range(substeps):                  │
│           substep(f)   # 执行子步                │
└────────────────────┬─────────────────────────────┘
                     ↓
┌──────────────────────────────────────────────────┐
│ Simulator.substep(f)  # 单个子步                 │
│   1. coupler.preprocess(f)                       │
│   2. substep_pre_coupling(f)                     │
│   3. coupler.couple(f)                           │
│   4. substep_post_coupling(f)                    │
└──┬──────────────┬──────────────┬─────────────┬───┘
   ↓              ↓              ↓             ↓
[预处理]      [前向计算]       [耦合]      [后处理]
```

#### 4.2.2 substep_pre_coupling 详解

```
Simulator.substep_pre_coupling(f)
        ↓
for solver in active_solvers:
    solver.substep_pre_coupling(f)
        ↓
┌─────────────────────────────────────────┐
│ 各求解器的前向计算（并行执行）            │
├─────────────────────────────────────────┤
│ RigidSolver:                            │
│   • 前向运动学                           │
│   • 质量矩阵计算                         │
│   • 约束装配                             │
│   • 约束求解                             │
│   • 积分                                 │
├─────────────────────────────────────────┤
│ MPMSolver:                              │
│   • P2G（粒子到网格）                    │
│   • 网格归一化                           │
│   • 应用重力                             │
│   • 边界条件                             │
├─────────────────────────────────────────┤
│ SPHSolver:                              │
│   • 邻居搜索                             │
│   • 密度计算                             │
│   • DFSPH 散度求解                       │
│   • DFSPH 密度求解                       │
├─────────────────────────────────────────┤
│ PBDSolver:                              │
│   • 外力应用                             │
│   • 碰撞检测                             │
│   • 约束投影迭代                         │
├─────────────────────────────────────────┤
│ FEMSolver:                              │
│   • 应力计算                             │
│   • Newton-PCG 求解（隐式）              │
│   • 或显式积分                           │
└─────────────────────────────────────────┘
```

#### 4.2.3 Coupler.couple 详解

```
Coupler.couple(f)
        ↓
┌──────────────────────────────────────────────┐
│ 跨求解器物理交互                              │
├──────────────────────────────────────────────┤
│ 1. MPM ↔ Rigid:                              │
│    • mpm_grid_op(f)                          │
│      - 检测 MPM 网格与刚体碰撞                │
│      - 修改网格速度                           │
│      - 应用法向反作用力到刚体                 │
├──────────────────────────────────────────────┤
│ 2. SPH ↔ Rigid:                              │
│    • sph_rigid(f)                            │
│      - 检测 SPH 粒子与刚体碰撞                │
│      - 计算边界力                             │
│      - 修改粒子速度                           │
├──────────────────────────────────────────────┤
│ 3. PBD ↔ Rigid:                              │
│    • pbd_rigid_collide()                     │
│      - PBD 粒子与刚体碰撞响应                 │
│      - 链接约束处理                           │
├──────────────────────────────────────────────┤
│ 4. FEM ↔ Rigid:                              │
│    • fem_surface_force(f)                    │
│      - 计算 FEM 表面力                        │
│      - 传递到刚体链接                         │
│    • fem_rigid_link_constraints()            │
│      - 处理焊接约束                           │
├──────────────────────────────────────────────┤
│ 5. MPM ↔ SPH:                                │
│    • mpm_sph_interaction()                   │
│      - 网格-粒子交互                          │
├──────────────────────────────────────────────┤
│ 6. MPM ↔ PBD:                                │
│    • mpm_pbd_interaction()                   │
│      - 网格-粒子交互                          │
└──────────────────────────────────────────────┘
```

#### 4.2.4 substep_post_coupling 详解

```
Simulator.substep_post_coupling(f)
        ↓
for solver in active_solvers:
    solver.substep_post_coupling(f)
        ↓
┌─────────────────────────────────────────┐
│ 各求解器的后处理（并行执行）              │
├─────────────────────────────────────────┤
│ RigidSolver:                            │
│   • （已在 pre_coupling 完成）           │
├─────────────────────────────────────────┤
│ MPMSolver:                              │
│   • G2P（网格到粒子）                    │
│     - 更新粒子速度                       │
│     - 更新粒子位置                       │
│     - 更新变形梯度                       │
├─────────────────────────────────────────┤
│ SPHSolver:                              │
│   • 速度积分                             │
│   • 位置更新                             │
│   • 边界条件                             │
├─────────────────────────────────────────┤
│ PBDSolver:                              │
│   • 速度更新                             │
│   • 位置积分                             │
├─────────────────────────────────────────┤
│ FEMSolver:                              │
│   • 位置更新                             │
│   • 速度更新                             │
└─────────────────────────────────────────┘
```

### 4.3 梯度反向传播管线

```
用户调用 loss.backward()
        ↓
Taichi 自动微分启动
        ↓
┌──────────────────────────────────────────┐
│ Simulator._step_grad()                   │
│   for _ in range(substeps-1, -1, -1):    │
│       load_ckpt()  # 恢复前向状态         │
│       sub_step_grad(f)                   │
│   process_input_grad()                   │
└────────────────────┬─────────────────────┘
                     ↓
┌──────────────────────────────────────────┐
│ Simulator.sub_step_grad(f)  # 反向子步   │
│   1. substep_post_coupling_grad(f)       │
│   2. coupler.couple_grad(f)              │
│   3. substep_pre_coupling_grad(f)        │
└──┬──────────────┬──────────────┬──────┘
   ↓              ↓              ↓
[后处理梯度]   [耦合梯度]    [前向梯度]
        ↓
┌──────────────────────────────────────────┐
│ 各求解器的梯度反向传播                     │
├──────────────────────────────────────────┤
│ MPMSolver:                               │
│   • g2p_grad(f)  # G2P 反向               │
│   • grid_op_grad(f)                      │
│   • p2g_grad(f)  # P2G 反向               │
├──────────────────────────────────────────┤
│ SPHSolver:                               │
│   • 速度积分反向                          │
│   • 压力求解反向                          │
│   • 密度计算反向                          │
├──────────────────────────────────────────┤
│ FEMSolver:                               │
│   • 隐式求解反向                          │
│   • 应力计算反向                          │
└──────────────────────────────────────────┘
```

### 4.4 状态管理管线

```
┌──────────────────────────────────────────┐
│ 1. 用户查询状态                           │
│    state = scene.get_state()             │
└────────────────────┬─────────────────────┘
                     ↓
┌──────────────────────────────────────────┐
│ 2. Scene 调用 Simulator.get_state()      │
└────────────────────┬─────────────────────┘
                     ↓
┌──────────────────────────────────────────┐
│ 3. Simulator 收集所有 Solver 状态        │
│    state = SimState(scene)               │
│    for solver in solvers:                │
│        state.append(solver.get_state(f)) │
│    # 缓存状态用于梯度追踪                 │
│    self._queried_states.append(state)    │
└────────────────────┬─────────────────────┘
                     ↓
┌──────────────────────────────────────────┐
│ 4. 各 Solver 返回其状态                  │
│    MPMSolverState:                       │
│      • particles.pos                     │
│      • particles.vel                     │
│      • particles.F                       │
│    RigidSolverState:                     │
│      • qpos                              │
│      • qvel                              │
│      • links_pos / links_quat            │
└────────────────────┬─────────────────────┘
                     ↓
┌──────────────────────────────────────────┐
│ 5. 状态作为 PyTorch Tensor 返回          │
│    用户可以:                              │
│    • 用于强化学习奖励计算                 │
│    • 用于监督学习目标                     │
│    • 用于可视化                           │
└──────────────────────────────────────────┘
```

---

## 5. 数据流分析

### 5.1 前向数据流

```
用户输入（控制命令）
        ↓
┌───────────────────────┐
│ Entity.set_dofs_kp() │
│ Entity.set_dofs_kv() │
│ Entity.set_dofs_force() │
└───────┬───────────────┘
        ↓
┌─────────────────────────────┐
│ Solver.process_input()      │
│ • 解析命令                   │
│ • 设置目标状态               │
└───────┬─────────────────────┘
        ↓
┌──────────────────────────────────┐
│ Solver.substep_pre_coupling()    │
│ • 物理计算                        │
│ • 状态更新                        │
└───────┬──────────────────────────┘
        ↓
┌──────────────────────────────┐
│ Coupler.couple()             │
│ • 跨求解器交互                │
└───────┬──────────────────────┘
        ↓
┌────────────────────────────────────┐
│ Solver.substep_post_coupling()     │
│ • 最终状态更新                      │
└───────┬────────────────────────────┘
        ↓
┌──────────────────────────┐
│ State (位置、速度等)      │
│ • 用于可视化              │
│ • 用于奖励计算            │
│ • 用于梯度反向传播        │
└──────────────────────────┘
```

### 5.2 反向数据流（梯度）

```
损失函数 L
        ↓
∂L/∂state (状态梯度)
        ↓
┌─────────────────────────────────┐
│ Solver.add_grad_from_state()    │
│ • 将梯度添加到求解器状态字段     │
└───────┬─────────────────────────┘
        ↓
┌───────────────────────────────────┐
│ Solver.substep_post_coupling_grad() │
│ • 反向计算后处理                   │
└───────┬───────────────────────────┘
        ↓
┌────────────────────────────┐
│ Coupler.couple_grad()      │
│ • 耦合反向传播              │
└───────┬────────────────────┘
        ↓
┌────────────────────────────────────┐
│ Solver.substep_pre_coupling_grad() │
│ • 反向计算前向过程                  │
└───────┬────────────────────────────┘
        ↓
┌────────────────────────────┐
│ Solver.process_input_grad() │
│ • 控制输入梯度              │
└───────┬────────────────────┘
        ↓
∂L/∂control (控制梯度)
        ↓
优化器更新参数
```

---

## 6. 关键设计决策

### 6.1 延迟构建（Lazy Build）

**为什么**：
- 用户可以动态添加实体而无需重新初始化
- 仅在 `build()` 时分配 GPU 内存，避免浪费
- 允许根据实体数量优化数据结构大小

**实现**：
```python
def __init__(self, ...):
    # 仅配置，不分配内存
    self._entities = []
    self._is_built = False

def add_entity(self, ...):
    # 添加实体，但不分配内存
    if self._is_built:
        raise Exception("Cannot add entity after build()")
    self._entities.append(entity)

def build(self):
    # 根据实体数量分配内存
    n_particles = sum(e.n_particles for e in self._entities)
    self.particles = ti.field(..., shape=(n_particles,))
    self._is_built = True
```

### 6.2 材料驱动的求解器路由

**为什么**：
- 用户无需了解底层求解器
- 材料类型自然决定物理行为
- 易于扩展新材料和新求解器

**实现**：
```python
def _add_entity(self, morph, material, surface):
    if isinstance(material, gs.materials.Rigid):
        entity = self.rigid_solver.add_entity(...)
    elif isinstance(material, gs.materials.MPM.Base):
        entity = self.mpm_solver.add_entity(...)
    elif isinstance(material, gs.materials.SPH.Base):
        entity = self.sph_solver.add_entity(...)
    # ...
```

### 6.3 统一的子步接口

**为什么**：
- 简化 Simulator 逻辑
- 易于添加新求解器
- 支持灵活的耦合策略

**实现**：
```python
# 所有求解器实现相同接口
class Solver:
    def substep_pre_coupling(self, f): pass
    def substep_post_coupling(self, f): pass

# Simulator 统一调用
for solver in active_solvers:
    solver.substep_pre_coupling(f)
coupler.couple(f)
for solver in active_solvers:
    solver.substep_post_coupling(f)
```

### 6.4 批处理优先

**为什么**：
- 支持并行环境训练
- GPU 利用率最大化
- 统一的批处理接口简化代码

**实现**：
```python
# 所有数据结构都包含批次维度
self.particles = ti.field(
    shape=(B, substeps_local + 1, n_particles),
    ...
)

# 内核自动并行化批次维度
@ti.kernel
def p2g(self, f: ti.i32):
    for i_p, i_b in ti.ndrange(n_particles, B):
        # i_b 是批次索引，自动并行
        ...
```

### 6.5 可微分优先

**为什么**：
- 支持基于梯度的优化
- 强化学习和逆向设计
- Taichi 自动微分简化实现

**实现**：
```python
# 前向字段需要梯度
self.particles = ti.field(..., needs_grad=True)

# 梯度反向传播
def _step_grad(self):
    for f in range(substeps-1, -1, -1):
        self.sub_step_grad(f)

# Taichi 自动计算梯度
# 无需手动实现 ∂f/∂x
```

---

## 7. 性能优化策略

### 7.1 纯刚体优化路径

```python
if self._rigid_only:
    # 跳过耦合开销，直接调用刚体求解器
    for _ in range(substeps):
        self.rigid_solver.substep()
```

**优势**：
- 减少函数调用开销
- 避免无用的耦合检查
- 刚体仿真可达 10,000+ FPS

### 7.2 接触岛屿和休眠

```python
# RigidSolver
if self._use_contact_island:
    # 将静止物体分组到岛屿
    self._func_aggregate_awake_entities()
    
if self._use_hibernation:
    # 休眠静止岛屿
    self._func_hibernate_check()
```

**优势**：
- 大幅减少刚体计算量
- 仅计算活跃的接触对
- 适合大规模场景

### 7.3 空间哈希加速

```python
# SPH/PBD Solver
self.sh = SpatialHasher(
    cell_size=particle_size,
    grid_res=(128, 128, 128),
)

# 邻居查询 O(1)
for i_p in range(n_particles):
    for j in sh.query_neighbors(i_p):
        # 仅计算邻居交互
        ...
```

**优势**：
- 邻居查询从 O(n²) 降到 O(n)
- GPU 友好的数据结构
- 支持大规模粒子仿真

### 7.4 SoA 数据布局

```python
# Structure of Arrays (SoA)
self.particles = struct.field(
    shape=(...),
    layout=ti.Layout.SOA,  # 关键优化
)
```

**优势**：
- GPU 合并内存访问
- 缓存友好
- SIMD 向量化

### 7.5 编译缓存

```python
# RigidSolver
self._static_rigid_sim_cache_key = \
    array_class.get_static_rigid_sim_cache_key(self)
```

**优势**：
- Taichi kernel 编译结果缓存
- 相同配置无需重新编译
- 加速场景加载

---

## 8. 扩展点

### 8.1 添加新求解器

```python
# 1. 创建求解器类
class NewSolver(Solver):
    def substep_pre_coupling(self, f):
        # 实现计算逻辑
        ...
    
    def substep_post_coupling(self, f):
        ...

# 2. 在 Simulator 中注册
class Simulator:
    def __init__(self, ...):
        self.new_solver = NewSolver(...)
        self._solvers.append(self.new_solver)

# 3. 添加材料类型
class NewMaterial(Material):
    pass

# 4. 路由到求解器
def _add_entity(self, morph, material, surface):
    if isinstance(material, NewMaterial):
        return self.new_solver.add_entity(...)
```

### 8.2 添加新耦合

```python
# 在 Coupler.couple() 中添加
def couple(self, f):
    if self._new_coupling:
        self.new_solver_interaction(f)

@ti.kernel
def new_solver_interaction(self, f: ti.i32):
    # 实现耦合逻辑
    ...
```

### 8.3 添加新实体类型

```python
# 1. 创建实体类
class NewEntity(Entity):
    def __init__(self, ...):
        super().__init__(...)
        # 自定义数据
    
    def _build(self):
        # 初始化逻辑
        ...

# 2. 在 Solver 中使用
def add_entity(self, ...):
    entity = NewEntity(...)
    self._entities.append(entity)
    return entity
```

---

## 9. 常见问题

### 9.1 为什么分两阶段初始化（__init__ 和 build）？

**答**：允许用户动态添加实体，仅在 `build()` 时根据实际数量分配内存，避免浪费。

### 9.2 为什么需要 Coupler？

**答**：不同求解器处理不同类型的物理（刚体、流体、软体），Coupler 负责它们之间的物理交互（碰撞、力传递）。

### 9.3 子步（substep）和步（step）的区别？

**答**：
- **步（step）**：用户可见的时间步，通常对应控制频率（如 50Hz）
- **子步（substep）**：内部物理计算的时间步，更小以保证稳定性（如 500Hz）
- 关系：`dt_substep = dt_step / n_substeps`

### 9.4 为什么梯度传播是反向的？

**答**：遵循自动微分的链式法则，从损失函数向后传播梯度到控制输入。

### 9.5 如何选择求解器？

**答**：
- **刚体、机器人**：RigidSolver
- **流体**：SPHSolver
- **大变形、断裂**：MPMSolver
- **布料、软体**：PBDSolver
- **精确变形**：FEMSolver
- **烟雾、气体**：SFSolver
- **动画角色**：AvatarSolver

---

## 10. 总结

### 10.1 架构优势

1. **分层清晰**：用户层 → 调度层 → 计算层 → 数据层，职责明确
2. **高度解耦**：各求解器独立实现，通过统一接口交互
3. **易于扩展**：添加新求解器、新材料、新耦合都很简单
4. **性能优化**：批处理、SoA 布局、空间哈希、休眠等优化
5. **可微分**：大部分流程支持自动微分

### 10.2 核心工作流

```
初始化: Scene → Simulator → Solvers → Entities
        ↓
仿真:   process_input → [substep_pre → couple → substep_post] × n
        ↓
查询:   get_state → SimState (用于奖励/可视化)
        ↓
梯度:   loss.backward() → 反向传播 → 控制梯度
```

### 10.3 设计哲学

1. **材料驱动**：用户通过材料选择物理行为
2. **统一接口**：所有求解器遵循相同的生命周期
3. **延迟构建**：灵活添加实体，高效内存使用
4. **批处理优先**：支持并行环境训练
5. **可微分优先**：支持基于梯度的优化

---

**创建日期**：2025年  
**基于代码**：Genesis 引擎 `genesis/engine/`  
**作者**：Genesis 架构分析
