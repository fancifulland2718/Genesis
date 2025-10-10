# Genesis engine 核心文件架构分析

## 1. 概述

`genesis/engine` 根目录包含5个核心文件，它们构成了 Genesis 仿真系统的顶层架构。这些文件定义了场景管理、仿真协调、网格处理、力场和空间加速结构。

**核心文件统计**:
```
scene.py         : 1,530 行 - 场景管理和用户接口
simulator.py     : 501 行  - 仿真协调器
mesh.py          : 491 行  - 网格处理和优化
force_fields.py  : 544 行  - 力场系统
bvh.py           : 550 行  - 碰撞检测加速结构
__init__.py      : 0 行    - 模块导出
总计             : 3,616 行
```

## 2. 文件间关系

```
┌──────────────────────────────────────────┐
│              Scene                       │
│  - 用户接口层                             │
│  - 实体管理                               │
│  - 可视化控制                             │
└──────────────────────────────────────────┘
              ↓ 持有
┌──────────────────────────────────────────┐
│           Simulator                      │
│  - 时间步进                               │
│  - 求解器协调                             │
│  - 梯度管理                               │
└──────────────────────────────────────────┘
              ↓ 使用
┌───────────┬──────────┬──────────────────┐
│   Mesh    │ForceField│      BVH         │
│ 网格工具  │  力场    │  碰撞加速        │
└───────────┴──────────┴──────────────────┘
```

## 3. Scene - 场景管理 (1,530行)

### 3.1 核心职责

**Scene 是用户的主要接口**，封装了整个仿真环境：

```python
@gs.assert_initialized
class Scene(RBC):
    """
    场景对象包装仿真环境的所有组件：
    - Simulator (物理仿真器)
    - Entities (实体列表)
    - Visualizer (可视化)
    - Recorders (录制器)
    - Sensors (传感器)
    """
```

### 3.2 初始化流程

```python
def __init__(
    self,
    sim_options: SimOptions = None,
    coupler_options: BaseCouplerOptions = None,
    tool_options: ToolOptions = None,
    rigid_options: RigidOptions = None,
    avatar_options: AvatarOptions = None,
    mpm_options: MPMOptions = None,
    sph_options: SPHOptions = None,
    fem_options: FEMOptions = None,
    sf_options: SFOptions = None,
    pbd_options: PBDOptions = None,
    vis_options: VisOptions = None,
    viewer_options: ViewerOptions = None,
    renderer: RendererOptions = None,
    show_viewer: bool = True,
    show_FPS: bool = False,
):
    # 1. 创建 Simulator
    self._sim = Simulator(
        scene=self,
        options=self.sim_options,
        coupler_options=self.coupler_options,
        # ... 所有求解器选项
    )
    
    # 2. 创建 Visualizer
    self._visualizer = Visualizer(
        scene=self,
        options=self.vis_options,
        viewer_options=self.viewer_options,
        renderer_options=self.renderer_options,
    )
    
    # 3. 管理组件
    self._entities = gs.List()
    self._materials = gs.List()
    self._sensors = SensorManager(self)
    self._recorders = RecorderManager(self)
```

### 3.3 主要接口

#### 3.3.1 实体管理

```python
def add_entity(
    self,
    morph: Morph,
    material: Material | None = None,
    surface: Surface | None = None,
    visualize_contact: bool = False,
) -> Entity:
    """
    添加实体到场景
    
    工作流程:
    1. 验证参数组合
    2. 选择对应的求解器
    3. 创建实体对象
    4. 注册到求解器和场景
    """
    gs.assert_unbuilt(self)
    
    # 根据材料类型选择求解器
    if isinstance(material, gs.materials.Rigid):
        solver = self.sim.rigid_solver
        entity = RigidEntity(...)
    elif isinstance(material, gs.materials.MPM.Base):
        solver = self.sim.mpm_solver
        entity = MPMEntity(...)
    # ... 其他材料类型
    
    # 注册
    self._entities.append(entity)
    solver.add_entity(entity)
    
    return entity
```

#### 3.3.2 力场管理

```python
def add_force_field(
    self,
    force_field: ForceField,
) -> ForceField:
    """添加全局力场 (如重力、风场)"""
    self._sim.add_force_field(force_field)
    return force_field
```

#### 3.3.3 发射器

```python
def add_emitter(
    self,
    material: Material,
    max_particles: int,
    surface: Surface | None = None,
    particle_size: float | None = None,
) -> Emitter:
    """添加粒子发射器"""
    emitter = Emitter(
        scene=self,
        solver=solver,
        material=material,
        max_particles=max_particles,
        particle_size=particle_size,
    )
    self._entities.append(emitter)
    return emitter
```

### 3.4 仿真控制

#### 3.4.1 构建

```python
def build(self, n_envs: int = 1, env_spacing: tuple = (0.0, 0.0, 0.0)):
    """
    构建场景 (分配资源、初始化数据结构)
    
    流程:
    1. 设置环境数量和间距
    2. 构建 Simulator (各个求解器)
    3. 构建 Visualizer
    4. 标记为已构建
    """
    gs.assert_unbuilt(self)
    
    self._n_envs = n_envs
    self._env_spacing = env_spacing
    
    # 构建仿真器
    self._sim.build()
    
    # 构建可视化
    if self.show_viewer or self.viewer_options.camera_pos is not None:
        self._visualizer.build()
    
    self._is_built = True
    gs.logger.info(f"Scene built with {n_envs} environments.")
```

#### 3.4.2 时间步进

```python
def step(self):
    """
    推进仿真一个时间步
    
    流程:
    1. Simulator.step() - 物理仿真
    2. Visualizer.update() - 更新渲染
    3. Recorder.record() - 录制帧
    4. FPS 追踪
    """
    gs.assert_built(self)
    
    # 物理仿真
    self._sim.step()
    
    # 可视化更新
    if self.visualizer.viewer is not None:
        self.visualizer.update()
    
    # 录制
    for recorder in self._recorders:
        recorder.record()
    
    # FPS 计数
    if self.show_FPS:
        self._fps_tracker.update()
```

#### 3.4.3 重置

```python
def reset(self, state: SimState | None = None):
    """
    重置场景到初始状态或指定状态
    
    参数:
        state: 目标状态 (None 表示重置到初始状态)
    """
    gs.assert_built(self)
    
    if state is None:
        # 重置到初始状态
        self._sim.reset()
    else:
        # 重置到指定状态
        self._sim.set_state(state)
    
    # 重置可视化
    self.visualizer.reset()
```

### 3.5 状态管理

```python
def get_state(self) -> SimState:
    """获取当前仿真状态 (所有求解器和实体)"""
    return self._sim.get_state()

def set_state(self, state: SimState):
    """设置场景到指定状态"""
    self._sim.set_state(state)
```

### 3.6 设计模式

#### Facade Pattern (外观模式)

Scene 隐藏内部复杂性，提供简单接口：

```python
# 用户只需要与 Scene 交互
scene = gs.Scene()
scene.add_entity(...)
scene.build()
scene.step()

# 内部协调多个子系统
# - Simulator
# - Visualizer
# - Sensors
# - Recorders
```

#### Builder Pattern (构建者模式)

分步构建场景：

```python
scene = gs.Scene(
    rigid_options=gs.RigidOptions(...),
    mpm_options=gs.MPMOptions(...),
)

# 添加组件
scene.add_entity(...)
scene.add_force_field(...)

# 延迟构建
scene.build()
```

## 4. Simulator - 仿真协调器 (501行)

### 4.1 核心职责

**Simulator 协调多个求解器和耦合器**：

```python
@ti.data_oriented
class Simulator(RBC):
    """
    场景级仿真管理器:
    - 管理多个求解器 (Rigid, MPM, FEM, etc.)
    - 管理求解器间耦合 (Coupler)
    - 时间步进协调
    - 梯度管理
    """
```

### 4.2 架构设计

```
Simulator
    ├── Solvers (列表)
    │   ├── ToolSolver
    │   ├── RigidSolver
    │   ├── AvatarSolver
    │   ├── MPMSolver
    │   ├── SPHSolver
    │   ├── PBDSolver
    │   ├── FEMSolver
    │   └── SFSolver
    └── Coupler (耦合器)
        ├── LegacyCoupler
        └── SAPCoupler
```

### 4.3 初始化

```python
def __init__(
    self,
    scene: "Scene",
    options: SimOptions,
    coupler_options: BaseCouplerOptions,
    # ... 各个求解器选项
):
    self._scene = scene
    self.options = options
    
    # 创建所有求解器
    self.tool_solver = ToolSolver(self.scene, self, self.tool_options)
    self.rigid_solver = RigidSolver(self.scene, self, self.rigid_options)
    self.avatar_solver = AvatarSolver(self.scene, self, self.avatar_options)
    self.mpm_solver = MPMSolver(self.scene, self, self.mpm_options)
    self.sph_solver = SPHSolver(self.scene, self, self.sph_options)
    self.pbd_solver = PBDSolver(self.scene, self, self.pbd_options)
    self.fem_solver = FEMSolver(self.scene, self, self.fem_options)
    self.sf_solver = SFSolver(self.scene, self, self.sf_options)
    
    # 求解器列表
    self._solvers = gs.List([
        self.tool_solver,
        self.rigid_solver,
        self.avatar_solver,
        self.mpm_solver,
        self.sph_solver,
        self.pbd_solver,
        self.fem_solver,
        self.sf_solver,
    ])
    
    # 活跃求解器 (有实体的求解器)
    self._active_solvers = gs.List()
    
    # 创建耦合器
    if isinstance(coupler_options, SAPCouplerOptions):
        self._coupler = SAPCoupler(self, coupler_options)
    else:
        self._coupler = LegacyCoupler(self, coupler_options)
```

### 4.4 时间步进

```python
def step(self):
    """
    推进仿真一个时间步
    
    流程:
    1. 子步循环
    2. 每个子步包括:
       - 求解器预处理
       - 求解器前向
       - 耦合处理
       - 求解器后处理
    """
    self.cur_step_global += 1
    
    for substep in range(self._substeps):
        self.cur_substep_local = substep
        
        # 前向求解
        for solver in self._active_solvers:
            solver.substep_pre_coupling(substep)
        
        # 耦合
        self._coupler.couple(substep)
        
        # 后续处理
        for solver in self._active_solvers:
            solver.substep_post_coupling(substep)
    
    # 梯度收集 (可微分仿真)
    if self.requires_grad:
        self.collect_output_grads()
```

### 4.5 时间坐标系统

```python
# 三种时间索引:
# 1. f_global: 全局帧索引 (从 0 开始)
# 2. f_local: 局部帧索引 (从 -self._substeps_reset 开始)
# 3. s_global: 全局步索引 (全局帧 / 子步数)

def f_global_to_f_local(self, f_global: int) -> int:
    return f_global - self.cur_step_global * self._substeps

def f_local_to_s_local(self, f_local: int) -> int:
    return f_local // self._substeps

def f_global_to_s_global(self, f_global: int) -> int:
    return f_global // self._substeps
```

### 4.6 梯度管理

```python
def collect_output_grads(self):
    """
    从下游查询的状态收集梯度
    
    用于可微分仿真的反向传播
    """
    # Simulator 级别的状态
    if self.cur_step_global in self._queried_states:
        for state in self._queried_states[self.cur_step_global]:
            self.add_grad_from_state(state)
    
    # 每个求解器的实体状态
    for solver in self._active_solvers:
        solver.collect_output_grads()

def reset_grad(self):
    """重置所有梯度"""
    for solver in self._active_solvers:
        solver.reset_grad()
```

### 4.7 设计模式

#### Coordinator Pattern (协调器模式)

Simulator 协调多个求解器的执行：

```python
# 不是各个求解器独立运行
rigid_solver.step()
mpm_solver.step()

# 而是协调器统一调度
simulator.step()
    └─> for solver in active_solvers:
            solver.substep(...)
```

#### Mediator Pattern (中介者模式)

Simulator 作为求解器间的中介：

```python
# 求解器不直接通信
# Rigid → Simulator → Coupler → MPM

self._coupler.couple(substep)
    └─> 检测 RigidSolver 和 MPMSolver 的碰撞
    └─> 施加耦合力
```

## 5. Mesh - 网格处理 (491行)

### 5.1 核心职责

**Mesh 类是 Genesis 的三角网格对象**：

```python
class Mesh(RBC):
    """
    Genesis 的三角网格对象
    
    封装 trimesh.Trimesh 并添加:
    - 凸包化 (convexify)
    - 简化/抽取 (decimate)
    - 表面属性管理
    - UV 坐标处理
    """
```

### 5.2 初始化

```python
def __init__(
    self,
    mesh: trimesh.Trimesh,
    surface: Surface | None = None,
    uvs: np.ndarray | None = None,
    convexify: bool = False,
    decimate: bool = False,
    decimate_face_num: int = 500,
    decimate_aggressiveness: int = 0,
    metadata: dict = None,
):
    self._uid = gs.UID()
    self._mesh = mesh  # trimesh.Trimesh 对象
    self._surface = surface
    self._uvs = uvs
    self._metadata = metadata or {}
    
    # 预处理
    if convexify:
        self.convexify()
    
    if decimate:
        self.decimate(decimate_face_num, decimate_aggressiveness, convexify)
```

### 5.3 网格处理

#### 5.3.1 凸包化

```python
def convexify(self):
    """
    将网格转换为凸包
    
    用于碰撞检测优化
    """
    if self._mesh.vertices.shape[0] > 3:
        self._mesh = trimesh.convex.convex_hull(self._mesh)
        self._metadata["convexified"] = True
    self.clear_visuals()
```

#### 5.3.2 简化

```python
def decimate(self, target_face_num, aggressiveness, convexify):
    """
    网格简化 (减少面数)
    
    使用 fast_simplification 库
    
    参数:
        target_face_num: 目标面数
        aggressiveness: 0 (无损) ~ 8 (激进)
    """
    if len(self._mesh.faces) > target_face_num:
        self._mesh.process(validate=True)
        self._mesh = trimesh.Trimesh(
            *fast_simplification.simplify(
                self._mesh.vertices,
                self._mesh.faces,
                target_count=target_face_num,
                agg=aggressiveness,
                lossless=(aggressiveness == 0),
            ),
        )
        self._metadata["decimated"] = True
```

### 5.4 属性访问

```python
@property
def vertices(self) -> np.ndarray:
    """顶点坐标 (N, 3)"""
    return self._mesh.vertices

@property
def faces(self) -> np.ndarray:
    """面索引 (M, 3)"""
    return self._mesh.faces

@property
def trimesh(self) -> trimesh.Trimesh:
    """内部 trimesh 对象"""
    return self._mesh

@property
def surface(self) -> Surface:
    """表面属性"""
    return self._surface
```

### 5.5 工厂方法

```python
@staticmethod
def load(file: str, **kwargs) -> "Mesh":
    """从文件加载网格"""
    mesh = load_mesh(file)
    return Mesh(mesh, **kwargs)

@staticmethod
def create_box(extents, **kwargs) -> "Mesh":
    """创建盒子网格"""
    mesh = mu.create_box(extents)
    return Mesh(mesh, **kwargs)

@staticmethod
def create_sphere(radius, **kwargs) -> "Mesh":
    """创建球体网格"""
    mesh = mu.create_sphere(radius)
    return Mesh(mesh, **kwargs)
```

### 5.6 设计特点

**Wrapper Pattern (包装器模式)**:

```python
# 包装 trimesh.Trimesh
class Mesh:
    def __init__(self, mesh: trimesh.Trimesh):
        self._mesh = mesh  # 内部 trimesh 对象
    
    # 暴露必要属性
    @property
    def vertices(self):
        return self._mesh.vertices
    
    # 添加额外功能
    def convexify(self):
        self._mesh = trimesh.convex.convex_hull(self._mesh)
```

## 6. ForceField - 力场系统 (544行)

### 6.1 核心职责

**ForceField 定义空间中的加速度场**：

```python
@ti.data_oriented
class ForceField(RBC):
    """
    力场基类 (实际上是加速度场)
    
    注意: 力场没有空间密度概念，所以实际是加速度场
    """
    
    def __init__(self):
        self._active = ti.field(gs.ti_bool, shape=())
        self._active[None] = False
    
    @ti.func
    def get_acc(self, pos, vel, t, i) -> ti.Vector:
        """
        获取位置 pos 处的加速度
        
        参数:
            pos: 位置
            vel: 速度 (用于速度相关力)
            t: 时间
            i: 粒子索引
        """
        acc = ti.Vector.zero(gs.ti_float, 3)
        if self._active[None]:
            acc = self._get_acc(pos, vel, t, i)
        return acc
    
    @ti.func
    def _get_acc(self, pos, vel, t, i):
        """子类实现"""
        raise NotImplementedError
```

### 6.2 具体力场类型

#### 6.2.1 Constant - 恒定力场

```python
class Constant(ForceField):
    """
    恒定加速度场 (如重力)
    
    参数:
        direction: 方向 (归一化)
        strength: 强度 (m/s²)
    """
    
    def __init__(self, direction=(0, 0, -1), strength=9.81):
        super().__init__()
        
        direction = np.array(direction)
        self._direction = direction / np.linalg.norm(direction)
        self._strength = strength
        self._acc_ti = ti.Vector(self._direction * self._strength, dt=gs.ti_float)
    
    @ti.func
    def _get_acc(self, pos, vel, t, i):
        return self._acc_ti
```

**使用示例**:
```python
gravity = gs.force_fields.Constant(direction=(0, 0, -1), strength=9.81)
scene.add_force_field(gravity)
```

#### 6.2.2 Wind - 风场

```python
class Wind(ForceField):
    """
    圆柱形风场
    
    参数:
        direction: 风向
        strength: 风强度
        radius: 圆柱半径
        center: 圆柱中心
    """
    
    def __init__(self, direction=(1, 0, 0), strength=10.0, radius=0.5, center=(0, 0, 0)):
        super().__init__()
        
        self._direction = direction / np.linalg.norm(direction)
        self._strength = strength
        self._radius = radius
        self._center = ti.Vector(center, dt=gs.ti_float)
        self._acc_ti = ti.Vector(self._direction * self._strength, dt=gs.ti_float)
    
    @ti.func
    def _get_acc(self, pos, vel, t, i):
        # 计算到圆柱轴的距离
        rel_pos = pos - self._center
        proj = rel_pos.dot(self._acc_ti.normalized())
        perp_dist = (rel_pos - proj * self._acc_ti.normalized()).norm()
        
        # 圆柱内部施加风力
        acc = ti.Vector.zero(gs.ti_float, 3)
        if perp_dist < self._radius:
            acc = self._acc_ti
        return acc
```

#### 6.2.3 Vortex - 涡旋场

```python
class Vortex(ForceField):
    """
    涡旋加速度场
    
    参数:
        center: 涡旋中心
        axis: 涡旋轴
        strength: 涡旋强度
        radius: 影响半径
    """
    
    @ti.func
    def _get_acc(self, pos, vel, t, i):
        rel_pos = pos - self._center
        
        # 计算切向分量
        proj = rel_pos.dot(self._axis)
        perp = rel_pos - proj * self._axis
        dist = perp.norm()
        
        # 涡旋加速度 (切向)
        acc = ti.Vector.zero(gs.ti_float, 3)
        if dist > 0 and dist < self._radius:
            tangent = self._axis.cross(perp.normalized())
            acc = tangent * self._strength * (1.0 - dist / self._radius)
        
        return acc
```

#### 6.2.4 Attractor - 吸引力场

```python
class Attractor(ForceField):
    """
    点吸引力场 (引力)
    
    参数:
        center: 吸引中心
        strength: 吸引强度
        radius: 影响半径
    """
    
    @ti.func
    def _get_acc(self, pos, vel, t, i):
        rel_pos = self._center - pos
        dist = rel_pos.norm()
        
        # 距离平方反比律
        acc = ti.Vector.zero(gs.ti_float, 3)
        if dist > 0 and dist < self._radius:
            acc = rel_pos.normalized() * self._strength / (dist * dist + 1e-6)
        
        return acc
```

### 6.3 使用流程

```python
# 1. 创建力场
gravity = gs.force_fields.Constant(direction=(0, 0, -1), strength=9.81)
wind = gs.force_fields.Wind(direction=(1, 0, 0), strength=5.0)

# 2. 添加到场景
scene.add_force_field(gravity)
scene.add_force_field(wind)

# 3. 激活/停用
gravity.activate()
wind.deactivate()

# 4. 在求解器中应用
@ti.kernel
def apply_force_fields():
    for i in particles:
        acc = ti.Vector.zero(gs.ti_float, 3)
        for ff in force_fields:
            acc += ff.get_acc(pos[i], vel[i], t, i)
        vel[i] += acc * dt
```

### 6.4 设计模式

**Strategy Pattern (策略模式)**:

不同的力场实现不同的 `_get_acc()` 策略。

## 7. BVH - 碰撞检测加速 (550行)

### 7.1 核心职责

**BVH (Bounding Volume Hierarchy) 加速空间查询**：

```python
@ti.data_oriented
class LBVH(RBC):
    """
    线性 BVH (Linear BVH)
    
    用于加速碰撞检测和最近邻查询
    
    特点:
    - 并行构建
    - GPU 友好
    - 支持批处理
    """
```

### 7.2 数据结构

#### 7.2.1 AABB - 轴对齐包围盒

```python
@ti.data_oriented
class AABB(RBC):
    """轴对齐包围盒管理器"""
    
    def __init__(self, n_batches, n_aabbs):
        @ti.dataclass
        class ti_aabb:
            min: gs.ti_vec3
            max: gs.ti_vec3
            
            @ti.func
            def intersects(self, other) -> bool:
                """检查两个 AABB 是否相交"""
                return (
                    self.min[0] <= other.max[0] and self.max[0] >= other.min[0] and
                    self.min[1] <= other.max[1] and self.max[1] >= other.min[1] and
                    self.min[2] <= other.max[2] and self.max[2] >= other.min[2]
                )
        
        self.ti_aabb = ti_aabb
        self.aabbs = ti_aabb.field(shape=(n_batches, n_aabbs), layout=ti.Layout.SOA)
```

#### 7.2.2 BVH Node - BVH 节点

```python
@ti.dataclass
class Node:
    """BVH 树节点"""
    
    left: ti.i32          # 左子节点索引 (-1 表示叶节点)
    right: ti.i32         # 右子节点索引
    parent: ti.i32        # 父节点索引
    aabb: ti_aabb         # 包围盒
```

### 7.3 构建算法

```python
def build(self):
    """
    并行构建 BVH
    
    算法: Morton Code + Radix Sort
    
    步骤:
    1. 计算 AABB 中心
    2. 计算 Morton Code (空间曲线编码)
    3. Radix Sort 排序
    4. 自底向上构建树
    """
    # 1. 计算中心和 Morton Code
    self._compute_morton_codes()
    
    # 2. Radix Sort
    self._radix_sort()
    
    # 3. 构建树
    self._build_tree()
    
    # 4. 自底向上计算包围盒
    self._compute_bounding_boxes()
```

#### Morton Code 编码

```python
@ti.func
def morton_encode(x: ti.f32, y: ti.f32, z: ti.f32) -> ti.u32:
    """
    将 3D 坐标编码为 Morton Code (Z-order curve)
    
    保持空间局部性
    """
    x = ti.cast(x * 1024.0, ti.u32) & 0x3FF
    y = ti.cast(y * 1024.0, ti.u32) & 0x3FF
    z = ti.cast(z * 1024.0, ti.u32) & 0x3FF
    
    # 交错位
    code = expand_bits(x) | (expand_bits(y) << 1) | (expand_bits(z) << 2)
    return code
```

### 7.4 查询算法

```python
@ti.kernel
def query_overlaps(self, query_aabb: ti_aabb) -> ti.types.ndarray():
    """
    查询与给定 AABB 重叠的所有 AABB
    
    算法: 深度优先遍历
    
    使用栈避免递归
    """
    results = []
    stack = [root_node]
    
    while len(stack) > 0:
        node = stack.pop()
        
        if node.aabb.intersects(query_aabb):
            if node.is_leaf():
                # 叶节点: 记录结果
                results.append(node.aabb_idx)
            else:
                # 内部节点: 继续遍历
                stack.append(node.left)
                stack.append(node.right)
    
    return results
```

### 7.5 性能特点

**优势**:
- **并行构建**: O(N log N) 复杂度
- **GPU 加速**: Taichi 内核
- **批处理**: 支持多个环境并行查询

**开销**:
- 每帧重建 (动态场景)
- 内存占用 (2N-1 个节点)

### 7.6 设计模式

**Spatial Indexing Pattern (空间索引模式)**:

```python
# 不是暴力检测
for i in objects:
    for j in objects:
        if collide(i, j):
            handle_collision(i, j)
# O(N²)

# 而是使用 BVH 加速
bvh.build(objects)
for i in objects:
    candidates = bvh.query_overlaps(objects[i].aabb)
    for j in candidates:
        if collide(i, j):
            handle_collision(i, j)
# O(N log N)
```

## 8. 模块间协作

### 8.1 典型工作流

```
用户
  ↓ 创建场景
Scene(sim_options=..., vis_options=...)
  ↓ 创建
Simulator(scene, options, ...)
  ↓ 创建
Solvers + Coupler
  ↓ 用户添加实体
Scene.add_entity(morph=Mesh(...), material=..., surface=...)
  ↓ 创建网格
Mesh(file="model.obj", convexify=True)
  ↓ 用户添加力场
Scene.add_force_field(Constant(...))
  ↓ 构建
Scene.build(n_envs=4)
  ↓ 分配资源
Simulator.build()
  ↓ 构建 BVH
BVH.build(aabbs)
  ↓ 仿真循环
while True:
    Scene.step()
        ↓
    Simulator.step()
        ↓
    Solvers.substep()
        ↓
    Apply ForceFields
        ↓
    BVH query for collision
```

### 8.2 依赖图

```
Scene
  └── Simulator
       ├── Solvers
       │   └── Entities
       │        └── Mesh
       ├── Coupler
       │   └── BVH
       └── ForceFields
```

## 9. 代码风格总结

### 9.1 命名规范

- **类名**: PascalCase (Scene, Simulator, Mesh)
- **文件名**: snake_case (scene.py, force_fields.py)
- **私有属性**: `_` 前缀 (self._sim, self._entities)
- **公开接口**: 简洁动词 (add_entity, build, step)

### 9.2 设计原则

1. **单一职责**: 每个文件有明确职责
2. **高内聚低耦合**: Scene 作为外观，隐藏内部细节
3. **可扩展**: 通过继承和组合扩展功能
4. **批处理优先**: 所有数据结构支持批次维度

### 9.3 文档风格

使用 NumPy 风格的 docstring:
```python
def add_entity(self, morph, material, surface):
    """
    Add an entity to the scene.
    
    Parameters
    ----------
    morph : Morph
        The shape of the entity.
    material : Material
        The material of the entity.
    surface : Surface
        The surface properties.
    
    Returns
    -------
    Entity
        The created entity.
    """
```

## 10. 总结

### 10.1 模块职责划分

| 文件 | 职责 | 复杂度 |
|------|------|--------|
| scene.py | 用户接口 + 场景管理 | ⭐⭐⭐⭐⭐ |
| simulator.py | 仿真协调 + 梯度管理 | ⭐⭐⭐⭐ |
| mesh.py | 网格处理 + 优化 | ⭐⭐⭐ |
| force_fields.py | 力场系统 | ⭐⭐ |
| bvh.py | 碰撞加速 | ⭐⭐⭐⭐ |

### 10.2 设计亮点

1. **分层清晰**: Scene → Simulator → Solvers
2. **Facade 模式**: Scene 隐藏复杂性
3. **可扩展**: ForceField 和 Mesh 支持继承
4. **高性能**: BVH + GPU 加速

### 10.3 改进建议

1. Scene.py 过大 (1,530行)，可拆分为:
   - SceneCore (核心逻辑)
   - SceneEntity (实体管理)
   - SceneVis (可视化接口)

2. 增加类型注解和文档

3. 提供更多单元测试

---

**总计**:
- 总行数: 3,616 行
- 核心类数: 10+ 个
- 设计模式: 5+ 种
- 平均每个文件: 723 行
