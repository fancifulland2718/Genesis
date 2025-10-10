# Genesis entities 模块架构分析

## 1. 概述

`genesis/engine/entities` 模块定义了仿真系统中的实体类型。实体（Entity）是仿真世界中的基本对象，可以是刚体、软体、流体、粒子系统等。该模块是连接几何形状（Morph）、材料（Material）和求解器（Solver）的桥梁。

**模块位置**: `genesis/engine/entities/`

**代码统计**:
- 总代码量: 10,985 行
- Python 文件数: 25 个
- 实体类型: 10+ 种

**目录结构**:
```
entities/
├── __init__.py                # 导出所有实体类
├── base_entity.py             # 实体基类 (78行)
├── rigid_entity/              # 刚体实体 (4,879行)
│   ├── __init__.py
│   ├── rigid_entity.py        # 主类 (3,026行)
│   ├── rigid_link.py          # 连杆 (736行)
│   ├── rigid_joint.py         # 关节 (456行)
│   ├── rigid_geom.py          # 几何 (1,111行)
│   └── rigid_equality.py      # 约束 (140行)
├── avatar_entity/             # Avatar实体 (260行)
│   ├── __init__.py
│   ├── avatar_entity.py       # 主类 (172行)
│   ├── avatar_link.py         # 连杆 (45行)
│   ├── avatar_joint.py        # 关节
│   └── avatar_geom.py         # 几何 (15行)
├── tool_entity/               # Tool实体 (716行)
│   ├── __init__.py
│   ├── tool_entity.py         # 主类 (507行)
│   └── mesh.py                # 网格 (208行)
├── particle_entity.py         # 粒子实体基类 (854行)
├── mpm_entity.py              # MPM实体 (564行)
├── fem_entity.py              # FEM实体 (1,021行)
├── pbd_entity.py              # PBD实体 (667行)
├── sph_entity.py              # SPH实体 (185行)
├── sf_entity.py               # SF实体 (36行)
├── hybrid_entity.py           # 混合实体 (724行)
├── drone_entity.py            # 无人机实体 (132行)
└── emitter.py                 # 发射器 (277行)
```

## 2. 架构设计

### 2.1 继承体系

```
                   Entity (基类)
                      │
        ┌─────────────┼─────────────┐
        │             │             │
    RigidEntity  ParticleEntity  AvatarEntity
        │             │             
        │        ┌────┴────┐
        │        │         │
        │    MPMEntity  SPHEntity
        │        │         │
        │    FEMEntity  PBDEntity
        │        
   DroneEntity (继承 RigidEntity)
   
   ToolEntity (独立分支)
   HybridEntity (组合 Rigid + MPM)
   Emitter (发射粒子)
   SFParticleEntity
```

### 2.2 组件化设计

**RigidEntity 的组件**:
```
RigidEntity
    ├── RigidLink (连杆)
    │   ├── pos, quat (位置、旋转)
    │   ├── mass, inertia (质量、惯性)
    │   └── geoms (几何体列表)
    ├── RigidJoint (关节)
    │   ├── type (关节类型)
    │   ├── axis (关节轴)
    │   └── limits (关节限制)
    ├── RigidGeom (几何体)
    │   ├── type (几何类型)
    │   ├── mesh (网格)
    │   └── friction (摩擦系数)
    └── RigidEquality (约束)
        ├── type (约束类型)
        └── params (约束参数)
```

### 2.3 数据流

```
Morph (几何描述)
    ↓
Entity._load_model()
    ↓
Entity Components (Links, Joints, Geoms)
    ↓
Entity._add_to_solver()
    ↓
Solver (物理仿真)
    ↓
Entity.get_state() → EntityState
```

## 3. 核心类详解

### 3.1 Entity - 实体基类

```python
@ti.data_oriented
class Entity(RBC):
    """所有实体的基类"""
    
    def __init__(self, idx, scene, morph, solver, material, surface):
        self._uid = gs.UID()        # 唯一标识符
        self._idx = idx             # 在场景中的索引
        self._scene = scene         # 所属场景
        self._solver = solver       # 所属求解器
        self._material = material   # 材料
        self._morph = morph         # 几何形状
        self._surface = surface     # 表面属性
        self._sim = scene.sim       # 仿真器
```

**职责**:
- 存储实体的基本属性
- 提供统一的属性访问接口
- 记录日志

**设计特点**:
- 使用 `@ti.data_oriented` 支持 GPU 加速
- 继承 `RBC` (ReprBaseClass) 提供统一的字符串表示
- 只读属性通过 `@property` 装饰器

### 3.2 RigidEntity - 刚体实体

**最复杂的实体类型 (3,026行)**

```python
@ti.data_oriented
class RigidEntity(Entity):
    """刚体实体：机器人、地形、漂浮物体等"""
    
    def __init__(self, scene, solver, material, morph, surface, idx, ...):
        super().__init__(idx, scene, morph, solver, material, surface)
        
        # 索引管理
        self._link_start = link_start
        self._joint_start = joint_start
        self._geom_start = geom_start
        
        # 组件列表
        self._links = gs.List()
        self._joints = gs.List()
        self._equalities = gs.List()
        
        # 加载模型
        self._load_model()
```

**支持的 Morph 类型**:
1. **URDF/MJCF**: 机器人描述文件
2. **Mesh**: 自定义网格
3. **Primitive**: 原始几何体 (Box, Sphere, Cylinder, Capsule)
4. **Terrain**: 地形

**加载流程**:
```python
def _load_model(self):
    if isinstance(self._morph, gs.morphs.Mesh):
        self._load_mesh(self._morph, self._surface)
    elif isinstance(self._morph, (gs.morphs.MJCF, gs.morphs.URDF)):
        self._load_scene(self._morph, self._surface)
    elif isinstance(self._morph, gs.morphs.Primitive):
        self._load_primitive(self._morph, self._surface)
    elif isinstance(self._morph, gs.morphs.Terrain):
        self._load_terrain(self._morph, self._surface)
```

**组件类**:

#### RigidLink - 连杆

```python
class RigidLink:
    def __init__(self, entity, idx, idx_local, name, ...):
        self._entity = entity
        self._idx = idx              # 全局索引
        self._idx_local = idx_local  # 实体内索引
        self._name = name
        
        # 物理属性
        self._mass = mass
        self._inertia = inertia
        self._pos = pos
        self._quat = quat
        
        # 拓扑关系
        self._parent_idx = parent_idx
        self._child_idxs = []
        
        # 几何体
        self._geoms = []
```

#### RigidJoint - 关节

```python
class RigidJoint:
    def __init__(self, entity, idx, idx_local, name, type, ...):
        self._type = type  # FIXED, FREE, REVOLUTE, PRISMATIC, etc.
        self._axis = axis
        self._limit_lower = limit_lower
        self._limit_upper = limit_upper
        self._stiffness = stiffness
        self._damping = damping
```

**关节类型**:
- `FIXED`: 固定关节 (焊接)
- `FREE`: 自由关节 (6 DOF)
- `REVOLUTE`: 旋转关节 (1 DOF)
- `PRISMATIC`: 平移关节 (1 DOF)
- `BALL`: 球形关节 (3 DOF)
- `PLANAR`: 平面关节 (3 DOF)

#### RigidGeom - 几何体

```python
class RigidGeom:
    def __init__(self, entity, idx, idx_local, type, ...):
        self._type = type  # BOX, SPHERE, CYLINDER, MESH, etc.
        self._pos = pos
        self._quat = quat
        self._friction = friction
        self._mesh = mesh  # 如果是 MESH 类型
```

### 3.3 ParticleEntity - 粒子实体基类

```python
@ti.data_oriented
class ParticleEntity(Entity):
    """粒子系统实体的基类"""
    
    def __init__(self, scene, solver, material, morph, surface, 
                 particle_size, idx, particle_start, ...):
        super().__init__(idx, scene, morph, solver, material, surface)
        
        self._particle_size = particle_size
        self._particle_start = particle_start
        self._n_particles = 0
        
        # 粒子数据
        self._particles = None  # numpy 数组
        
        # 采样粒子
        self._sample_particles()
```

**粒子采样**:
```python
def _sample_particles(self):
    if isinstance(self._morph, gs.morphs.Mesh):
        # 网格体素化
        self._particles = pu.sample_mesh_interior(
            mesh, self._particle_size, sampler=self.material.sampler
        )
    elif isinstance(self._morph, gs.morphs.Box):
        # 规则网格采样
        self._particles = pu.sample_box(...)
```

### 3.4 MPMEntity - MPM 粒子实体

```python
@ti.data_oriented
class MPMEntity(ParticleEntity):
    """MPM (Material Point Method) 实体"""
    
    def init_tgt_keys(self):
        if isinstance(self.material, gs.materials.MPM.Muscle):
            self._tgt_keys = ("pos", "vel", "act", "actu")
        else:
            self._tgt_keys = ("pos", "vel", "act")
```

**特殊功能**:

1. **肌肉激活** (仅限 Muscle 材料):
```python
@assert_muscle
def set_muscle_group(self, group_ids):
    """设置肌肉分组"""
    self._muscle_group = group_ids

@assert_muscle
def set_actuation(self, actuation, envs_idx=None):
    """设置肌肉激活度 (0.0~1.0)"""
    self._kernel_set_actuation(actuation, envs_idx)
```

2. **蒙皮渲染** (软体可变形):
```python
if need_skinning:
    self._setup_skinning(mesh)
    # 运行时更新顶点位置
```

### 3.5 FEMEntity - FEM 有限元实体

```python
@ti.data_oriented
class FEMEntity(Entity):
    """FEM (Finite Element Method) 实体"""
    
    def __init__(self, scene, solver, material, morph, surface, idx, ...):
        super().__init__(idx, scene, morph, solver, material, surface)
        
        # 网格数据
        self._n_vertices = n_vertices
        self._n_elements = n_elements
        self._vertices = vertices      # 顶点位置
        self._elements = elements      # 四面体索引
        
        # 四面体化
        self._tetrahedralize()
```

**四面体化**:
```python
def _tetrahedralize(self):
    if isinstance(self._morph, gs.morphs.Mesh):
        # 使用 TetGen 或类似工具
        self._vertices, self._elements = mu.tetrahedralize_mesh(mesh)
```

**肌肉支持**:
```python
@assert_muscle
def set_muscle_direction(self, direction):
    """设置肌肉纤维方向"""
    self._muscle_direction = direction

@assert_muscle
def set_actuation(self, actuation, envs_idx=None):
    """设置肌肉激活度"""
    # 在应力计算中添加主动收缩力
```

### 3.6 PBDEntity - PBD 基于位置动力学实体

```python
@ti.data_oriented
class PBD3DEntity(ParticleEntity):
    """3D 可变形体 (四面体网格)"""
    
    def __init__(self, scene, solver, material, morph, surface, ...):
        super().__init__(...)
        
        # 拓扑约束
        self._edges = None       # 边
        self._tetras = None      # 四面体
        
        # 空间约束
        self._add_topology_constraints()
```

**约束类型**:
1. **拓扑约束**:
   - Edge Constraint (边长约束)
   - Tetrahedral Volume Constraint (体积约束)
   - Bending Constraint (弯曲约束)

2. **空间约束**:
   - Collision Constraint (碰撞约束)
   - Attachment Constraint (附着约束)

### 3.7 HybridEntity - 混合实体

```python
@ti.data_oriented
class HybridEntity(Entity):
    """刚柔耦合实体 (如软体机器人)"""
    
    def __init__(self, idx, scene, material, morph, surface):
        super().__init__(idx, scene, morph, None, material, surface)
        
        # 刚体部分
        self._part_rigid = scene.add_entity(
            material=material.material_rigid,
            morph=morph,
            surface=surface_rigid,
        )
        
        # 软体部分
        self._part_soft = self._instantiate_soft_from_rigid(
            part_rigid=self._part_rigid,
            material_soft=material.material_soft,
        )
        
        # 耦合
        self._setup_coupling()
```

**耦合机制**:
```python
def _setup_coupling(self):
    # 1. 识别接触点
    coupling_links = material.coupling_links
    
    # 2. 为软体粒子附着到刚体
    for particle in soft_particles:
        link = find_nearest_link(particle.pos)
        if link in coupling_links:
            attach_particle_to_link(particle, link)
    
    # 3. 双向力传递
    # 刚体 → 软体: 位置约束
    # 软体 → 刚体: 接触力
```

### 3.8 AvatarEntity - Avatar 实体

```python
@ti.data_oriented
class AvatarEntity(Entity):
    """Avatar 实体 (人形角色、动物等)"""
    
    def __init__(self, idx, scene, solver, material, morph, surface, ...):
        super().__init__(idx, scene, morph, solver, material, surface)
        
        # 从 URDF/MJCF 加载
        self._load_model()
        
        # Avatar-specific 数据
        self._links = []
        self._joints = []
```

**与 RigidEntity 的区别**:
- 简化的碰撞检测
- 专注于动画和运动控制
- 不参与复杂的物理交互

### 3.9 ToolEntity - 工具实体

```python
@ti.data_oriented
class ToolEntity(Entity):
    """工具实体 (用于单向刚-柔耦合)"""
    
    def __init__(self, scene, solver, material, morph, surface, idx, ...):
        super().__init__(idx, scene, morph, solver, material, surface)
        
        # 网格 SDF
        self._mesh = ToolMesh(mesh, material.sdf_cell_size)
```

**应用场景**:
- 刀具切割软体
- 夹爪抓取可变形物体
- 外部工具操控

### 3.10 Emitter - 发射器

```python
@ti.data_oriented
class Emitter(Entity):
    """粒子发射器 (持续生成新粒子)"""
    
    def __init__(self, scene, solver, material, morph, surface, idx, ...):
        super().__init__(idx, scene, morph, solver, material, surface)
        
        self._emit_rate = emit_rate
        self._emit_velocity = emit_velocity
    
    def emit(self):
        """每帧发射新粒子"""
        new_particles = self._generate_particles()
        self._solver.add_particles(new_particles)
```

## 4. 代码风格分析

### 4.1 命名规范

```python
# 类名: PascalCase
RigidEntity, MPMEntity, ParticleEntity

# 属性: snake_case 带下划线前缀
self._idx, self._scene, self._n_particles

# 公开属性: 通过 @property
@property
def idx(self):
    return self._idx

# 组件列表: 复数形式
self._links, self._joints, self._geoms

# 索引: _start 后缀表示起始位置
self._link_start, self._particle_start
```

### 4.2 装饰器模式

```python
# 数据导向装饰器 (所有实体类)
@ti.data_oriented
class Entity:
    pass

# 断言装饰器 (条件方法)
@assert_active
def set_velocity(self, vel):
    # 只有激活的实体才能设置速度
    pass

@assert_muscle
def set_actuation(self, actu):
    # 只有肌肉材料才支持激活
    pass
```

### 4.3 工厂方法模式

```python
def _load_model(self):
    """根据 Morph 类型选择加载方法"""
    if isinstance(self._morph, gs.morphs.Mesh):
        return self._load_mesh()
    elif isinstance(self._morph, gs.morphs.URDF):
        return self._load_scene()
    elif isinstance(self._morph, gs.morphs.Primitive):
        return self._load_primitive()
```

### 4.4 延迟初始化

```python
def build(self):
    """实体在 scene.build() 时才真正初始化"""
    if self._is_built:
        return
    
    self._add_to_solver()
    self._allocate_buffers()
    self._is_built = True
```

## 5. 设计模式

### 5.1 Composite Pattern (组合模式)

```
RigidEntity (整体)
    ├── RigidLink (部分)
    │   └── RigidGeom (部分)
    └── RigidJoint (部分)
```

实体由多个组件组成，可递归访问。

### 5.2 Strategy Pattern (策略模式)

不同的粒子采样策略:
```python
if sampler == "pbs":
    particles = sample_pbs(mesh)
elif sampler == "random":
    particles = sample_random(mesh)
elif sampler == "regular":
    particles = sample_regular(mesh)
```

### 5.3 Template Method Pattern (模板方法)

```python
class Entity:
    def build(self):
        # 模板方法
        self._load_model()      # 子类实现
        self._add_to_solver()   # 子类实现
        self._is_built = True

class RigidEntity(Entity):
    def _load_model(self):
        # 具体实现
        ...
```

### 5.4 Observer Pattern (观察者模式)

实体状态变化通知求解器:
```python
def set_velocity(self, vel):
    self._velocity = vel
    self._solver.mark_dirty(self)  # 通知求解器更新
```

### 5.5 Builder Pattern (构建者模式)

复杂实体的分步构建:
```python
entity = scene.add_entity(morph=..., material=..., surface=...)
# 此时实体未完全构建

scene.build()  # 构建所有实体
# 此时实体完全构建
```

## 6. 数据结构

### 6.1 索引管理

**分层索引**:
```python
# 全局索引 (在求解器中)
self._idx = 5

# 局部索引 (在实体内)
link._idx_local = 2

# 起始索引 (在求解器缓冲区中)
self._particle_start = 1000
```

**索引映射**:
```
Scene Entities: [0, 1, 2, 3, 4]
    ↓
Solver Entities: [0, 1, 2] (只包含该求解器的实体)
    ↓
Solver Particles: [0...999, 1000...1499, 1500...2000]
                       ↑
                entity._particle_start = 1000
```

### 6.2 批处理布局

**SoA (Structure of Arrays)**:
```python
# 不是
entities = [Entity1(pos, vel), Entity2(pos, vel), ...]

# 而是
pos_field = ti.field(shape=(B, N, 3))  # 所有位置
vel_field = ti.field(shape=(B, N, 3))  # 所有速度
```

**批次维度**:
```python
base_shape = (sim._B, n_particles)
#             ^^^^^^
#             批次数 (并行环境数)
```

### 6.3 树形结构

**RigidEntity 的关节树**:
```
root_link (idx=0)
    ├── child_link_1 (idx=1)
    │   └── child_link_2 (idx=3)
    └── child_link_3 (idx=2)
```

**遍历方法**:
```python
def traverse_tree(link):
    visit(link)
    for child_idx in link.child_idxs:
        traverse_tree(links[child_idx])
```

## 7. 状态管理

### 7.1 状态获取

```python
def get_state(self, f):
    """获取实体在帧 f 的状态"""
    state = EntityState(self, s_global)
    self._kernel_get_state(state, f)
    return state
```

### 7.2 状态设置

```python
def set_state(self, f, state, envs_idx=None):
    """设置实体到指定状态"""
    self._kernel_set_state(state, f, envs_idx)
```

### 7.3 检查点

```python
def save_ckpt(self, ckpt_name):
    """保存检查点"""
    self._ckpts[ckpt_name] = self.get_state(f=0)

def load_ckpt(self, ckpt_name):
    """加载检查点"""
    state = self._ckpts[ckpt_name]
    self.set_state(f=0, state=state)
```

## 8. 性能优化

### 8.1 内存优化

**共享几何数据**:
```python
# 多个实体共享同一网格
mesh = gs.Mesh(file="cube.obj")
entity1 = scene.add_entity(morph=gs.morphs.Mesh(mesh=mesh))
entity2 = scene.add_entity(morph=gs.morphs.Mesh(mesh=mesh))
# mesh 只存储一次
```

**延迟分配**:
```python
def build(self):
    # 只有在 build 时才分配 GPU 内存
    self._allocate_buffers()
```

### 8.2 GPU 加速

**Taichi 内核**:
```python
@ti.kernel
def _kernel_set_velocity(self, vel: ti.types.ndarray(), envs_idx: ti.types.ndarray()):
    for i_b_ in range(envs_idx.shape[0]):
        i_b = envs_idx[i_b_]
        for i_p in range(self._n_particles):
            idx = self._particle_start + i_p
            self._solver.particles_vel[i_b, idx] = vel[i_b_, i_p]
```

### 8.3 批处理

**并行环境**:
```python
# 一次仿真 B 个环境
vel = torch.zeros((B, n_particles, 3))
entity.set_velocity(vel, envs_idx=torch.arange(B))
```

## 9. 可扩展性

### 9.1 添加新实体类型

**步骤**:
1. 继承 `Entity` 或其子类
2. 实现 `_load_model()`
3. 实现 `_add_to_solver()`
4. 实现 `get_state()` 和 `set_state()`

**示例**:
```python
@ti.data_oriented
class CustomEntity(Entity):
    def __init__(self, ...):
        super().__init__(...)
        self._custom_data = ...
    
    def _load_model(self):
        # 加载自定义模型
        pass
    
    def _add_to_solver(self):
        # 添加到求解器
        self._solver.add_custom_entity(self)
```

### 9.2 当前限制

- 实体类型与求解器强耦合
- 缺少通用的实体接口
- 难以支持多求解器实体

### 9.3 改进建议

**接口抽象**:
```python
class IEntity(ABC):
    @abstractmethod
    def build(self) -> None:
        pass
    
    @abstractmethod
    def get_state(self, f: int) -> EntityState:
        pass
    
    @abstractmethod
    def set_state(self, f: int, state: EntityState) -> None:
        pass
```

## 10. 典型使用流程

### 10.1 创建刚体机器人

```python
scene = gs.Scene()

robot = scene.add_entity(
    morph=gs.morphs.URDF(file="robot.urdf"),
    material=gs.materials.Rigid(rho=1000.0, friction=0.5),
    surface=gs.surfaces.Default(),
)

scene.build()

# 控制机器人
robot.set_dofs_position([0.0, 0.5, ...])
robot.set_dofs_velocity([0.0, 0.0, ...])
```

### 10.2 创建软体

```python
soft_body = scene.add_entity(
    morph=gs.morphs.Mesh(file="bunny.obj"),
    material=gs.materials.MPM.Elastic(E=1e6, nu=0.2),
    surface=gs.surfaces.Default(),
)

scene.build()

# 施加外力
soft_body.set_velocity(velocity)
```

### 10.3 创建混合机器人

```python
hybrid_robot = scene.add_entity(
    morph=gs.morphs.URDF(file="soft_robot.urdf"),
    material=gs.materials.Hybrid(
        material_rigid=gs.materials.Rigid(),
        material_soft=gs.materials.MPM.Elastic(),
        coupling_links=["link1", "link2"],
    ),
    surface=gs.surfaces.Default(),
)
```

## 11. 与其他模块的集成

### 11.1 与 Solver 的关系

```
Entity
    ↓ _add_to_solver()
Solver._entities.append(entity)
    ↓ build()
Solver._allocate_buffers()
    ↓ step()
Solver._update_entities()
```

### 11.2 与 Material 的关系

```
Entity.material → Material
    ↓ 物理参数
Solver.compute_forces()
    ↓ 本构关系
Material.update_stress()
```

### 11.3 与 Morph 的关系

```
Entity.morph → Morph
    ↓ 几何描述
Entity._load_model()
    ↓ 采样/网格化
Entity Components (particles, vertices, faces)
```

## 12. 代码质量评估

### 12.1 优点

- ✅ 清晰的继承体系
- ✅ 组件化设计
- ✅ 支持多种实体类型
- ✅ GPU 加速
- ✅ 批处理友好

### 12.2 缺点

- ⚠️ RigidEntity 过于复杂 (3,026行)
- ⚠️ 缺少统一接口
- ⚠️ 文档不够完善
- ⚠️ 缺少单元测试
- ⚠️ 与求解器耦合紧密

### 12.3 复杂度分析

**最复杂的类**:
1. RigidEntity (3,026行) - 支持多种模型格式
2. RigidGeom (1,111行) - 复杂的碰撞检测
3. FEMEntity (1,021行) - 四面体网格处理

**最简单的类**:
1. SFParticleEntity (36行) - 烟雾粒子
2. Avatar 相关类 (< 200行) - 简化的刚体

## 13. 总结

### 13.1 模块职责

entities 模块是 Genesis 仿真系统的"演员"：
- 封装物理对象
- 连接几何、材料和求解器
- 提供状态管理接口
- 支持多种物理模型

### 13.2 设计亮点

1. **多样性**: 支持刚体、软体、流体、混合等多种类型
2. **组件化**: RigidEntity 的 Link-Joint-Geom 结构
3. **可扩展**: 清晰的继承体系
4. **高性能**: GPU 加速 + 批处理

### 13.3 架构总结

```
用户
  ↓ add_entity()
Scene
  ↓ 创建
Entity (几何 + 材料)
  ↓ _add_to_solver()
Solver (物理仿真)
  ↓ step()
Entity.get_state() → 状态快照
```

### 13.4 未来改进方向

1. **接口统一**: 定义 `IEntity` 抽象接口
2. **解耦**: 减少与求解器的直接依赖
3. **模块化**: 拆分 RigidEntity 为多个子模块
4. **文档**: 补充完整的 API 文档
5. **测试**: 增加单元测试和集成测试

---

**代码统计**:
- 总行数: 10,985 行
- 实体类数: 10+ 种
- 组件类数: 10+ 种
- 最大文件: rigid_entity.py (3,026行)
- 平均每个实体类: ~500 行
