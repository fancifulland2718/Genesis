# Genesis Engine 架构总览

## 1. 概述

本文档总结了 Genesis 物理引擎 `genesis/engine` 模块的完整架构分析，包括代码风格、设计模式和架构设计。

**分析范围**: 除 solvers 和 couplers 外的所有 genesis/engine 模块

**总代码量**: 约 18,000 行

**分析文档**:
1. [boundaries_architecture_analysis.md](boundaries_architecture_analysis.md) - 边界条件模块
2. [states_architecture_analysis.md](states_architecture_analysis.md) - 状态管理模块
3. [materials_architecture_analysis.md](materials_architecture_analysis.md) - 材料模块
4. [entities_architecture_analysis.md](entities_architecture_analysis.md) - 实体模块
5. [core_files_architecture_analysis.md](core_files_architecture_analysis.md) - 核心文件
6. 本文档 (overview.md) - 总览

## 2. 模块统计

### 2.1 代码量统计

| 模块 | 文件数 | 代码行数 | 类数量 | 主要功能 |
|------|--------|----------|--------|----------|
| **core files** | 5 | 3,616 | 10+ | 场景管理、仿真协调 |
| **entities** | 25 | 10,985 | 20+ | 实体定义和管理 |
| **materials** | 30 | 2,132 | 20+ | 材料属性和本构关系 |
| **states** | 4 | 531 | 14 | 状态快照和管理 |
| **boundaries** | 2 | 74 | 2 | 边界条件 |
| **总计** | **66** | **~18,000** | **66+** | - |

### 2.2 模块复杂度

```
复杂度排名:
1. entities    - ⭐⭐⭐⭐⭐ (最复杂，10,985行)
2. core files  - ⭐⭐⭐⭐⭐ (核心架构，3,616行)
3. materials   - ⭐⭐⭐⭐   (物理模型复杂，2,132行)
4. states      - ⭐⭐⭐     (状态管理，531行)
5. boundaries  - ⭐         (最简单，74行)
```

## 3. 架构总览

### 3.1 分层架构

```
┌─────────────────────────────────────────────────────────┐
│                     用户接口层                           │
│                  Scene (scene.py)                       │
│  - add_entity()                                         │
│  - add_force_field()                                    │
│  - build(), step(), reset()                             │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│                    仿真协调层                            │
│                Simulator (simulator.py)                 │
│  - 多求解器协调                                          │
│  - 时间步进管理                                          │
│  - 梯度追踪                                              │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│                     求解器层                             │
│   Solvers (在 solver_architecture_analysis.md 中分析)   │
│  - RigidSolver, MPMSolver, FEMSolver, etc.              │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│                     实体层                               │
│              Entities (entities/)                       │
│  - RigidEntity, MPMEntity, FEMEntity, etc.              │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│                   数据与工具层                           │
│  Materials  │  States  │  Boundaries  │  Mesh  │  BVH   │
│  材料模型   │  状态管理 │  边界条件    │ 网格   │ 加速   │
└─────────────────────────────────────────────────────────┘
```

### 3.2 模块依赖图

```
Scene
  ├── Simulator
  │   ├── Solvers (8个)
  │   │   └── Entities
  │   │        ├── Materials
  │   │        ├── Mesh
  │   │        └── Boundaries
  │   └── Coupler
  │        └── BVH
  ├── ForceFields
  ├── Visualizer
  └── States
```

### 3.3 数据流

```
用户输入
    ↓
Morph + Material + Surface
    ↓
Entity (实体创建)
    ↓
Solver (物理计算)
    ↓
State (状态快照)
    ↓
Visualizer (渲染输出)
```

## 4. 设计模式总结

### 4.1 创建型模式

#### 4.1.1 Factory Pattern (工厂模式)

**Scene.add_entity()** 根据材料类型创建实体:

```python
def add_entity(self, morph, material, surface):
    if isinstance(material, gs.materials.Rigid):
        return RigidEntity(...)
    elif isinstance(material, gs.materials.MPM.Base):
        return MPMEntity(...)
    elif isinstance(material, gs.materials.FEM.Base):
        return FEMEntity(...)
```

**使用场景**: entities/, core files

#### 4.1.2 Builder Pattern (构建者模式)

**Scene 的分步构建**:

```python
scene = gs.Scene(sim_options=..., rigid_options=...)
scene.add_entity(...)
scene.add_force_field(...)
scene.build()  # 延迟构建
```

**使用场景**: scene.py, entities/

#### 4.1.3 Singleton Pattern (单例模式)

**全局求解器实例**:

```python
class Scene:
    def __init__(self):
        self._sim = Simulator(...)  # 一个场景一个 Simulator
        self._sim.rigid_solver = RigidSolver(...)  # 一个求解器实例
```

**使用场景**: simulator.py

### 4.2 结构型模式

#### 4.2.1 Facade Pattern (外观模式)

**Scene 隐藏内部复杂性**:

```python
# 用户只需与 Scene 交互
scene.add_entity(...)
scene.build()
scene.step()

# 内部协调多个子系统
# - Simulator
# - Visualizer
# - Sensors
# - Recorders
```

**使用场景**: scene.py

#### 4.2.2 Composite Pattern (组合模式)

**RigidEntity 的组件结构**:

```
RigidEntity (整体)
    ├── RigidLink (部分)
    │   └── RigidGeom (部分)
    └── RigidJoint (部分)
```

**使用场景**: entities/rigid_entity/

#### 4.2.3 Wrapper Pattern (包装器模式)

**Mesh 包装 trimesh.Trimesh**:

```python
class Mesh:
    def __init__(self, mesh: trimesh.Trimesh):
        self._mesh = mesh  # 包装第三方库
    
    # 添加额外功能
    def convexify(self):
        ...
```

**使用场景**: mesh.py

#### 4.2.4 Flyweight Pattern (享元模式)

**材料共享**:

```python
material = gs.materials.Rigid(rho=200.0)
entity1 = scene.add_entity(material=material, ...)
entity2 = scene.add_entity(material=material, ...)  # 共享材料
```

**使用场景**: materials/

### 4.3 行为型模式

#### 4.3.1 Template Method Pattern (模板方法模式)

**Entity 的生命周期**:

```python
class Entity:
    def build(self):
        self._load_model()      # 子类实现
        self._add_to_solver()   # 子类实现
        self._is_built = True

class RigidEntity(Entity):
    def _load_model(self):
        # 具体实现
```

**使用场景**: entities/, materials/

#### 4.3.2 Strategy Pattern (策略模式)

**多种本构模型**:

```python
if model == "linear":
    self.update_stress = self.update_stress_linear
elif model == "stable_neohookean":
    self.update_stress = self.update_stress_stable_neohookean
```

**使用场景**: materials/FEM/

#### 4.3.3 Observer Pattern (观察者模式)

**状态变化通知**:

```python
def set_velocity(self, vel):
    self._velocity = vel
    self._solver.mark_dirty(self)  # 通知求解器
```

**使用场景**: entities/

#### 4.3.4 Memento Pattern (备忘录模式)

**状态快照和恢复**:

```python
# 保存状态
state = scene.get_state()
queried_states.append(state)

# 恢复状态
scene.set_state(state)
```

**使用场景**: states/

#### 4.3.5 Mediator Pattern (中介者模式)

**Simulator 作为求解器间的中介**:

```python
class Simulator:
    def step(self):
        for solver in self._active_solvers:
            solver.substep_pre_coupling()
        
        self._coupler.couple()  # 中介协调
        
        for solver in self._active_solvers:
            solver.substep_post_coupling()
```

**使用场景**: simulator.py

## 5. 代码风格规范

### 5.1 命名规范

#### 5.1.1 类名

```python
# PascalCase
Scene, Simulator, Entity, Material

# 模块前缀
RigidEntity, MPMEntity, FEMEntity
MPM.Elastic, FEM.Muscle, PBD.Cloth
```

#### 5.1.2 变量和方法名

```python
# snake_case
def add_entity(self, morph, material, surface):
    pass

# 私有属性: 下划线前缀
self._idx, self._scene, self._material

# 公开属性: @property
@property
def idx(self):
    return self._idx
```

#### 5.1.3 常量

```python
# UPPER_CASE
MAX_PARTICLES = 1000000
DEFAULT_DENSITY = 1000.0

# 枚举
class JOINT_TYPE:
    FIXED = 0
    REVOLUTE = 1
    PRISMATIC = 2
```

#### 5.1.4 Taichi 相关

```python
# Taichi 函数: _kernel_ 或 _func_ 前缀
@ti.kernel
def _kernel_update_particles(self):
    pass

@ti.func
def _func_compute_force(self, pos, vel):
    pass

# Taichi 变量: _ti 后缀
self.pos_ti = ti.Vector(pos, dt=gs.ti_float)
```

### 5.2 文档字符串

**使用 NumPy 风格**:

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
    
    Examples
    --------
    >>> scene = gs.Scene()
    >>> entity = scene.add_entity(
    ...     morph=gs.morphs.Box(size=(1, 1, 1)),
    ...     material=gs.materials.Rigid(rho=1000.0),
    ... )
    """
```

### 5.3 代码组织

#### 5.3.1 类内部结构

```python
class Entity:
    # 1. 构造函数
    def __init__(self, ...):
        pass
    
    # 2. 属性 (@property)
    @property
    def idx(self):
        return self._idx
    
    # 3. 公开方法 (按字母顺序或逻辑顺序)
    def build(self):
        pass
    
    def get_state(self):
        pass
    
    # 4. 私有方法 (_前缀)
    def _load_model(self):
        pass
    
    # 5. Taichi 函数
    @ti.kernel
    def _kernel_update(self):
        pass
```

#### 5.3.2 文件组织

```python
# 1. 导入 (标准库 → 第三方 → 本地)
import os
import pickle

import numpy as np
import trimesh

import genesis as gs
from genesis.engine.base import Entity

# 2. 类型提示 (避免循环导入)
if TYPE_CHECKING:
    from genesis.engine.scene import Scene

# 3. 常量和辅助函数
MAX_PARTICLES = 1000000

def helper_function():
    pass

# 4. 主类
class MainClass:
    pass
```

### 5.4 类型注解

```python
# 函数参数和返回值
def add_entity(
    self,
    morph: Morph,
    material: Material | None = None,
    surface: Surface | None = None,
) -> Entity:
    pass

# 属性注解
class Scene:
    _sim: Simulator
    _entities: list[Entity]
    _is_built: bool
```

## 6. 技术栈

### 6.1 核心依赖

| 技术 | 用途 | 备注 |
|------|------|------|
| **Taichi** | GPU 加速计算 | @ti.data_oriented, @ti.kernel |
| **NumPy** | 数值计算 | 数组操作 |
| **Trimesh** | 网格处理 | 网格加载、凸包、简化 |
| **PyTorch** | 自动微分 | 可微分仿真 |
| **fast_simplification** | 网格简化 | 减少面数 |

### 6.2 设计理念

#### 6.2.1 数据驱动

**SoA (Structure of Arrays)**:

```python
# 不是 AoS (Array of Structures)
particles = [Particle(pos, vel, mass), ...]

# 而是 SoA
pos_field = ti.field(shape=(B, N, 3))  # 位置数组
vel_field = ti.field(shape=(B, N, 3))  # 速度数组
mass_field = ti.field(shape=(B, N))    # 质量数组
```

**批处理优先**:

```python
# 所有操作支持批次维度
base_shape = (sim._B, n_particles)
#             ^^^^^^
#             批次数 (并行环境数)
```

#### 6.2.2 GPU 加速

**Taichi 装饰器**:

```python
@ti.data_oriented  # 类级别
class Solver:
    
    @ti.kernel  # 并行核函数
    def update_particles(self):
        for i in particles:
            # GPU 并行执行
            pass
    
    @ti.func  # 内联函数
    def compute_force(self, pos, vel):
        # 可在 kernel 中调用
        pass
```

#### 6.2.3 可微分仿真

**梯度追踪**:

```python
# 1. 启用梯度
scene = gs.Scene(requires_grad=True)

# 2. 正向仿真
state = scene.get_state()
queried_states.append(state)

# 3. 计算损失
loss = compute_loss(state)

# 4. 反向传播
loss.backward()

# 5. 梯度收集
simulator.collect_output_grads()
```

#### 6.2.4 延迟初始化

**两阶段构建**:

```python
# 阶段 1: 配置 (不分配 GPU 内存)
scene = gs.Scene()
scene.add_entity(...)

# 阶段 2: 构建 (分配 GPU 内存)
scene.build()
```

## 7. 关键特性

### 7.1 多物理场仿真

**支持的物理模型**:

| 求解器 | 物理模型 | 典型应用 |
|--------|----------|----------|
| Rigid | 刚体动力学 | 机器人、碰撞 |
| MPM | 物质点法 | 雪、沙、软体 |
| FEM | 有限元 | 软体、肌肉 |
| PBD | 基于位置动力学 | 布料、绳索 |
| SPH | 光滑粒子流体动力学 | 液体 |
| SF | 烟雾流体 | 烟雾、气体 |
| Avatar | 简化刚体 | 人形角色 |

### 7.2 混合仿真

**刚柔耦合**:

```python
hybrid_robot = scene.add_entity(
    material=gs.materials.Hybrid(
        material_rigid=gs.materials.Rigid(),
        material_soft=gs.materials.MPM.Elastic(),
    ),
    morph=gs.morphs.URDF(file="soft_robot.urdf"),
)
```

### 7.3 并行环境

**批处理仿真**:

```python
# 一次仿真 4 个环境
scene.build(n_envs=4)

# 每个环境独立运行
for i in range(100):
    scene.step()
    
    # 获取所有环境的状态
    states = scene.get_state()  # shape: (4, ...)
```

### 7.4 可扩展性

**添加新材料**:

```python
# 1. 继承基类
class CustomMaterial(gs.materials.MPM.Base):
    @ti.func
    def update_stress(self, ...):
        # 自定义本构关系
        pass

# 2. 使用
material = CustomMaterial(E=1e6, nu=0.2)
entity = scene.add_entity(material=material, ...)
```

**添加新实体类型**:

```python
# 1. 继承基类
class CustomEntity(gs.Entity):
    def _load_model(self):
        # 自定义加载逻辑
        pass
    
    def _add_to_solver(self):
        # 添加到求解器
        pass

# 2. 注册到 Scene
# (需要修改 Scene.add_entity)
```

## 8. 性能优化

### 8.1 内存优化

**共享数据**:
- 材料共享 (Flyweight Pattern)
- 网格共享 (Mesh 对象)

**延迟分配**:
- build() 之前不分配 GPU 内存
- 按需创建状态快照

**SoA 布局**:
- 提高缓存命中率
- 支持 SIMD 向量化

### 8.2 计算优化

**GPU 加速**:
- Taichi 自动并行化
- 内联函数避免开销

**空间加速**:
- BVH 加速碰撞检测 (O(N log N))
- SDF 加速距离查询

**批处理**:
- 并行仿真多个环境
- 单次内核调用处理批次

## 9. 代码质量评估

### 9.1 优点

- ✅ **架构清晰**: 分层设计，职责明确
- ✅ **可扩展**: 继承体系，策略模式
- ✅ **高性能**: GPU 加速，批处理
- ✅ **功能完备**: 支持多种物理模型
- ✅ **用户友好**: Scene 外观模式

### 9.2 缺点

- ⚠️ **文档不足**: 部分类缺少 docstring
- ⚠️ **测试缺失**: 缺少单元测试
- ⚠️ **类型注解不完整**: 部分函数没有类型提示
- ⚠️ **代码重复**: 部分逻辑在多个类中重复
- ⚠️ **耦合度**: Entity 与 Solver 耦合紧密

### 9.3 改进建议

#### 9.3.1 代码结构

1. **拆分大文件**:
   - `scene.py` (1,530行) → SceneCore + SceneEntity + SceneVis
   - `rigid_entity.py` (3,026行) → 多个子模块

2. **提取接口**:
   ```python
   class IEntity(ABC):
       @abstractmethod
       def build(self) -> None: pass
       
       @abstractmethod
       def get_state(self, f: int) -> EntityState: pass
   ```

3. **减少耦合**:
   - Entity 不直接访问 Solver 的内部字段
   - 使用事件系统代替直接调用

#### 9.3.2 文档和测试

1. **补全文档**:
   - 所有公开类和方法添加 docstring
   - 提供使用示例
   - API 参考文档

2. **增加测试**:
   ```python
   def test_add_entity():
       scene = gs.Scene()
       entity = scene.add_entity(
           morph=gs.morphs.Box(size=(1, 1, 1)),
           material=gs.materials.Rigid(rho=1000.0),
       )
       assert entity is not None
       assert entity.idx == 0
   ```

3. **类型注解**:
   - 使用 `mypy` 检查类型
   - 完善所有函数签名

#### 9.3.3 性能

1. **缓存优化**:
   ```python
   @functools.lru_cache(maxsize=128)
   def compute_expensive_property(self):
       pass
   ```

2. **并行构建**:
   - BVH 构建并行化
   - 网格处理并行化

3. **内存池**:
   - 复用粒子内存
   - 减少分配开销

## 10. 总结

### 10.1 架构总结

Genesis Engine 采用**分层架构 + 组件化设计**：

```
用户接口层 (Scene)
    ↓
仿真协调层 (Simulator)
    ↓
求解器层 (Solvers)
    ↓
实体层 (Entities)
    ↓
数据层 (Materials, States, Mesh, etc.)
```

**关键特性**:
- 多物理场仿真
- GPU 加速
- 可微分
- 批处理
- 可扩展

### 10.2 设计模式总结

**使用频率排序**:
1. Template Method (模板方法) - 最常用
2. Strategy (策略) - 常用
3. Factory (工厂) - 常用
4. Composite (组合) - 实体层
5. Facade (外观) - Scene
6. Memento (备忘录) - States

### 10.3 代码风格总结

**命名规范**:
- 类名: PascalCase
- 函数/变量: snake_case
- 私有: `_` 前缀
- Taichi: `_ti` 后缀

**文档风格**: NumPy docstring

**组织风格**: 模块化 + 继承体系

### 10.4 未来发展

**短期改进**:
- 补全文档和测试
- 拆分大文件
- 增加类型注解

**长期规划**:
- 支持更多物理模型
- 提高性能
- 简化用户接口
- 云端部署

---

**完整分析文档**:
- [boundaries_architecture_analysis.md](boundaries_architecture_analysis.md) (74行代码)
- [states_architecture_analysis.md](states_architecture_analysis.md) (531行代码)
- [materials_architecture_analysis.md](materials_architecture_analysis.md) (2,132行代码)
- [entities_architecture_analysis.md](entities_architecture_analysis.md) (10,985行代码)
- [core_files_architecture_analysis.md](core_files_architecture_analysis.md) (3,616行代码)
- [overview.md](overview.md) (本文档)

**总计**: 约 18,000 行代码，66+ 个类，5 个主要模块
