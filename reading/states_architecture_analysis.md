# Genesis states 模块架构分析

## 1. 概述

`genesis/engine/states` 模块负责管理仿真系统的状态快照和状态查询功能。该模块是实现可微分仿真、时间回溯和状态检查点的关键基础设施。

**模块位置**: `genesis/engine/states/`

**核心文件**:
- `solvers.py` (297行) - 求解器级别的状态类
- `entities.py` (190行) - 实体级别的状态类
- `cache.py` (44行) - 状态缓存和查询管理
- `__init__.py` (0行) - 模块导出

**总代码量**: 531 行

## 2. 核心依赖关系

```
states
    ├── genesis core (gs)
    ├── genesis.repr_base (RBC)
    ├── Taichi (间接，通过实体和求解器)
    └── Tensor (gs.zeros, 支持自动微分)
```

## 3. 模块架构

### 3.1 三层状态体系

```
┌──────────────────────────────────────┐
│         SimState (最高层)             │
│  包含所有求解器的状态快照              │
│  - s_global (全局步数)                │
│  - solvers_state (求解器状态列表)     │
└──────────────────────────────────────┘
              ↓
┌──────────────────────────────────────┐
│      Solver State (中间层)            │
│  ToolSolverState                     │
│  RigidSolverState                    │
│  AvatarSolverState                   │
│  MPMSolverState                      │
│  SPHSolverState                      │
│  FEMSolverState                      │
│  PBDSolverState                      │
│  SFSolverState                       │
└──────────────────────────────────────┘
              ↓
┌──────────────────────────────────────┐
│      Entity State (底层)              │
│  ToolEntityState                     │
│  MPMEntityState                      │
│  SPHEntityState                      │
│  FEMEntityState                      │
└──────────────────────────────────────┘
```

### 3.2 状态缓存机制

```
┌───────────────────────────┐
│    QueriedStates          │
│  (字典: s_global -> list) │
├───────────────────────────┤
│ + append(state)           │
│ + clear()                 │
│ + __contains__(key)       │
│ + __getitem__(key)        │
└───────────────────────────┘
         ↕
┌───────────────────────────┐
│      StateList            │
│  (list 的子类)            │
├───────────────────────────┤
│ + __getitem__(slice)      │
│ + __repr__()              │
└───────────────────────────┘
```

## 4. 核心类详解

### 4.1 SimState - 仿真状态

**功能**: 捕获整个仿真器在某一时刻的完整状态

```python
class SimState(RBC):
    def __init__(self, scene, s_global, f_local, solvers):
        self._scene = scene
        self._s_global = s_global  # 全局步数
        self._solvers_state = [solver.get_state(f_local) for solver in solvers]
```

**属性**:
- `scene`: 反向引用场景对象
- `s_global`: 全局时间步索引
- `solvers_state`: 所有求解器的状态列表

**方法**:
- `serializable()`: 断开与场景的引用，准备序列化
- `__iter__()`: 支持迭代求解器状态

**设计特点**:
- 使用 `RBC` 基类 (ReprBaseClass) 提供统一的 repr
- 懒加载：状态从求解器按需获取
- 支持序列化：可保存到磁盘

### 4.2 Solver State 类族

#### 4.2.1 MPMSolverState - MPM 求解器状态

```python
class MPMSolverState(RBC):
    def __init__(self, scene):
        # 为每个实体存储状态
        self.entities = []
```

**存储内容**:
- 实体状态列表 (`MPMEntityState`)
- 粒子位置、速度、变形梯度等

**特点**:
- 支持索引访问 `state[i]`
- 支持长度查询 `len(state)`
- 可迭代

#### 4.2.2 AvatarSolverState - Avatar 求解器状态

```python
class AvatarSolverState:
    def __init__(self, scene):
        self.scene = scene
        self.qpos = None        # 关节位置
        self.dofs_vel = None    # 自由度速度
        self.links_pos = None   # 连杆位置
        self.links_quat = None  # 连杆四元数
```

**可微分支持**:
```python
def serializable(self):
    self.qpos = self.qpos.detach()
    self.dofs_vel = self.dofs_vel.detach()
    # ...
```

使用 `detach()` 断开计算图，避免梯度泄漏。

#### 4.2.3 ToolSolverState - Tool 求解器状态

**特殊性**: 用于临时工具（刚体到柔体的单向耦合）

```python
class ToolSolverState:
    def __init__(self, scene):
        self.scene = scene
        self.entities = []  # ToolEntityState 列表
```

**设计特点**:
- 结构简单，只包含实体列表
- 支持批处理多个工具实体

### 4.3 Entity State 类族

#### 4.3.1 MPMEntityState - MPM 实体状态

```python
class MPMEntityState(RBC):
    def __init__(self, entity, s_global):
        base_shape = (entity.sim._B, entity.n_particles)
        
        # 创建可微分张量
        args = {
            "dtype": gs.tc_float,
            "requires_grad": entity.scene.requires_grad,
            "scene": entity.scene,
        }
        
        self._pos = gs.zeros(base_shape + (3,), **args)
        self._vel = gs.zeros(base_shape + (3,), **args)
        self._C = gs.zeros(base_shape + (3, 3), **args)
        self._F = gs.zeros(base_shape + (3, 3), **args)
        self._Jp = gs.zeros(base_shape, **args)
        
        # 不可微分的活跃标记
        self._active = gs.zeros(base_shape, dtype=int, requires_grad=False)
```

**物理量**:
- `pos`: 粒子位置
- `vel`: 粒子速度
- `C`: APIC 仿射矩阵
- `F`: 变形梯度
- `Jp`: 塑性体积变化

**属性封装**:
```python
@property
def pos(self):
    return self._pos
```

所有物理量通过 `@property` 只读访问，保护内部状态。

#### 4.3.2 ToolEntityState - Tool 实体状态

```python
class ToolEntityState:
    def __init__(self, entity, s_global):
        args = {
            "dtype": gs.tc_float,
            "requires_grad": entity.scene.requires_grad,
            "scene": entity.scene,
        }
        
        self.pos = gs.zeros((entity.sim._B, 3), **args)
        self.quat = gs.zeros((entity.sim._B, 4), **args)
        self.vel = gs.zeros((entity.sim._B, 3), **args)
        self.ang = gs.zeros((entity.sim._B, 3), **args)
```

**刚体六自由度**:
- `pos`: 位置 (3D)
- `quat`: 旋转四元数 (4D)
- `vel`: 线速度 (3D)
- `ang`: 角速度 (3D)

#### 4.3.3 FEMEntityState - FEM 实体状态

```python
class FEMEntityState:
    def __init__(self, entity, s_global):
        base_shape = (entity.sim._B, entity.n_vertices, 3)
        
        self._pos = gs.zeros(base_shape, **args)
        self._vel = gs.zeros(base_shape, **args)
        self._active = gs.zeros((entity.sim._B, entity.n_elements), dtype=int)
```

**网格顶点状态**:
- `pos`: 顶点位置
- `vel`: 顶点速度
- `active`: 元素活跃标记

### 4.4 状态缓存管理

#### 4.4.1 QueriedStates - 查询状态字典

```python
class QueriedStates:
    def __init__(self):
        self.states = dict()  # {s_global: StateList([state1, state2, ...])}
    
    def append(self, state):
        if state.s_global not in self.states:
            self.states[state.s_global] = StateList([state])
        else:
            self.states[state.s_global].append(state)
```

**功能**:
- 按时间步索引状态
- 支持同一时间步多个查询（不同用途）
- 用于反向传播时收集梯度

**使用场景**:
```python
# 正向模拟
state = scene.get_state()
queried_states.append(state)

# 反向传播
if s_global in queried_states:
    for state in queried_states[s_global]:
        solver.add_grad_from_state(state)
```

#### 4.4.2 StateList - 状态列表

```python
class StateList(list):
    def __getitem__(self, item):
        result = super().__getitem__(item)
        if isinstance(item, slice):
            return StateList(result)  # 切片返回 StateList
        else:
            return result  # 单个元素返回原始对象
```

**特性**:
- 继承自 `list`，保持兼容性
- 切片操作返回 `StateList` 而非普通 `list`
- 自定义 `__repr__` 显示类型信息

## 5. 代码风格分析

### 5.1 命名规范

```python
# 状态类: [Solver/Entity]State
SimState, MPMSolverState, ToolEntityState

# 私有属性: 下划线前缀
self._scene, self._s_global, self._pos

# 公开属性: 无下划线 (或通过 property)
self.scene, self.entities

# 属性访问器: @property 装饰器
@property
def pos(self):
    return self._pos
```

### 5.2 类型提示

```python
# 使用 TYPE_CHECKING 避免循环导入
if TYPE_CHECKING:
    from genesis.engine.scene import Scene
```

### 5.3 数据初始化模式

**统一的张量创建参数**:
```python
args = {
    "dtype": gs.tc_float,
    "requires_grad": entity.scene.requires_grad,
    "scene": entity.scene,
}

self._pos = gs.zeros(shape, **args)
self._vel = gs.zeros(shape, **args)
```

这种模式:
- 集中管理参数
- 确保一致性
- 易于修改

### 5.4 批处理设计

所有状态都支持批处理:
```python
base_shape = (entity.sim._B, entity.n_particles)
#             ^^^^^^^^^^^^^^^^
#             批次维度
```

`_B` 是批次大小，支持并行仿真多个环境。

## 6. 设计模式

### 6.1 Snapshot Pattern (快照模式)

状态对象是不可变快照:
- 捕获某一时刻的完整状态
- 可序列化保存
- 可用于时间回溯

### 6.2 Composite Pattern (组合模式)

```
SimState
  └─ List[SolverState]
       └─ List[EntityState]
```

树形结构，逐层包含。

### 6.3 Memento Pattern (备忘录模式)

`QueriedStates` 维护状态历史:
```python
# 保存状态
queried_states.append(state)

# 恢复状态
if s_global in queried_states:
    scene.set_state(queried_states[s_global][0])
```

### 6.4 Lazy Initialization (懒加载)

状态从求解器和实体按需获取:
```python
# SimState 中
self._solvers_state = [solver.get_state(f_local) for solver in solvers]
```

避免不必要的内存分配。

## 7. 可微分支持

### 7.1 梯度追踪

**启用梯度**:
```python
args = {
    "requires_grad": entity.scene.requires_grad,
    # ...
}
self._pos = gs.zeros(shape, **args)
```

**断开梯度**:
```python
def serializable(self):
    self._pos = self._pos.detach()
    self._vel = self._vel.detach()
```

### 7.2 反向传播流程

```
1. 正向模拟
   scene.step()
   state = scene.get_state()
   queried_states.append(state)

2. 计算损失
   loss = compute_loss(state)

3. 反向传播
   loss.backward()
   
4. 梯度收集
   simulator.collect_output_grads()
   └─ for state in queried_states[s_global]:
          solver.add_grad_from_state(state)
```

### 7.3 选择性梯度

某些物理量不参与梯度:
```python
# 活跃标记不需要梯度
args["dtype"] = int
args["requires_grad"] = False
self._active = gs.zeros(base_shape, **args)
```

## 8. 序列化机制

### 8.1 serializable() 方法

**作用**: 准备状态对象以便序列化

```python
def serializable(self):
    # 1. 断开场景引用 (避免循环引用)
    self.scene = None
    
    # 2. 断开计算图
    self.pos = self.pos.detach()
    
    # 3. 递归调用子对象
    for entity_state in self.entities:
        entity_state.serializable()
```

### 8.2 使用场景

```python
# 保存检查点
state = scene.get_state()
state.serializable()
pickle.dump(state, file)

# 加载检查点
state = pickle.load(file)
scene.set_state(state)
```

## 9. 性能优化

### 9.1 内存管理

**按需分配**:
- 只有查询状态时才创建状态对象
- 使用 `gs.zeros` 延迟初始化

**批处理**:
- 所有状态支持批次维度
- 单次内存分配处理多个环境

### 9.2 避免拷贝

状态对象存储张量引用而非拷贝:
```python
# 不是 self.pos = entity.pos.clone()
# 而是在 get_state 时从 Taichi field 拷贝到 tensor
```

## 10. 与其他模块的集成

### 10.1 Simulator 中的使用

```python
class Simulator:
    def get_state(self, f):
        return SimState(
            scene=self._scene,
            s_global=self.f_global_to_s_global(f),
            f_local=self.f_global_to_f_local(f),
            solvers=self._solvers,
        )
```

### 10.2 Solver 中的使用

```python
class MPMSolver:
    def get_state(self, f):
        state = MPMSolverState(self.scene)
        for entity in self._entities:
            entity_state = entity.get_state(f)
            state.entities.append(entity_state)
        return state
```

### 10.3 Entity 中的使用

```python
class MPMEntity:
    def get_state(self, f):
        state = MPMEntityState(self, s_global)
        # 从 Taichi field 拷贝到 state tensor
        self._kernel_get_state(state, f)
        return state
```

## 11. 扩展性分析

### 11.1 添加新状态类型

**步骤**:
1. 在 `entities.py` 中定义 `NewEntityState`
2. 在 `solvers.py` 中定义 `NewSolverState`
3. 在 Solver 中实现 `get_state()` 和 `set_state()`
4. 在 Entity 中实现 `get_state()` 和 `set_state()`

**示例**:
```python
# entities.py
class CustomEntityState:
    def __init__(self, entity, s_global):
        self._custom_field = gs.zeros(...)
    
    def serializable(self):
        self._custom_field = self._custom_field.detach()

# solvers.py
class CustomSolverState:
    def __init__(self, scene):
        self.entities = []
```

### 11.2 当前限制

- 状态类没有统一基类（除了部分使用 RBC）
- 接口不够严格（duck typing）
- 缺少类型注解

### 11.3 改进建议

```python
# 定义统一接口
class StateInterface(ABC):
    @abstractmethod
    def serializable(self) -> None:
        pass
    
    @abstractmethod
    def deserializable(self, scene) -> None:
        pass

class EntityState(StateInterface):
    pass

class SolverState(StateInterface):
    pass
```

## 12. 代码质量评估

### 12.1 优点

- ✅ 结构清晰，层次分明
- ✅ 支持可微分仿真
- ✅ 批处理友好
- ✅ 序列化支持完善
- ✅ 内存管理高效

### 12.2 缺点

- ⚠️ 缺少完整的 docstring
- ⚠️ 接口不够统一（部分类继承 RBC，部分不继承）
- ⚠️ 类型注解不完整
- ⚠️ 缺少单元测试

### 12.3 文档完善度

```python
# 现状
class MPMEntityState(RBC):
    """
    Dynamic state queried from a genesis MPMEntity.
    """
    # 缺少属性和方法的详细说明

# 建议
class MPMEntityState(RBC):
    """
    Dynamic state queried from a genesis MPMEntity.
    
    Attributes:
        pos: Particle positions, shape (B, n_particles, 3)
        vel: Particle velocities, shape (B, n_particles, 3)
        C: APIC affine matrices, shape (B, n_particles, 3, 3)
        F: Deformation gradients, shape (B, n_particles, 3, 3)
        Jp: Plastic volume changes, shape (B, n_particles)
    """
```

## 13. 总结

### 13.1 模块职责

states 模块是 Genesis 仿真系统的"快照相机"：
- 捕获仿真状态
- 支持时间回溯
- 实现梯度追踪
- 提供检查点机制

### 13.2 设计亮点

1. **三层架构**: Sim → Solver → Entity，清晰分层
2. **可微分支持**: 与自动微分系统深度集成
3. **批处理优先**: 原生支持并行环境
4. **序列化友好**: 可保存和恢复完整状态

### 13.3 典型使用流程

```python
# 1. 正向仿真
scene.build()
for i in range(100):
    scene.step()
    if i % 10 == 0:
        state = scene.get_state()
        states_history.append(state)

# 2. 时间回溯
scene.set_state(states_history[5])

# 3. 检查点保存
state = scene.get_state()
state.serializable()
with open('ckpt.pkl', 'wb') as f:
    pickle.dump(state, f)

# 4. 可微分仿真
scene = gs.Scene(requires_grad=True)
# ... 模拟 ...
loss = compute_loss(scene.get_state())
loss.backward()
```

### 13.4 与其他模块的关系

```
Scene
  └─ Simulator
       └─ Solvers
            └─ Entities
                  ↓
            get_state()
                  ↓
       ┌──────────────────┐
       │   States Module   │
       │  - SimState       │
       │  - SolverState    │
       │  - EntityState    │
       └──────────────────┘
```

---

**代码统计**:
- 总行数: 531 行
- 状态类数量: 12 个
- 辅助类数量: 2 个 (QueriedStates, StateList)
- 平均每个状态类: ~40 行
