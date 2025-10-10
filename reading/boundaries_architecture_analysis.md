# Genesis boundaries 模块架构分析

## 1. 概述

`genesis/engine/boundaries` 模块提供了边界条件实现，用于限制粒子和实体的运动范围。该模块设计简洁，只包含两个核心类，但在物理仿真中扮演重要角色。

**模块位置**: `genesis/engine/boundaries/`

**核心文件**:
- `boundaries.py` (73行) - 边界条件实现
- `__init__.py` (1行) - 模块导出

## 2. 核心依赖

```
boundaries
    ├── gstaichi (Taichi)
    ├── numpy
    └── genesis core
```

## 3. 架构设计

### 3.1 类图结构

```
┌─────────────────────────────┐
│      CubeBoundary           │
│  @ti.data_oriented          │
├─────────────────────────────┤
│ - upper, lower              │
│ - restitution               │
├─────────────────────────────┤
│ + impose_pos_vel()          │
│ + impose_pos()              │
│ + is_inside()               │
└─────────────────────────────┘

┌─────────────────────────────┐
│      FloorBoundary          │
│  @ti.data_oriented          │
├─────────────────────────────┤
│ - height                    │
│ - restitution               │
├─────────────────────────────┤
│ + impose_pos_vel()          │
│ + impose_pos()              │
└─────────────────────────────┘
```

### 3.2 设计特点

1. **简洁性**: 只有两个边界类型，满足大多数仿真需求
2. **装饰器模式**: 使用 `@ti.data_oriented` 支持GPU加速
3. **统一接口**: 两个类都提供 `impose_pos_vel()` 和 `impose_pos()` 方法

## 4. 核心类详解

### 4.1 CubeBoundary - 立方体边界

**功能**: 定义一个轴对齐的立方体空间边界

```python
@ti.data_oriented
class CubeBoundary:
    def __init__(self, lower, upper, restitution=0.0)
```

**属性**:
- `lower`, `upper`: 边界的最小和最大坐标 (numpy + Taichi)
- `restitution`: 恢复系数，控制碰撞弹性

**核心方法**:

1. **`impose_pos_vel(pos, vel)` - Taichi 函数**
   - 同时修正位置和速度
   - 碰撞时根据 restitution 反转速度
   - 将位置钳制在边界内

2. **`impose_pos(pos)` - Taichi 函数**
   - 只修正位置
   - 使用 `ti.max` 和 `ti.min` 钳制坐标

3. **`is_inside(pos)` - Python 函数**
   - 检查点是否在边界内
   - 使用 NumPy 向量化操作

### 4.2 FloorBoundary - 地板边界

**功能**: 定义一个水平地板边界 (只限制 Z 轴)

```python
@ti.data_oriented
class FloorBoundary:
    def __init__(self, height, restitution=0.0)
```

**属性**:
- `height`: 地板高度 (Z坐标)
- `restitution`: 恢复系数

**核心方法**: 与 CubeBoundary 类似，但只作用于 Z 轴

## 5. 代码风格分析

### 5.1 命名规范

```python
# 类名: PascalCase
CubeBoundary, FloorBoundary

# 方法名: snake_case
impose_pos_vel, is_inside

# 私有属性: 无下划线前缀 (公开属性)
self.upper, self.lower

# Taichi 属性: _ti 后缀
self.upper_ti, self.lower_ti
```

### 5.2 类型处理

**双重存储策略**:
```python
# NumPy 版本 (用于 Python 端操作)
self.upper = np.array(upper, dtype=gs.np_float)

# Taichi 版本 (用于 GPU 内核)
self.upper_ti = ti.Vector(upper, dt=gs.ti_float)
```

这种设计允许:
- Python 端进行逻辑判断
- GPU 端进行高性能计算

### 5.3 装饰器使用

```python
@ti.data_oriented  # 类级别装饰器
class CubeBoundary:
    
    @ti.func  # Taichi 内联函数 (可在 kernel 中调用)
    def impose_pos_vel(self, pos, vel):
        # GPU 加速代码
```

### 5.4 代码简洁性

- **无继承**: 两个类独立实现，没有共同基类
- **方法少**: 每个类只有 3 个方法
- **逻辑清晰**: 碰撞检测和响应逻辑直观

## 6. 设计模式

### 6.1 Strategy Pattern (策略模式)

不同的边界类型实现相同的接口:
```python
boundary.impose_pos_vel(pos, vel)  # 统一接口
```

求解器可以无差别地使用任何边界类型。

### 6.2 Immutable Configuration (不可变配置)

边界参数在构造后不可更改:
- 简化线程安全
- 避免意外修改
- 符合物理世界的静态边界假设

## 7. 物理算法

### 7.1 位置钳制

```python
pos = ti.max(ti.min(pos, self.upper_ti), self.lower_ti)
```

使用 `min` 和 `max` 实现向量钳制，等价于:
```python
pos[i] = clamp(pos[i], lower[i], upper[i])
```

### 7.2 速度反转

```python
if pos[i] >= self.upper_ti[i] and vel[i] >= 0:
    vel[i] *= -self.restitution
```

**物理意义**:
- 检查碰撞条件 (位置超界 且 速度朝向边界)
- 反转速度方向
- 乘以恢复系数模拟能量损失

**restitution 取值**:
- `0.0`: 完全非弹性碰撞 (停止)
- `1.0`: 完全弹性碰撞 (反弹)
- `0.5`: 部分能量损失

## 8. 使用场景

### 8.1 CubeBoundary 应用

```python
# MPM/SPH 求解器中的粒子边界
boundary = CubeBoundary(
    lower=[-1.0, -1.0, 0.0],
    upper=[1.0, 1.0, 2.0],
    restitution=0.0  # 粒子碰到边界后停止
)
```

### 8.2 FloorBoundary 应用

```python
# 刚体求解器中的地板
boundary = FloorBoundary(
    height=0.0,
    restitution=0.3  # 轻微反弹
)
```

## 9. 与求解器的集成

### 9.1 MPM/SPH 求解器

在粒子时间步进中调用:
```python
@ti.kernel
def substep_update_particles():
    for i in particles:
        pos, vel = boundary.impose_pos_vel(pos, vel)
```

### 9.2 PBD 求解器

在约束投影阶段调用:
```python
@ti.kernel
def solve_constraints():
    for i in vertices:
        pos = boundary.impose_pos(pos)
```

## 10. 扩展性分析

### 10.1 当前限制

- 只支持轴对齐边界 (AABB)
- 不支持旋转边界
- 不支持动态移动边界
- 不支持复杂几何边界

### 10.2 潜在扩展

可能的新边界类型:
```python
class SphereBoundary:
    """球形边界"""
    
class CylinderBoundary:
    """柱形边界"""
    
class MeshBoundary:
    """任意网格边界 (使用 SDF)"""
```

## 11. 性能特点

### 11.1 优势

- **GPU 加速**: `@ti.func` 内联展开，零函数调用开销
- **向量化**: Taichi Vector 操作并行化
- **无分支**: `ti.max/min` 避免条件分支

### 11.2 开销

- **双重存储**: NumPy + Taichi 版本占用更多内存
- **类型转换**: Python 到 Taichi 有转换开销 (构造时一次性)

## 12. 代码质量

### 12.1 可读性

- ✅ 代码短小精悍 (73行)
- ✅ 函数命名清晰
- ✅ 逻辑直观易懂
- ✅ 包含 `__repr__` 方法便于调试

### 12.2 健壮性

- ✅ 边界检查 (`assert (self.upper >= self.lower).all()`)
- ✅ 类型转换明确
- ✅ 向量维度固定 (3D)

### 12.3 文档

- ⚠️ 缺少 docstring
- ⚠️ 参数说明不完整
- ✅ 变量名自解释

## 13. 总结

### 13.1 设计亮点

1. **极简主义**: 只有两个类，满足 80% 需求
2. **高性能**: Taichi 函数内联，GPU 友好
3. **统一接口**: 多态性支持，易于扩展
4. **双层设计**: Python 和 Taichi 分离

### 13.2 改进建议

1. 添加完整的 docstring
2. 考虑提取共同基类 (如果需要更多边界类型)
3. 支持动态边界 (移动平台等场景)
4. 添加单元测试

### 13.3 适用场景

- ✅ 基础仿真场景
- ✅ 粒子系统边界
- ✅ 刚体地板约束
- ⚠️ 复杂几何边界 (需要扩展)
- ⚠️ 可变形边界 (不支持)

---

**代码统计**:
- 总行数: 74 行
- 类数量: 2
- Taichi 函数: 4
- Python 函数: 2
