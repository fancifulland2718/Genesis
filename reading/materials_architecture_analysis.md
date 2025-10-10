# Genesis materials 模块架构分析

## 1. 概述

`genesis/engine/materials` 模块定义了物理仿真中使用的各种材料模型。该模块为不同的求解器（MPM、FEM、PBD、SPH、SF、Rigid、Avatar、Tool）提供材料属性和本构关系。

**模块位置**: `genesis/engine/materials/`

**核心文件统计**:
- 总代码量: 2132 行
- 子模块数: 6 个 (MPM, FEM, PBD, SPH, SF, 通用)
- 材料类数: 20+ 个

**子模块结构**:
```
materials/
├── __init__.py           # 导出所有材料类
├── base.py               # 材料基类
├── rigid.py              # 刚体材料
├── avatar.py             # Avatar 材料
├── tool.py               # Tool 材料
├── hybrid.py             # 混合材料
├── MPM/                  # MPM 材料 (8个文件)
├── FEM/                  # FEM 材料 (3个文件)
├── PBD/                  # PBD 材料 (5个文件)
├── SPH/                  # SPH 材料 (2个文件)
└── SF/                   # SF 材料 (3个文件)
```

## 2. 架构设计

### 2.1 继承体系

```
         Material (基类)
              │
    ┌─────────┴─────────┐
    │                   │
Solver-Specific    General Materials
    │                   │
    ├── MPM.Base        ├── Rigid
    ├── FEM.Base        ├── Avatar
    ├── PBD.Base        ├── Tool
    ├── SPH.Base        └── Hybrid
    └── SF.Base
         │
    ┌────┴────┐
    │         │
Concrete   Concrete
Materials  Materials
```

### 2.2 类型层次

```
┌─────────────────────────────────────────┐
│          Material (基类)                 │
│  - uid: 唯一标识符                       │
│  - @ti.data_oriented                    │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│      Domain-Specific Base               │
│  MPM.Base / FEM.Base / PBD.Base         │
│  - E, nu, rho (弹性模量、泊松比、密度)   │
│  - lam, mu (拉梅参数)                    │
│  - update_stress() (本构关系)            │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│      Concrete Materials                 │
│  MPM.Elastic, FEM.Muscle, PBD.Cloth     │
│  - 特定材料参数                          │
│  - 特定物理模型                          │
└─────────────────────────────────────────┘
```

## 3. 核心基类

### 3.1 Material - 材料基类

```python
@ti.data_oriented
class Material(RBC):
    """材料基类"""
    
    def __init__(self):
        self._uid = gs.UID()  # 唯一标识符
    
    @property
    def uid(self):
        return self._uid
    
    @classmethod
    def _repr_type(cls):
        return f"<gs.{cls.__module__.split('.')[-2]}.{cls.__name__}>"
```

**设计特点**:
- 使用 `@ti.data_oriented` 支持 GPU 加速
- 继承 `RBC` (ReprBaseClass) 提供统一的字符串表示
- 每个材料实例有唯一 ID (UID)
- 定制化的类型表示 (`_repr_type`)

### 3.2 Domain-Specific Base Classes

每个求解器都有自己的材料基类：

#### MPM.Base - MPM 材料基类

```python
@ti.data_oriented
class Base(Material):
    def __init__(self, E=1e6, nu=0.2, rho=1000.0, lam=None, mu=None, sampler=None):
        super().__init__()
        
        self._E = E            # 杨氏模量
        self._nu = nu          # 泊松比
        self._rho = rho        # 密度
        self._sampler = sampler  # 粒子采样器
        
        # 自动计算拉梅参数
        if mu is None:
            self._mu = E / (2.0 * (1.0 + nu))
        if lam is None:
            self._lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    
    @ti.func
    def update_stress(self, U, S, V, F_tmp, F_new, J, Jp, actu, m_dir):
        """计算应力张量 (本构关系)"""
        # 子类重写
        raise NotImplementedError
```

**职责**:
- 存储基本物理参数
- 自动计算衍生参数（如拉梅参数）
- 定义本构关系接口 `update_stress()`

#### FEM.Base - FEM 材料基类

```python
@ti.data_oriented
class Base(Material):
    def __init__(self, E=1e6, nu=0.2, rho=1000.0, hydroelastic_modulus=1e7, friction_mu=0.1):
        super().__init__()
        
        self._E = E
        self._nu = nu
        self._rho = rho
        self._hydroelastic_modulus = hydroelastic_modulus
        self._friction_mu = friction_mu
        
        # 计算拉梅参数
        self._mu = E / (2.0 * (1.0 + nu))
        self._lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    
    @ti.func
    def update_stress(self, mu, lam, J, F, actu, m_dir):
        """计算应力张量"""
        raise NotImplementedError
```

**与 MPM.Base 的差异**:
- 增加了 `hydroelastic_modulus` (水弹性模量)
- 增加了 `friction_mu` (摩擦系数)
- 没有粒子采样器参数

#### PBD.Base - PBD 材料基类

```python
@ti.data_oriented
class Base(Material):
    def __init__(self, n_substeps, rho=1000.0, gravity=None, sampler=None):
        super().__init__()
        
        self._n_substeps = n_substeps
        self._rho = rho
        self._gravity = gravity or gs.ti_vec3(0, 0, -9.81)
        self._sampler = sampler
```

**特点**:
- 基于位置的动力学，不需要弹性模量
- 包含子步数参数
- 包含重力参数

## 4. 具体材料类

### 4.1 刚体材料

#### Rigid - 刚体材料

```python
@ti.data_oriented
class Rigid(Material):
    def __init__(
        self,
        rho=200.0,
        friction=None,
        needs_coup=True,
        coup_friction=0.1,
        coup_softness=0.002,
        coup_restitution=0.0,
        sdf_cell_size=0.005,
        sdf_min_res=32,
        sdf_max_res=128,
        gravity_compensation=0,
    ):
```

**参数分类**:

1. **基本物理参数**:
   - `rho`: 密度
   - `friction`: 摩擦系数

2. **耦合参数** (与其他求解器交互):
   - `needs_coup`: 是否参与耦合
   - `coup_friction`: 耦合摩擦系数
   - `coup_softness`: 耦合软度
   - `coup_restitution`: 恢复系数

3. **SDF (Signed Distance Field) 参数**:
   - `sdf_cell_size`: 单元格大小
   - `sdf_min_res`, `sdf_max_res`: 网格分辨率范围

4. **其他**:
   - `gravity_compensation`: 重力补偿

**验证逻辑**:
```python
if friction is not None:
    if friction < 1e-2 or friction > 5.0:
        gs.raise_exception("`friction` must be in the range [1e-2, 5.0]")

if coup_restitution != 0:
    gs.logger.warning("Non-zero `coup_restitution` could lead to instability.")
```

### 4.2 MPM 材料

#### MPM.Elastic - 弹性材料

```python
@ti.data_oriented
class Elastic(Base):
    def __init__(self, E=1e6, nu=0.2, rho=1000.0, lam=None, mu=None, sampler=None):
        super().__init__(E, nu, rho, lam, mu, sampler)
    
    @ti.func
    def update_F_S_Jp(self, J, F_tmp, U, S, V, Jp):
        # 弹性变形，Jp 不变
        return F_tmp, S, Jp
```

**特点**: 纯弹性，无塑性变形

#### MPM.Snow - 雪材料

```python
@ti.data_oriented
class Snow(Base):
    def __init__(
        self,
        E=1e6, nu=0.2, rho=400.0,
        theta_c=2.5e-2,  # 临界压缩
        theta_s=7.5e-3,  # 临界拉伸
        lam=None, mu=None, sampler=None,
    ):
```

**塑性模型**:
```python
@ti.func
def update_F_S_Jp(self, J, F_tmp, U, S, V, Jp):
    # 雪的塑性流动
    S_new = ti.Vector.zero(gs.ti_float, 3)
    for d in ti.static(range(3)):
        S_new[d] = ti.min(ti.max(S[d], 1 - self._theta_c), 1 + self._theta_s)
    
    Jp_new = Jp * J / S_new.prod()
    F_new = U @ ti.Matrix.diag(3, S_new) @ V.transpose()
    
    return F_new, S_new, Jp_new
```

**物理意义**:
- `theta_c`: 压缩阈值，超过则发生塑性变形
- `theta_s`: 拉伸阈值，超过则发生断裂
- 通过钳制奇异值实现塑性

#### MPM.Sand - 沙土材料

```python
@ti.data_oriented
class Sand(Base):
    def __init__(
        self,
        E=1e6, nu=0.2, rho=1000.0,
        friction_angle=30.0,  # 内摩擦角
        lam=None, mu=None, sampler=None,
    ):
```

**Drucker-Prager 屈服准则**:
```python
@ti.func
def update_F_S_Jp(self, J, F_tmp, U, S, V, Jp):
    e = ti.log(S)
    e_hat = e - e.sum() / 3
    e_hat_norm = e_hat.norm() + 1e-20
    
    if e_hat_norm >= self._yield_surface:
        # 投影到屈服面
        e -= (e_hat_norm - self._yield_surface) / e_hat_norm * e_hat
        S_new = ti.exp(e)
        Jp_new = Jp * J / S_new.prod()
        F_new = U @ ti.Matrix.diag(3, S_new) @ V.transpose()
        return F_new, S_new, Jp_new
    else:
        return F_tmp, S, Jp
```

### 4.3 FEM 材料

#### FEM.Elastic - 弹性材料

**支持多种本构模型**:
```python
def __init__(self, E=1e6, nu=0.2, rho=1000.0, model="linear"):
    if model == "linear":
        self.update_stress = self.update_stress_linear
    elif model == "stable_neohookean":
        self.update_stress = self.update_stress_stable_neohookean
    elif model == "linear_corotated":
        self.update_stress = self.update_stress_linear_corotated
```

**线性弹性应力**:
```python
@ti.func
def update_stress_linear(self, mu, lam, J, F, actu, m_dir):
    I = ti.Matrix.identity(dt=gs.ti_float, n=3)
    stress = mu * (F + F.transpose() - 2 * I) + lam * (F - I).trace() * I
    return stress
```

**稳定 Neo-Hookean 应力**:
```python
@ti.func
def update_stress_stable_neohookean(self, mu, lam, J, F, actu, m_dir):
    IC = F.norm_sqr()
    dJdF = partialJpartialF(F)
    
    # 稳定化处理
    alpha = 1 + mu / lam - mu / (4 * lam)
    P = mu * (1 - 1 / (IC + 1)) * F + lam * (J - alpha) * dJdF
    stress = P @ F.transpose()
    return stress
```

#### FEM.Muscle - 肌肉材料

```python
@ti.data_oriented
class Muscle(Elastic):
    def __init__(
        self,
        E=1e6, nu=0.2, rho=1000.0,
        model="stable_neohookean",
        activation=0.0,  # 肌肉激活度
        f_max=3e5,       # 最大肌肉力
    ):
```

**各向异性应力** (沿肌肉纤维方向):
```python
@ti.func
def update_stress_stable_neohookean(self, mu, lam, J, F, actu, m_dir):
    # 基础 Neo-Hookean 应力
    stress = super().update_stress_stable_neohookean(mu, lam, J, F, actu, m_dir)
    
    # 添加肌肉主动应力
    m_dir_curr = F @ m_dir
    m_dir_curr_norm = m_dir_curr.norm()
    muscle_P = actu * self._f_max * m_dir_curr / m_dir_curr_norm * m_dir.transpose()
    stress += muscle_P @ F.transpose()
    
    return stress
```

### 4.4 PBD 材料

#### PBD.Cloth - 布料材料

```python
@ti.data_oriented
class Cloth(Base):
    def __init__(
        self,
        n_substeps,
        rho=100.0,
        stretch_compliance=1e-6,   # 拉伸柔度
        bending_compliance=1e-3,   # 弯曲柔度
        shear_compliance=1e-4,     # 剪切柔度
        damping=0.0,
    ):
```

**柔度参数**: 值越小越硬

#### PBD.Liquid - 液体材料

```python
@ti.data_oriented
class Liquid(Base):
    def __init__(
        self,
        n_substeps,
        rho=1000.0,
        rest_density=1000.0,       # 静止密度
        viscosity=0.01,            # 黏度
        surface_tension=0.0,       # 表面张力
        vorticity_confinement=0.0, # 涡量增强
    ):
```

### 4.5 SPH 材料

#### SPH.Liquid - SPH 液体

```python
@ti.data_oriented
class Liquid(Base):
    def __init__(
        self,
        rho=1000.0,
        gamma=7.0,         # 气体常数
        c=20.0,            # 声速
        viscosity=0.01,
        surface_tension=0.0,
    ):
```

**状态方程** (Tait 方程):
```python
@ti.func
def compute_pressure(self, rho):
    return self._B * (ti.pow(rho / self._rho, self._gamma) - 1.0)
```

### 4.6 SF 材料

#### SF.Smoke - 烟雾材料

```python
@ti.data_oriented
class Smoke(Base):
    def __init__(
        self,
        rho=1.0,
        velocity_dissipation=0.99,  # 速度耗散
        density_dissipation=0.99,   # 密度耗散
        temperature_dissipation=0.99,
        buoyancy=0.5,               # 浮力
    ):
```

## 5. 代码风格分析

### 5.1 命名规范

```python
# 类名: PascalCase
Elastic, Snow, Liquid, Muscle

# 模块名: 大写缩写
MPM, FEM, PBD, SPH, SF

# 属性: snake_case 带下划线前缀
self._E, self._nu, self._rho

# 物理参数: 符号或描述性名称
E (杨氏模量), nu (泊松比), lam (拉梅参数), mu (剪切模量)
```

### 5.2 参数验证模式

```python
# 范围检查
if friction < 1e-2 or friction > 5.0:
    gs.raise_exception("...")

# 逻辑检查
if sdf_min_res > sdf_max_res:
    gs.raise_exception("...")

# 警告提示
if coup_restitution != 0:
    gs.logger.warning("...")
```

### 5.3 默认值策略

**智能默认值**:
```python
# 根据平台选择采样器
if sampler is None:
    sampler = "pbs" if gs.platform == "Linux" else "random"

# 根据参数计算默认值
if mu is None:
    self._mu = E / (2.0 * (1.0 + nu))
```

### 5.4 文档字符串风格

```python
class Elastic(Base):
    """
    The elastic material class for FEM.
    
    Parameters
    ----------
    E: float, optional
        Young's modulus, which controls stiffness. Default is 1e6.
    nu: float, optional
        Poisson ratio, describing the material's volume change under stress.
    """
```

**使用 NumPy 风格文档字符串**

## 6. 设计模式

### 6.1 Template Method Pattern (模板方法模式)

基类定义算法骨架，子类实现具体步骤:

```python
class Base:
    @ti.func
    def update_stress(...):
        raise NotImplementedError

class Elastic(Base):
    @ti.func
    def update_stress(...):
        # 具体实现
        return stress
```

### 6.2 Strategy Pattern (策略模式)

FEM.Elastic 支持多种本构模型:

```python
if model == "linear":
    self.update_stress = self.update_stress_linear
elif model == "stable_neohookean":
    self.update_stress = self.update_stress_stable_neohookean
```

运行时选择算法。

### 6.3 Factory Pattern (工厂模式的变体)

材料注册到求解器:

```python
# 用户代码
material = gs.materials.MPM.Elastic(E=1e6)
entity = scene.add_entity(material=material, ...)

# 内部
solver.register_material(material)
material._idx = solver.n_materials
```

### 6.4 Flyweight Pattern (享元模式)

相同材料可被多个实体共享:

```python
material = gs.materials.Rigid(rho=200.0)

entity1 = scene.add_entity(material=material, ...)
entity2 = scene.add_entity(material=material, ...)  # 共享材料
```

通过 `material.uid` 判断是否为同一材料。

## 7. 物理模型总结

### 7.1 本构模型分类

| 材料 | 本构模型 | 特点 |
|------|---------|------|
| MPM.Elastic | 线性弹性 | 纯弹性变形 |
| MPM.Snow | Von Mises 塑性 | 压缩+拉伸阈值 |
| MPM.Sand | Drucker-Prager | 摩擦角屈服 |
| FEM.Elastic | 线性/Neo-Hookean/线性共旋 | 多模型支持 |
| FEM.Muscle | Neo-Hookean + 主动应力 | 各向异性 |
| PBD.* | 基于位置 | 无应力概念 |
| SPH.Liquid | Tait 状态方程 | 压力-密度关系 |

### 7.2 参数范围建议

| 参数 | 典型范围 | 单位 |
|------|---------|------|
| E (杨氏模量) | 1e4 ~ 1e9 | Pa |
| nu (泊松比) | 0.0 ~ 0.5 | 无量纲 |
| rho (密度) | 100 ~ 10000 | kg/m³ |
| friction | 0.01 ~ 5.0 | 无量纲 |

## 8. 扩展性分析

### 8.1 添加新材料

**步骤**:

1. 创建材料类文件:
```python
# genesis/engine/materials/MPM/custom.py
@ti.data_oriented
class Custom(Base):
    def __init__(self, E=1e6, nu=0.2, custom_param=1.0):
        super().__init__(E, nu)
        self._custom_param = custom_param
    
    @ti.func
    def update_stress(self, ...):
        # 自定义本构关系
        pass
```

2. 在 `__init__.py` 中导出:
```python
from .custom import Custom
```

3. 用户使用:
```python
material = gs.materials.MPM.Custom(custom_param=2.0)
```

### 8.2 当前限制

- 材料参数在构造后不可修改（不可变设计）
- 缺少材料参数的自动校验
- 缺少材料参数的单位转换

### 8.3 改进建议

1. **参数验证器**:
```python
class Material:
    def __init__(self):
        self._validate_params()
    
    def _validate_params(self):
        if self._nu < 0 or self._nu > 0.5:
            raise ValueError("Poisson ratio must be in [0, 0.5]")
```

2. **单位系统**:
```python
E = 200 * GPa  # 自动转换为 Pa
```

3. **材料库**:
```python
material = gs.materials.from_library("steel")
```

## 9. 性能特点

### 9.1 优势

- **GPU 加速**: 所有 `update_stress` 函数都是 `@ti.func`
- **内联展开**: Taichi 编译器优化
- **共享材料**: 减少内存占用

### 9.2 开销

- **类型转换**: Python 到 Taichi 的参数传递
- **分支预测**: `if model == "linear"` 类型的运行时分支

## 10. 总结

### 10.1 设计亮点

1. **清晰的层次结构**: Base → Specific，易于扩展
2. **物理完备性**: 覆盖弹性、塑性、流体等多种材料
3. **GPU 友好**: Taichi 装饰器，高性能计算
4. **参数验证**: 确保物理合理性

### 10.2 代码质量

- ✅ 模块化设计
- ✅ 文档较完善
- ✅ 参数验证
- ⚠️ 缺少单元测试
- ⚠️ 部分类缺少 docstring

### 10.3 适用场景

- ✅ 多物理场仿真
- ✅ 高性能GPU计算
- ✅ 可微分仿真
- ⚠️ 复杂材料非线性（需扩展）

---

**代码统计**:
- 总行数: 2132 行
- 材料类数: 20+ 个
- 支持的求解器: 8 个
- 本构模型: 10+ 种
