# SAP Coupler 架构分析

## 概述

`SAPCoupler` 是 Genesis 中基于 SAP (Semi-Analytic Primal，半解析主问题) 方法实现的跨求解器耦合器，用于统一处理刚体求解器（Rigid Solver）与有限元求解器（FEM Solver）之间的接触、摩擦和约束。本文从介观角度分析其核心机制。

**参考文献**：
- Paper: https://arxiv.org/abs/2110.10107
- Drake实现: https://drake.mit.edu/release_notes/v1.5.0.html

## 1. 刚体与有限元求解器的信息交换机制

### 1.1 信息交换的层次结构

SAP Coupler 通过以下三个层次实现信息交换：

#### 1.1.1 状态层 (State Layer)
- **FEM 状态**：`fem_state_v` 存储 FEM 顶点的速度、速度差、梯度和冲量
  - `v`: 当前顶点速度
  - `v_diff`: 与未约束速度的差值 (v - v*)
  - `gradient`: SAP 问题的梯度
  - `impulse`: 累积冲量

- **刚体状态**：`rigid_state_dof` 存储刚体 DOF 的对应量
  - `v`: DOF 速度
  - `v_diff`: 速度差
  - `mass_v_diff`: 质量矩阵与速度差的乘积
  - `gradient`: SAP 梯度
  - `impulse`: 冲量

#### 1.1.2 接触处理层 (Contact Handler Layer)
通过多个 ContactHandler 实现不同类型的接触处理：

```python
# 代码位置：sap_coupler.py, line 248-307
self.contact_handlers = []

# FEM 相关接触
if self._fem_floor_contact_type == FEMFloorContactType.TET:
    self.fem_floor_tet_contact = FEMFloorTetContactHandler(self.sim)
    self.contact_handlers.append(self.fem_floor_tet_contact)

# 刚体相关接触
if self._rigid_floor_contact_type == RigidFloorContactType.VERT:
    self.rigid_floor_vert_contact = RigidFloorVertContactHandler(self.sim)
    self.contact_handlers.append(self.rigid_floor_vert_contact)

# 刚体-FEM 耦合接触
if self._enable_rigid_fem_contact:
    self.rigid_fem_contact = RigidFemTriTetContactHandler(self.sim)
    self.contact_handlers.append(self.rigid_fem_contact)
```

每个 ContactHandler 负责：
1. **宽相检测** (Broad Phase)：使用 BVH 快速筛选潜在接触对
2. **窄相检测** (Narrow Phase)：精确计算接触几何
3. **雅可比构建** (Jacobian Construction)：建立接触约束与系统速度的关系

#### 1.1.3 求解层 (Solver Layer)
通过统一的 SAP 迭代框架整合两个求解器：

```python
# 代码位置：sap_coupler.py, line 866-888
def sap_solve(self, i_step):
    self._init_sap_solve(i_step)
    for iter in range(self._n_sap_iterations):
        # 1. 计算无约束项（各求解器独立）
        self.compute_unconstrained_gradient_diag(i_step, iter)
        
        # 2. 计算接触/约束项（耦合发生）
        self.compute_constraint_contact_gradient_hessian_diag_prec()
        self.check_sap_convergence()
        
        # 3. 使用 PCG 求解速度修正 dv
        self.pcg_solve()
        
        # 4. 精确线搜索更新速度
        self.exact_linesearch(i_step)
```

### 1.2 信息交换的具体流程

#### 1.2.1 前向传递：从状态到接触力

**步骤 1：读取初始速度**
```python
# 代码位置：line 994-1016
@ti.func
def _init_v(self, i_step: ti.i32):
    # FEM: 从 elements_v 读取
    for i_b, i_v in ti.ndrange(self.fem_solver._B, self.fem_solver.n_vertices):
        self.fem_state_v.v[i_b, i_v] = self.fem_solver.elements_v[i_step + 1, i_v, i_b].vel
    
    # 刚体: 从 dofs_state 读取
    for i_b, i_d in ti.ndrange(self.rigid_solver._B, self.rigid_solver.n_dofs):
        self.rigid_state_dof.v[i_b, i_d] = self.rigid_solver.dofs_state.vel[i_d, i_b]
```

**步骤 2：通过接触雅可比计算相对速度**

以刚体-FEM 接触为例（`RigidFemTriTetContactHandler`）：

```python
# 代码位置：line 4106-4131
@ti.func
def compute_Jx(self, i_p, x0, x1):
    """计算 J @ x，其中 x0 是 FEM 顶点速度，x1 是刚体 DOF 速度"""
    i_b = self.contact_pairs[i_p].batch_idx
    i_g0 = self.contact_pairs[i_p].geom_idx0
    Jx = ti.Vector.zero(gs.ti_float, 3)
    
    # FEM 部分：重心插值
    for i in ti.static(range(4)):
        i_v = self.fem_solver.elements_i[i_g0].el2v[i]
        Jx = Jx + self.contact_pairs[i_p].barycentric0[i] * x0[i_b, i_v]
    
    # 刚体部分：运动学链传递
    for i in range(self.rigid_solver.n_dofs):
        Jx = Jx - self.Jt[i_p, i] * x1[i_b, i]
    
    # 投影到接触系 (tangent0, tangent1, normal)
    return ti.Vector([
        Jx.dot(self.contact_pairs[i_p].tangent0),
        Jx.dot(self.contact_pairs[i_p].tangent1),
        Jx.dot(self.contact_pairs[i_p].normal),
    ])
```

**关键机制**：
- FEM 顶点通过**重心坐标**插值到接触点
- 刚体通过**运动学链雅可比** `Jt` 映射 DOF 速度到接触点速度
- 最终投影到**接触坐标系** (切向0, 切向1, 法向)

**步骤 3：计算接触冲量**

```python
# 代码位置：line 1244-1258（FEMContactHandler 的通用逻辑）
@ti.func
def compute_gradient_hessian_diag(self):
    """计算接触梯度与 Hessian 对角"""
    for i_p in range(self.n_contact_pairs[None]):
        # 1. 计算 Delassus 矩阵 W = J M^{-1} J^T
        W = self.compute_delassus(i_p)
        
        # 2. 计算正则化参数 R 和稳定速度 v_hat
        self.compute_regularization_per_pair(i_p, W)
        
        # 3. 求解接触冲量 gamma = -R^{-1}(J*v + v_hat)
        Jv = self.compute_Jx(i_p, self.coupler.fem_state_v.v)
        gamma = self.compute_gamma(i_p, Jv)
        
        # 4. 累加到梯度：gradient += J^T @ gamma
        self.add_Jt_x(self.coupler.fem_state_v.gradient, i_p, gamma)
```

#### 1.2.2 反向传递：从接触力到速度修正

**通过雅可比转置回传冲量**：

```python
# 代码位置：line 4133-4157
@ti.func
def add_Jt_x(self, y0, y1, i_p, x):
    """y0/y1 += J^T @ x，x 在接触系中给出"""
    i_b = self.contact_pairs[i_p].batch_idx
    i_g0 = self.contact_pairs[i_p].geom_idx0
    
    # 转换到世界系
    world = ti.Matrix.cols([
        self.contact_pairs[i_p].tangent0,
        self.contact_pairs[i_p].tangent1,
        self.contact_pairs[i_p].normal
    ])
    x_ = world @ x
    
    # FEM: 按重心权分配到 4 个顶点
    for i in ti.static(range(4)):
        i_v = self.fem_solver.elements_i[i_g0].el2v[i]
        y0[i_b, i_v] += self.contact_pairs[i_p].barycentric0[i] * x_
    
    # 刚体: 沿运动学树的 Jt 行向量累加
    for i in range(self.rigid_solver.n_dofs):
        y1[i_b, i] -= self.Jt[i_p, i].dot(x_)
```

**最终写回**：

```python
# 代码位置：line 708-732
@ti.kernel
def update_vel(self, i_step: ti.i32):
    # FEM
    for i_b, i_v in ti.ndrange(self.fem_solver._B, self.fem_solver.n_vertices):
        self.fem_solver.elements_v[i_step + 1, i_v, i_b].vel = self.fem_state_v.v[i_b, i_v]
    
    # 刚体
    for i_b, i_d in ti.ndrange(self.rigid_solver._B, self.rigid_solver.n_dofs):
        self.rigid_solver.dofs_state.vel[i_d, i_b] = self.rigid_state_dof.v[i_b, i_d]
```

### 1.3 信息交换的时序

```
时间步 t:
1. 预处理 (preprocess)
   ├─ 更新 FEM 压力梯度 (fem_compute_pressure_gradient)
   ├─ 更新刚体顶点位置 (func_update_all_verts)
   └─ 更新刚体体场姿态 (rigid_update_volume_verts_pressure_gradient)

2. BVH 更新 (update_bvh)
   ├─ FEM 表面四面体 AABB
   ├─ 刚体三角面片 AABB
   └─ 刚体体四面体 AABB

3. 接触检测 (update_contact)
   ├─ 宽相：BVH 查询潜在接触对
   ├─ 窄相：几何裁剪与压力场测试
   └─ 雅可比构建：J, Jt

4. 正则化计算 (compute_regularization)
   └─ 基于 Delassus 矩阵计算 R, R_inv, v_hat

5. SAP 迭代求解 (sap_solve)
   for iter in [0, n_sap_iterations):
       ├─ 计算未约束梯度 (compute_unconstrained_gradient_diag)
       ├─ 计算接触梯度与 Hessian (compute_constraint_contact_gradient_hessian_diag_prec)
       ├─ PCG 求解 dv (pcg_solve)
       └─ 线搜索更新 v (exact_linesearch)

6. 写回速度 (update_vel)
   ├─ FEM: fem_state_v.v → elements_v[t+1].vel
   └─ 刚体: rigid_state_dof.v → dofs_state.vel
```

## 2. 求解过程中的数据组织形式

### 2.1 数据结构层次

#### 2.1.1 接触对数据 (Contact Pair Data)

以 `RigidFemTriTetContactHandler` 为例：

```python
# 代码位置：line 3827-3848
self.contact_pair_type = ti.types.struct(
    batch_idx=gs.ti_int,           # batch 索引
    normal=gs.ti_vec3,              # 接触法向
    tangent0=gs.ti_vec3,            # 切向 0
    tangent1=gs.ti_vec3,            # 切向 1
    geom_idx0=gs.ti_int,            # FEM 元素索引
    barycentric0=gs.ti_vec4,        # 四面体重心坐标
    link_idx=gs.ti_int,             # 刚体 link 索引
    contact_pos=gs.ti_vec3,         # 接触点位置
    sap_info=self.sap_contact_info_type,  # SAP 参数
)
```

其中 `sap_info` 包含：

```python
# 代码位置：line 1207-1214（FEMContactHandler 基类）
self.sap_contact_info_type = ti.types.struct(
    k=gs.ti_float,        # 接触刚度
    phi0=gs.ti_float,     # 初始穿透深度
    mu=gs.ti_float,       # 摩擦系数
    R=gs.ti_mat3,         # 正则化矩阵
    R_inv=gs.ti_mat3,     # 正则化矩阵逆
    v_hat=gs.ti_vec3,     # 稳定速度
)
```

**数据组织特点**：
- **Structure of Arrays (SOA)**：Taichi 自动优化内存布局
- **批次优先**：`batch_idx` 作为第一维，支持并行 batch 处理
- **几何-物理分离**：几何信息（normal, tangent）与物理参数（k, phi0, mu）分开存储

#### 2.1.2 求解器状态数据

**FEM 状态**：

```python
# 代码位置：line 508-540
fem_state_v = ti.types.struct(
    v=gs.ti_vec3,        # 顶点速度
    v_diff=gs.ti_vec3,   # v - v*
    gradient=gs.ti_vec3, # ∂ℓ/∂v
    impulse=gs.ti_vec3,  # ∫γ dt
)

pcg_fem_state_v = ti.types.struct(
    diag3x3=gs.ti_mat3,  # Hessian 对角块
    prec=gs.ti_mat3,     # 预条件器
    x=gs.ti_vec3,        # PCG 解向量
    r=gs.ti_vec3,        # 残差
    z=gs.ti_vec3,        # 预条件残差
    p=gs.ti_vec3,        # 搜索方向
    Ap=gs.ti_vec3,       # A @ p
)

linesearch_fem_state_v = ti.types.struct(
    x_prev=gs.ti_vec3,  # 前一解
    dp=gs.ti_vec3,      # A @ dv
)
```

**形状**：`(B, n_fem_vertices)` 其中 B 是 batch 数量

**刚体状态**：

```python
# 代码位置：line 542-576
rigid_state_dof = ti.types.struct(
    v=gs.ti_float,        # DOF 速度（标量）
    v_diff=gs.ti_float,
    mass_v_diff=gs.ti_float,
    gradient=gs.ti_float,
    impulse=gs.ti_float,
)
```

**形状**：`(B, n_rigid_dofs)`

**关键差异**：
- FEM 以**顶点**为自由度，速度为 3D 向量
- 刚体以**广义坐标**为自由度，速度为标量

#### 2.1.3 雅可比矩阵的存储

对于刚体-FEM 接触：

```python
# 代码位置：line 3849-3851
self.Jt = ti.field(gs.ti_vec3, shape=(max_contact_pairs, n_dofs))
self.M_inv_Jt = ti.field(gs.ti_vec3, shape=(max_contact_pairs, n_dofs))
self.W = ti.field(gs.ti_mat3, shape=(max_contact_pairs,))
```

**含义**：
- `Jt[i_p, i_d]`：第 `i_p` 个接触对，第 `i_d` 个 DOF 对接触点速度的贡献（3D 向量）
- `M_inv_Jt[i_p, i_d]`：质量矩阵逆乘 `Jt`，用于快速计算 Delassus
- `W[i_p]`：第 `i_p` 个接触对的 Delassus 矩阵（3x3，世界系）

**为何这样存储**：
1. **避免显式构建完整 J**：完整雅可比 `(3*n_contacts, n_dofs + 3*n_fem_verts)` 极其稀疏
2. **利用局部性**：每个接触对只涉及少数自由度（4个FEM顶点 + 刚体树路径上的DOF）
3. **支持并行**：按接触对并行计算，无需同步

### 2.2 数据访问模式

#### 2.2.1 接触检测阶段：写模式

```python
# 宽相：BVH 查询并行写入候选
for i_r in range(bvh_result_count):
    i_c = ti.atomic_add(self.n_contact_candidates[None], 1)  # 原子递增
    if i_c < max_candidates:
        self.contact_candidates[i_c] = ...  # 独占写入

# 窄相：候选并行转换为接触对
for i_c in range(n_candidates):
    i_p = ti.atomic_add(self.n_contact_pairs[None], 1)  # 原子递增
    if i_p < max_pairs:
        self.contact_pairs[i_p] = ...  # 独占写入
```

**特点**：使用原子操作保证线程安全的递增计数

#### 2.2.2 梯度计算阶段：读-写模式

```python
# 读取接触对，写入梯度（需要原子操作）
for i_p in range(n_contact_pairs):
    Jv = self.compute_Jx(i_p, v)  # 读 v
    gamma = self.compute_gamma(i_p, Jv)  # 局部计算
    self.add_Jt_x(gradient, i_p, gamma)  # 原子累加到 gradient
```

**为何需要原子操作**：
- 多个接触对可能共享同一 FEM 顶点或刚体 DOF
- 并行写入需要原子累加避免竞态条件

#### 2.2.3 PCG 求解阶段：全局归约

```python
# 代码位置：line 1519-1538（计算 PCG 标量）
@ti.func
def pcg_inner_product(self, x, y):
    """计算 <x, y> = sum(x_i * y_i)"""
    result = 0.0
    # FEM 部分
    for i_b, i_v in ti.ndrange(B, n_fem_verts):
        result += x[i_b, i_v].dot(y[i_b, i_v])
    # 刚体部分
    for i_b, i_d in ti.ndrange(B, n_rigid_dofs):
        result += x[i_b, i_d] * y[i_b, i_d]
    return result
```

**优化**：Taichi 自动处理并行归约

### 2.3 内存布局优化

#### 2.3.1 SOA vs AOS

**选择 SOA（Structure of Arrays）**：

```python
# SOA：Taichi 默认
field(shape=(B, N), layout=ti.Layout.SOA)
# 内存：[v0_x, v0_y, v0_z, v1_x, v1_y, v1_z, ...]

# AOS（未使用）：
# 内存：[{v0_x, v0_y, v0_z}, {v1_x, v1_y, v1_z}, ...]
```

**优势**：
- GPU 上连续访问同一分量（如所有 x 坐标）
- 向量化友好

#### 2.3.2 Batch 维度优化

**批次作为外层循环**：

```python
for i_b, i_v in ti.ndrange(B, N):  # B 在外层
    # 访问 state[i_b, i_v]
```

**原因**：
- 不同 batch 完全独立，无需同步
- GPU 可按 batch 分配到不同 SM（Streaming Multiprocessor）

## 3. 数值稳定性的工程修正

### 3.1 正则化 (Regularization)

#### 3.1.1 SAP 正则化参数

```python
# 代码位置：line 182-192
self._sap_taud = options.sap_taud      # 阻尼时间常数（默认 0.01）
self._sap_beta = options.sap_beta      # 近似阻尼系数（默认 0.8）
self._sap_sigma = options.sap_sigma    # 正则化系数（默认 1e-6）
```

**作用**：

**R 矩阵计算**（代码位置：line 1300-1325）：

```python
@ti.func
def compute_regularization_per_pair(self, i_p, W):
    """计算接触对的正则化参数"""
    k = self.contact_pairs[i_p].sap_info.k
    dt = self.sim._substep_dt
    
    # 法向正则化
    R_nn = (1.0 + self.coupler._sap_beta) * dt / self.coupler._sap_taud + dt**2 * k
    
    # 切向正则化（摩擦）
    R_tt = self.coupler._sap_sigma / dt
    
    R = ti.Matrix.zero(gs.ti_float, 3, 3)
    R[0, 0] = R_tt  # tangent0
    R[1, 1] = R_tt  # tangent1
    R[2, 2] = R_nn  # normal
    
    # v_hat：稳定速度项
    phi0 = self.contact_pairs[i_p].sap_info.phi0
    v_hat_n = -(phi0 / dt + self.coupler._sap_beta * phi0 / self.coupler._sap_taud)
    v_hat = ti.Vector([0.0, 0.0, v_hat_n])
    
    self.contact_pairs[i_p].sap_info.R = R
    self.contact_pairs[i_p].sap_info.R_inv = R.inverse()
    self.contact_pairs[i_p].sap_info.v_hat = v_hat
```

**物理含义**：

1. **法向正则 R_nn**：
   - 第一项 `(1+β)dt/τ_d`：阻尼项，防止震荡
   - 第二项 `dt²k`：接触刚度，控制穿透量
   
2. **切向正则 R_tt**：
   - `σ/dt`：摩擦正则化，避免摩擦锥退化
   
3. **稳定速度 v_hat_n**：
   - `-φ₀/dt`：Baumgarte 稳定项，修正穿透
   - `-βφ₀/τ_d`：阻尼校正，防止过度反弹

**数值效果**：
- 避免条件数爆炸（接触刚度 → ∞ 时）
- 软化接触：允许微小穿透，换取稳定性

#### 3.1.2 预条件器正则化

**FEM 预条件器**（代码位置：line 1131-1141）：

```python
@ti.func
def compute_preconditioner_fem(self):
    """计算 FEM 的 3x3 对角预条件器"""
    for i_b, i_v in ti.ndrange(B, n_verts):
        diag = self.pcg_fem_state_v[i_b, i_v].diag3x3
        
        # 正则化：加入小量确保可逆
        reg = ti.Matrix.identity(gs.ti_float, 3) * 1e-10
        diag_reg = diag + reg
        
        # 近似逆：对角化假设
        prec = diag_reg.inverse()
        self.pcg_fem_state_v[i_b, i_v].prec = prec
```

**作用**：
- 避免对角块奇异（如约束顶点质量为 0）
- 加速 PCG 收敛（条件数从 O(h⁻²) 降至 O(1)）

### 3.2 几何退化处理

#### 3.2.1 三角形面积检查

```python
# 代码位置：line 3884-3888（rigid_fem_contact）
normal = (pos_v1 - pos_v0).cross(pos_v2 - pos_v0)
magnitude_sqr = normal.norm_sqr()
if magnitude_sqr < gs.EPS:  # 退化三角形
    continue  # 跳过
normal *= ti.rsqrt(magnitude_sqr)  # 快速归一化
```

**阈值**：`gs.EPS = 1e-10`（双精度）或 `1e-6`（单精度）

#### 3.2.2 压力梯度检查

```python
# 代码位置：line 3890-3893
g0 = self.coupler.fem_pressure_gradient[i_b, i_q]
if g0.dot(normal) < gs.EPS:
    continue  # 法向与压力梯度垂直，接触无效
```

**原因**：压力梯度垂直于法向时，无法产生法向接触力

#### 3.2.3 体积计算保护

```python
# 代码位置：line 762-767（fem_compute_pressure_gradient）
signed_volume = area_vector.dot(e10)
if ti.abs(signed_volume) > gs.EPS:
    grad_i = area_vector / signed_volume
    gradient += grad_i * pressure
# else: 退化四面体，跳过
```

**防止除零**：四面体体积接近 0 时不参与压力梯度计算

### 3.3 线搜索保护

#### 3.3.1 Armijo 条件失败处理

```python
# 代码位置：line 1797-1825（exact_linesearch）
@ti.kernel
def exact_linesearch(self, i_step: ti.i32):
    # 初始化步长范围
    self.init_linesearch(i_step)
    
    for iter in range(self._n_linesearch_iterations):
        self.compute_linesearch_energy_and_derivatives()
        self.check_linesearch_convergence()
        self.update_linesearch_alpha()
    
    # 失败回退
    for i_b in range(B):
        if self.linesearch_state[i_b].step_size < 1e-12:
            # 步长过小，回退到安全值
            self.linesearch_state[i_b].step_size = 0.1 * self._linesearch_max_step_size
```

**机制**：
- rtsafe 混合 Newton 与二分法
- 步长限制在 `[0, max_step_size]`（默认 1.0）
- 失败时使用保守步长（10% 最大值）

#### 3.3.2 能量非凸性处理

```python
# 代码位置：line 1910-1925
@ti.func
def update_linesearch_alpha(self):
    """更新线搜索步长（混合 Newton/二分）"""
    for i_b in range(B):
        f = self.linesearch_state[i_b].f
        df = self.linesearch_state[i_b].df
        
        if ti.abs(df) < 1e-12:
            # 导数过小，使用二分法
            minus_dalpha = 0.5 * (alpha_max - alpha_min)
        elif f * df < 0:
            # Newton 步不在区间内，使用二分法
            minus_dalpha = 0.5 * (alpha_max - alpha_min)
        else:
            # Newton 步
            minus_dalpha = f / df
            # 截断到区间
            minus_dalpha = ti.max(minus_dalpha, alpha_min - alpha)
            minus_dalpha = ti.min(minus_dalpha, alpha_max - alpha)
```

**原因**：接触能量非光滑（摩擦锥、单边约束），Newton 法可能失效

### 3.4 PCG 收敛性保障

#### 3.4.1 条件数控制

```python
# 代码位置：line 1118-1141
@ti.kernel
def compute_fem_prec(self):
    """FEM 预条件器：近似 Hessian 对角逆"""
    for i_b, i_v in ti.ndrange(B, n_verts):
        diag = self.pcg_fem_state_v[i_b, i_v].diag3x3
        
        # 对角块包含：
        # 1. 质量矩阵（dt²缩放）
        # 2. 弹性项对角
        # 3. 接触 Hessian 对角
        
        # 正则化后求逆
        reg = ti.Matrix.identity(gs.ti_float, 3) * 1e-10
        prec = (diag + reg).inverse()
        self.pcg_fem_state_v[i_b, i_v].prec = prec
```

**效果**：
- 理想情况：条件数 κ(P⁻¹A) ≈ 1，PCG 在 O(1) 次迭代收敛
- 实际：κ 取决于接触数量与分布，通常 < 100

#### 3.4.2 早停机制

```python
# 代码位置：line 1437-1463（pcg_solve）
def pcg_solve(self):
    self.init_pcg()
    self.compute_fem_prec()
    self.compute_rigid_prec()
    
    for pcg_iter in range(self._n_pcg_iterations):
        self.pcg_step()  # 一次 PCG 迭代
        
        # 检查收敛（在 pcg_step 中）
        for i_b in range(B):
            residual_norm = ti.sqrt(self.pcg_state[i_b].rTr_new)
            if residual_norm < self._pcg_threshold:
                self.batch_pcg_active[i_b] = False  # 停止迭代
```

**阈值**：`pcg_threshold`（默认 1e-8）

### 3.5 批次独立性保障

#### 3.5.1 每批独立收敛判定

```python
# 代码位置：line 890-902
@ti.kernel
def check_sap_convergence(self):
    self.clear_sap_norms()
    self.add_fem_norms()
    self.add_rigid_norms()
    self.update_batch_active()

@ti.func
def update_batch_active(self):
    for i_b in range(B):
        norm_thr = atol + rtol * ti.max(momentum_norm, impulse_norm)
        if gradient_norm < norm_thr:
            self.batch_active[i_b] = False  # 该 batch 停止迭代
```

**优势**：
- 避免快速收敛的 batch 等待慢速 batch
- 节省计算资源

#### 3.5.2 防止跨 batch 数据竞争

```python
# 所有并行循环都带 batch 索引
for i_b, i_v in ti.ndrange(B, N):
    # 访问 state[i_b, i_v]，不同 batch 完全独立
```

**实现**：Taichi 保证不同 `i_b` 间无数据竞争

## 4. 除 SAPCoupler 外的支持类作用

### 4.1 接触处理器 (Contact Handlers)

SAP Coupler 通过多态接口 `ContactHandler` 支持不同接触类型：

#### 4.1.1 基类：FEMContactHandler

```python
# 代码位置：line 1198-1215
@ti.data_oriented
class FEMContactHandler(RBC):
    """FEM 接触处理器基类"""
    
    def __init__(self, simulator):
        self.sim = simulator
        self.coupler = simulator.coupler
        self.fem_solver = simulator.fem_solver
        
        # SAP 参数类型
        self.sap_contact_info_type = ti.types.struct(
            k=gs.ti_float, phi0=gs.ti_float, mu=gs.ti_float,
            R=gs.ti_mat3, R_inv=gs.ti_mat3, v_hat=gs.ti_vec3,
        )
        
        self.n_contact_pairs = ti.field(gs.ti_int, shape=())
```

**核心接口**：

```python
@ti.func
def detection(self, f: ti.i32) -> bool:
    """接触检测，返回是否溢出"""
    pass

@ti.func
def compute_jacobian(self):
    """构建接触雅可比"""
    pass

@ti.func
def compute_regularization(self):
    """计算正则化参数 R, v_hat"""
    pass

@ti.func
def compute_gradient_hessian_diag(self):
    """计算梯度与 Hessian 对角"""
    pass
```

#### 4.1.2 具体实现

**1. FEMFloorTetContactHandler**（FEM-地面四面体接触）

- **特点**：Hydroelastic 接触，使用压力场
- **检测**：
  - 宽相：自接触 BVH（表面四面体）
  - 窄相：Marching Tetrahedra 截取等压面
- **雅可比**：基于重心坐标的插值

**2. FEMFloorVertContactHandler**（FEM-地面点接触）

- **特点**：点-平面接触，无压力场
- **检测**：直接距离测试
- **雅可比**：单位映射（J = I）

**3. FEMSelfTetContactHandler**（FEM 自接触）

- **特点**：两个 FEM 四面体间的 Hydroelastic 接触
- **检测**：
  - 宽相：表面四面体 BVH
  - 窄相：双向等压面裁剪
- **雅可比**：双重重心插值（两个四面体）

**4. RigidFemTriTetContactHandler**（刚体-FEM 接触）

- **特点**：刚体三角形与 FEM 四面体的混合接触
- **检测**：
  - 宽相：刚体三角 BVH vs FEM 表面四面体 AABB
  - 窄相：三角形被四面体四个半空间裁剪
- **雅可比**：
  - FEM 侧：重心插值
  - 刚体侧：运动学链 `Jt`

**5. RigidFloorVertContactHandler**（刚体-地面点接触）

- 类似 FEMFloorVertContactHandler，但应用于刚体顶点

**6. RigidFloorTetContactHandler**（刚体-地面四面体接触）

- 使用刚体的"压力场"（体四面体化）

**7. RigidRigidTetContactHandler**（刚体-刚体四面体接触）

- **特点**：两个刚体的体四面体间 Hydroelastic 接触
- **检测**：双重等压面裁剪
- **雅可比**：两条运动学链的 `Jt`

### 4.2 约束处理器：RigidConstraintHandler

```python
# 代码位置：line 479-490
class RigidConstraintHandler:
    """处理刚体关节等式约束"""
    
    def build_constraints(self, equalities_info, joints_info, ...):
        """从刚体求解器提取约束信息"""
        pass
    
    @ti.func
    def compute_regularization(self):
        """约束的正则化（通常无正则）"""
        pass
    
    @ti.func
    def compute_gradient_hessian_diag(self):
        """约束拉格朗日乘子对梯度的贡献"""
        pass
```

**作用**：
- 将刚体关节约束（如铰链、滑移）嵌入 SAP 框架
- 统一处理接触与约束

### 4.3 BVH 结构

#### 4.3.1 AABB (Axis-Aligned Bounding Box)

```python
# 代码位置：bvh.py
class AABB:
    """轴对齐包围盒"""
    
    def __init__(self, B, n_primitives):
        self.aabbs = ti.field(
            dtype=ti.types.struct(min=gs.ti_vec3, max=gs.ti_vec3),
            shape=(B, n_primitives)
        )
```

**用途**：
- 快速相交测试：`O(1)` 判断两 AABB 是否重叠
- 作为 BVH 的叶子节点

#### 4.3.2 LBVH (Linear Bounding Volume Hierarchy)

```python
# 代码位置：bvh.py
class LBVH:
    """线性 BVH（基于 Morton 码）"""
    
    def build(self):
        """构建 BVH 树"""
        self.compute_morton_codes()  # Z曲线编码
        self.sort_morton_codes()     # 基数排序
        self.build_tree()            # 自底向上构建
    
    def query(self, query_aabbs):
        """返回所有相交的原语对"""
        pass
```

**优势**：
- GPU 友好：完全并行构建
- O(n log n) 构建，O(log n) 查询

#### 4.3.3 特化 BVH

**FEMSurfaceTetLBVH**：
- 专用于 FEM 表面四面体
- 集成到 FEM Solver 的数据结构

**RigidTetLBVH**：
- 专用于刚体体四面体
- 访问 `coupler.rigid_volume_verts`

### 4.4 辅助函数

#### 4.4.1 几何函数

```python
# 代码位置：line 93-146
@ti.func
def tri_barycentric(p, tri_vertices, normal):
    """计算点 p 在三角形中的重心坐标"""
    pass

@ti.func
def tet_barycentric(p, tet_vertices):
    """计算点 p 在四面体中的重心坐标"""
    # 用于插值顶点量（速度、压力）到接触点
    pass
```

#### 4.4.2 Marching Tetrahedra 表

```python
# 代码位置：line 22-50
MARCHING_TETS_EDGE_TABLE = (...)  # 16种截交模式
TET_EDGES = ((0,1), (1,2), (2,0), (0,3), (1,3), (2,3))  # 6条边
```

**用途**：快速确定平面与四面体相交的边

## 5. sap_coupler 中其余代码的作用

### 5.1 枚举类型

```python
# 代码位置：line 59-91
class FEMFloorContactType(IntEnum):
    NONE = 0  # 不参与地面接触
    TET = 1   # 四面体（Hydroelastic）
    VERT = 2  # 顶点点接触

class RigidFloorContactType(IntEnum):
    NONE = 0
    VERT = 1
    TET = 2

class RigidRigidContactType(IntEnum):
    NONE = 0
    TET = 1
```

**作用**：
- 编译时分支：Taichi 可优化掉未使用的代码路径
- 配置灵活性：用户通过 `SAPCouplerOptions` 选择接触类型

### 5.2 Hydroelastic 压力场初始化

#### 5.2.1 FEM 压力场

```python
# 代码位置：line 331-341
def _init_hydroelastic_fem_fields_and_info(self):
    """从 FEM 实体收集预先生成的压力场"""
    fem_pressure_np = np.concatenate([
        fem_entity.pressure_field_np 
        for fem_entity in self.fem_solver.entities
    ])
    self.fem_pressure.from_numpy(fem_pressure_np)
    self.fem_pressure_gradient = ti.field(gs.ti_vec3, shape=(B, n_elements))
```

**来源**：`fem_entity.pressure_field_np` 在 FEM Entity 初始化时生成（基于 signed distance）

#### 5.2.2 刚体压力场

```python
# 代码位置：line 342-402
def _init_hydroelastic_rigid_fields_and_info(self):
    """为刚体构造体四面体化与压力场"""
    for geom in self.rigid_solver.geoms:
        # 1. 四面体剖分
        volume = geom.get_trimesh().volume
        tet_cfg = {"maxvolume": volume / 100}
        verts, elems = mesh_to_elements(geom.get_trimesh(), tet_cfg)
        
        # 2. 计算 signed distance
        signed_distance, *_ = igl.signed_distance(
            verts, geom.init_verts, geom.init_faces
        )
        
        # 3. 归一化为压力场
        distance_max = np.max(np.abs(signed_distance))
        pressure_field = np.abs(signed_distance) / distance_max * stiffness
        
        # 4. 预计算压力梯度（静止系）
        self.rigid_compute_pressure_gradient_rest()
```

**关键**：
- 刚体视为"柔顺体"：内部压力场，外部刚性运动
- 压力梯度随姿态旋转：`gradient_world = R @ gradient_rest`

### 5.3 状态管理

#### 5.3.1 批次激活标记

```python
# 代码位置：line 492-502
self.batch_active = ti.field(dtype=gs.ti_bool, shape=B)  # SAP 迭代
self.batch_pcg_active = ti.field(dtype=gs.ti_bool, shape=B)  # PCG 迭代
self.batch_linesearch_active = ti.field(dtype=gs.ti_bool, shape=B)  # 线搜索
```

**作用**：
- 独立控制每个 batch 的求解状态
- 提前终止收敛的 batch

#### 5.3.2 范数统计

```python
# 代码位置：line 496-502
sap_state = ti.types.struct(
    gradient_norm=gs.ti_float,   # ||g||²/M⁻¹
    momentum_norm=gs.ti_float,   # ||v||²ₘ
    impulse_norm=gs.ti_float,    # ||γ||²/M⁻¹
)
```

**用途**：
- 收敛判据：`||g|| < atol + rtol * max(||p||, ||γ||)`
- 监控求解质量

### 5.4 PCG 求解框架

#### 5.4.1 矩阵-向量乘积

```python
# 代码位置：line 1539-1600（伪代码简化）
@ti.func
def pcg_matvec_Ap(self, p, Ap):
    """计算 Ap = A @ p，A 是系统矩阵"""
    Ap.fill(0.0)
    
    # 1. 未约束项：M, K, D
    self.compute_fem_matrix_vector_product(p, Ap, batch_active)
    self.compute_rigid_matrix_vector_product(p, Ap, batch_active)
    
    # 2. 接触/约束项：J^T R^{-1} J
    for contact in ti.static(self.contact_handlers):
        contact.add_Ap_contact(p, Ap)
    
    if ti.static(self.rigid_solver.n_equalities > 0):
        self.equality_constraint_handler.add_Ap_constraint(p, Ap)
```

**矩阵隐式表示**：
- 无需显式构建 `A`（稀疏矩阵 O(n²) 存储）
- 通过 `matvec` 接口隐式作用

#### 5.4.2 预条件器应用

```python
@ti.func
def pcg_apply_preconditioner(self, r, z):
    """z = P^{-1} @ r，P 是预条件器"""
    # FEM: 3x3 对角块逆
    for i_b, i_v in ti.ndrange(B, n_fem_verts):
        z[i_b, i_v] = prec[i_b, i_v] @ r[i_b, i_v]
    
    # 刚体: 标量对角逆
    for i_b, i_d in ti.ndrange(B, n_rigid_dofs):
        z[i_b, i_d] = r[i_b, i_d] / mass_mat[i_d, i_d, i_b]
```

### 5.5 线搜索能量计算

#### 5.5.1 能量分解

```python
# 代码位置：line 1850-1910（简化）
@ti.kernel
def compute_linesearch_energy_and_derivatives(self):
    """计算能量 ℓ(α) 及其导数"""
    for i_b in range(B):
        alpha = self.linesearch_state[i_b].step_size
        v_trial = v_prev + alpha * dv
        
        # 1. 动力项：½(v - v*)^T M (v - v*)
        energy_kinetic = 0.5 * (v_trial - v_star) @ M @ (v_trial - v_star)
        dell_kinetic = dv @ M @ (v_trial - v_star)
        d2ell_kinetic = dv @ M @ dv
        
        # 2. 接触/约束项：Σ Φ(γ)
        energy_contact = 0.0
        dell_contact = 0.0
        for contact in ti.static(self.contact_handlers):
            Jv = contact.compute_Jx(v_trial)
            gamma = contact.compute_gamma(Jv)  # γ = -R^{-1}(Jv + v_hat)
            energy_contact += contact.potential(gamma)
            dell_contact += gamma @ J @ dv
        
        # 3. 总能量
        energy = energy_kinetic + energy_contact
        dell = dell_kinetic + dell_contact
        d2ell = d2ell_kinetic  # 近似：忽略 Hessian 的接触项
```

**SAP 能量泛函**（理论背景）：

$$
\ell(v) = \frac{1}{2}(v-v^*)^T M (v-v^*) + \sum_{c} \Phi_c(\gamma_c(v))
$$

其中 $\Phi_c(\gamma) = \frac{1}{2} \gamma^T R_c \gamma + \gamma^T v_{\text{hat},c}$ 是接触势能。

#### 5.5.2 rtsafe 算法

```python
# 代码位置：line 1910-1950
@ti.func
def update_linesearch_alpha(self):
    """混合 Newton 与二分法求解 dℓ/dα = 0"""
    for i_b in range(B):
        f = dell / dell_scale  # 归一化一阶导
        df = d2ell / dell_scale  # 归一化二阶导
        
        # Newton 步
        if ti.abs(df) > 1e-12 and f * df < 0:
            minus_dalpha = f / df
            # 截断到 [alpha_min, alpha_max]
            minus_dalpha = clamp(minus_dalpha, alpha_min - alpha, alpha_max - alpha)
        else:
            # 退化到二分法
            minus_dalpha = 0.5 * (alpha_max + alpha_min) - alpha
        
        # 更新步长
        alpha_new = alpha - minus_dalpha
        
        # 更新区间
        if f < 0:
            alpha_min = alpha
            f_lower = f
        else:
            alpha_max = alpha
            f_upper = f
```

**收敛条件**：
- `|α_max - α_min| < α_tol`（默认 1e-10）
- 或 `|f| < ftol`（默认 1e-10）

### 5.6 重置与回调

```python
# 代码位置：line 317-321
def reset(self, envs_idx=None):
    """复位接口（当前无需特殊处理）"""
    pass
```

**说明**：SAP Coupler 无持久状态需要重置（每步重新检测接触）

### 5.7 梯度接口（占位）

```python
# 代码位置：line 702-706
def couple_grad(self, i_step):
    """SAP 模式暂不提供反向传播"""
    gs.raise_exception(
        "couple_grad is not available for SAPCoupler. "
        "Please use LegacyCoupler instead."
    )
```

**原因**：
- SAP 的非光滑性（摩擦锥、单边约束）使梯度计算复杂
- 需要使用可微分的近似（如 LegacyCoupler 的光滑罚函数）

## 6. 总结

### 6.1 核心设计思想

1. **统一框架**：
   - 将接触、摩擦、约束统一为优化问题
   - 刚体与 FEM 在同一数值框架下求解

2. **模块化**：
   - 接触检测、雅可比构建、求解器解耦
   - 通过 `ContactHandler` 接口扩展新接触类型

3. **数值稳定性优先**：
   - 正则化、预条件、线搜索三重保障
   - 几何退化检查与批次独立收敛

4. **GPU 并行优化**：
   - SOA 布局、批次并行、原子操作
   - BVH 加速宽相检测

### 6.2 关键技术特点

| 方面 | 技术选择 | 优势 |
|------|---------|------|
| **接触模型** | Hydroelastic（压力场） | 避免刚性碰撞，稳定柔顺 |
| **线性求解** | PCG + 对角预条件 | 稀疏矩阵友好，GPU 并行 |
| **步长控制** | 精确线搜索（rtsafe） | 保证能量单调递减 |
| **正则化** | R = R(k, τ_d, β) | 可调刚度-稳定性权衡 |
| **宽相** | LBVH（Morton 码） | O(n log n) 构建，完全并行 |

### 6.3 使用建议

1. **接触类型选择**：
   - 精度优先：`TET`（Hydroelastic）
   - 性能优先：`VERT`（点接触）

2. **参数调优**：
   - `sap_taud`：↓ 增加阻尼，↑ 减少穿透
   - `hydroelastic_stiffness`：↑ 减少穿透，↓ 增加稳定性
   - `sap_convergence_atol/rtol`：权衡精度与性能

3. **精度要求**：
   - SAP Coupler **必须** 使用 `precision='64'`
   - FEM 必须启用 `use_implicit_solver=True`

4. **性能优化**：
   - 减少不必要的接触类型（如禁用自接触）
   - 调整 `n_sap_iterations`（默认 50）
   - 使用 batch 并行提升吞吐

### 6.4 与 LegacyCoupler 对比

| 特性 | SAPCoupler | LegacyCoupler |
|------|-----------|---------------|
| **理论基础** | 半解析主问题（SAP） | 罚函数 + LCP |
| **稳定性** | 高（线搜索保证收敛） | 中（依赖时间步长） |
| **精度** | 高（精确满足约束） | 中（软约束） |
| **可微分性** | 否 | 是 |
| **性能** | 中（多次迭代） | 高（一次求解） |
| **适用场景** | 复杂接触、高精度 | 梯度优化、实时仿真 |

---

本文档基于代码版本：`sap_coupler.py`（4494 行）
