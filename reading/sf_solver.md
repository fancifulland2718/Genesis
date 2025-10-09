# SFSolver 求解器文档

## 概述
SFSolver (Stable Fluid Solver) 是一个稳定流体求解器，用于基于欧拉网格的气体模拟。实现了Jos Stam的稳定流体方法。

**涉及的类**: `SFSolver`, `TexPair`
**涉及的辅助类**: `TexPair` (纹理对，用于交换缓冲区)

## 功能模块划分

### 1. 初始化模块 (Initialization)
**功能**: 设置SF求解器的网格和场结构

#### 涉及的类:
- **`TexPair`**
  - 功能: 纹理对，用于双缓冲交换
  - 属性:
    - `cur`: 当前缓冲区
    - `nxt`: 下一个缓冲区
  - 方法:
    - `swap()`: 交换当前和下一个缓冲区

#### 涉及的函数:
- **`__init__(self, scene, sim, options)`**
  - 功能: 构造函数，初始化SF求解器
  - 关键配置:
    - `res`: 网格分辨率(n_grid)
    - `solver_iters`: 压力投影求解器迭代次数
    - `decay`: 衰减系数
    - `inlet_s`: 入口标量
  - 计算参数:
    - `dx`: 网格间距 = 1.0 / n_grid
    - `res`: 三维网格分辨率 (n_grid, n_grid, n_grid)
  - 其他:
    - `jets`: 喷射源列表

- **`setup_fields(self)`**
  - 功能: 设置场结构
  - 字段包括:
    - `grid`: 单元状态
      - `v`: 速度场
      - `v_tmp`: 临时速度场
      - `div`: 散度场
      - `p`: 压力场
      - `q`: 标量场（对每个jet）
    - `p_swap`: 压力交换缓冲区（用于Jacobi迭代）

- **`init_fields(self)`** (Taichi Kernel)
  - 功能: 初始化场
  - 操作: 将标量场初始化为0

- **`build(self)`**
  - 功能: 构建求解器
  - 操作:
    1. 调用父类build
    2. 如果活跃，设置和初始化场

### 2. 喷射源管理模块 (Jet Management)
**功能**: 管理气体喷射源

#### 涉及的函数:
- **`set_jets(self, jets)`**
  - 功能: 设置喷射源
  - 参数: `jets` - 喷射源列表

- **`is_active(self)`**
  - 功能: 检查求解器是否活跃
  - 返回: 是否有喷射源 (len(jets) > 0)

### 3. 压力投影模块 (Pressure Projection)
**功能**: 实现压力投影以保证速度场无散度

#### 涉及的函数:
- **`pressure_jacobi(self, pf, new_pf)`** (Taichi Kernel)
  - 功能: Jacobi迭代求解压力泊松方程
  - 方程: ∇²p = ∇·v
  - 迭代公式: p_new[i] = (p[left] + p[right] + p[bottom] + p[top] + p[prev] + p[next] - div[i]) / 6
  - 参数:
    - `pf`: 当前压力场
    - `new_pf`: 新压力场

- **`divergence(self)`** (Taichi Kernel)
  - 功能: 计算速度场的散度
  - 公式: div = (v[right].x - v[left].x + v[top].y - v[bottom].y + v[next].z - v[prev].z) / (2*dx)

- **`subtract_gradient(self)`** (Taichi Kernel)
  - 功能: 从速度场中减去压力梯度，使速度场无散度
  - 公式: v = v - nabla(p)
  - 具体:
    - v.x -= (p[right] - p[left]) / (2*dx)
    - v.y -= (p[top] - p[bottom]) / (2*dx)
    - v.z -= (p[next] - p[prev]) / (2*dx)

- **`reset_swap(self)`**
  - 功能: 重置交换缓冲区
  - 操作: 将p_swap的cur和nxt清零

### 4. 平流模块 (Advection)
**功能**: 实现半拉格朗日平流

#### 涉及的函数:
- **`advect_and_impulse(self, f, t)`** (Taichi Kernel)
  - 功能: 平流速度场和标量场，并添加喷射源
  - 步骤:
    1. 对每个网格单元，使用半拉格朗日法回溯轨迹
    2. 在回溯点进行三线性插值
    3. 添加喷射源的影响
    4. 应用衰减

- **`backtrace(self, vf, p, dt)`** (Taichi函数)
  - 功能: 半拉格朗日回溯
  - 算法: RK1 (一阶龙格库塔，即欧拉法)
  - 公式: p_new = p - dt * v(p)
  - 返回: 回溯后的位置

- **`trilerp(self, qf, p)`** (Taichi函数)
  - 功能: 三线性插值速度场
  - 参数:
    - `qf`: 场
    - `p`: 插值位置
  - 返回: 插值后的速度向量

- **`trilerp_scalar(self, qf, p, qf_idx)`** (Taichi函数)
  - 功能: 三线性插值标量场
  - 参数:
    - `qf`: 场
    - `p`: 插值位置
    - `qf_idx`: 标量场索引
  - 返回: 插值后的标量值

### 5. 边界条件模块 (Boundary Conditions)
**功能**: 处理网格边界条件

#### 涉及的函数:
- **`compute_location(self, u, v, w, du, dv, dw)`** (Taichi函数)
  - 功能: 计算边界处的网格位置
  - 参数: (u,v,w) - 当前位置, (du,dv,dw) - 偏移
  - 返回: 考虑边界的位置坐标
  - 处理: 使用周期性边界条件

- **`is_free(self, u, v, w, du, dv, dw)`** (Taichi函数)
  - 功能: 检查网格单元是否自由（不在固体内）
  - 返回: 布尔值

- **`pressure_to_swap(self)`** (Taichi Kernel)
  - 功能: 将压力从grid.p复制到p_swap.cur

- **`pressure_from_swap(self)`** (Taichi Kernel)
  - 功能: 将压力从p_swap.cur复制到grid.p

### 6. 时间步进模块 (Time Stepping)
**功能**: 管理时间步进流程

#### 涉及的函数:
- **`process_input(self, in_backward)`**
  - 功能: 处理输入
  - 操作: 空实现（SF求解器不需要处理输入）

- **`substep_pre_coupling(self, f)`**
  - 功能: 耦合前的子步骤
  - 步骤:
    1. 平流: advect_and_impulse(f, t) - 平流速度和标量，添加喷射源
    2. 压力投影:
       - reset_swap() - 重置压力缓冲
       - divergence() - 计算散度
       - pressure_to_swap() - 准备压力场
       - solver_iters次Jacobi迭代
       - pressure_from_swap() - 获取压力
       - subtract_gradient() - 投影到无散度场
    3. 更新时间: t += dt

- **`substep_post_coupling(self, f)`**
  - 功能: 耦合后的子步骤
  - 操作: 空实现

### 7. 梯度管理模块 (Gradient Management)
**功能**: 管理梯度（不支持可微分）

#### 涉及的函数:
- **`reset_grad(self)`**
  - 功能: 重置梯度
  - 操作: 空实现

- **`collect_output_grads(self)`**
  - 功能: 收集输出梯度
  - 操作: 空实现（不支持可微分）

- **`add_grad_from_state(self, state)`**
  - 功能: 从状态添加梯度
  - 操作: 空实现

### 8. 状态管理模块 (State Management)
**功能**: 管理状态的保存、加载和查询

#### 涉及的函数:
- **`get_state(self, f)`**
  - 功能: 获取状态
  - 返回: 速度场和标量场的状态

- **`set_state(self, f, state, envs_idx=None)`**
  - 功能: 设置状态
  - 操作: 设置速度场和标量场

- **`save_ckpt(self, ckpt_name)`**
  - 功能: 保存检查点
  - 操作: 空实现

- **`load_ckpt(self, ckpt_name)`**
  - 功能: 加载检查点
  - 操作: 空实现

## 主要功能管线

### 稳定流体时间步进管线 (Stable Fluid Time Stepping Pipeline)
```
1. substep_pre_coupling(f) - 主要计算
   ├─ A. 平流阶段 (Advection)
   │  └─ advect_and_impulse(f, t)
   │     ├─ 对每个网格单元:
   │     ├─ 1) 半拉格朗日回溯: p_prev = backtrace(v, p, dt)
   │     ├─ 2) 三线性插值速度: v_new = trilerp(v, p_prev)
   │     ├─ 3) 三线性插值标量: q_new = trilerp_scalar(q, p_prev, idx)
   │     ├─ 4) 添加喷射源影响
   │     └─ 5) 应用衰减: v *= decay
   │  
   ├─ B. 压力投影阶段 (Pressure Projection)
   │  ├─ 1) divergence() - 计算散度: div = ∇·v
   │  ├─ 2) reset_swap() - 重置压力缓冲
   │  ├─ 3) pressure_to_swap() - 准备压力场
   │  ├─ 4) solver_iters次迭代:
   │  │  └─ pressure_jacobi() - Jacobi迭代求解: ∇²p = div
   │  │     └─ p_new = (sum of 6 neighbors - div) / 6
   │  ├─ 5) pressure_from_swap() - 获取最终压力
   │  └─ 6) subtract_gradient() - 投影: v = v - ∇p
   │  
   └─ C. 更新时间: t += dt
```

### 半拉格朗日平流详细流程 (Semi-Lagrangian Advection Detail)
```
对于网格点 p:
1. backtrace(v, p, dt) - 回溯粒子轨迹
   └─ p_prev = p - dt * v(p)
   
2. trilerp(v, p_prev) - 在回溯点插值速度
   ├─ 找到包含p_prev的网格单元
   ├─ 获取8个顶点的速度值
   ├─ 计算插值权重（三线性）
   └─ v_new = sum(weight_i * v_i)
   
3. 类似地插值标量场
   
4. 添加喷射源
   └─ 如果p在jet范围内，增强速度和标量
   
5. 应用衰减
   └─ v *= decay
```

### 压力泊松方程求解流程 (Pressure Poisson Equation Solve)
```
目标: 求解 ∇²p = ∇·v，使得 v_new = v - ∇p 满足 ∇·v_new = 0

1. 计算右端项 (RHS)
   └─ divergence() : div[i,j,k] = ∇·v[i,j,k]
   
2. Jacobi迭代求解
   └─ 迭代 solver_iters 次:
      └─ pressure_jacobi(p_cur, p_next):
         └─ 对每个网格点 [i,j,k]:
            p_next[i,j,k] = (
               p_cur[i-1,j,k] + p_cur[i+1,j,k] +
               p_cur[i,j-1,k] + p_cur[i,j+1,k] +
               p_cur[i,j,k-1] + p_cur[i,j,k+1] -
               div[i,j,k]
            ) / 6
      └─ swap(p_cur, p_next)
      
3. 投影到无散度空间
   └─ subtract_gradient():
      v[i,j,k] -= (
         ∂p/∂x,
         ∂p/∂y,
         ∂p/∂z
      )
```

## 设计特点
1. **欧拉视角**: 基于固定网格的欧拉方法，适合气体模拟
2. **稳定性**: Jos Stam的稳定流体方法，无条件稳定
3. **半拉格朗日平流**: 避免CFL条件限制，允许大时间步长
4. **压力投影**: Helmholtz-Hodge分解，保证速度场无散度
5. **Jacobi迭代**: 简单的压力泊松方程求解器
6. **喷射源**: 支持多个气体喷射源
7. **双缓冲**: 使用TexPair实现高效的缓冲区交换
8. **周期性边界**: 默认使用周期性边界条件
9. **衰减**: 模拟能量耗散
10. **不支持可微分**: 当前实现不支持梯度反向传播

## 算法基础
基于Jos Stam的"Stable Fluids"论文 (SIGGRAPH 1999):
- 分步法: 平流 → 添加力 → 压力投影
- 半拉格朗日平流保证稳定性
- 压力投影保证不可压缩性（∇·v = 0）

## 应用场景
- 烟雾模拟
- 气体扩散
- 抽象流体艺术效果
- 实时流体可视化

## 限制
- 不支持可微分模拟
- 相对粗糙的网格分辨率
- 不适合精确的流体力学模拟
- 主要用于视觉效果
