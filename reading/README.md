# Genesis Engine 架构分析文档

本文件夹包含 Genesis 引擎的完整架构分析，包括求解器、耦合器以及其他核心模块的详细文档。

## 文档结构

### 第一部分：求解器和耦合器分析 (已完成)

这部分文档分析了 `genesis/engine/solvers` 和 `genesis/engine/couplers` 模块。

#### 求解器文档列表

### 1. [ToolSolver](tool_solver.md) - 工具求解器
- **功能**: 临时解决方案，用于刚体到柔体的单向可微耦合
- **特点**: 简单的实体管理、边界条件、梯度支持
- **涉及的类**: `ToolSolver`
- **行数**: 174行

### 2. [RigidSolver](rigid_solver.md) - 刚体求解器
- **功能**: 铰接刚体系统动力学模拟
- **特点**: 约束求解、碰撞检测、接触岛、休眠机制
- **涉及的类**: `RigidSolver`, `StaticRigidSimConfig`
- **应用**: 机器人仿真、机械臂、腿足机器人
- **行数**: 461行

### 3. [AvatarSolver](avatar_solver.md) - 角色求解器
- **功能**: 纯运动学角色动画
- **特点**: 继承RigidSolver但不考虑物理、无约束求解
- **涉及的类**: `AvatarSolver` (继承自 `RigidSolver`)
- **应用**: 角色动画、运动学控制
- **行数**: 148行

### 4. [MPMSolver](mpm_solver.md) - 物质点法求解器
- **功能**: 连续体材料模拟（弹性体、塑性体、液体、雪等）
- **特点**: 粒子-网格混合方法、SVD分解、可微分
- **涉及的类**: `MPMSolver`
- **辅助函数**: `signmax()`, `backward_svd()`
- **应用**: 软体、液体、雪、沙土
- **行数**: 314行

### 5. [SPHSolver](sph_solver.md) - 光滑粒子流体动力学求解器
- **功能**: 液体模拟
- **特点**: WCSPH和DFSPH两种压力求解方法、空间哈希
- **涉及的类**: `SPHSolver`
- **应用**: 水、液体模拟
- **行数**: 404行

### 6. [FEMSolver](fem_solver.md) - 有限元法求解器
- **功能**: 可变形固体模拟（四面体网格）
- **特点**: 显式和隐式时间积分、牛顿法+PCG+线搜索、可微分
- **涉及的类**: `FEMSolver`
- **应用**: 软体、肌肉、变形物体
- **行数**: 398行

### 7. [SFSolver](sf_solver.md) - 稳定流体求解器
- **功能**: 欧拉网格气体模拟
- **特点**: Jos Stam稳定流体方法、半拉格朗日平流、压力投影
- **涉及的类**: `SFSolver`, `TexPair`
- **应用**: 烟雾、气体扩散、流体艺术效果
- **行数**: 316行

### 8. [PBDSolver](pbd_solver.md) - 基于位置动力学求解器
- **功能**: 布料、弹性体、液体、粒子模拟
- **特点**: 无条件稳定、约束投影、XPBD、多材料
- **涉及的类**: `PBDSolver`, `MATERIAL`枚举
- **应用**: 布料、软体、液体、粒子效果
- **行数**: 440行

#### 架构总览

- **[solver_architecture_analysis.md](solver_architecture_analysis.md)** - 求解器架构深度分析
  - 代码量: 包含所有求解器
  - 核心内容: 求解器架构、设计模式、代码风格
  - 关键特性: 统一接口、批处理、可微分设计

### 第二部分：其他核心模块分析 (新增)

这部分文档分析了 `genesis/engine` 中除求解器和耦合器外的其他模块。

#### 核心模块文档

1. **[engine_overview.md](engine_overview.md)** - Genesis Engine 总览 ⭐
   - 整体架构概览
   - 模块统计和依赖关系
   - 设计模式总结
   - 代码风格规范
   - 技术栈和关键特性

2. **[boundaries_architecture_analysis.md](boundaries_architecture_analysis.md)** - 边界条件模块
   - 代码量: 74行, 2个类
   - 核心内容: CubeBoundary, FloorBoundary
   - 应用场景: 粒子边界、地板约束

3. **[states_architecture_analysis.md](states_architecture_analysis.md)** - 状态管理模块
   - 代码量: 531行, 14个类
   - 核心内容: SimState, SolverState, EntityState 三层架构
   - 关键特性: 状态快照、可微分、时间回溯

4. **[materials_architecture_analysis.md](materials_architecture_analysis.md)** - 材料模块
   - 代码量: 2,132行, 20+个类
   - 核心内容: 材料继承体系、本构模型
   - 支持材料: Rigid, MPM (Elastic, Snow, Sand), FEM (Elastic, Muscle), PBD, SPH, SF

5. **[entities_architecture_analysis.md](entities_architecture_analysis.md)** - 实体模块
   - 代码量: 10,985行, 20+个类
   - 核心内容: 10种实体类型、组件化设计
   - 实体类型: Rigid, MPM, FEM, PBD, SPH, SF, Avatar, Tool, Hybrid, Emitter

6. **[core_files_architecture_analysis.md](core_files_architecture_analysis.md)** - 核心文件
   - 代码量: 3,616行, 10+个类
   - 核心文件: scene.py, simulator.py, mesh.py, force_fields.py, bvh.py
   - 关键组件: Scene (用户接口), Simulator (仿真协调), Mesh (网格处理), ForceField (力场), BVH (碰撞加速)

## 快速导航

### 按复杂度排序
1. 🔴 entities (10,985行) - 最复杂
2. 🔴 core files (3,616行) - 核心架构
3. 🟡 materials (2,132行) - 物理模型
4. 🟢 states (531行) - 状态管理
5. 🟢 boundaries (74行) - 最简单

### 按主题浏览

**架构设计**
- [engine_overview.md](engine_overview.md) - 整体架构
- [solver_architecture_analysis.md](solver_architecture_analysis.md) - 求解器架构
- [core_files_architecture_analysis.md](core_files_architecture_analysis.md) - 核心组件

**物理模拟**
- [materials_architecture_analysis.md](materials_architecture_analysis.md) - 材料和本构模型
- [entities_architecture_analysis.md](entities_architecture_analysis.md) - 实体类型
- [boundaries_architecture_analysis.md](boundaries_architecture_analysis.md) - 边界条件

**状态和数据**
- [states_architecture_analysis.md](states_architecture_analysis.md) - 状态管理

## 文档结构

每个模块文档都按照统一的结构组织：

### 1. 概述
- 求解器的主要功能和用途
- 涉及的类和辅助函数

### 2. 功能模块划分
将求解器的功能划分为若干相对独立的模块，每个模块包含：
- **模块功能说明**
- **涉及的函数列表**
- **每个函数的详细说明**：
  - 功能描述
  - 参数说明
  - 返回值
  - 操作流程或算法

### 3. 主要功能管线
定义复杂功能的实现管线，通常包括：
- **时间步进管线**: 模拟的主循环流程
- **反向传播管线**: 梯度计算流程（如果支持可微分）
- **特定算法管线**: 如约束求解、压力求解等

### 4. 设计特点
总结求解器的设计亮点和技术特色

## 求解器对比

| 求解器 | 类型 | 可微分 | 主要应用 | 特点 |
|--------|------|--------|----------|------|
| ToolSolver | 刚体 | 部分 | 刚软耦合 | 临时方案 |
| RigidSolver | 刚体 | 否 | 机器人 | 约束求解、接触岛 |
| AvatarSolver | 刚体 | 否 | 角色动画 | 纯运动学 |
| MPMSolver | 连续体 | 是 | 软体/液体 | 粒子-网格混合 |
| SPHSolver | 流体 | 否 | 液体 | WCSPH/DFSPH |
| FEMSolver | 固体 | 是 | 可变形固体 | 隐式/显式积分 |
| SFSolver | 气体 | 否 | 烟雾 | 欧拉网格 |
| PBDSolver | 多用途 | 否 | 布料/软体/液体 | 无条件稳定 |

## 技术术语表

### 时间积分
- **显式积分**: 直接从当前状态计算下一状态，速度快但有稳定性限制
- **隐式积分**: 求解非线性方程组，稳定但计算量大
- **半隐式**: 介于两者之间的折中方案

### 求解器
- **PCG (Preconditioned Conjugate Gradient)**: 预条件共轭梯度法
- **牛顿法**: 求解非线性方程的迭代方法
- **Jacobi迭代**: 简单的线性系统求解器
- **Gauss-Seidel**: 另一种线性系统求解器

### 约束
- **等式约束**: 必须严格满足的约束（如焊接）
- **不等式约束**: 单向约束（如接触、关节限制）
- **XPBD**: 扩展PBD，使用柔度参数

### 核函数
- **Poly6核**: 用于密度计算的SPH核函数
- **Spiky核**: 用于压力计算的SPH核函数
- **Cubic spline**: 三次样条核函数

### 其他
- **SVD (Singular Value Decomposition)**: 奇异值分解
- **RNEA**: 递归牛顿-欧拉算法
- **CRB**: 复合刚体算法
- **接触岛**: 将刚体分组为独立接触岛
- **休眠**: 静止物体进入休眠状态以节省计算

## 使用建议

1. **选择合适的求解器**:
   - 刚体机器人 → RigidSolver
   - 角色动画 → AvatarSolver
   - 软体、液体 → MPMSolver
   - 水模拟 → SPHSolver
   - 精确固体 → FEMSolver
   - 烟雾气体 → SFSolver
   - 布料、快速软体 → PBDSolver

2. **可微分需求**:
   - 需要梯度 → MPMSolver或FEMSolver
   - 不需要梯度 → 其他求解器

3. **性能考虑**:
   - 实时模拟 → PBDSolver, SFSolver
   - 高精度 → FEMSolver (隐式), MPMSolver
   - 大规模 → RigidSolver (接触岛+休眠)

## 设计模式速查

### 创建型模式
- **Factory Pattern** (工厂): Scene.add_entity(), entities/
- **Builder Pattern** (构建者): Scene 构建流程, entities/
- **Singleton Pattern** (单例): Simulator 求解器实例

### 结构型模式
- **Facade Pattern** (外观): Scene 隐藏复杂性
- **Composite Pattern** (组合): RigidEntity 组件结构
- **Wrapper Pattern** (包装器): Mesh 包装 trimesh
- **Flyweight Pattern** (享元): 材料共享

### 行为型模式
- **Template Method** (模板方法): Entity 生命周期, 材料本构模型
- **Strategy Pattern** (策略): 多种本构模型, 粒子采样
- **Observer Pattern** (观察者): 实体状态变化通知
- **Memento Pattern** (备忘录): 状态快照和恢复
- **Mediator Pattern** (中介者): Simulator 协调求解器

## 代码风格规范

### 命名规范
- **类名**: PascalCase (Scene, Simulator, Entity)
- **函数/变量**: snake_case (add_entity, particle_size)
- **私有属性**: 下划线前缀 (self._idx, self._scene)
- **Taichi 变量**: _ti 后缀 (self.pos_ti)

### 文档风格
- NumPy docstring 格式
- 参数、返回值、示例完整

### 代码组织
- 模块化设计
- 清晰的继承体系
- SoA (Structure of Arrays) 数据布局

## 文档统计

### 总体统计
- **总文档数**: 14个 (8个求解器 + 6个模块分析)
- **总代码量**: ~18,000行 (不含求解器)
- **总类数**: 66+个
- **设计模式**: 10+种

### 模块统计
| 模块 | 文档 | 代码行数 | 类数量 |
|------|------|----------|--------|
| 核心文件 | 1 | 3,616 | 10+ |
| 实体 | 1 | 10,985 | 20+ |
| 材料 | 1 | 2,132 | 20+ |
| 状态 | 1 | 531 | 14 |
| 边界 | 1 | 74 | 2 |
| 总览 | 1 | - | - |

## 技术栈

### 核心依赖
- **Taichi** - GPU 加速计算
- **NumPy** - 数值计算
- **PyTorch** - 自动微分
- **Trimesh** - 网格处理
- **fast_simplification** - 网格简化

### 关键特性
- 多物理场仿真 (Rigid, MPM, FEM, PBD, SPH, SF)
- GPU 加速 (Taichi)
- 可微分仿真 (PyTorch)
- 批处理 (并行环境)
- 混合仿真 (刚柔耦合)

## 贡献

本文档集是对 Genesis 引擎完整架构的深度分析和总结。

### 源代码路径
- 求解器: `genesis/engine/solvers/`
- 耦合器: `genesis/engine/couplers/`
- 实体: `genesis/engine/entities/`
- 材料: `genesis/engine/materials/`
- 状态: `genesis/engine/states/`
- 边界: `genesis/engine/boundaries/`
- 核心文件: `genesis/engine/*.py`

## 版本信息

- **文档创建日期**: 2024年10月
- **基于代码版本**: Genesis 最新版本
- **最后更新**: 2025年10月 (新增模块分析)

## 推荐阅读顺序

### 初学者
1. [engine_overview.md](engine_overview.md) - 了解整体架构
2. [core_files_architecture_analysis.md](core_files_architecture_analysis.md) - 理解核心组件
3. [entities_architecture_analysis.md](entities_architecture_analysis.md) - 学习实体系统
4. 选择感兴趣的求解器文档深入学习

### 高级用户
1. [solver_architecture_analysis.md](solver_architecture_analysis.md) - 求解器深度分析
2. [materials_architecture_analysis.md](materials_architecture_analysis.md) - 材料和本构模型
3. [states_architecture_analysis.md](states_architecture_analysis.md) - 状态管理和可微分
4. 特定求解器的详细文档

### 开发者
1. [engine_overview.md](engine_overview.md) - 设计模式和代码规范
2. 所有模块分析文档 - 理解实现细节
3. 源代码 - 结合文档阅读实现

---

**文档维护**: 这些文档随着代码库的更新可能需要更新。如发现不一致，请参考最新源代码。
