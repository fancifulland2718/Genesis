# Entity-Solver 交互架构与完整计算流程
# Entity-Solver Interaction Architecture and Complete Computation Flow

本文档详细分析Entity和Solver的交互过程，并提供架构图和完整的计算流程图（包含耦合机制）。

This document provides detailed analysis of Entity-Solver interaction with architecture diagrams and complete computation flow including coupling mechanisms.

---

## 1. 整体架构概览 (Overall Architecture Overview)

### 1.1 三层架构 (Three-Tier Architecture)

```mermaid
graph TB
    subgraph "用户层 User Layer"
        User[用户代码<br/>User Code]
        API[Scene API]
    end
    
    subgraph "实体层 Entity Layer"
        Entity[Entity Base]
        RigidEntity[RigidEntity]
        FEMEntity[FEMEntity]
        MPMEntity[MPMEntity]
        PBDEntity[PBDEntity]
        ToolEntity[ToolEntity]
        HybridEntity[HybridEntity]
    end
    
    subgraph "求解器层 Solver Layer"
        RigidSolver[RigidSolver]
        FEMSolver[FEMSolver]
        MPMSolver[MPMSolver]
        PBDSolver[PBDSolver]
        ToolSolver[ToolSolver]
    end
    
    subgraph "耦合层 Coupling Layer"
        SAPCoupler[SAP Coupler<br/>刚柔耦合]
        LegacyCoupler[Legacy Coupler<br/>传统耦合]
    end
    
    User --> API
    API --> Entity
    Entity --> RigidEntity
    Entity --> FEMEntity
    Entity --> MPMEntity
    Entity --> PBDEntity
    Entity --> ToolEntity
    Entity --> HybridEntity
    
    RigidEntity --> RigidSolver
    FEMEntity --> FEMSolver
    MPMEntity --> MPMSolver
    PBDEntity --> PBDSolver
    ToolEntity --> ToolSolver
    HybridEntity --> RigidSolver
    HybridEntity --> FEMSolver
    
    RigidSolver --> SAPCoupler
    FEMSolver --> SAPCoupler
    PBDSolver --> SAPCoupler
    ToolSolver --> SAPCoupler
    
    SAPCoupler --> LegacyCoupler
    
    style User fill:#e1f5ff
    style API fill:#e1f5ff
    style Entity fill:#ffe1e1
    style SAPCoupler fill:#e1ffe1
    style LegacyCoupler fill:#e1ffe1
```

### 1.2 核心交互原则 (Core Interaction Principles)

1. **单向依赖**: Entity 依赖 Solver，Solver 不依赖具体 Entity
2. **状态管理**: Entity 提供状态查询接口，Solver 管理物理状态
3. **批量处理**: Solver 批量处理同类型的所有 Entity
4. **耦合解耦**: 跨Solver交互通过Coupler进行，保持Solver独立性

---

## 2. Entity-Solver 交互详解 (Detailed Entity-Solver Interaction)

### 2.1 RigidEntity-RigidSolver 交互 (最复杂)

```mermaid
sequenceDiagram
    participant User as 用户代码
    participant RE as RigidEntity
    participant RS as RigidSolver
    participant CS as ConstraintSolver
    participant Col as Collider
    
    Note over User,Col: 初始化阶段 Initialization Phase
    User->>RE: 创建entity (从URDF/MJCF)
    RE->>RE: _load_model()
    RE->>RE: 创建Links, Joints, Geoms
    RE->>RS: 注册entity到solver
    RS->>RS: 分配内存和索引
    RS->>CS: 初始化约束求解器
    RS->>Col: 初始化碰撞器
    
    Note over User,Col: 仿真步骤 Simulation Step
    User->>RE: set_dofs_position(q)
    RE->>RS: 写入q到solver状态
    
    User->>RE: control_dofs_force(tau)
    RE->>RS: 写入控制力到solver
    
    User->>User: scene.step()
    activate RS
    
    Note over RS: Step 1: 运动学更新
    RS->>RS: kernel_step_1()
    RS->>RS: func_forward_kinematics()
    RS->>RS: 更新连杆位置和速度
    
    Note over RS: Step 2: 动力学计算
    RS->>RS: func_forward_dynamics()
    RS->>RS: func_compute_mass_matrix()
    RS->>RS: func_factor_mass()
    RS->>RS: func_torque_and_passive_force()
    RS->>RS: func_compute_qacc()
    
    Note over RS,Col: Step 3: 约束和碰撞
    RS->>Col: detect_collisions()
    Col-->>RS: 碰撞对列表
    RS->>CS: 装配约束
    RS->>CS: resolve_constraints()
    CS-->>RS: 约束力
    
    Note over RS: Step 4: 积分
    RS->>RS: kernel_step_2()
    RS->>RS: 更新速度和位置
    RS->>RS: 更新几何体位置
    
    deactivate RS
    
    User->>RE: get_dofs_position()
    RE->>RS: 读取q从solver状态
    RS-->>RE: 返回q
    RE-->>User: 返回关节位置
    
    User->>RE: get_links_pos()
    RE->>RS: 读取连杆位置
    RS-->>RE: 返回位置数组
    RE-->>User: 返回连杆位置
```

#### 2.1.1 数据流向 (Data Flow)

**Entity → Solver (写入)**:
```
用户设置 → Entity接口 → Solver状态数组
例如:
- set_dofs_position(q) → solver.dofs_state.q[entity_dof_range]
- control_dofs_force(tau) → solver.dofs_state.qf_applied[entity_dof_range]
- set_links_velocity(v) → solver.links_state.vel[entity_link_range]
```

**Solver → Entity (读取)**:
```
Solver状态数组 → Entity接口 → 用户获取
例如:
- solver.links_state.pos[entity_link_range] → get_links_pos() → 用户
- solver.dofs_state.q[entity_dof_range] → get_dofs_position() → 用户
- solver.geoms_state.pos[entity_geom_range] → get_geoms_pos() → 用户
```

#### 2.1.2 RigidSolver 核心计算流程 (Core Computation Pipeline)

```mermaid
graph TD
    Start[开始 substep]
    
    FK[正向运动学<br/>Forward Kinematics<br/>func_forward_kinematics]
    COM[计算质心<br/>Compute COM]
    
    MM[计算质量矩阵<br/>Mass Matrix<br/>func_compute_mass_matrix]
    Factor[质量矩阵分解<br/>Factor Mass<br/>func_factor_mass]
    
    Force[计算力项<br/>Compute Forces<br/>func_torque_and_passive_force]
    Acc[计算加速度<br/>Compute Acceleration<br/>func_compute_qacc]
    
    DetectCol[碰撞检测<br/>Collision Detection<br/>collider.detect]
    AssembleConst[装配约束<br/>Assemble Constraints]
    SolveConst[求解约束<br/>Solve Constraints<br/>constraint_solver.resolve]
    
    Integrate[积分<br/>Integration<br/>update q, qd]
    UpdateGeom[更新几何<br/>Update Geometry]
    
    Hibernate[休眠检测<br/>Hibernation Check]
    
    End[结束 substep]
    
    Start --> FK
    FK --> COM
    COM --> MM
    MM --> Factor
    Factor --> Force
    Force --> Acc
    
    Acc --> DetectCol
    DetectCol --> AssembleConst
    AssembleConst --> SolveConst
    SolveConst --> Integrate
    
    Integrate --> UpdateGeom
    UpdateGeom --> Hibernate
    Hibernate --> End
    
    style Start fill:#e1f5ff
    style End fill:#e1f5ff
    style FK fill:#ffe1e1
    style MM fill:#fff4e1
    style DetectCol fill:#e1ffe1
    style SolveConst fill:#e1ffe1
    style Integrate fill:#f0e1ff
```

---

### 2.2 FEMEntity-FEMSolver 交互

```mermaid
sequenceDiagram
    participant User as 用户代码
    participant FE as FEMEntity
    participant FS as FEMSolver
    
    Note over User,FS: 初始化阶段
    User->>FE: 创建entity (从网格文件)
    FE->>FE: _load_mesh()
    FE->>FE: 生成四面体网格
    FE->>FS: 注册entity到solver
    FS->>FS: 分配顶点/单元内存
    FS->>FS: 初始化刚度矩阵
    
    Note over User,FS: 仿真步骤
    User->>FE: set_state(pos, vel)
    FE->>FS: 写入状态到solver
    
    User->>User: scene.step()
    activate FS
    
    Note over FS: Step 1: 计算内力
    FS->>FS: compute_elastic_force()
    FS->>FS: 基于变形计算应力
    
    Note over FS: Step 2: 组装系统
    FS->>FS: assemble_system()
    FS->>FS: 构建刚度矩阵
    
    Note over FS: Step 3: 求解
    FS->>FS: solve_linear_system()
    FS->>FS: 隐式/显式积分
    
    Note over FS: Step 4: 更新状态
    FS->>FS: update_positions()
    FS->>FS: update_velocities()
    
    deactivate FS
    
    User->>FE: get_state()
    FE->>FS: 读取状态
    FS-->>FE: 返回pos, vel
    FE-->>User: 返回状态
```

#### 2.2.1 FEMSolver 核心流程

```mermaid
graph TD
    Start[开始 substep]
    
    ComputeDef[计算变形<br/>Compute Deformation<br/>F = ∇x]
    ComputeStress[计算应力<br/>Compute Stress<br/>P = ∂W/∂F]
    ComputeForce[计算内力<br/>Compute Internal Force<br/>f_int = -B^T P]
    
    AddExtForce[添加外力<br/>Add External Forces<br/>f_ext = gravity + user]
    
    AssembleRHS[组装右端<br/>Assemble RHS<br/>b = f_ext + f_int]
    AssembleLHS[组装左端<br/>Assemble LHS<br/>A = M - dt²K]
    
    Solve[求解线性系统<br/>Solve Linear System<br/>Ax = b]
    
    UpdateVel[更新速度<br/>Update Velocity<br/>v = v + dt*a]
    UpdatePos[更新位置<br/>Update Position<br/>x = x + dt*v]
    
    End[结束 substep]
    
    Start --> ComputeDef
    ComputeDef --> ComputeStress
    ComputeStress --> ComputeForce
    ComputeForce --> AddExtForce
    
    AddExtForce --> AssembleRHS
    AssembleRHS --> AssembleLHS
    AssembleLHS --> Solve
    
    Solve --> UpdateVel
    UpdateVel --> UpdatePos
    UpdatePos --> End
    
    style Start fill:#e1f5ff
    style End fill:#e1f5ff
    style ComputeStress fill:#ffe1e1
    style Solve fill:#e1ffe1
```

---

### 2.3 MPMEntity-MPMSolver 交互

```mermaid
sequenceDiagram
    participant User as 用户代码
    participant ME as MPMEntity
    participant MS as MPMSolver
    
    Note over User,MS: 初始化阶段
    User->>ME: 创建entity (粒子初始化)
    ME->>ME: 分配粒子内存
    ME->>MS: 注册entity到solver
    MS->>MS: 分配网格内存
    MS->>MS: 初始化背景网格
    
    Note over User,MS: 仿真步骤
    User->>User: scene.step()
    activate MS
    
    Note over MS: Step 1: P2G (粒子到网格)
    MS->>MS: kernel_p2g()
    MS->>MS: 传输质量和动量到网格
    
    Note over MS: Step 2: 网格操作
    MS->>MS: kernel_grid_op()
    MS->>MS: 更新网格速度
    MS->>MS: 应用边界条件
    
    Note over MS: Step 3: G2P (网格到粒子)
    MS->>MS: kernel_g2p()
    MS->>MS: 更新粒子位置和速度
    MS->>MS: 更新变形梯度
    
    deactivate MS
    
    User->>ME: get_pos()
    ME->>MS: 读取粒子位置
    MS-->>ME: 返回位置数组
    ME-->>User: 返回粒子位置
```

#### 2.3.1 MPMSolver P2G-G2P 流程

```mermaid
graph TD
    Start[开始 substep]
    
    ClearGrid[清空网格<br/>Clear Grid<br/>grid_v = 0, grid_m = 0]
    
    P2G_Mass[P2G: 传输质量<br/>Transfer Mass<br/>grid_m += w * m_p]
    P2G_Mom[P2G: 传输动量<br/>Transfer Momentum<br/>grid_mv += w * m_p * v_p]
    P2G_Force[P2G: 传输力<br/>Transfer Force<br/>grid_f += w * f_p]
    
    GridVel[计算网格速度<br/>Grid Velocity<br/>grid_v = grid_mv / grid_m]
    GridUpdate[更新网格速度<br/>Update Grid Velocity<br/>grid_v += dt * grid_f / grid_m]
    GridBC[应用边界条件<br/>Apply Boundary Conditions]
    
    G2P_Vel[G2P: 更新粒子速度<br/>Update Particle Velocity<br/>v_p = Σ w * grid_v]
    G2P_Pos[G2P: 更新粒子位置<br/>Update Particle Position<br/>x_p += dt * v_p]
    G2P_DefGrad[G2P: 更新变形梯度<br/>Update Deformation Gradient<br/>F_p = (I + dt∇v) * F_p]
    
    End[结束 substep]
    
    Start --> ClearGrid
    ClearGrid --> P2G_Mass
    P2G_Mass --> P2G_Mom
    P2G_Mom --> P2G_Force
    
    P2G_Force --> GridVel
    GridVel --> GridUpdate
    GridUpdate --> GridBC
    
    GridBC --> G2P_Vel
    G2P_Vel --> G2P_Pos
    G2P_Pos --> G2P_DefGrad
    G2P_DefGrad --> End
    
    style Start fill:#e1f5ff
    style End fill:#e1f5ff
    style P2G_Mass fill:#ffe1e1
    style P2G_Mom fill:#ffe1e1
    style GridUpdate fill:#fff4e1
    style G2P_Vel fill:#e1ffe1
    style G2P_Pos fill:#e1ffe1
```

---

### 2.4 PBDEntity-PBDSolver 交互

```mermaid
sequenceDiagram
    participant User as 用户代码
    participant PE as PBDEntity
    participant PS as PBDSolver
    
    Note over User,PS: 初始化阶段
    User->>PE: 创建entity (布料/软体)
    PE->>PE: _build_tet_mesh()
    PE->>PE: _init_constraints()
    PE->>PS: 注册entity到solver
    PS->>PS: 分配粒子和约束内存
    
    Note over User,PS: 仿真步骤
    User->>User: scene.step()
    activate PS
    
    Note over PS: Step 1: 预测位置
    PS->>PS: kernel_predict_pos()
    PS->>PS: x* = x + dt * v
    
    Note over PS: Step 2: 约束迭代
    loop 迭代求解
        PS->>PS: kernel_solve_constraints()
        PS->>PS: 距离约束
        PS->>PS: 弯曲约束
        PS->>PS: 碰撞约束
    end
    
    Note over PS: Step 3: 更新速度
    PS->>PS: kernel_update_velocity()
    PS->>PS: v = (x* - x) / dt
    
    Note over PS: Step 4: 更新位置
    PS->>PS: x = x*
    
    deactivate PS
    
    User->>PE: get_state()
    PE->>PS: 读取状态
    PS-->>PE: 返回pos, vel
    PE-->>User: 返回状态
```

#### 2.4.1 PBDSolver 约束投影流程

```mermaid
graph TD
    Start[开始 substep]
    
    Predict[预测位置<br/>Predict Position<br/>x* = x + dt*v + dt²*f/m]
    
    InitIter[初始化迭代<br/>iter = 0]
    
    SolveDist[求解距离约束<br/>Distance Constraints<br/>|x_i - x_j| = rest_length]
    SolveBend[求解弯曲约束<br/>Bending Constraints<br/>angle = rest_angle]
    SolveVol[求解体积约束<br/>Volume Constraints<br/>V = rest_volume]
    SolveCol[求解碰撞约束<br/>Collision Constraints<br/>penetration > 0]
    
    CheckConv{收敛?<br/>Converged?}
    IncIter[iter++]
    CheckMaxIter{达到最大迭代?}
    
    UpdateVel[更新速度<br/>Update Velocity<br/>v = (x* - x) / dt]
    UpdatePos[更新位置<br/>Update Position<br/>x = x*]
    
    ApplyDamp[应用阻尼<br/>Apply Damping<br/>v *= damping]
    
    End[结束 substep]
    
    Start --> Predict
    Predict --> InitIter
    InitIter --> SolveDist
    SolveDist --> SolveBend
    SolveBend --> SolveVol
    SolveVol --> SolveCol
    SolveCol --> CheckConv
    
    CheckConv -->|否| IncIter
    IncIter --> CheckMaxIter
    CheckMaxIter -->|否| SolveDist
    CheckMaxIter -->|是| UpdateVel
    CheckConv -->|是| UpdateVel
    
    UpdateVel --> UpdatePos
    UpdatePos --> ApplyDamp
    ApplyDamp --> End
    
    style Start fill:#e1f5ff
    style End fill:#e1f5ff
    style Predict fill:#ffe1e1
    style SolveDist fill:#fff4e1
    style SolveBend fill:#fff4e1
    style UpdateVel fill:#e1ffe1
```

---

### 2.5 ToolEntity-ToolSolver 交互

ToolEntity 比较特殊，它在 Entity 层面就实现了大量的物理更新逻辑：

```mermaid
sequenceDiagram
    participant User as 用户代码
    participant TE as ToolEntity
    participant TS as ToolSolver
    participant SC as SAPCoupler
    
    Note over User,SC: 初始化阶段
    User->>TE: 创建entity (从网格)
    TE->>TE: 加载网格顶点和面
    TE->>TS: 注册entity到solver
    TS->>TS: 分配内存
    
    Note over User,SC: 仿真步骤
    User->>TE: set_frame(pos, quat, vel, ang)
    TE->>TE: 直接更新entity内部状态
    
    User->>User: scene.step()
    
    Note over TE: Entity层面计算
    TE->>TE: substep_pre_coupling()
    TE->>TE: 更新顶点位置
    TE->>TE: 计算速度
    
    Note over SC: 耦合计算
    TE->>SC: 提供顶点信息
    SC->>SC: 检测tool与其他entity碰撞
    SC->>SC: 计算接触力
    SC-->>TE: 返回接触力
    
    Note over TE: Entity层面更新
    TE->>TE: substep_post_coupling()
    TE->>TE: 应用接触力（如需要）
    TE->>TE: 更新最终状态
    
    User->>TE: get_frame()
    TE-->>User: 返回位置和姿态
```

**特点**: ToolEntity 在 entity 层面实现了22个 `@ti.kernel` 函数，大部分物理更新逻辑不需要经过 ToolSolver

---

## 3. 跨Solver耦合机制 (Cross-Solver Coupling)

### 3.1 SAP Coupler 架构

SAP (Sweep and Prune) Coupler 负责处理不同 Solver 之间的交互，主要包括：
- Rigid-FEM 刚柔耦合
- Rigid-PBD 刚体-软体耦合
- FEM-Floor 柔体-地面碰撞
- Tool-Other entity 工具与其他物体交互

```mermaid
graph TB
    subgraph "Solver 层 Solver Layer"
        RS[RigidSolver]
        FS[FEMSolver]
        PS[PBDSolver]
        TS[ToolSolver]
    end
    
    subgraph "SAP Coupler"
        AABB[AABB 构建<br/>Build AABBs]
        BVH[BVH 构建<br/>Build BVH Tree]
        Sweep[扫描与剪枝<br/>Sweep and Prune]
        Detect[精确检测<br/>Narrow Phase Detection]
        Resolve[接触求解<br/>Contact Resolution]
    end
    
    subgraph "接触类型 Contact Types"
        RF[Rigid-FEM]
        RP[Rigid-PBD]
        RR[Rigid-Rigid]
        FF[FEM-Floor]
        TF[Tool-FEM]
        TP[Tool-PBD]
    end
    
    RS --> AABB
    FS --> AABB
    PS --> AABB
    TS --> AABB
    
    AABB --> BVH
    BVH --> Sweep
    Sweep --> Detect
    Detect --> Resolve
    
    Resolve --> RF
    Resolve --> RP
    Resolve --> RR
    Resolve --> FF
    Resolve --> TF
    Resolve --> TP
    
    RF --> RS
    RF --> FS
    RP --> RS
    RP --> PS
    TF --> TS
    TF --> FS
    
    style AABB fill:#e1f5ff
    style BVH fill:#ffe1e1
    style Sweep fill:#fff4e1
    style Detect fill:#e1ffe1
    style Resolve fill:#f0e1ff
```

### 3.2 完整的多物理仿真流程（含耦合）

```mermaid
graph TD
    Start[开始时间步<br/>Start Timestep]
    
    subgraph "各Solver独立计算 Independent Solver Computation"
        RS_Compute[RigidSolver:<br/>运动学+动力学<br/>FK, Mass Matrix, Forces]
        FS_Compute[FEMSolver:<br/>变形计算<br/>Deformation, Stress]
        MS_Compute[MPMSolver:<br/>P2G传输<br/>P2G Transfer]
        PS_Compute[PBDSolver:<br/>位置预测<br/>Position Prediction]
        TS_Compute[ToolSolver:<br/>更新Tool状态<br/>Update Tool State]
    end
    
    BuildAABB[构建AABB<br/>Build AABBs for all entities]
    BuildBVH[构建BVH树<br/>Build BVH Tree]
    
    subgraph "耦合检测 Coupling Detection"
        SweepPhase[扫描阶段<br/>Sweep Phase<br/>粗筛选potential contacts]
        NarrowPhase[精确检测<br/>Narrow Phase<br/>四面体-四面体精确测试]
    end
    
    subgraph "耦合求解 Coupling Resolution"
        CalcForce[计算接触力<br/>Calculate Contact Forces<br/>基于penetration depth]
        DistForce_Rigid[分发力到Rigid<br/>Distribute to Rigid]
        DistForce_FEM[分发力到FEM<br/>Distribute to FEM]
        DistForce_PBD[分发力到PBD<br/>Distribute to PBD]
        DistForce_Tool[分发力到Tool<br/>Distribute to Tool]
    end
    
    subgraph "各Solver完成计算 Complete Solver Computation"
        RS_Finish[RigidSolver:<br/>约束求解+积分<br/>Constraint + Integration]
        FS_Finish[FEMSolver:<br/>系统求解+积分<br/>System Solve + Integration]
        MS_Finish[MPMSolver:<br/>G2P传输<br/>G2P Transfer]
        PS_Finish[PBDSolver:<br/>约束投影<br/>Constraint Projection]
        TS_Finish[ToolSolver:<br/>完成更新<br/>Finish Update]
    end
    
    UpdateVis[更新可视化<br/>Update Visualization]
    
    End[结束时间步<br/>End Timestep]
    
    Start --> RS_Compute
    Start --> FS_Compute
    Start --> MS_Compute
    Start --> PS_Compute
    Start --> TS_Compute
    
    RS_Compute --> BuildAABB
    FS_Compute --> BuildAABB
    MS_Compute --> BuildAABB
    PS_Compute --> BuildAABB
    TS_Compute --> BuildAABB
    
    BuildAABB --> BuildBVH
    BuildBVH --> SweepPhase
    SweepPhase --> NarrowPhase
    
    NarrowPhase --> CalcForce
    CalcForce --> DistForce_Rigid
    CalcForce --> DistForce_FEM
    CalcForce --> DistForce_PBD
    CalcForce --> DistForce_Tool
    
    DistForce_Rigid --> RS_Finish
    DistForce_FEM --> FS_Finish
    DistForce_PBD --> PS_Finish
    DistForce_Tool --> TS_Finish
    
    RS_Finish --> UpdateVis
    FS_Finish --> UpdateVis
    MS_Finish --> UpdateVis
    PS_Finish --> UpdateVis
    TS_Finish --> UpdateVis
    
    UpdateVis --> End
    
    style Start fill:#e1f5ff
    style End fill:#e1f5ff
    style RS_Compute fill:#ffe1e1
    style FS_Compute fill:#ffe1e1
    style MS_Compute fill:#ffe1e1
    style PS_Compute fill:#ffe1e1
    style SweepPhase fill:#fff4e1
    style NarrowPhase fill:#fff4e1
    style CalcForce fill:#e1ffe1
    style RS_Finish fill:#f0e1ff
    style FS_Finish fill:#f0e1ff
```

### 3.3 刚柔耦合详细流程 (Rigid-FEM Coupling Detail)

```mermaid
sequenceDiagram
    participant RS as RigidSolver
    participant SAP as SAPCoupler
    participant FS as FEMSolver
    
    Note over RS,FS: 阶段1: 构建空间索引
    RS->>SAP: 提供刚体四面体网格
    FS->>SAP: 提供FEM四面体网格
    SAP->>SAP: 为所有四面体构建AABB
    SAP->>SAP: 构建LBVH树
    
    Note over RS,FS: 阶段2: 粗筛选
    SAP->>SAP: Sweep Phase
    SAP->>SAP: 查询AABB重叠对
    SAP->>SAP: 过滤明显不相交的对
    
    Note over RS,FS: 阶段3: 精确检测
    loop 对每个候选接触对
        SAP->>SAP: 四面体-四面体相交测试
        SAP->>SAP: Marching Tetrahedra算法
        SAP->>SAP: 计算接触面积和法向
    end
    
    Note over RS,FS: 阶段4: 计算接触力
    SAP->>SAP: 基于penetration depth计算力
    SAP->>SAP: F = k * depth * area
    SAP->>SAP: 计算接触点重心坐标
    
    Note over RS,FS: 阶段5: 分发接触力
    SAP->>RS: 将力分发到刚体顶点
    SAP->>FS: 将力分发到FEM节点
    
    Note over RS,FS: 阶段6: 各自完成求解
    RS->>RS: 包含接触力的约束求解
    FS->>FS: 包含接触力的隐式积分
```

### 3.4 HybridEntity 的特殊处理

HybridEntity 结合了刚体和柔体，需要同时与 RigidSolver 和 FEMSolver 交互：

```mermaid
graph TD
    HE[HybridEntity]
    
    subgraph "刚体部分 Rigid Part"
        RigidLinks[刚体Links]
        RigidJoints[刚体Joints]
    end
    
    subgraph "柔体部分 FEM Part"
        FEMNodes[FEM节点]
        FEMElements[FEM单元]
    end
    
    subgraph "附着约束 Attachment Constraints"
        AttachPts[附着点<br/>Attachment Points]
        AttachConst[附着约束<br/>刚体Link ↔ FEM节点]
    end
    
    RS[RigidSolver]
    FS[FEMSolver]
    SAP[SAPCoupler]
    
    HE --> RigidLinks
    HE --> RigidJoints
    HE --> FEMNodes
    HE --> FEMElements
    HE --> AttachPts
    
    RigidLinks --> RS
    RigidJoints --> RS
    FEMNodes --> FS
    FEMElements --> FS
    
    AttachPts --> AttachConst
    AttachConst --> RS
    AttachConst --> FS
    AttachConst --> SAP
    
    RS --> SAP
    FS --> SAP
    
    style HE fill:#e1f5ff
    style RigidLinks fill:#ffe1e1
    style FEMNodes fill:#fff4e1
    style AttachConst fill:#e1ffe1
    style SAP fill:#f0e1ff
```

---

## 4. 关键数据结构 (Key Data Structures)

### 4.1 Solver 状态数组 (Solver State Arrays)

每个 Solver 维护结构化数组 (Structure of Arrays, SoA) 存储所有实体的状态：

```python
# RigidSolver 状态
class RigidSolverState:
    # 连杆状态 Link State (n_links)
    links_state.pos: ti.field(ti.math.vec3)      # 位置
    links_state.quat: ti.field(ti.math.vec4)     # 四元数
    links_state.vel: ti.field(ti.math.vec3)      # 线速度
    links_state.ang: ti.field(ti.math.vec3)      # 角速度
    
    # 自由度状态 DOF State (n_dofs)
    dofs_state.q: ti.field(ti.f32)               # 关节位置
    dofs_state.qd: ti.field(ti.f32)              # 关节速度
    dofs_state.qacc: ti.field(ti.f32)            # 关节加速度
    dofs_state.qf_applied: ti.field(ti.f32)      # 外力
    dofs_state.qf_constraint: ti.field(ti.f32)   # 约束力
    
    # 几何状态 Geom State (n_geoms)
    geoms_state.pos: ti.field(ti.math.vec3)      # 几何中心
    geoms_state.quat: ti.field(ti.math.vec4)     # 几何方向
```

```python
# FEMSolver 状态
class FEMSolverState:
    # 节点状态 Node State (n_nodes)
    nodes_state.pos: ti.field(ti.math.vec3)      # 当前位置
    nodes_state.vel: ti.field(ti.math.vec3)      # 速度
    nodes_state.force: ti.field(ti.math.vec3)    # 受力
    nodes_state.rest_pos: ti.field(ti.math.vec3) # 静态位置
    
    # 单元状态 Element State (n_elements)
    elements.vertices: ti.field(ti.i32, shape=4)  # 四面体顶点索引
    elements.F: ti.field(ti.math.mat3)            # 变形梯度
    elements.stress: ti.field(ti.math.mat3)       # 应力张量
```

```python
# MPMSolver 状态
class MPMSolverState:
    # 粒子状态 Particle State (n_particles)
    particles.pos: ti.field(ti.math.vec3)        # 位置
    particles.vel: ti.field(ti.math.vec3)        # 速度
    particles.F: ti.field(ti.math.mat3)          # 变形梯度
    particles.C: ti.field(ti.math.mat3)          # APIC矩阵
    particles.mass: ti.field(ti.f32)             # 质量
    
    # 网格状态 Grid State (grid_size^3)
    grid.v: ti.field(ti.math.vec3)               # 网格速度
    grid.m: ti.field(ti.f32)                     # 网格质量
```

### 4.2 Entity 索引范围 (Entity Index Ranges)

每个 Entity 存储其在 Solver 状态数组中的索引范围：

```python
class RigidEntity:
    _link_start: int       # 连杆起始索引
    _link_end: int         # 连杆结束索引（不含）
    _joint_start: int      # 关节起始索引
    _dof_start: int        # 自由度起始索引
    _geom_start: int       # 几何起始索引
    
    # 访问示例
    def get_links_pos(self):
        return self._solver.links_state.pos[self._link_start:self._link_end]
```

这种设计允许：
1. **批量处理**: Solver 可以高效地处理所有实体
2. **并行计算**: GPU kernel 可以并行访问数组
3. **清晰边界**: Entity 通过索引范围访问自己的数据

---

## 5. 总结 (Summary)

### 5.1 Entity-Solver 交互模式

1. **初始化**: Entity 加载模型 → 注册到 Solver → Solver 分配内存
2. **输入**: 用户 → Entity API → Solver 状态数组
3. **计算**: Solver 批量处理所有 Entity → 更新状态数组
4. **输出**: Solver 状态数组 → Entity API → 用户
5. **耦合**: SAPCoupler 协调跨 Solver 交互

### 5.2 设计优势

✅ **性能优化**: 批量处理 + GPU 并行  
✅ **清晰分层**: 用户层 → Entity 层 → Solver 层  
✅ **灵活扩展**: 新 Entity 类型易于添加  
✅ **多物理支持**: 通过 Coupler 实现跨 Solver 交互

### 5.3 关键组件职责

| 组件 | 职责 |
|------|------|
| **Entity** | 提供用户API，管理索引范围，数据访问接口 |
| **Solver** | 批量物理计算，状态更新，内存管理 |
| **Coupler** | 跨Solver交互，碰撞检测，接触求解 |
| **Scene** | 协调所有组件，驱动仿真循环 |

---

*生成时间: 2025-10-26*
*分析范围: genesis/engine/entities 和 genesis/engine/solvers*
