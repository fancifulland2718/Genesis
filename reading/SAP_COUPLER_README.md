# SAP Coupler 文档导航

本目录包含对 Genesis 物理引擎中 `sap_coupler` 模块的全面分析文档。

## 文档概览

### 1. [sap_coupler_analysis.md](./sap_coupler_analysis.md) - 架构分析（主文档）

**内容**：
- **第1节**：刚体与有限元求解器的信息交换机制
  - 信息交换的层次结构（状态层、接触处理层、求解层）
  - 前向传递：从状态到接触力
  - 反向传递：从接触力到速度修正
  - 信息交换的时序图

- **第2节**：求解过程中的数据组织形式
  - 数据结构层次（接触对、求解器状态、雅可比矩阵）
  - 数据访问模式（写模式、读-写模式、全局归约）
  - 内存布局优化（SOA vs AOS、批次维度）

- **第3节**：数值稳定性的工程修正
  - 正则化（SAP 参数、预条件器）
  - 几何退化处理
  - 线搜索保护
  - PCG 收敛性保障
  - 批次独立性保障

- **第4节**：除 SAPCoupler 外的支持类
  - 接触处理器（7种类型详解）
  - 约束处理器
  - BVH 结构
  - 辅助函数

- **第5节**：sap_coupler 中其余代码
  - 枚举类型
  - Hydroelastic 压力场
  - 状态管理
  - PCG 求解框架
  - 线搜索能量计算

- **第6节**：总结
  - 核心设计思想
  - 关键技术特点对比表
  - 使用建议
  - 与 LegacyCoupler 对比

**适合读者**：需要理解 SAP Coupler 整体架构和设计原理的开发者

---

### 2. [sap_coupler_implementation_details.md](./sap_coupler_implementation_details.md) - 实现细节（补充文档）

**内容**：
- **第1节**：完整的求解流程追踪
  - 单步仿真的完整调用链（带行号）
  - 关键数据流转示例（Franka 抓取 FEM 球体）
  - 典型性能分析（时间分解）

- **第2节**：接触处理器深度解析
  - RigidFemTriTetContactHandler 窄相裁剪算法
  - Hydroelastic 压力场计算（FEM/刚体）
  - 压力梯度计算与等效刚度

- **第3节**：数值算法细节
  - PCG 算法（理论背景、SAP 适配、收敛性）
  - 线搜索算法（能量泛函、rtsafe 混合算法）
  - SAP 收敛性分析

- **第4节**：工程优化技巧
  - 内存池化
  - 批次早停（27% 性能提升）
  - 原子操作优化
  - BVH 优化（Morton 码）
  - 预条件器质量提升

- **第5节**：调试与可视化
  - 接触可视化方法
  - Taichi Profiler 使用
  - 数值稳定性检查

- **第6节**：扩展与定制
  - 添加新接触类型的步骤
  - 自适应参数调整

- **第7节**：常见问题与解决方案
  - 接触溢出
  - 数值不稳定
  - 收敛缓慢
  - 性能瓶颈

**适合读者**：需要实际使用、调试或扩展 SAP Coupler 的开发者

---

## 文档特点

1. **介观视角**：聚焦架构和数据流，不深入数学公式推导
2. **代码关联**：大量引用源码位置（文件名 + 行号）
3. **实例驱动**：使用 Franka 抓取 FEM 球体作为贯穿示例
4. **工程导向**：重点关注数值稳定性、性能优化和常见问题

## 阅读建议

### 快速了解架构
1. 阅读 `sap_coupler_analysis.md` 第1节（信息交换）和第6节（总结）
2. 查看第4节的接触处理器列表

### 深入理解实现
1. 阅读 `sap_coupler_implementation_details.md` 第1节（完整调用链）
2. 重点关注第2节的窄相裁剪算法

### 解决实际问题
1. 查阅 `sap_coupler_implementation_details.md` 第7节（常见问题）
2. 使用第5节的调试方法定位问题

### 性能优化
1. 阅读 `sap_coupler_implementation_details.md` 第1.3节（性能分析）
2. 参考第4节的优化技巧

### 扩展功能
1. 学习 `sap_coupler_analysis.md` 第4节（支持类）
2. 参考 `sap_coupler_implementation_details.md` 第6.1节（添加新接触类型）

## 关键概念速查

| 概念 | 位置 |
|------|------|
| **信息交换机制** | analysis.md §1 |
| **数据组织（SOA/AOS）** | analysis.md §2.3 |
| **正则化参数（R, v_hat）** | analysis.md §3.1, details.md §2.2 |
| **接触检测（宽相/窄相）** | analysis.md §4.1, details.md §2.1 |
| **PCG 算法** | details.md §3.1 |
| **线搜索（rtsafe）** | details.md §3.2 |
| **BVH 结构** | analysis.md §4.3, details.md §4.4 |
| **性能优化** | details.md §4 |
| **调试方法** | details.md §5 |
| **常见问题** | details.md §7 |

## 源码参考

文档基于以下源文件：
- `genesis/engine/couplers/sap_coupler.py`（4494 行，核心实现）
- `genesis/engine/simulator.py`（模拟器主流程）
- `genesis/engine/entities/`（FEM/刚体实体）
- `genesis/engine/solvers/`（FEM/刚体求解器）
- `genesis/engine/bvh.py`（BVH 实现）

## 相关示例代码

- `examples/sap_coupling/franka_grasp_fem_sphere.py`
- `examples/sap_coupling/franka_grasp_rigid_cube.py`
- `examples/sap_coupling/fem_sphere_and_cube.py`
- `examples/sap_coupling/fem_fixed_constraint.py`

## 参考文献

- **SAP 原始论文**：https://arxiv.org/abs/2110.10107
- **Drake 实现**：https://drake.mit.edu/release_notes/v1.5.0.html
- **Drake 源码**：`sap_driver.cc`

---

**文档作者**：GitHub Copilot  
**基于代码版本**：Genesis main 分支（2025-10-19）  
**总文档量**：~64KB（2份 Markdown 文件）
