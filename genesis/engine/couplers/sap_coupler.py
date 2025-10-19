from typing import TYPE_CHECKING
import math

import igl
import numpy as np
import gstaichi as ti

import genesis as gs
from genesis.options.solvers import SAPCouplerOptions
from genesis.repr_base import RBC
from genesis.engine.bvh import AABB, LBVH, FEMSurfaceTetLBVH, RigidTetLBVH
import genesis.utils.element as eu
import genesis.utils.array_class as array_class
import genesis.utils.geom as gu
from genesis.constants import IntEnum, EQUALITY_TYPE
from genesis.engine.solvers.rigid.rigid_solver_decomp import func_update_all_verts

if TYPE_CHECKING:
    from genesis.engine.simulator import Simulator

# ───────────────────────────────── 常量/查找表 ─────────────────────────────────
# Marching Tetrahedra 的边查找表（每项给出交平面对应的四条候选边索引，-1 表示不存在）
MARCHING_TETS_EDGE_TABLE = (
    (-1, -1, -1, -1),
    (0, 3, 2, -1),
    (0, 1, 4, -1),
    (4, 3, 2, 1),
    (1, 2, 5, -1),
    (0, 3, 5, 1),
    (0, 2, 5, 4),
    (3, 5, 4, -1),
    (3, 4, 5, -1),
    (4, 5, 2, 0),
    (1, 5, 3, 0),
    (1, 5, 2, -1),
    (1, 2, 3, 4),
    (0, 4, 1, -1),
    (0, 2, 3, -1),
    (-1, -1, -1, -1),
)

# 四面体 6 条边（端点索引）
TET_EDGES = (
    (0, 1),
    (1, 2),
    (2, 0),
    (0, 3),
    (1, 3),
    (2, 3),
)

# 判断两个向量是否近似同向的余弦阈值（严格正向可置 0）
COS_ANGLE_THRESHOLD = math.cos(math.pi * 5.0 / 8.0)

# 单次 AABB 查询估计的最大接触对数量（用于防止溢出）
MAX_N_QUERY_RESULT_PER_AABB = 32


class FEMFloorContactType(IntEnum):
    """
    FEM 与地面接触的枚举类型：
      - NONE: 不参与地面接触
      - TET : 基于四面体（流固/柔顺）接触
      - VERT: 顶点点接触
    """
    NONE = 0
    TET = 1
    VERT = 2


class RigidFloorContactType(IntEnum):
    """
    刚体与地面接触的枚举类型：
      - NONE: 不参与地面接触
      - VERT: 顶点点接触
      - TET : 四面体（流固/柔顺）接触
    """
    NONE = 0
    VERT = 1
    TET = 2


class RigidRigidContactType(IntEnum):
    """
    刚体-刚体接触的枚举类型：
      - NONE: 不处理刚体间接触
      - TET : 四面体（流固/柔顺）接触
    """
    NONE = 0
    TET = 1


@ti.func
def tri_barycentric(p, tri_vertices, normal):
    """
    计算点 p 相对于三角形（tri_vertices 形状 3x3，列为顶点）的重心坐标。

    参数
    ----
    p:
        空间中的查询点。
    tri_vertices:
        3x3 矩阵，每列为三角形的一个顶点。
    normal:
        三角形法向量。

    说明
    ----
    假设三角形非退化。
    """
    v0 = tri_vertices[:, 0]
    v1 = tri_vertices[:, 1]
    v2 = tri_vertices[:, 2]

    # 用向量叉积与法向求面积倒数
    area_tri_inv = 1.0 / (v1 - v0).cross((v2 - v0)).dot(normal)

    # 重心坐标
    b0 = (v2 - v1).cross(p - v1).dot(normal) * area_tri_inv
    b1 = (v0 - v2).cross(p - v2).dot(normal) * area_tri_inv
    b2 = 1.0 - b0 - b1

    return gs.ti_vec3(b0, b1, b2)


@ti.func
def tet_barycentric(p, tet_vertices):
    """
    计算点 p 相对于四面体（tet_vertices 形状 3x4，列为顶点）的重心坐标 (b0..b3)。
    """
    v0 = tet_vertices[:, 0]
    v1 = tet_vertices[:, 1]
    v2 = tet_vertices[:, 2]
    v3 = tet_vertices[:, 3]

    # 体积倒数（混合积）
    vol_tet_inv = 1.0 / ((v1 - v0).dot((v2 - v0).cross(v3 - v0)))

    # 重心坐标
    b0 = (p - v1).dot((v3 - v1).cross(v2 - v1)) * vol_tet_inv
    b1 = (p - v2).dot((v3 - v2).cross(v0 - v2)) * vol_tet_inv
    b2 = (p - v3).dot((v1 - v3).cross(v0 - v3)) * vol_tet_inv
    b3 = 1.0 - b0 - b1 - b2

    return ti.Vector([b0, b1, b2, b3], dt=gs.ti_float)


@ti.data_oriented
class SAPCoupler(RBC):
    """
    SAP 耦合器：基于 SAP（Semi-Analytic Primal，半解析主问题）统一处理不同求解器之间的接触/摩擦/约束耦合。
    要点：
      1) 通过各 ContactHandler 构建接触对（含法向/切向、压力/梯度等）。
      2) 根据 Delassus 等效质量设置正则（法向/切向 R）与稳定速度 v_hat。
      3) 组装未约束项与接触/约束项，使用 PCG 求解 dv，再用精确线搜索更新 v。
      4) 支持 Hydroelastic（压力场+梯度）的柔顺接触。

    说明
    ----
    目前所有 batch 共享同一约束配置（如关节等式约束在所有 batch 一致）。
    参考：
      Paper: https://arxiv.org/abs/2110.10107
      Drake: https://drake.mit.edu/release_notes/v1.5.0.html
      Code : sap_driver.cc（Drake 实现参考）
    """

    # ------------------------------------------------------------------------------------
    # --------------------------------- 初始化 -------------------------------------------
    # ------------------------------------------------------------------------------------

    def __init__(
        self,
        simulator: "Simulator",
        options: "SAPCouplerOptions",
    ) -> None:
        self.sim = simulator
        self.options = options
        self.rigid_solver = self.sim.rigid_solver
        self.fem_solver = self.sim.fem_solver
        self._n_sap_iterations = options.n_sap_iterations
        self._n_pcg_iterations = options.n_pcg_iterations
        self._n_linesearch_iterations = options.n_linesearch_iterations
        self._sap_convergence_atol = options.sap_convergence_atol
        self._sap_convergence_rtol = options.sap_convergence_rtol
        self._sap_taud = options.sap_taud
        self._sap_beta = options.sap_beta
        self._sap_sigma = options.sap_sigma
        self._pcg_threshold = options.pcg_threshold
        self._linesearch_ftol = options.linesearch_ftol
        self._linesearch_max_step_size = options.linesearch_max_step_size
        self._hydroelastic_stiffness = options.hydroelastic_stiffness
        self._point_contact_stiffness = options.point_contact_stiffness
        if gs.ti_float == ti.f32:
            raise gs.GenesisException(
                "SAPCoupler 不支持 32 位精度，请在初始化 Genesis 时设置 precision='64'。"
            )
        # FEM 地面接触类型
        if options.fem_floor_contact_type == "tet":
            self._fem_floor_contact_type = FEMFloorContactType.TET
        elif options.fem_floor_contact_type == "vert":
            self._fem_floor_contact_type = FEMFloorContactType.VERT
        elif options.fem_floor_contact_type == "none":
            self._fem_floor_contact_type = FEMFloorContactType.NONE
        else:
            gs.raise_exception(
                f"非法 FEM 地面接触类型: {options.fem_floor_contact_type}，应为 'tet'、'vert' 或 'none'。"
            )
        self._enable_fem_self_tet_contact = options.enable_fem_self_tet_contact

        # 刚体地面接触类型
        if options.rigid_floor_contact_type == "vert":
            self._rigid_floor_contact_type = RigidFloorContactType.VERT
        elif options.rigid_floor_contact_type == "tet":
            self._rigid_floor_contact_type = RigidFloorContactType.TET
        elif options.rigid_floor_contact_type == "none":
            self._rigid_floor_contact_type = RigidFloorContactType.NONE
        else:
            gs.raise_exception(
                f"非法刚体地面接触类型: {options.rigid_floor_contact_type}，应为 'vert' 或 'none'。"
            )
        self._enable_rigid_fem_contact = options.enable_rigid_fem_contact

        # 刚体-刚体接触类型
        if options.rigid_rigid_contact_type == "tet":
            self._rigid_rigid_contact_type = RigidRigidContactType.TET
        elif options.rigid_rigid_contact_type == "none":
            self._rigid_rigid_contact_type = RigidRigidContactType.NONE
        else:
            gs.raise_exception(
                f"非法刚体-刚体接触类型: {options.rigid_rigid_contact_type}，应为 'tet' 或 'none'。"
            )

        self._rigid_compliant = False  # 是否启用刚体的“柔顺体场”（由是否构建体网格决定）

    # ------------------------------------------------------------------------------------
    # --------------------------------- 初始化 -------------------------------------------
    # ------------------------------------------------------------------------------------

    def build(self) -> None:
        """
        构建耦合器运行所需的字段与处理器：
          - 校验启用的求解器与接触开关
          - 初始化 FEM / 刚体 hydroelastic 相关数据
          - 构建 ContactHandlers / 约束处理器
          - 初始化 BVH / SAP / PCG / 线搜索字段
        """
        self._B = self.sim._B
        self.contact_handlers = []
        self._enable_rigid_fem_contact &= self.rigid_solver.is_active() and self.fem_solver.is_active()
        self._enable_fem_self_tet_contact &= self.fem_solver.is_active()

        init_tet_tables = False

        # FEM 相关
        if self.fem_solver.is_active():
            if self.fem_solver._use_implicit_solver is False:
                gs.raise_exception(
                    "SAPCoupler 要求 FEM 使用隐式求解，请在 FEM 选项设置 use_implicit_solver=True。"
                )
            if self._fem_floor_contact_type == FEMFloorContactType.TET or self._enable_fem_self_tet_contact:
                init_tet_tables = True
                self._init_hydroelastic_fem_fields_and_info()

            if self._fem_floor_contact_type == FEMFloorContactType.TET:
                self.fem_floor_tet_contact = FEMFloorTetContactHandler(self.sim)
                self.contact_handlers.append(self.fem_floor_tet_contact)

            if self._fem_floor_contact_type == FEMFloorContactType.VERT:
                self.fem_floor_vert_contact = FEMFloorVertContactHandler(self.sim)
                self.contact_handlers.append(self.fem_floor_vert_contact)

            if self._enable_fem_self_tet_contact:
                self.fem_self_tet_contact = FEMSelfTetContactHandler(self.sim)
                self.contact_handlers.append(self.fem_self_tet_contact)

            self._init_fem_fields()

        # 刚体相关
        if self.rigid_solver.is_active():
            if (
                self._rigid_floor_contact_type == RigidFloorContactType.TET
                or self._rigid_rigid_contact_type == RigidRigidContactType.TET
            ):
                init_tet_tables = True
                self._init_hydroelastic_rigid_fields_and_info()

            self._init_rigid_fields()

            if self._rigid_floor_contact_type == RigidFloorContactType.VERT:
                self.rigid_floor_vert_contact = RigidFloorVertContactHandler(self.sim)
                self.contact_handlers.append(self.rigid_floor_vert_contact)
            elif self._rigid_floor_contact_type == RigidFloorContactType.TET:
                self.rigid_floor_tet_contact = RigidFloorTetContactHandler(self.sim)
                self.contact_handlers.append(self.rigid_floor_tet_contact)

            if self._rigid_rigid_contact_type == RigidRigidContactType.TET:
                self.rigid_rigid_tet_contact = RigidRigidTetContactHandler(self.sim)
                self.contact_handlers.append(self.rigid_rigid_tet_contact)

            # TODO: 目前不支持运行中动态新增的约束（如运行期注册的焊接）
            if self.rigid_solver.n_equalities > 0:
                self._init_equality_constraint()

        # 刚体-FEM 接触（基于三角形-四面体）
        if self._enable_rigid_fem_contact:
            self.rigid_fem_contact = RigidFemTriTetContactHandler(self.sim)
            self.contact_handlers.append(self.rigid_fem_contact)

        # 初始化 BVH/表格与数值字段
        self._init_bvh()
        if init_tet_tables:
            self._init_tet_tables()
        self._init_sap_fields()
        self._init_pcg_fields()
        self._init_linesearch_fields()

    def reset(self, envs_idx=None):
        """
        复位接口（当前无需特殊处理）。
        """
        pass

    def _init_tet_tables(self):
        # 构建四面体截交的边查找表
        self.MarchingTetsEdgeTable = ti.field(gs.ti_ivec4, shape=len(MARCHING_TETS_EDGE_TABLE))
        self.MarchingTetsEdgeTable.from_numpy(np.array(MARCHING_TETS_EDGE_TABLE, dtype=gs.np_int))

        self.TetEdges = ti.field(gs.ti_ivec2, shape=len(TET_EDGES))
        self.TetEdges.from_numpy(np.array(TET_EDGES, dtype=gs.np_int))

    def _init_hydroelastic_fem_fields_and_info(self):
        """
        FEM 的压力场与压力梯度场：
          - 从各 FEM 实体收集预先生成的 pressure_field_np
          - 为每个元素分配压力梯度缓存（运行时计算）
        """
        self.fem_pressure = ti.field(gs.ti_float, shape=(self.fem_solver.n_vertices))
        fem_pressure_np = np.concatenate([fem_entity.pressure_field_np for fem_entity in self.fem_solver.entities])
        self.fem_pressure.from_numpy(fem_pressure_np)
        self.fem_pressure_gradient = ti.field(gs.ti_vec3, shape=(self.fem_solver._B, self.fem_solver.n_elements))

    def _init_hydroelastic_rigid_fields_and_info(self):
        """
        刚体的“体四面体化 + 压力场”：
          - 对每个参与碰撞的刚体几何做四面体剖分（根据体积估计分辨率）
          - 使用 signed distance 归一化构造压力场（0..stiffness）
          - 预计算每个四面体的压力梯度（静止系），运行时按姿态旋转
        """
        rigid_volume_verts = []
        rigid_volume_elems = []
        rigid_volume_verts_geom_idx = []
        rigid_volume_elems_geom_idx = []
        rigid_pressure_field = []
        offset = 0
        for geom in self.rigid_solver.geoms:
            if geom.contype or geom.conaffinity:
                if geom.type == gs.GEOM_TYPE.PLANE:
                    raise gs.GenesisException("不支持平面作为用户指定的体碰撞几何。")
                volume = geom.get_trimesh().volume
                tet_cfg = {"nobisect": False, "maxvolume": volume / 100}
                verts, elems = eu.split_all_surface_tets(*eu.mesh_to_elements(file=geom.get_trimesh(), tet_cfg=tet_cfg))
                rigid_volume_verts.append(verts)
                rigid_volume_elems.append(elems + offset)
                rigid_volume_verts_geom_idx.append(np.full(len(verts), geom.idx, dtype=np.int32))
                rigid_volume_elems_geom_idx.append(np.full(len(elems), geom.idx, dtype=np.int32))
                signed_distance, *_ = igl.signed_distance(verts, geom.init_verts, geom.init_faces)
                signed_distance = signed_distance.astype(gs.np_float, copy=False)

                distance_unsigned = np.abs(signed_distance)
                distance_max = np.max(distance_unsigned)
                if distance_max < gs.EPS:
                    gs.raise_exception(
                        f"压力场最大距离过小: {distance_max}，可能是网格没有内部顶点。"
                    )
                pressure_field_np = distance_unsigned / distance_max * self._hydroelastic_stiffness
                rigid_pressure_field.append(pressure_field_np)
                offset += len(verts)
        if not rigid_volume_verts:
            gs.raise_exception("未找到刚体碰撞几何。")
        rigid_volume_verts_np = np.concatenate(rigid_volume_verts, axis=0, dtype=np.float32)
        rigid_volume_elems_np = np.concatenate(rigid_volume_elems, axis=0, dtype=np.float32)
        rigid_volume_verts_geom_idx_np = np.concatenate(rigid_volume_verts_geom_idx, axis=0, dtype=np.float32)
        rigid_volume_elems_geom_idx_np = np.concatenate(rigid_volume_elems_geom_idx, axis=0, dtype=np.float32)
        rigid_pressure_field_np = np.concatenate(rigid_pressure_field, axis=0, dtype=np.float32)

        self.n_rigid_volume_verts = len(rigid_volume_verts_np)
        self.n_rigid_volume_elems = len(rigid_volume_elems_np)
        self.rigid_volume_verts_rest = ti.field(gs.ti_vec3, shape=(self.n_rigid_volume_verts,))
        self.rigid_volume_verts_rest.from_numpy(rigid_volume_verts_np)
        self.rigid_volume_verts = ti.field(gs.ti_vec3, shape=(self._B, self.n_rigid_volume_verts))
        self.rigid_volume_elems = ti.field(gs.ti_ivec4, shape=(self.n_rigid_volume_elems,))
        self.rigid_volume_elems.from_numpy(rigid_volume_elems_np)
        self.rigid_volume_verts_geom_idx = ti.field(gs.ti_int, shape=(self.n_rigid_volume_verts,))
        self.rigid_volume_verts_geom_idx.from_numpy(rigid_volume_verts_geom_idx_np)
        self.rigid_volume_elems_geom_idx = ti.field(gs.ti_int, shape=(self.n_rigid_volume_elems,))
        self.rigid_volume_elems_geom_idx.from_numpy(rigid_volume_elems_geom_idx_np)
        self.rigid_pressure_field = ti.field(gs.ti_float, shape=(self.n_rigid_volume_verts,))
        self.rigid_pressure_field.from_numpy(rigid_pressure_field_np)
        self.rigid_pressure_gradient_rest = ti.field(gs.ti_vec3, shape=(self.n_rigid_volume_elems,))
        self.rigid_pressure_gradient = ti.field(gs.ti_vec3, shape=(self._B, self.n_rigid_volume_elems))
        self.rigid_compute_pressure_gradient_rest()
        self._rigid_compliant = True

    @ti.func
    def rigid_update_volume_verts_pressure_gradient(self):
        """
        根据当前刚体姿态，更新体顶点坐标与压力梯度（旋转到世界系）。
        """
        for i_b, i_v in ti.ndrange(self._B, self.n_rigid_volume_verts):
            i_g = self.rigid_volume_verts_geom_idx[i_v]
            pos = self.rigid_solver.geoms_state.pos[i_g, i_b]
            quat = self.rigid_solver.geoms_state.quat[i_g, i_b]
            R = gu.ti_quat_to_R(quat)
            self.rigid_volume_verts[i_b, i_v] = R @ self.rigid_volume_verts_rest[i_v] + pos

        for i_b, i_e in ti.ndrange(self._B, self.n_rigid_volume_elems):
            i_g = self.rigid_volume_elems_geom_idx[i_e]
            pos = self.rigid_solver.geoms_state.pos[i_g, i_b]
            quat = self.rigid_solver.geoms_state.quat[i_g, i_b]
            R = gu.ti_quat_to_R(quat)
            self.rigid_pressure_gradient[i_b, i_e] = R @ self.rigid_pressure_gradient_rest[i_e]

    @ti.kernel
    def rigid_compute_pressure_gradient_rest(self):
        """
        预计算刚体四面体的“静止系”压力梯度（按顶点压力对三角面贡献的近似累积）。
        """
        grad = ti.static(self.rigid_pressure_gradient_rest)
        for i_e in range(self.n_rigid_volume_elems):
            grad[i_e].fill(0.0)
            for i in ti.static(range(4)):
                i_v0 = self.rigid_volume_elems[i_e][i]
                i_v1 = self.rigid_volume_elems[i_e][(i + 1) % 4]
                i_v2 = self.rigid_volume_elems[i_e][(i + 2) % 4]
                i_v3 = self.rigid_volume_elems[i_e][(i + 3) % 4]
                pos_v0 = self.rigid_volume_verts_rest[i_v0]
                pos_v1 = self.rigid_volume_verts_rest[i_v1]
                pos_v2 = self.rigid_volume_verts_rest[i_v2]
                pos_v3 = self.rigid_volume_verts_rest[i_v3]

                e10 = pos_v0 - pos_v1
                e12 = pos_v2 - pos_v1
                e13 = pos_v3 - pos_v1

                area_vector = e12.cross(e13)
                signed_volume = area_vector.dot(e10)
                if ti.abs(signed_volume) > gs.EPS:
                    grad_i = area_vector / signed_volume
                    grad[i_e] += grad_i * self.rigid_pressure_field[i_v0]

    def _init_bvh(self):
        """
        初始化各类 BVH（AABB + LBVH），用于联系对宽相查询：
          - FEM 表面四面体自接触
          - 刚体三角（用于刚体-FEM）
          - 刚体体四面体（用于刚体-刚体）
        """
        if self._enable_fem_self_tet_contact:
            self.fem_surface_tet_aabb = AABB(self.fem_solver._B, self.fem_solver.n_surface_elements)
            self.fem_surface_tet_bvh = FEMSurfaceTetLBVH(
                self.fem_solver, self.fem_surface_tet_aabb, max_n_query_result_per_aabb=MAX_N_QUERY_RESULT_PER_AABB
            )

        if self._enable_rigid_fem_contact:
            self.rigid_tri_aabb = AABB(self.sim._B, self.rigid_solver.n_faces)
            max_n_query_result_per_aabb = (
                max(self.rigid_solver.n_faces, self.fem_solver.n_surface_elements)
                * MAX_N_QUERY_RESULT_PER_AABB
                // self.rigid_solver.n_faces
            )
            self.rigid_tri_bvh = LBVH(self.rigid_tri_aabb, max_n_query_result_per_aabb)

        if self.rigid_solver.is_active() and self._rigid_rigid_contact_type == RigidRigidContactType.TET:
            self.rigid_tet_aabb = AABB(self.sim._B, self.n_rigid_volume_elems)
            self.rigid_tet_bvh = RigidTetLBVH(
                self, self.rigid_tet_aabb, max_n_query_result_per_aabb=MAX_N_QUERY_RESULT_PER_AABB
            )

    def _init_equality_constraint(self):
        """
        初始化刚体等式约束（当前仅支持关节等式），将其接入 SAP 管线。
        """
        # TODO: 动态注册的焊接等约束目前不支持（需传入 constraint_state）
        self.equality_constraint_handler = RigidConstraintHandler(self.sim)
        self.equality_constraint_handler.build_constraints(
            self.rigid_solver.equalities_info,
            self.rigid_solver.joints_info,
            self.rigid_solver._static_rigid_sim_config,
            self.rigid_solver._static_rigid_sim_cache_key,
        )

    def _init_sap_fields(self):
        """
        初始化 SAP 全局状态（batch 激活标记、范数统计）。
        """
        self.batch_active = ti.field(dtype=gs.ti_bool, shape=self.sim._B, needs_grad=False)
        sap_state = ti.types.struct(
            gradient_norm=gs.ti_float,  # 梯度范数
            momentum_norm=gs.ti_float,  # 动量范数
            impulse_norm=gs.ti_float,  # 冲量范数
        )
        self.sap_state = sap_state.field(shape=self.sim._B, needs_grad=False, layout=ti.Layout.SOA)

    def _init_fem_fields(self):
        """
        初始化 FEM 速度相关的求解/PCG/线搜索缓存。
        """
        fem_state_v = ti.types.struct(
            v=gs.ti_vec3,        # 顶点速度
            v_diff=gs.ti_vec3,   # 当前速度与上一帧的差
            gradient=gs.ti_vec3, # 梯度
            impulse=gs.ti_vec3,  # 冲量
        )

        self.fem_state_v = fem_state_v.field(
            shape=(self.sim._B, self.fem_solver.n_vertices), needs_grad=False, layout=ti.Layout.SOA
        )

        pcg_fem_state_v = ti.types.struct(
            diag3x3=gs.ti_mat3,  # Hessian 对角 3x3 块
            prec=gs.ti_mat3,     # 预条件器
            x=gs.ti_vec3,        # 解向量
            r=gs.ti_vec3,        # 残差
            z=gs.ti_vec3,        # 预条件残差
            p=gs.ti_vec3,        # 搜索方向
            Ap=gs.ti_vec3,       # A @ p
        )

        self.pcg_fem_state_v = pcg_fem_state_v.field(
            shape=(self.sim._B, self.fem_solver.n_vertices), needs_grad=False, layout=ti.Layout.SOA
        )

        linesearch_fem_state_v = ti.types.struct(
            x_prev=gs.ti_vec3,  # 线搜索的前一解（v_prev）
            dp=gs.ti_vec3,      # A @ dv
        )

        self.linesearch_fem_state_v = linesearch_fem_state_v.field(
            shape=(self.sim._B, self.fem_solver.n_vertices), needs_grad=False, layout=ti.Layout.SOA
        )

    def _init_rigid_fields(self):
        """
        初始化刚体 DOF 的求解/PCG/线搜索缓存。
        """
        rigid_state_dof = ti.types.struct(
            v=gs.ti_float,        # DOF 速度
            v_diff=gs.ti_float,   # 速度差
            mass_v_diff=gs.ti_float,  # M v_diff
            gradient=gs.ti_float, # 梯度
            impulse=gs.ti_float,  # 冲量
        )

        self.rigid_state_dof = rigid_state_dof.field(
            shape=(self.sim._B, self.rigid_solver.n_dofs), needs_grad=False, layout=ti.Layout.SOA
        )

        pcg_rigid_state_dof = ti.types.struct(
            x=gs.ti_float,  # 解向量
            r=gs.ti_float,  # 残差
            z=gs.ti_float,  # 预条件残差
            p=gs.ti_float,  # 搜索方向
            Ap=gs.ti_float, # A @ p
        )

        self.pcg_rigid_state_dof = pcg_rigid_state_dof.field(
            shape=(self.sim._B, self.rigid_solver.n_dofs), needs_grad=False, layout=ti.Layout.SOA
        )

        linesearch_rigid_state_dof = ti.types.struct(
            x_prev=gs.ti_float,  # 线搜索的前一解
            dp=gs.ti_float,      # A @ dv
        )
        self.linesearch_rigid_state_dof = linesearch_rigid_state_dof.field(
            shape=(self.sim._B, self.rigid_solver.n_dofs), needs_grad=False, layout=ti.Layout.SOA
        )

    def _init_pcg_fields(self):
        """
        初始化 PCG 的批激活与标量状态。
        """
        self.batch_pcg_active = ti.field(dtype=gs.ti_bool, shape=self.sim._B, needs_grad=False)

        pcg_state = ti.types.struct(
            rTr=gs.ti_float,
            rTz=gs.ti_float,
            rTr_new=gs.ti_float,
            rTz_new=gs.ti_float,
            pTAp=gs.ti_float,
            alpha=gs.ti_float,
            beta=gs.ti_float,
        )

        self.pcg_state = pcg_state.field(shape=self.sim._B, needs_grad=False, layout=ti.Layout.SOA)

    def _init_linesearch_fields(self):
        """
        初始化精确线搜索（rtsafe）所需状态。
        """
        self.batch_linesearch_active = ti.field(dtype=gs.ti_bool, shape=self.sim._B, needs_grad=False)

        linesearch_state = ti.types.struct(
            prev_energy=gs.ti_float,
            energy=gs.ti_float,
            step_size=gs.ti_float,
            m=gs.ti_float,
            dell_dalpha=gs.ti_float,        # 总能量对 alpha 的一阶导
            d2ellA_dalpha2=gs.ti_float,     # 动力项二阶导
            d2ell_dalpha2=gs.ti_float,      # 总能量二阶导
            dell_scale=gs.ti_float,         # 一阶导缩放因子
            alpha_min=gs.ti_float,          # 步长下界
            alpha_max=gs.ti_float,          # 步长上界
            alpha_tol=gs.ti_float,          # 步长收敛阈值
            f_lower=gs.ti_float,            # f 下界值
            f_upper=gs.ti_float,            # f 上界值
            f=gs.ti_float,                  # 归一化一阶导
            df=gs.ti_float,                 # 归一化二阶导
            minus_dalpha=gs.ti_float,       # 负步长（Newton/二分混合）
            minus_dalpha_prev=gs.ti_float,  # 上一次负步长
        )

        self.linesearch_state = linesearch_state.field(shape=self.sim._B, needs_grad=False, layout=ti.Layout.SOA)

    # ------------------------------------------------------------------------------------
    # -------------------------------------- 主流程 ---------------------------------------
    # ------------------------------------------------------------------------------------

    def preprocess(self, i_step):
        """
        步前预处理：
          1) precompute：更新 FEM 压力梯度、刚体顶点、刚体体场姿态
          2) update_bvh：构建 BVH
          3) update_contact：由各 handler 进行接触检测并构建 Jacobian
          4) compute_regularization：根据 Delassus 计算接触/约束正则与 v_hat
        """
        self.precompute(i_step)
        self.update_bvh(i_step)
        self.has_contact, overflow = self.update_contact(i_step)
        if overflow:
            message = "接触查询溢出：\n"
            for contact in self.contact_handlers:
                if contact.n_contact_pairs[None] > contact.max_contact_pairs:
                    message += (
                        f"{contact.name} 最大接触对 {contact.max_contact_pairs}，实际 {contact.n_contact_pairs[None]}\n"
                    )
            gs.raise_exception(message)
        self.compute_regularization()

    @ti.kernel
    def precompute(self, i_step: ti.i32):
        """
        预计算：FEM 压力梯度、刚体所有顶点（含固定/自由）、刚体体场旋转。
        """
        if ti.static(self.fem_solver.is_active()):
            if ti.static(self._fem_floor_contact_type == FEMFloorContactType.TET or self._enable_fem_self_tet_contact):
                self.fem_compute_pressure_gradient(i_step)

        if ti.static(self.rigid_solver.is_active()):
            func_update_all_verts(
                self.rigid_solver.geoms_state,
                self.rigid_solver.verts_info,
                self.rigid_solver.free_verts_state,
                self.rigid_solver.fixed_verts_state,
            )

        if ti.static(self._rigid_compliant):
            self.rigid_update_volume_verts_pressure_gradient()

    @ti.kernel
    def update_contact(self, i_step: ti.i32) -> tuple[bool, bool]:
        """
        更新当前步的接触信息，并为每个接触处理器构建 Jacobian。

        流程
        ----
        1) 逐个遍历已注册的接触处理器（contact_handlers）；
        2) 调用各处理器的 detection(i_step) 完成宽相/窄相检测与接触对收集，累计是否溢出；
        3) 根据接触对数量更新 has_contact 标记；
        4) 调用各处理器的 compute_jacobian() 构建雅可比（J，或 Jt）；
        5) 返回 (has_contact, overflow)，分别表示是否存在接触、是否发生候选/配对上限溢出。
        """
        has_contact = False
        overflow = False
        # 遍历所有接触处理器，汇总检测与构建雅可比
        for contact in ti.static(self.contact_handlers):
            # detection: 宽相查询 + 窄相几何/压力场裁剪，写入 contact_pairs
            overflow |= contact.detection(i_step)
            # 统计是否有任意接触对
            has_contact |= contact.n_contact_pairs[None] > 0
            # compute_jacobian: 基于接触点/切法向构建 J 或 J^T
            contact.compute_jacobian()
        return has_contact, overflow

    def couple(self, i_step):
        """
        SAP 耦合主入口：若存在接触则执行一次 SAP 迭代求解并写回速度。
        """
        if self.has_contact:
            self.sap_solve(i_step)
            self.update_vel(i_step)

    def couple_grad(self, i_step):
        """
        SAP 模式暂不提供反向传播接口；如需梯度，请使用 LegacyCoupler。
        """
        gs.raise_exception("couple_grad is not available for SAPCoupler. Please use LegacyCoupler instead.")

    @ti.kernel
    def update_vel(self, i_step: ti.i32):
        """
        将 SAP 解出的状态速度写回对应求解器的状态缓存，用于步进更新。
        """
        if ti.static(self.fem_solver.is_active()):
            self.update_fem_vel(i_step)
        if ti.static(self.rigid_solver.is_active()):
            self.update_rigid_vel()

    @ti.func
    def update_fem_vel(self, i_step: ti.i32):
        """
        写回 FEM 顶点速度：elements_v[i_step + 1].vel <- fem_state_v.v
        """
        for i_b, i_v in ti.ndrange(self.fem_solver._B, self.fem_solver.n_vertices):
            self.fem_solver.elements_v[i_step + 1, i_v, i_b].vel = self.fem_state_v.v[i_b, i_v]

    @ti.func
    def update_rigid_vel(self):
        """
        写回刚体 DOF 速度：dofs_state.vel <- rigid_state_dof.v
        """
        for i_b, i_d in ti.ndrange(self.rigid_solver._B, self.rigid_solver.n_dofs):
            self.rigid_solver.dofs_state.vel[i_d, i_b] = self.rigid_state_dof.v[i_b, i_d]

    @ti.func
    def fem_compute_pressure_gradient(self, i_step: ti.i32):
        """
        计算 FEM 元素的压力梯度（逐四面体累加基于混合积的几何贡献）。

        说明
        ----
        - 对每个元素，以四个顶点构造边向量 e10、e12、e13；
        - 由 area_vector = e12 x e13 与 e10 的混合积得到有符号体积；
        - grad_i = area_vector / signed_volume，并用顶点压力进行加权累加；
        - 写入 fem_pressure_gradient[batch, elem]。
        """
        for i_b, i_e in ti.ndrange(self.fem_solver._B, self.fem_solver.n_elements):
            self.fem_pressure_gradient[i_b, i_e].fill(0.0)

            for i in ti.static(range(4)):
                i_v0 = self.fem_solver.elements_i[i_e].el2v[i]
                i_v1 = self.fem_solver.elements_i[i_e].el2v[(i + 1) % 4]
                i_v2 = self.fem_solver.elements_i[i_e].el2v[(i + 2) % 4]
                i_v3 = self.fem_solver.elements_i[i_e].el2v[(i + 3) % 4]
                pos_v0 = self.fem_solver.elements_v[i_step, i_v0, i_b].pos
                pos_v1 = self.fem_solver.elements_v[i_step, i_v1, i_b].pos
                pos_v2 = self.fem_solver.elements_v[i_step, i_v2, i_b].pos
                pos_v3 = self.fem_solver.elements_v[i_step, i_v3, i_b].pos

                e10 = pos_v0 - pos_v1
                e12 = pos_v2 - pos_v1
                e13 = pos_v3 - pos_v1

                area_vector = e12.cross(e13)
                signed_volume = area_vector.dot(e10)
                if ti.abs(signed_volume) > gs.EPS:
                    grad_i = area_vector / signed_volume
                    self.fem_pressure_gradient[i_b, i_e] += grad_i * self.fem_pressure[i_v0]

    # ------------------------------------------------------------------------------------
    # -------------------------------------- BVH -----------------------------------------
    # ------------------------------------------------------------------------------------

    def update_bvh(self, i_step: ti.i32):
        """
        更新各类 BVH 结构（AABB/LBVH），为接触检测提供宽相加速结构。
        """
        if self._enable_fem_self_tet_contact:
            self.update_fem_surface_tet_bvh(i_step)

        if self._enable_rigid_fem_contact:
            self.update_rigid_tri_bvh()

        if self.rigid_solver.is_active() and self._rigid_rigid_contact_type == RigidRigidContactType.TET:
            self.update_rigid_tet_bvh()

    def update_fem_surface_tet_bvh(self, i_step: ti.i32):
        """
        更新 FEM 表面四面体的 AABB 与 LBVH。
        """
        self.compute_fem_surface_tet_aabb(i_step)
        self.fem_surface_tet_bvh.build()

    def update_rigid_tri_bvh(self):
        """
        更新刚体三角面片的 AABB 与 LBVH（用于刚体-柔体或刚体-FEM 接触）。
        """
        self.compute_rigid_tri_aabb()
        self.rigid_tri_bvh.build()

    def update_rigid_tet_bvh(self):
        """
        更新刚体体四面体（体场）的 AABB 与 LBVH（用于刚体-刚体 hydroelastic 接触）。
        """
        self.compute_rigid_tet_aabb()
        self.rigid_tet_bvh.build()

    @ti.kernel
    def compute_fem_surface_tet_aabb(self, i_step: ti.i32):
        """
        计算 FEM 表面元素的逐批次 AABB（min/max），作为 BVH 的叶子包围盒。
        """
        aabbs = ti.static(self.fem_surface_tet_aabb.aabbs)
        for i_b, i_se in ti.ndrange(self.fem_solver._B, self.fem_solver.n_surface_elements):
            i_e = self.fem_solver.surface_elements[i_se]
            i_vs = self.fem_solver.elements_i[i_e].el2v

            aabbs[i_b, i_se].min.fill(np.inf)
            aabbs[i_b, i_se].max.fill(-np.inf)
            for i in ti.static(range(4)):
                pos_v = self.fem_solver.elements_v[i_step, i_vs[i], i_b].pos
                aabbs[i_b, i_se].min = ti.min(aabbs[i_b, i_se].min, pos_v)
                aabbs[i_b, i_se].max = ti.max(aabbs[i_b, i_se].max, pos_v)

    @ti.kernel
    def compute_rigid_tri_aabb(self):
        """
        计算刚体三角形面片的逐批次 AABB。
        """
        aabbs = ti.static(self.rigid_tri_aabb.aabbs)
        for i_b, i_f in ti.ndrange(self.rigid_solver._B, self.rigid_solver.n_faces):
            tri_vertices = ti.Matrix.zero(gs.ti_float, 3, 3)
            for i in ti.static(range(3)):
                i_v = self.rigid_solver.faces_info.verts_idx[i_f][i]
                i_fv = self.rigid_solver.verts_info.verts_state_idx[i_v]
                if self.rigid_solver.verts_info.is_fixed[i_v]:
                    tri_vertices[:, i] = self.rigid_solver.fixed_verts_state.pos[i_fv]
                else:
                    tri_vertices[:, i] = self.rigid_solver.free_verts_state.pos[i_fv, i_b]
            pos_v0, pos_v1, pos_v2 = tri_vertices[:, 0], tri_vertices[:, 1], tri_vertices[:, 2]

            aabbs[i_b, i_f].min = ti.min(pos_v0, pos_v1, pos_v2)
            aabbs[i_b, i_f].max = ti.max(pos_v0, pos_v1, pos_v2)

    @ti.kernel
    def compute_rigid_tet_aabb(self):
        """
        计算刚体体四面体（体场）单元的逐批次 AABB。
        """
        aabbs = ti.static(self.rigid_tet_aabb.aabbs)
        for i_b, i_e in ti.ndrange(self._B, self.n_rigid_volume_elems):
            i_v0 = self.rigid_volume_elems[i_e][0]
            i_v1 = self.rigid_volume_elems[i_e][1]
            i_v2 = self.rigid_volume_elems[i_e][2]
            i_v3 = self.rigid_volume_elems[i_e][3]
            pos_v0 = self.rigid_volume_verts[i_b, i_v0]
            pos_v1 = self.rigid_volume_verts[i_b, i_v1]
            pos_v2 = self.rigid_volume_verts[i_b, i_v2]
            pos_v3 = self.rigid_volume_verts[i_b, i_v3]
            aabbs[i_b, i_e].min = ti.min(pos_v0, pos_v1, pos_v2, pos_v3)
            aabbs[i_b, i_e].max = ti.max(pos_v0, pos_v1, pos_v2, pos_v3)

    # ------------------------------------------------------------------------------------
    # ------------------------------------- 求解 ------------------------------------------
    # ------------------------------------------------------------------------------------

    def sap_solve(self, i_step):
        """
        SAP 主循环管线：
          1) 初始化当前步状态（速度 v 等）
          2) 计算无约束梯度与对角（质量/弹性/阻尼等）
          3) 叠加接触/约束项的梯度与 Hessian 对角，并构建预条件器
          4) 使用预条件共轭梯度（PCG）迭代解 dv
          5) 精确线搜索（rtsafe）确定步长并更新 v
        """
        self._init_sap_solve(i_step)
        for iter in range(self._n_sap_iterations):
            # init gradient and preconditioner
            self.compute_unconstrained_gradient_diag(i_step, iter)

            # compute contact hessian and gradient
            self.compute_constraint_contact_gradient_hessian_diag_prec()
            self.check_sap_convergence()
            # solve for the vertex velocity
            self.pcg_solve()

            # line search
            self.exact_linesearch(i_step)

    @ti.kernel
    def check_sap_convergence(self):
        """
        Check SAP convergence by assembling norms and updating active batches.

        统计当前梯度/动量/冲量范数并更新 batch 激活标记，用于判断 SAP 收敛性。
        """
        self.clear_sap_norms()
        if ti.static(self.fem_solver.is_active()):
            self.add_fem_norms()
        if ti.static(self.rigid_solver.is_active()):
            self.add_rigid_norms()
        self.update_batch_active()

    @ti.func
    def clear_sap_norms(self):
        """
        Reset per-batch SAP norm accumulators to zeros.

        清空每个 batch 的 SAP 范数累加器（梯度/动量/冲量）。
        """
        for i_b in range(self._B):
            if not self.batch_active[i_b]:
                continue
            self.sap_state[i_b].gradient_norm = 0.0
            self.sap_state[i_b].momentum_norm = 0.0
            self.sap_state[i_b].impulse_norm = 0.0

    @ti.func
    def add_fem_norms(self):
        """
        Accumulate FEM contributions to SAP norms.

        将 FEM 部分对 SAP 范数（梯度/动量/冲量）的贡献进行累加。
        """
        for i_b, i_v in ti.ndrange(self._B, self.fem_solver.n_vertices):
            if not self.batch_active[i_b]:
                continue
            self.sap_state[i_b].gradient_norm += (
                self.fem_state_v.gradient[i_b, i_v].norm_sqr() / self.fem_solver.elements_v_info[i_v].mass
            )
            self.sap_state[i_b].momentum_norm += (
                self.fem_state_v.v[i_b, i_v].norm_sqr() * self.fem_solver.elements_v_info[i_v].mass
            )
            self.sap_state[i_b].impulse_norm += (
                self.fem_state_v.impulse[i_b, i_v].norm_sqr() / self.fem_solver.elements_v_info[i_v].mass
            )

    @ti.func
    def add_rigid_norms(self):
        """
        Accumulate rigid-body contributions to SAP norms.

        将刚体部分对 SAP 范数（梯度/动量/冲量）的贡献进行累加。
        """
        for i_b, i_d in ti.ndrange(self._B, self.rigid_solver.n_dofs):
            if not self.batch_active[i_b]:
                continue
            self.sap_state[i_b].gradient_norm += (
                self.rigid_state_dof.gradient[i_b, i_d] ** 2 / self.rigid_solver.mass_mat[i_d, i_d, i_b]
            )
            self.sap_state[i_b].momentum_norm += (
                self.rigid_state_dof.v[i_b, i_d] ** 2 * self.rigid_solver.mass_mat[i_d, i_d, i_b]
            )
            self.sap_state[i_b].impulse_norm += (
                self.rigid_state_dof.impulse[i_b, i_d] ** 2 / self.rigid_solver.mass_mat[i_d, i_d, i_b]
            )

    @ti.func
    def update_batch_active(self):
        """
        Update which batches remain active based on convergence thresholds.

        依据收敛阈值（绝对/相对）更新仍需参与迭代的 batch 激活状态。
        """
        for i_b in range(self._B):
            if not self.batch_active[i_b]:
                continue
            norm_thr = self._sap_convergence_atol + self._sap_convergence_rtol * ti.max(
                self.sap_state[i_b].momentum_norm, self.sap_state[i_b].impulse_norm
            )

    @ti.kernel
    def compute_regularization(self):
        """
        Compute regularization terms (R, R_inv, v_hat) for all contacts/constraints.

        为所有接触/约束计算正则化参数（R、R_inv、v_hat）。
        """
        for contact in ti.static(self.contact_handlers):
            contact.compute_regularization()
        if ti.static(self.rigid_solver.is_active() and self.rigid_solver.n_equalities > 0):
            self.equality_constraint_handler.compute_regularization()

    @ti.kernel
    def _init_sap_solve(self, i_step: ti.i32):
        """
        Initialize SAP solve state for this step.

        初始化本步 SAP 求解的状态（写入初值 v，并激活所有 batch）。
        """
        self._init_v(i_step)
        self.batch_active.fill(True)

    @ti.func
    def _init_v(self, i_step: ti.i32):
        if ti.static(self.fem_solver.is_active()):
            self._init_v_fem(i_step)
        if ti.static(self.rigid_solver.is_active()):
            self._init_v_rigid(i_step)

    @ti.func
    def _init_v_fem(self, i_step: ti.i32):
        """
        写回 FEM 顶点速度：elements_v[i_step + 1].vel <- fem_state_v.v
        将 SAP 解出的 FEM 顶点速度写回元素状态，用于后续步进更新。
        """
        for i_b, i_v in ti.ndrange(self.fem_solver._B, self.fem_solver.n_vertices):
            self.fem_state_v.v[i_b, i_v] = self.fem_solver.elements_v[i_step + 1, i_v, i_b].vel

    @ti.func
    def _init_v_rigid(self, i_step: ti.i32):
        """
        写回刚体 DOF 速度：dofs_state.vel <- rigid_state_dof.v
        将 SAP 解出的刚体 DOF 速度写回状态，用于后续步进更新。
        """
        for i_b, i_d in ti.ndrange(self.rigid_solver._B, self.rigid_solver.n_dofs):
            self.rigid_state_dof.v[i_b, i_d] = self.rigid_solver.dofs_state.vel[i_d, i_b]

    def compute_unconstrained_gradient_diag(self, i_step: ti.i32, iter: int):
        """
        Compute unconstrained gradient and diagonal blocks for the system.

        计算“未加接触/约束”的梯度与对角项；迭代 0 时跳过乘 A(v-v*)。
        """
        self.init_unconstrained_gradient_diag(i_step)
        # No need to do this for iter=0 because v=v* and A(v-v*) = 0
        if iter > 0:
            self.compute_unconstrained_gradient()

    def init_unconstrained_gradient_diag(self, i_step: ti.i32):
        """
        Initialize gradient and diagonal blocks for unconstrained system.

        初始化未约束系统的梯度与对角（质量/弹性/阻尼的对角近似）。
        """
        if self.fem_solver.is_active():
            self.init_fem_unconstrained_gradient_diag(i_step)
        if self.rigid_solver.is_active():
            self.init_rigid_unconstrained_gradient()

    @ti.kernel
    def init_fem_unconstrained_gradient_diag(self, i_step: ti.i32):
        """
        Initialize FEM unconstrained gradient and 3x3 diagonal blocks.

        初始化 FEM 未约束梯度与 3x3 对角块（按 dt^2 缩放质量/阻尼影响）。
        """
        dt2 = self.fem_solver._substep_dt**2
        for i_b, i_v in ti.ndrange(self.fem_solver._B, self.fem_solver.n_vertices):
            self.fem_state_v.gradient[i_b, i_v].fill(0.0)
            # was using position now using velocity, need to multiply dt^2
            self.pcg_fem_state_v[i_b, i_v].diag3x3 = self.fem_solver.pcg_state_v[i_b, i_v].diag3x3 * dt2
            self.fem_state_v.v_diff[i_b, i_v] = (
                self.fem_state_v.v[i_b, i_v] - self.fem_solver.elements_v[i_step + 1, i_v, i_b].vel
            )

    @ti.kernel
    def init_rigid_unconstrained_gradient(self):
        """
        Initialize rigid-body unconstrained gradient.

        初始化刚体未约束梯度与速度差（v - v*）。
        """
        for i_b, i_d in ti.ndrange(self.rigid_solver._B, self.rigid_solver.n_dofs):
            self.rigid_state_dof.gradient[i_b, i_d] = 0.0
            self.rigid_state_dof.v_diff[i_b, i_d] = (
                self.rigid_state_dof.v[i_b, i_d] - self.rigid_solver.dofs_state.vel[i_d, i_b]
            )

    def compute_unconstrained_gradient(self):
        """
        Apply system matrix to v_diff for unconstrained part.

        对未约束系统计算 A @ (v - v*)，累加到梯度。
        """
        if self.fem_solver.is_active():
            self.compute_fem_unconstrained_gradient()
        if self.rigid_solver.is_active():
            self.compute_rigid_unconstrained_gradient()

    @ti.kernel
    def compute_fem_unconstrained_gradient(self):
        """
        FEM part of unconstrained gradient: Ap = A_fem @ v_diff.

        FEM 未约束梯度：Ap = A_fem @ v_diff。
        """
        self.compute_fem_matrix_vector_product(self.fem_state_v.v_diff, self.fem_state_v.gradient, self.batch_active)

    @ti.kernel
    def compute_rigid_unconstrained_gradient(self):
        """
        Rigid part of unconstrained gradient: Ap = M @ v_diff.

        刚体未约束梯度：Ap = 质量矩阵 @ v_diff。
        """
        self.pcg_rigid_state_dof.Ap.fill(0.0)
        for i_b, i_d0, i_d1 in ti.ndrange(self.rigid_solver._B, self.rigid_solver.n_dofs, self.rigid_solver.n_dofs):
            if not self.batch_active[i_b]:
                continue
            self.rigid_state_dof.gradient[i_b, i_d1] += (
                self.rigid_solver.mass_mat[i_d1, i_d0, i_b] * self.rigid_state_dof.v_diff[i_b, i_d0]
            )

    @ti.kernel
    def compute_constraint_contact_gradient_hessian_diag_prec(self):
        """
        Accumulate constraint/contact gradient, Hessian diagonals and build preconditioner.

        累计接触/等式约束的梯度与 Hessian 对角，并构建预条件器。
        """
        self.clear_impulses()
        if ti.static(self.rigid_solver.is_active() and self.rigid_solver.n_equalities > 0):
            self.equality_constraint_handler.compute_gradient_hessian_diag()
        for contact in ti.static(self.contact_handlers):
            contact.compute_gradient_hessian_diag()
        self.compute_preconditioner()

    @ti.func
    def clear_impulses(self):
        """
        Reset contact and constraint impulses to zero.

        清空接触与约束的冲量。
        """
        if ti.static(self.fem_solver.is_active()):
            self.clear_fem_impulses()
        if ti.static(self.rigid_solver.is_active()):
            self.clear_rigid_impulses()

    @ti.func
    def clear_fem_impulses(self):
        """
        Clear FEM impulses.

        清空 FEM 的冲量。
        """
        for i_b, i_v in ti.ndrange(self.fem_solver._B, self.fem_solver.n_vertices):
            if not self.batch_active[i_b]:
                continue
            self.fem_state_v[i_b, i_v].impulse.fill(0.0)

    @ti.func
    def clear_rigid_impulses(self):
        """
        Clear rigid-body impulses.

        清空刚体的冲量。
        """
        for i_b, i_d in ti.ndrange(self.rigid_solver._B, self.rigid_solver.n_dofs):
            if not self.batch_active[i_b]:
                continue
            self.rigid_state_dof[i_b, i_d].impulse = 0.0

    @ti.func
    def compute_preconditioner(self):
        """
        Build block-diagonal preconditioners for enabled subsystems.

        为各子系统（如 FEM）构建块对角预条件器。
        """
        if ti.static(self.fem_solver.is_active()):
            self.compute_fem_preconditioner()

    @ti.func
    def compute_fem_preconditioner(self):
        """
        FEM preconditioner as inverse of local 3x3 diagonal blocks.

        FEM 预条件器为局部 3x3 对角块的逆矩阵。
        """
        for i_b, i_v in ti.ndrange(self.fem_solver._B, self.fem_solver.n_vertices):
            if not self.batch_active[i_b]:
                continue
            self.pcg_fem_state_v[i_b, i_v].prec = self.pcg_fem_state_v[i_b, i_v].diag3x3.inverse()

    @ti.func
    def compute_fem_pcg_matrix_vector_product(self):
        """
        Compute Ap for FEM during PCG (using current search direction p).

        PCG 中 FEM 的矩阵-向量乘法 Ap（以当前搜索方向 p）。
        """
        self.compute_fem_matrix_vector_product(self.pcg_fem_state_v.p, self.pcg_fem_state_v.Ap, self.batch_pcg_active)

    @ti.func
    def compute_rigid_pcg_matrix_vector_product(self):
        """
        Compute Ap for rigid during PCG (mass-matrix product).

        PCG 中刚体的矩阵-向量乘法 Ap（质量矩阵乘法）。
        """
        self.compute_rigid_mass_mat_vec_product(
            self.pcg_rigid_state_dof.p, self.pcg_rigid_state_dof.Ap, self.batch_pcg_active
        )

    @ti.func
    def compute_elastic_products(self, i_b, i_e, S, i_vs, src):
        """
        Helper for elasticity: pack 12-DOF vector to p9 and compute H9 @ p9.

        弹性项辅助：将 4 顶点的 12 自由度打包为 p9，并计算 H9 @ p9。
        """
        p9 = ti.Vector.zero(gs.ti_float, 9)
        for i, j in ti.static(ti.ndrange(3, 4)):
            p9[i * 3 : i * 3 + 3] = p9[i * 3 : i * 3 + 3] + (S[j, i] * src[i_b, i_vs[j]])

        H9_p9 = ti.Vector.zero(gs.ti_float, 9)

        for i, j in ti.static(ti.ndrange(3, 3)):
            H9_p9[i * 3 : i * 3 + 3] = H9_p9[i * 3 : i * 3 + 3] + (
                self.fem_solver.elements_el_hessian[i_b, i, j, i_e] @ p9[j * 3 : j * 3 + 3]
            )
        return p9, H9_p9

    @ti.func
    def compute_fem_matrix_vector_product(self, src, dst, active):
        """
        Compute the FEM matrix-vector product, including mass matrix and elasticity stiffness matrix.
        计算 FEM 的矩阵-向量乘法，包含质量项与弹性刚度项（并考虑 Rayleigh 阻尼）。
        """
        dt2 = self.fem_solver._substep_dt**2
        damping_alpha_factor = self.fem_solver._damping_alpha * self.fem_solver._substep_dt + 1.0
        damping_beta_factor = self.fem_solver._damping_beta / self.fem_solver._substep_dt + 1.0

        # Inerita
        for i_b, i_v in ti.ndrange(self.fem_solver._B, self.fem_solver.n_vertices):
            if not active[i_b]:
                continue
            dst[i_b, i_v] = (
                self.fem_solver.elements_v_info[i_v].mass_over_dt2 * src[i_b, i_v] * dt2 * damping_alpha_factor
            )

        # Elasticity
        for i_b, i_e in ti.ndrange(self.fem_solver._B, self.fem_solver.n_elements):
            if not active[i_b]:
                continue
            V_dt2 = self.fem_solver.elements_i[i_e].V * dt2
            B = self.fem_solver.elements_i[i_e].B
            S = ti.Matrix.zero(gs.ti_float, 4, 3)
            S[:3, :] = B
            S[3, :] = -B[0, :] - B[1, :] - B[2, :]
            i_vs = self.fem_solver.elements_i[i_e].el2v

            if ti.static(self.fem_solver._enable_vertex_constraints):
                for i in ti.static(range(4)):
                    if self.fem_solver.vertex_constraints.is_constrained[i_vs[i], i_b]:
                        S[i, :] = ti.Vector.zero(gs.ti_float, 3)

            _, new_p9 = self.compute_elastic_products(i_b, i_e, S, i_vs, src)
            # atomic
            scale = V_dt2 * damping_beta_factor
            for i in ti.static(range(4)):
                dst[i_b, i_vs[i]] += (S[i, 0] * new_p9[0:3] + S[i, 1] * new_p9[3:6] + S[i, 2] * new_p9[6:9]) * scale

    @ti.kernel
    def init_pcg_solve(self):
        """
        Initialize PCG state, per-subsystem vectors, and active mask.

        初始化 PCG 状态、各子系统向量以及 batch 激活标记。
        """
        self.init_pcg_state()
        if ti.static(self.fem_solver.is_active()):
            self.init_fem_pcg_solve()
        if ti.static(self.rigid_solver.is_active()):
            self.init_rigid_pcg_solve()
        self.init_pcg_active()

    @ti.func
    def init_pcg_state(self):
        """
        Reset scalar accumulators for PCG (rTr, rTz, etc).

        重置 PCG 的标量累加器（rTr、rTz 等）。
        """
        for i_b in ti.ndrange(self._B):
            self.batch_pcg_active[i_b] = self.batch_active[i_b]
            if not self.batch_pcg_active[i_b]:
                continue
            self.pcg_state[i_b].rTr = 0.0
            self.pcg_state[i_b].rTz = 0.0

    @ti.func
    def init_fem_pcg_solve(self):
        """
        Initialize FEM PCG vectors: x=0, r=-g, z=Prec r, p=z; accumulate rTr/rTz.

        初始化 FEM PCG 向量：x=0, r=-g, z=Prec r, p=z；并累加 rTr/rTz。
        """
        for i_b, i_v in ti.ndrange(self._B, self.fem_solver.n_vertices):
            if not self.batch_pcg_active[i_b]:
                continue
            self.pcg_fem_state_v[i_b, i_v].x = 0.0
            self.pcg_fem_state_v[i_b, i_v].r = -self.fem_state_v.gradient[i_b, i_v]
            self.pcg_fem_state_v[i_b, i_v].z = self.pcg_fem_state_v[i_b, i_v].prec @ self.pcg_fem_state_v[i_b, i_v].r
            self.pcg_fem_state_v[i_b, i_v].p = self.pcg_fem_state_v[i_b, i_v].z
            self.pcg_state[i_b].rTr += self.pcg_fem_state_v[i_b, i_v].r.dot(self.pcg_fem_state_v[i_b, i_v].r)
            self.pcg_state[i_b].rTz += self.pcg_fem_state_v[i_b, i_v].r.dot(self.pcg_fem_state_v[i_b, i_v].z)

    @ti.func
    def compute_rigid_mass_mat_vec_product(self, vec, out, active):
        """
        Compute the rigid mass matrix-vector product.
        计算刚体质量矩阵与向量的乘积。
        """
        out.fill(0.0)
        for i_b, i_d0, i_d1 in ti.ndrange(self._B, self.rigid_solver.n_dofs, self.rigid_solver.n_dofs):
            if not active[i_b]:
                continue
            out[i_b, i_d1] += self.rigid_solver.mass_mat[i_d1, i_d0, i_b] * vec[i_b, i_d0]

    # FIXME: This following two rigid solves are duplicated with the one in rigid_solver_decomp.py:func_solve_mass_batched
    # Consider refactoring.
    @ti.func
    def rigid_solve_pcg(self, vec, out):
        """
        Apply factorized mass preconditioner (LL^T with diagonal) to vec.

        对 vec 应用刚体质量矩阵的分解预条件器（L、D、L^T 三步回代）。
        """
        # Step 1: Solve w st. L^T @ w = y
        for i_b, i_e in ti.ndrange(self._B, self.rigid_solver.n_entities):
            if not self.batch_pcg_active[i_b]:
                continue
            entity_dof_start = self.rigid_solver.entities_info.dof_start[i_e]
            entity_dof_end = self.rigid_solver.entities_info.dof_end[i_e]
            n_dofs = self.rigid_solver.entities_info.n_dofs[i_e]
            for i_d_ in range(n_dofs):
                i_d = entity_dof_end - i_d_ - 1
                out[i_b, i_d] = vec[i_b, i_d]
                for j_d in range(i_d + 1, entity_dof_end):
                    out[i_b, i_d] -= self.rigid_solver.mass_mat_L[j_d, i_d, i_b] * out[i_b, j_d]

        # Step 2: z = D^{-1} w
        for i_b, i_d in ti.ndrange(self._B, self.rigid_solver.n_dofs):
            if not self.batch_pcg_active[i_b]:
                continue
            out[i_b, i_d] *= self.rigid_solver.mass_mat_D_inv[i_d, i_b]

        # Step 3: Solve x st. L @ x = z
        for i_b, i_e in ti.ndrange(self._B, self.rigid_solver.n_entities):
            if not self.batch_pcg_active[i_b]:
                continue
            entity_dof_start = self.rigid_solver.entities_info.dof_start[i_e]
            entity_dof_end = self.rigid_solver.entities_info.dof_end[i_e]
            n_dofs = self.rigid_solver.entities_info.n_dofs[i_e]
            for i_d in range(entity_dof_start, entity_dof_end):
                for j_d in range(entity_dof_start, i_d):
                    out[i_b, i_d] -= self.rigid_solver.mass_mat_L[i_d, j_d, i_b] * out[i_b, j_d]

    @ti.func
    def rigid_solve_jacobian(self, vec, out, n_contact_pairs, i_bs, dim):
        """
        Batched triangular solves for Jt blocks under mass factorization.

        在质量矩阵分解下，对接触对的雅可比块执行批量三角回代（J^T 的求解）。
        """
        # Step 1: Solve w st. L^T @ w = y
        for i_p, i_e, k in ti.ndrange(n_contact_pairs, self.rigid_solver.n_entities, dim):
            i_b = i_bs[i_p]
            entity_dof_start = self.rigid_solver.entities_info.dof_start[i_e]
            entity_dof_end = self.rigid_solver.entities_info.dof_end[i_e]
            n_dofs = self.rigid_solver.entities_info.n_dofs[i_e]
            for i_d_ in range(n_dofs):
                i_d = entity_dof_end - i_d_ - 1
                out[i_p, i_d][k] = vec[i_p, i_d][k]
                for j_d in range(i_d + 1, entity_dof_end):
                    out[i_p, i_d][k] -= self.rigid_solver.mass_mat_L[j_d, i_d, i_b] * out[i_p, j_d][k]

        # Step 2: z = D^{-1} w
        for i_p, i_d, k in ti.ndrange(n_contact_pairs, self.rigid_solver.n_dofs, dim):
            i_b = i_bs[i_p]
            out[i_p, i_d][k] *= self.rigid_solver.mass_mat_D_inv[i_d, i_b]

        # Step 3: Solve x st. L @ x = z
        for i_p, i_e, k in ti.ndrange(n_contact_pairs, self.rigid_solver.n_entities, dim):
            i_b = i_bs[i_p]
            entity_dof_start = self.rigid_solver.entities_info.dof_start[i_e]
            entity_dof_end = self.rigid_solver.entities_info.dof_end[i_e]
            n_dofs = self.rigid_solver.entities_info.n_dofs[i_e]
            for i_d in range(entity_dof_start, entity_dof_end):
                for j_d in range(entity_dof_start, i_d):
                    out[i_p, i_d][k] -= self.rigid_solver.mass_mat_L[i_d, j_d, i_b] * out[i_p, j_d][k]

    @ti.func
    def init_rigid_pcg_solve(self):
        """
        Initialize rigid PCG vectors and apply mass preconditioner for z.

        初始化刚体 PCG 向量，并通过质量预条件器求 z。
        """
        for i_b, i_d in ti.ndrange(self._B, self.rigid_solver.n_dofs):
            if not self.batch_pcg_active[i_b]:
                continue
            self.pcg_rigid_state_dof[i_b, i_d].x = 0.0
            self.pcg_rigid_state_dof[i_b, i_d].r = -self.rigid_state_dof.gradient[i_b, i_d]
            self.pcg_state[i_b].rTr += self.pcg_rigid_state_dof[i_b, i_d].r ** 2

        self.rigid_solve_pcg(self.pcg_rigid_state_dof.r, self.pcg_rigid_state_dof.z)

        for i_b, i_d in ti.ndrange(self._B, self.rigid_solver.n_dofs):
            if not self.batch_pcg_active[i_b]:
                continue
            self.pcg_rigid_state_dof[i_b, i_d].p = self.pcg_rigid_state_dof[i_b, i_d].z
            self.pcg_state[i_b].rTz += self.pcg_rigid_state_dof[i_b, i_d].r * self.pcg_rigid_state_dof[i_b, i_d].z

    @ti.func
    def init_pcg_active(self):
        """
        Enable PCG for batches whose initial residual exceeds threshold.

        仅对初始残差超过阈值的 batch 启用 PCG 迭代。
        """
        for i_b in ti.ndrange(self._B):
            if not self.batch_pcg_active[i_b]:
                continue
            self.batch_pcg_active[i_b] = self.pcg_state[i_b].rTr > self._pcg_threshold

    def one_pcg_iter(self):
        """
        One full PCG iteration step.

        执行一次完整的 PCG 迭代。
        """
        self._kernel_one_pcg_iter()

    @ti.kernel
    def _kernel_one_pcg_iter(self):
        self.compute_pcg_matrix_vector_product()
        self.clear_pcg_state()
        self.compute_pcg_pTAp()
        self.compute_alpha()
        self.compute_pcg_state()
        self.check_pcg_convergence()
        self.compute_p()

    @ti.func
    def compute_pcg_matrix_vector_product(self):
        """
        Compute Ap for all subsystems plus constraints/contacts.

        对各子系统以及约束/接触计算 Ap 并汇总。
        """
        if ti.static(self.fem_solver.is_active()):
            self.compute_fem_pcg_matrix_vector_product()
        if ti.static(self.rigid_solver.is_active()):
            self.compute_rigid_pcg_matrix_vector_product()
        # Constraint
        if ti.static(self.rigid_solver.is_active() and self.rigid_solver.n_equalities > 0):
            self.equality_constraint_handler.compute_Ap()
        # Contact
        for contact in ti.static(self.contact_handlers):
            contact.compute_pcg_matrix_vector_product()

    @ti.func
    def clear_pcg_state(self):
        """
        Reset per-iteration PCG accumulators.

        重置本次迭代使用的 PCG 标量累加器。
        """
        for i_b in ti.ndrange(self._B):
            if not self.batch_pcg_active[i_b]:
                continue
            self.pcg_state[i_b].pTAp = 0.0
            self.pcg_state[i_b].rTr_new = 0.0
            self.pcg_state[i_b].rTz_new = 0.0

    @ti.func
    def compute_pcg_pTAp(self):
        """
        Compute the product p^T @ A @ p used in the Preconditioned Conjugate Gradient method.

        Notes
        -----
        Reference: https://en.wikipedia.org/wiki/Conjugate_gradient_method#The_preconditioned_conjugate_gradient_method

        计算 PCG 所需的二次型 p^T A p（跨子系统累加）。
        参考：上面维基条目中的“预条件共轭梯度法”部分。
        """
        if ti.static(self.fem_solver.is_active()):
            self.compute_fem_pcg_pTAp()
        if ti.static(self.rigid_solver.is_active()):
            self.compute_rigid_pcg_pTAp()

    @ti.func
    def compute_alpha(self):
        """
        Compute step size alpha = (r^T z)/(p^T A p).

        计算 PCG 步长 alpha = (r^T z)/(p^T A p)。
        """
        for i_b in ti.ndrange(self._B):
            if not self.batch_pcg_active[i_b]:
                continue
            self.pcg_state[i_b].alpha = self.pcg_state[i_b].rTz / self.pcg_state[i_b].pTAp

    @ti.func
    def compute_pcg_state(self):
        """
        Update x, r and z after taking step alpha; accumulate new rTr/rTz.

        按步长 alpha 更新 x、r、z，并累计新的 rTr/rTz。
        """
        if ti.static(self.fem_solver.is_active()):
            self.compute_fem_pcg_state()
        if ti.static(self.rigid_solver.is_active()):
            self.compute_rigid_pcg_state()

    @ti.func
    def check_pcg_convergence(self):
        """
        Check PCG convergence by residual norm and update beta, scalars.

        依据残差范数检查 PCG 收敛，并更新 beta 与标量状态。
        """
        # check convergence
        for i_b in ti.ndrange(self._B):
            if not self.batch_pcg_active[i_b]:
                continue
            self.batch_pcg_active[i_b] = self.pcg_state[i_b].rTr_new > self._pcg_threshold
        # update beta, rTr, rTz
        for i_b in ti.ndrange(self._B):
            if not self.batch_pcg_active[i_b]:
                continue
            self.pcg_state[i_b].beta = self.pcg_state[i_b].rTz_new / self.pcg_state[i_b].rTz
            self.pcg_state[i_b].rTr = self.pcg_state[i_b].rTr_new
            self.pcg_state[i_b].rTz = self.pcg_state[i_b].rTz_new

    @ti.func
    def compute_p(self):
        """
        Update search direction p = z + beta * p.

        更新搜索方向 p = z + beta * p。
        """
        if ti.static(self.fem_solver.is_active()):
            self.compute_fem_p()
        if ti.static(self.rigid_solver.is_active()):
            self.compute_rigid_p()

    def pcg_solve(self):
        """
        Run PCG: initialization followed by fixed-number iterations.

        运行 PCG：初始化后执行固定次数迭代。
        """
        self.init_pcg_solve()
        for i in range(self._n_pcg_iterations):
            self.one_pcg_iter()

    @ti.func
    def compute_total_energy(self, i_step: ti.i32, energy: ti.template()):
        """
        Compute total energy (inertia + elastic + constraints + contacts).

        计算总能量（惯性 + 弹性 + 约束 + 接触）。
        """
        energy.fill(0.0)
        if ti.static(self.fem_solver.is_active()):
            self.compute_fem_energy(i_step, energy)
        if ti.static(self.rigid_solver.is_active()):
            self.compute_rigid_energy(energy)
        # Constraint
        if ti.static(self.rigid_solver.is_active() and self.rigid_solver.n_equalities > 0):
            self.equality_constraint_handler.compute_energy(energy)
        # Contact
        for contact in ti.static(self.contact_handlers):
            contact.compute_energy(energy)

    @ti.func
    def compute_fem_energy(self, i_step: ti.i32, energy: ti.template()):
        """
        FEM part of energy including inertia (in velocity-space) and elasticity.

        FEM 能量项：基于速度空间的惯性能与弹性能。
        """
        dt2 = self.fem_solver._substep_dt**2
        damping_alpha_factor = self.fem_solver._damping_alpha * self.fem_solver._substep_dt + 1.0
        damping_beta_factor = self.fem_solver._damping_beta / self.fem_solver._substep_dt + 1.0

        # Inertia
        for i_b, i_v in ti.ndrange(self._B, self.fem_solver.n_vertices):
            if not self.batch_linesearch_active[i_b]:
                continue
            self.fem_state_v.v_diff[i_b, i_v] = (
                self.fem_state_v.v[i_b, i_v] - self.fem_solver.elements_v[i_step + 1, i_v, i_b].vel
            )
            energy[i_b] += (
                0.5
                * self.fem_solver.elements_v_info[i_v].mass_over_dt2
                * self.fem_state_v.v_diff[i_b, i_v].norm_sqr()
                * dt2
                * damping_alpha_factor
            )

        # Elastic
        for i_b, i_e in ti.ndrange(self._B, self.fem_solver.n_elements):
            if not self.batch_linesearch_active[i_b]:
                continue

            V_dt2 = self.fem_solver.elements_i[i_e].V * dt2
            B = self.fem_solver.elements_i[i_e].B
            S = ti.Matrix.zero(gs.ti_float, 4, 3)
            S[:3, :] = B
            S[3, :] = -B[0, :] - B[1, :] - B[2, :]
            i_vs = self.fem_solver.elements_i[i_e].el2v

            if ti.static(self.fem_solver._enable_vertex_constraints):
                for i in ti.static(range(4)):
                    if self.fem_solver.vertex_constraints.is_constrained[i_vs[i], i_b]:
                        S[i, :] = ti.Vector.zero(gs.ti_float, 3)

            p9, H9_p9 = self.compute_elastic_products(i_b, i_e, S, i_vs, self.fem_state_v.v_diff)
            energy[i_b] += 0.5 * p9.dot(H9_p9) * damping_beta_factor * V_dt2

    @ti.func
    def compute_rigid_energy(self, energy: ti.template()):
        """
        Rigid kinetic energy using mass matrix.

        刚体动能（质量矩阵形式）。
        """
        # Kinetic energy
        for i_b, i_d in ti.ndrange(self._B, self.rigid_solver.n_dofs):
            if not self.batch_linesearch_active[i_b]:
                continue
            self.rigid_state_dof.v_diff[i_b, i_d] = (
                self.rigid_state_dof.v[i_b, i_d] - self.rigid_solver.dofs_state.vel[i_d, i_b]
            )
        self.compute_rigid_mass_mat_vec_product(
            self.rigid_state_dof.v_diff, self.rigid_state_dof.mass_v_diff, self.batch_linesearch_active
        )
        for i_b, i_d in ti.ndrange(self._B, self.rigid_solver.n_dofs):
            if not self.batch_linesearch_active[i_b]:
                continue
            energy[i_b] += 0.5 * self.rigid_state_dof.v_diff[i_b, i_d] * self.rigid_state_dof.mass_v_diff[i_b, i_d]

    @ti.kernel
    def init_exact_linesearch(self, i_step: ti.i32):
        """
        Initialize exact line search (rtsafe) state and precompute derivatives.

        初始化精确线搜索（rtsafe）所需状态，并预计算能量与导数。
        """
        self._func_init_linesearch(self._linesearch_max_step_size)
        self.compute_total_energy(i_step, self.linesearch_state.prev_energy)
        self.prepare_search_direction_data()
        self.update_velocity_linesearch()
        self.compute_line_energy_gradient_hessian(i_step)
        self.check_initial_exact_linesearch_convergence()
        self.init_newton_linesearch()

    @ti.func
    def init_newton_linesearch(self):
        """
        Initialize bracket and Newton step for rtsafe hybrid method.

        初始化 rtsafe 的区间与牛顿步，作为牛顿/二分混合根查找。
        """
        for i_b in ti.ndrange(self._B):
            if not self.batch_linesearch_active[i_b]:
                continue
            self.linesearch_state[i_b].dell_scale = -self.linesearch_state[i_b].m
            self.linesearch_state[i_b].step_size = ti.min(
                -self.linesearch_state[i_b].m / self.linesearch_state[i_b].d2ell_dalpha2, self._linesearch_max_step_size
            )
            self.linesearch_state[i_b].alpha_min = 0.0
            self.linesearch_state[i_b].alpha_max = self._linesearch_max_step_size
            self.linesearch_state[i_b].f_lower = -1.0
            self.linesearch_state[i_b].f_upper = (
                self.linesearch_state[i_b].dell_dalpha / self.linesearch_state[i_b].dell_scale
            )
            self.linesearch_state[i_b].alpha_tol = self._linesearch_ftol * self.linesearch_state[i_b].step_size
            self.linesearch_state[i_b].minus_dalpha = (
                self.linesearch_state[i_b].alpha_min - self.linesearch_state[i_b].alpha_max
            )
            self.linesearch_state[i_b].minus_dalpha_prev = self.linesearch_state[i_b].minus_dalpha
            if ti.abs(self.linesearch_state[i_b].f_lower) < self._linesearch_ftol:
                self.batch_linesearch_active[i_b] = False
                self.linesearch_state[i_b].step_size = self.linesearch_state[i_b].alpha_min
            if ti.abs(self.linesearch_state[i_b].f_upper) < self._linesearch_ftol:
                self.batch_linesearch_active[i_b] = False
                self.linesearch_state[i_b].step_size = self.linesearch_state[i_b].alpha_max

    @ti.func
    def compute_line_energy_gradient_hessian(self, i_step: ti.i32):
        """
        Assemble line energy, gradient and Hessian along alpha.

        沿步长 alpha 方向组装能量、一阶导与二阶导。
        """
        self.init_linesearch_energy_gradient_hessian()
        if ti.static(self.fem_solver.is_active()):
            self.compute_fem_energy_alpha(i_step, self.linesearch_state.energy)
            self.compute_fem_gradient_alpha(i_step)

        if ti.static(self.rigid_solver.is_active()):
            self.compute_rigid_energy_alpha(self.linesearch_state.energy)
            self.compute_rigid_gradient_alpha()
        # Constraint
        if ti.static(self.rigid_solver.is_active() and self.rigid_solver.n_equalities > 0):
            self.equality_constraint_handler.compute_energy_gamma_G()
            self.equality_constraint_handler.update_gradient_hessian_alpha()
        # Contact
        for contact in ti.static(self.contact_handlers):
            contact.compute_energy_gamma_G()
            contact.update_gradient_hessian_alpha()

    @ti.func
    def init_linesearch_energy_gradient_hessian(self):
        """
        Reset energy/grad/hess accumulators for current alpha.

        重置当前 alpha 的能量/一阶/二阶导累加器。
        """
        energy = ti.static(self.linesearch_state.energy)
        alpha = ti.static(self.linesearch_state.step_size)
        for i_b in ti.ndrange(self._B):
            if not self.batch_linesearch_active[i_b]:
                continue

            # energy
            energy[i_b] = (
                self.linesearch_state.prev_energy[i_b]
                + 0.5 * alpha[i_b] ** 2 * self.linesearch_state[i_b].d2ellA_dalpha2
            )

            # gradient
            self.linesearch_state[i_b].dell_dalpha = 0.0

            # hessian
            self.linesearch_state.d2ell_dalpha2[i_b] = self.linesearch_state.d2ellA_dalpha2[i_b]

    @ti.func
    def compute_fem_gradient_alpha(self, i_step: ti.i32):
        """
        Contribution of FEM to dE/dalpha along line direction.

        FEM 对线搜索方向导数 dE/dalpha 的贡献。
        """
        dp = ti.static(self.linesearch_fem_state_v.dp)
        v = ti.static(self.fem_state_v.v)
        v_star = ti.static(self.fem_solver.elements_v.vel)
        for i_b, i_v in ti.ndrange(self._B, self.fem_solver.n_vertices):
            if not self.batch_linesearch_active[i_b]:
                continue
            self.linesearch_state.dell_dalpha[i_b] += dp[i_b, i_v].dot(v[i_b, i_v] - v_star[i_step + 1, i_v, i_b])

    @ti.func
    def compute_rigid_gradient_alpha(self):
        """
        Contribution of rigid to dE/dalpha along line direction.

        刚体对线搜索方向导数 dE/dalpha 的贡献。
        """
        dp = ti.static(self.linesearch_rigid_state_dof.dp)
        v = ti.static(self.rigid_state_dof.v)
        v_star = ti.static(self.rigid_solver.dofs_state.vel)
        for i_b, i_d in ti.ndrange(self._B, self.rigid_solver.n_dofs):
            if not self.batch_linesearch_active[i_b]:
                continue
            self.linesearch_state.dell_dalpha[i_b] += dp[i_b, i_d] * (v[i_b, i_d] - v_star[i_d, i_b])

    @ti.func
    def compute_fem_energy_alpha(self, i_step: ti.i32, energy: ti.template()):
        """
        FEM linear term alpha * dp·(v - v*).

        FEM 线性项：alpha * dp·(v - v*) 的能量贡献。
        """
        alpha = ti.static(self.linesearch_state.step_size)
        dp = ti.static(self.linesearch_fem_state_v.dp)
        v = ti.static(self.fem_state_v.v)
        v_star = ti.static(self.fem_solver.elements_v.vel)
        for i_b, i_v in ti.ndrange(self._B, self.fem_solver.n_vertices):
            if not self.batch_linesearch_active[i_b]:
                continue
            energy[i_b] += alpha[i_b] * dp[i_b, i_v].dot(v[i_b, i_v] - v_star[i_step + 1, i_v, i_b])

    @ti.func
    def compute_rigid_energy_alpha(self, energy: ti.template()):
        """
        Rigid linear term alpha * dp*(v - v*).

        刚体线性项：alpha * dp*(v - v*) 的能量贡献。
        """
        alpha = ti.static(self.linesearch_state.step_size)
        dp = ti.static(self.linesearch_rigid_state_dof.dp)
        v = ti.static(self.rigid_state_dof.v)
        v_star = ti.static(self.rigid_solver.dofs_state.vel)
        for i_b, i_d in ti.ndrange(self._B, self.rigid_solver.n_dofs):
            if not self.batch_linesearch_active[i_b]:
                continue
            energy[i_b] += alpha[i_b] * dp[i_b, i_d] * (v[i_b, i_d] - v_star[i_d, i_b])

    @ti.func
    def prepare_search_direction_data(self):
        """
        Precompute A @ x (dp) for line search and d2E_A/dalpha^2.

        为线搜索预计算方向上的 A@x（dp），并统计动力项二阶导。
        """
        constraints = ti.static(self.constraints)
        sap_info = ti.static(constraints.sap_info)
        for i_c in range(self.n_constraints[None]):
            i_b = constraints[i_c].batch_idx
            if self.coupler.batch_linesearch_active[i_b]:
                sap_info[i_c].dvc = self.compute_Jx(i_c, self.coupler.pcg_rigid_state_dof.x)

    @ti.func
    def compute_d2ellA_dalpha2(self):
        """
        Accumulate quadratic term x^T A x across subsystems.

        累计二次项 x^T A x（各子系统求和）。
        """
        for i_b in ti.ndrange(self._B):
            self.linesearch_state[i_b].d2ellA_dalpha2 = 0.0
        if ti.static(self.fem_solver.is_active()):
            self.compute_fem_d2ellA_dalpha2()
        if ti.static(self.rigid_solver.is_active()):
            self.compute_rigid_d2ellA_dalpha2()

    @ti.func
    def compute_fem_d2ellA_dalpha2(self):
        """
        FEM contribution to x^T A x.

        FEM 对二次项 x^T A x 的贡献。
        """
        for i_b, i_v in ti.ndrange(self._B, self.fem_solver.n_vertices):
            if not self.batch_linesearch_active[i_b]:
                continue
            self.linesearch_state[i_b].d2ellA_dalpha2 += self.pcg_fem_state_v[i_b, i_v].x.dot(
                self.linesearch_fem_state_v[i_b, i_v].dp
            )

    @ti.func
    def compute_rigid_d2ellA_dalpha2(self):
        """
        Rigid contribution to x^T A x.

        刚体对二次项 x^T A x 的贡献。
        """
        for i_b, i_d in ti.ndrange(self._B, self.rigid_solver.n_dofs):
            if not self.batch_linesearch_active[i_b]:
                continue
            self.linesearch_state[i_b].d2ellA_dalpha2 += (
                self.pcg_rigid_state_dof[i_b, i_d].x * self.linesearch_rigid_state_dof[i_b, i_d].dp
            )

    @ti.func
    def prepare_fem_search_direction_data(self):
        """
        Compute dp = A_fem @ x for line search.

        计算线搜索所需 dp = A_fem @ x。
        """
        self.compute_fem_matrix_vector_product(
            self.pcg_fem_state_v.x, self.linesearch_fem_state_v.dp, self.batch_linesearch_active
        )

    @ti.func
    def prepare_rigid_search_direction_data(self):
        """
        Compute dp = M @ x for line search.

        计算线搜索所需 dp = M @ x（刚体）。
        """
        self.compute_rigid_mass_mat_vec_product(
            self.pcg_rigid_state_dof.x, self.linesearch_rigid_state_dof.dp, self.batch_linesearch_active
        )

    @ti.func
    def _func_init_linesearch(self, step_size: float):
        """
        Initialize per-batch line-search state and m = x^T g.

        初始化线搜索状态与方向导数首项 m = x^T g。
        """
        for i_b in ti.ndrange(self._B):
            self.batch_linesearch_active[i_b] = self.batch_active[i_b]
            if not self.batch_linesearch_active[i_b]:
                continue
            self.linesearch_state[i_b].step_size = step_size
            self.linesearch_state[i_b].m = 0.0

        if ti.static(self.fem_solver.is_active()):
            self._func_init_fem_linesearch()
        if ti.static(self.rigid_solver.is_active()):
            self._func_init_rigid_linesearch()

    @ti.func
    def _func_init_fem_linesearch(self):
        """
        Initialize FEM line-search accumulators.

      初始化 FEM 的线搜索累加量（m 与 x_prev）。
        """
        for i_b, i_v in ti.ndrange(self._B, self.fem_solver.n_vertices):
            if not self.batch_linesearch_active[i_b]:
                continue
            self.linesearch_state[i_b].m += self.pcg_fem_state_v[i_b, i_v].x.dot(self.fem_state_v.gradient[i_b, i_v])
            self.linesearch_fem_state_v[i_b, i_v].x_prev = self.fem_state_v.v[i_b, i_v]

    @ti.func
    def _func_init_rigid_linesearch(self):
        """
        Initialize rigid line-search accumulators.

        初始化刚体的线搜索累加量（m 与 x_prev）。
        """
        for i_b, i_d in ti.ndrange(self._B, self.rigid_solver.n_dofs):
            if not self.batch_linesearch_active[i_b]:
                continue
            self.linesearch_state[i_b].m += (
                self.pcg_rigid_state_dof[i_b, i_d].x * self.rigid_state_dof.gradient[i_b, i_d]
            )
            self.linesearch_rigid_state_dof[i_b, i_d].x_prev = self.rigid_state_dof.v[i_b, i_d]

    @ti.func
    def check_initial_exact_linesearch_convergence(self):
        """
        Early termination and special handling before iterations.

        在线搜索迭代前进行早停判断与特殊处理（如步长直接取 1 的情形）。
        """
        for i_b in ti.ndrange(self._B):
            if not self.batch_linesearch_active[i_b]:
                continue
            self.batch_linesearch_active[i_b] = self.linesearch_state[i_b].dell_dalpha > 0.0

        if ti.static(self.fem_solver.is_active()):
            self.update_initial_fem_state()
        if ti.static(self.rigid_solver.is_active()):
            self.update_initial_rigid_state()

        # When tolerance is small but gradient norm is small, take step 1.0 and end, this is a rare case, directly
        # copied from drake
        # Link: https://github.com/RobotLocomotion/drake/blob/3bb00e611983fb894151c547776d5aa85abe9139/multibody/contact_solvers/sap/sap_solver.cc#L625
        for i_b in range(self._B):
            if not self.batch_linesearch_active[i_b]:
                continue
            err_threshold = (
                self._sap_convergence_atol + self._sap_convergence_rtol * self.linesearch_state[i_b].prev_energy
            )
            if -self.linesearch_state[i_b].m < err_threshold:
                self.batch_linesearch_active[i_b] = False
                self.linesearch_state[i_b].step_size = 1.0

    @ti.func
    def update_initial_fem_state(self):
        """
        If early-accept, update v = x_prev + x (FEM).

        若早接收，更新 FEM 的 v = x_prev + x。
        """
        for i_b, i_v in ti.ndrange(self._B, self.fem_solver.n_vertices):
            if not self.batch_linesearch_active[i_b]:
                continue
            err_threshold = (
                self._sap_convergence_atol + self._sap_convergence_rtol * self.linesearch_state[i_b].prev_energy
            )
            if -self.linesearch_state[i_b].m < err_threshold:
                self.fem_state_v.v[i_b, i_v] = (
                    self.linesearch_fem_state_v[i_b, i_v].x_prev + self.pcg_fem_state_v[i_b, i_v].x
                )

    @ti.func
    def update_initial_rigid_state(self):
        """
        If early-accept, update v = x_prev + x (rigid).

        若早接收，更新刚体的 v = x_prev + x。
        """
        for i_b, i_d in ti.ndrange(self._B, self.rigid_solver.n_dofs):
            if not self.batch_linesearch_active[i_b]:
                continue
            err_threshold = (
                self._sap_convergence_atol + self._sap_convergence_rtol * self.linesearch_state[i_b].prev_energy
            )
            if -self.linesearch_state[i_b].m < err_threshold:
                self.rigid_state_dof.v[i_b, i_d] = (
                    self.linesearch_rigid_state_dof[i_b, i_d].x_prev + self.pcg_rigid_state_dof[i_b, i_d].x
                )

    def one_linesearch_iter(self, i_step: ti.i32):
        """
        One iteration of backtracking/energy-eval for line search.

        线搜索的一次迭代（更新 v、计算能量并检查收敛）。
        """
        self.update_velocity_linesearch()
        self.compute_total_energy(i_step, self.linesearch_state.energy)
        self.check_linesearch_convergence()

    @ti.func
    def update_velocity_linesearch(self):
        """
        Update velocities using current step size and direction.

        按当前步长与方向更新速度。
        """
        if ti.static(self.fem_solver.is_active()):
            self.update_fem_velocity_linesearch()
        if ti.static(self.rigid_solver.is_active()):
            self.update_rigid_velocity_linesearch()

    @ti.func
    def update_fem_velocity_linesearch(self):
        """
        Update FEM velocities along line-search direction.

        沿线搜索方向更新 FEM 速度。
        """
        for i_b, i_v in ti.ndrange(self._B, self.fem_solver.n_vertices):
            if not self.batch_linesearch_active[i_b]:
                continue
            self.fem_state_v.v[i_b, i_v] = (
                self.linesearch_fem_state_v[i_b, i_v].x_prev
                + self.linesearch_state[i_b].step_size * self.pcg_fem_state_v[i_b, i_v].x
            )

    @ti.func
    def update_rigid_velocity_linesearch(self):
        """
        Update rigid DOF velocities along line-search direction.

        沿线搜索方向更新刚体 DOF 速度。
        """
        for i_b, i_d in ti.ndrange(self._B, self.rigid_solver.n_dofs):
            if not self.batch_linesearch_active[i_b]:
                continue
            self.rigid_state_dof.v[i_b, i_d] = (
                self.linesearch_rigid_state_dof[i_b, i_d].x_prev
                + self.linesearch_state[i_b].step_size * self.pcg_rigid_state_dof[i_b, i_d].x
            )

    def exact_linesearch(self, i_step: ti.i32):
        """
        Note
        ------
        Exact line search using rtsafe
        https://github.com/RobotLocomotion/drake/blob/master/multibody/contact_solvers/sap/sap_solver.h#L393

        中文说明
        ------
        使用 rtsafe 的精确线搜索（牛顿-二分混合），参考上方 Drake 实现链接。
        """
        self.init_exact_linesearch(i_step)
        for i in range(self._n_linesearch_iterations):
            self.one_exact_linesearch_iter(i_step)

    @ti.kernel
    def one_exact_linesearch_iter(self, i_step: ti.i32):
        """
        One rtsafe iteration: update v, compute f/df, update bracket and alpha.

        一次 rtsafe 迭代：更新 v，计算 f/df，更新括号区间与步长。
        """
        self.update_velocity_linesearch()
        self.compute_line_energy_gradient_hessian(i_step)
        self.compute_f_df_bracket()
        self.find_next_step_size()

    @ti.func
    def compute_f_df_bracket(self):
        """
        Compute the function (derivative of total energy) value and its derivative to alpha.
        Update the bracket for the next step size.

        The bracket is defined by [alpha_min, alpha_max] which is the range that contains the root of df/dalpha = 0.

        计算总能量对步长的导数 f 及其导数 df，并更新包含根的区间 [alpha_min, alpha_max]。
        """
        for i_b in ti.ndrange(self._B):
            if not self.batch_linesearch_active[i_b]:
                continue
            self.linesearch_state[i_b].f = (
                self.linesearch_state[i_b].dell_dalpha / self.linesearch_state[i_b].dell_scale
            )
            self.linesearch_state[i_b].df = (
                self.linesearch_state[i_b].d2ell_dalpha2 / self.linesearch_state[i_b].dell_scale
            )
            if ti.math.sign(self.linesearch_state[i_b].f) != ti.math.sign(self.linesearch_state[i_b].f_upper):
                self.linesearch_state[i_b].alpha_min = self.linesearch_state[i_b].step_size
                self.linesearch_state[i_b].f_lower = self.linesearch_state[i_b].f
            else:
                self.linesearch_state[i_b].alpha_max = self.linesearch_state[i_b].step_size
                self.linesearch_state[i_b].f_upper = self.linesearch_state[i_b].f
            if ti.abs(self.linesearch_state[i_b].f) < self._linesearch_ftol:
                self.batch_linesearch_active[i_b] = False

    @ti.func
    def find_next_step_size(self):
        """
        Hybrid Newton-bisection update for next alpha.

        采用牛顿/二分混合策略更新下一次的步长 alpha。
        """
        for i_b in ti.ndrange(self._B):
            if not self.batch_linesearch_active[i_b]:
                continue
            newton_is_slow = 2.0 * ti.abs(self.linesearch_state[i_b].f) > ti.abs(
                self.linesearch_state[i_b].minus_dalpha_prev * self.linesearch_state[i_b].df
            )
            self.linesearch_state[i_b].minus_dalpha_prev = self.linesearch_state[i_b].minus_dalpha
            if newton_is_slow:
                # bisect
                self.linesearch_state[i_b].minus_dalpha = 0.5 * (
                    self.linesearch_state[i_b].alpha_min - self.linesearch_state[i_b].alpha_max
                )
                self.linesearch_state[i_b].step_size = (
                    self.linesearch_state[i_b].alpha_min - self.linesearch_state[i_b].minus_dalpha
                )
            else:
                # newton
                self.linesearch_state[i_b].minus_dalpha = self.linesearch_state[i_b].f / self.linesearch_state[i_b].df
                self.linesearch_state[i_b].step_size = (
                    self.linesearch_state[i_b].step_size - self.linesearch_state[i_b].minus_dalpha
                )
                if (
                    self.linesearch_state[i_b].step_size <= self.linesearch_state[i_b].alpha_min
                    or self.linesearch_state[i_b].step_size >= self.linesearch_state[i_b].alpha_max
                ):
                    # bisect
                    self.linesearch_state[i_b].minus_dalpha = 0.5 * (
                        self.linesearch_state[i_b].alpha_min - self.linesearch_state[i_b].alpha_max
                    )
                    self.linesearch_state[i_b].step_size = (
                        self.linesearch_state[i_b].alpha_min - self.linesearch_state[i_b].minus_dalpha
                    )
            if ti.abs(self.linesearch_state[i_b].minus_dalpha) < self.linesearch_state[i_b].alpha_tol:
                self.batch_linesearch_active[i_b] = False


    # ------------------------------------------------------------------------------------
    # ----------------------------------- Properties -------------------------------------
    # ------------------------------------------------------------------------------------
    @property
    def active_solvers(self):
        """所有活动求解器均由 scene.simulator 管理"""
        return self.sim.active_solvers


@ti.data_oriented
class BaseConstraintHandler(RBC):
    """
    Base class for constraint handling in SAPCoupler.

    SAP 耦合器中的约束处理基类。提供约束数据布局、正则化与能量/梯度/Hessian 计算的通用框架。
    """

    def __init__(
        self,
        simulator: "Simulator",
        stiffness: float = 1e8,
        beta: float = 0.1,
    ) -> None:
        self.sim = simulator
        self.stiffness = stiffness
        self.beta = beta
        self._B = simulator._B
        self.coupler = simulator.coupler
        self.sap_constraint_info_type = ti.types.struct(
            k=gs.ti_float,      # 约束刚度
            R=gs.ti_float,      # 正则化
            R_inv=gs.ti_float,  # 正则化的逆
            v_hat=gs.ti_float,  # 稳定化速度
            energy=gs.ti_float, # 能量
            gamma=gs.ti_float,  # 约束冲量
            G=gs.ti_float,      # Hessian（标量约束）
            dvc=gs.ti_float,    # 约束速度的变化量（用于线搜索）
        )

    @ti.func
    def compute_constraint_regularization(self, sap_info, i_c, w_rms, time_step):
        """
        Compute constraint regularization R and its inverse.

        计算约束正则化 R 与其逆 R_inv。R 基于等效质量 RMS 与时间步稳定性项。
        """
        beta_factor = self.beta**2 / (4.0 * ti.math.pi**2)
        dt2 = time_step**2
        k = sap_info[i_c].k
        R = max(beta_factor * w_rms, 1.0 / (dt2 * k))
        sap_info[i_c].R = R
        sap_info[i_c].R_inv = 1.0 / R

    @ti.func
    def compute_constraint_gamma_G(self, sap_info, i_c, vc):
        """
        Compute constraint impulse gamma and Hessian G.

        计算约束冲量 gamma 以及 Hessian G（此处为标量）。
        """
        y = (sap_info[i_c].v_hat - vc) * sap_info[i_c].R_inv
        sap_info[i_c].gamma = y
        sap_info[i_c].G = sap_info[i_c].R_inv

    @ti.func
    def compute_energy(self, energy: ti.template()):
        """
        Accumulate constraint energy to total energy.

        将约束能量累加到总能量（仅对激活的 batch）。
        """
        constraints = ti.static(self.constraints)
        sap_info = ti.static(constraints.sap_info)
        for i_c in range(self.n_constraints[None]):
            i_b = constraints[i_c].batch_idx
            if self.coupler.batch_linesearch_active[i_b]:
                vc = self.compute_vc(i_c)
                self.compute_constraint_energy(sap_info, i_c, vc)
                energy[i_b] += sap_info[i_c].energy

    @ti.func
    def compute_constraint_energy(self, sap_info, i_c, vc):
        """
        Compute constraint energy from quadratic regularization.

        按二次正则化形式计算约束能量：0.5 * (v_hat - vc)^2 / R_inv。
        """
        y = (sap_info[i_c].v_hat - vc) * sap_info[i_c].R_inv
        sap_info[i_c].energy = 0.5 * y**2 * sap_info[i_c].R


@ti.data_oriented
class RigidConstraintHandler(BaseConstraintHandler):
    """
    Rigid body constraints in SAPCoupler. Currently only support joint equality constraints.

    刚体约束处理器。目前仅支持关节等式约束，并以 SAP 形式接入。
    """

    def __init__(
        self,
        simulator: "Simulator",
        stiffness: float = 1e8,
        beta: float = 0.1,
    ) -> None:
        super().__init__(simulator, stiffness, beta)
        self.rigid_solver = simulator.rigid_solver
        self.constraint_solver = simulator.rigid_solver.constraint_solver
        self.max_constraints = simulator.rigid_solver.n_equalities * self._B
        self.n_constraints = ti.field(gs.ti_int, shape=())
        self.constraint_type = ti.types.struct(
            batch_idx=gs.ti_int,             # batch 索引
            i_dof1=gs.ti_int,               # 约束第一个 DOF
            i_dof2=gs.ti_int,               # 约束第二个 DOF
            sap_info=self.sap_constraint_info_type,  # SAP 约束信息
        )
        self.constraints = self.constraint_type.field(shape=(self.max_constraints,))
        self.Jt = ti.field(gs.ti_float, shape=(self.max_constraints, self.rigid_solver.n_dofs))
        self.M_inv_Jt = ti.field(gs.ti_float, shape=(self.max_constraints, self.rigid_solver.n_dofs))
        self.W = ti.field(gs.ti_float, shape=(self.max_constraints,))

    @ti.kernel
    def build_constraints(
        self,
        equalities_info: array_class.EqualitiesInfo,
        joints_info: array_class.JointsInfo,
        static_rigid_sim_config: ti.template(),
        static_rigid_sim_cache_key: array_class.StaticRigidSimCacheKey,
    ):
        """
        Build joint equality constraints and corresponding Jt rows.

        构建关节等式约束与对应的雅可比转置 Jt 行（各 batch 共享同一组约束）。
        """
        self.n_constraints[None] = 0
        self.Jt.fill(0.0)
        # 目前所有 batch 使用同一组约束；未来可考虑为不同 batch 配置不同约束。
        dt2 = self.sim._substep_dt**2
        for i_b, i_e in ti.ndrange(self._B, self.rigid_solver.n_equalities):
            if equalities_info.eq_type[i_e, i_b] == gs.EQUALITY_TYPE.JOINT:
                i_c = ti.atomic_add(self.n_constraints[None], 1)
                self.constraints[i_c].batch_idx = i_b
                I_joint1 = (
                    [equalities_info.eq_obj1id[i_e, i_b], i_b]
                    if ti.static(static_rigid_sim_config.batch_joints_info)
                    else equalities_info.eq_obj1id[i_e, i_b]
                )
                I_joint2 = (
                    [equalities_info.eq_obj2id[i_e, i_b], i_b]
                    if ti.static(static_rigid_sim_config.batch_joints_info)
                    else equalities_info.eq_obj2id[i_e, i_b]
                )
                i_dof1 = joints_info.dof_start[I_joint1]
                i_dof2 = joints_info.dof_start[I_joint2]
                self.constraints[i_c].i_dof1 = i_dof1
                self.constraints[i_c].i_dof2 = i_dof2
                self.constraints[i_c].sap_info.k = self.stiffness
                self.constraints[i_c].sap_info.R_inv = dt2 * self.stiffness
                self.constraints[i_c].sap_info.R = 1.0 / self.constraints[i_c].sap_info.R_inv
                self.constraints[i_c].sap_info.v_hat = 0.0
                self.Jt[i_c, i_dof1] = 1.0
                self.Jt[i_c, i_dof2] = -1.0

    @ti.func
    def compute_regularization(self):
        """
        Compute regularization for each constraint using Delassus estimate.

        基于 Delassus 等效质量为每个约束计算正则化，并设置 v_hat。
        """
        dt_inv = 1.0 / self.sim._substep_dt
        q = ti.static(self.rigid_solver.dofs_state.pos)
        sap_info = ti.static(self.constraints.sap_info)
        for i_c in range(self.n_constraints[None]):
            i_b = self.constraints[i_c].batch_idx
            g0 = q[self.constraints[i_c].i_dof1, i_b] - q[self.constraints[i_c].i_dof2, i_b]
            self.constraints[i_c].sap_info.v_hat = -g0 * dt_inv
            W = self.compute_delassus(i_c)
            self.compute_constraint_regularization(sap_info, i_c, W, self.sim._substep_dt)

    @ti.func
    def compute_delassus_world_frame(self):
        """
        Compute M^-1 J^T and W = J M^-1 J^T in world frame.

        计算质量矩阵分解下的 M^-1 J^T，并据此累积 W = J M^-1 J^T。
        """
        self.coupler.rigid_solve_jacobian(
            self.Jt, self.M_inv_Jt, self.n_constraints[None], self.constraints.batch_idx, 1
        )
        self.W.fill(0.0)
        for i_c, i_d in ti.ndrange(self.n_constraints[None], self.rigid_solver.n_dofs):
            self.W[i_c] += self.M_inv_Jt[i_c, i_d] * self.Jt[i_c, i_d]

    @ti.func
    def compute_delassus(self, i_c):
        """
        Return Delassus scalar for constraint i_c.

        返回第 i_c 个约束的 Delassus（标量）。
        """
        return self.W[i_c]

    @ti.func
    def compute_Jx(self, i_c, x):
        """
        Compute J @ x for 1D constraint.

        计算一维等式约束的 J @ x（x 为 DOF 速度/方向）。
        """
        i_b = self.constraints[i_c].batch_idx
        i_dof1 = self.constraints[i_c].i_dof1
        i_dof2 = self.constraints[i_c].i_dof2
        return x[i_b, i_dof1] - x[i_b, i_dof2]

    @ti.func
    def add_Jt_x(self, y, i_c, x):
        """
        y += J^T @ x for 1D constraint.

        执行 y += J^T @ x 的累加（标量约束向两 DOF 回写）。
        """
        i_b = self.constraints[i_c].batch_idx
        i_dof1 = self.constraints[i_c].i_dof1
        i_dof2 = self.constraints[i_c].i_dof2
        y[i_b, i_dof1] += x
        y[i_b, i_dof2] -= x

    @ti.func
    def compute_vc(self, i_c):
        """
        Compute constraint velocity vc = J @ v.

        计算当前约束速度 vc = J @ v。
        """
        return self.compute_Jx(i_c, self.coupler.rigid_state_dof.v)

    @ti.func
    def compute_gradient_hessian_diag(self):
        """
        Accumulate gradient, impulse and diagonal for constraints.

        累加等式约束的梯度、冲量与对角（G）。
        """
        constraints = ti.static(self.constraints)
        sap_info = ti.static(constraints.sap_info)
        for i_c in range(self.n_constraints[None]):
            vc = self.compute_vc(i_c)
            self.compute_constraint_gamma_G(sap_info, i_c, vc)
            self.add_Jt_x(self.coupler.rigid_state_dof.gradient, i_c, -sap_info[i_c].gamma)
            self.add_Jt_x(self.coupler.rigid_state_dof.impulse, i_c, sap_info[i_c].gamma)

    @ti.func
    def compute_Ap(self):
        """
        Accumulate Ap contribution from constraints: J^T G J p.

        计算并累加约束项的 Ap：J^T G J p。
        """
        constraints = ti.static(self.constraints)
        sap_info = ti.static(constraints.sap_info)
        for i_c in range(self.n_constraints[None]):
            # Jt @ G @ J @ p
            x = self.compute_Jx(i_c, self.coupler.pcg_rigid_state_dof.p)
            x = sap_info[i_c].G * x
            self.add_Jt_x(self.coupler.pcg_rigid_state_dof.Ap, i_c, x)

    @ti.func
    def prepare_search_direction_data(self):
        """
        Precompute dvc = J @ x for line-search.

        为线搜索预计算 dvc = J @ x。
        """
        constraints = ti.static(self.constraints)
        sap_info = ti.static(constraints.sap_info)
        for i_c in range(self.n_constraints[None]):
            i_b = constraints[i_c].batch_idx
            if self.coupler.batch_linesearch_active[i_b]:
                sap_info[i_c].dvc = self.compute_Jx(i_c, self.coupler.pcg_rigid_state_dof.x)

    @ti.func
    def compute_energy_gamma_G(self):
        """
        Update gamma/G and accumulate constraint energy for line-search.

        更新约束的 gamma/G，并在当前 v 下累计约束能量。
        """
        constraints = ti.static(self.constraints)
        sap_info = ti.static(constraints.sap_info)
        for i_c in range(self.n_constraints[None]):
            vc = self.compute_vc(i_c)
            self.compute_constraint_energy_gamma_G(sap_info, i_c, vc)

    @ti.func
    def compute_constraint_energy_gamma_G(self, sap_info, i_c, vc):
        """
        Update gamma/G and constraint energy.

        更新 gamma/G 并计算约束能量。
        """
        self.compute_constraint_gamma_G(sap_info, i_c, vc)
        sap_info[i_c].energy = 0.5 * sap_info[i_c].gamma ** 2 * sap_info[i_c].R

    @ti.func
    def update_gradient_hessian_alpha(self):
        """
        Accumulate derivatives wrt alpha for line-search.

        在线搜索中累计对步长 alpha 的一阶/二阶导数贡献。
        """
        dvc = ti.static(self.constraints.sap_info.dvc)
        gamma = ti.static(self.constraints.sap_info.gamma)
        G = ti.static(self.constraints.sap_info.G)
        for i_c in ti.ndrange(self.n_constraints[None]):
            i_b = self.constraints[i_c].batch_idx
            if self.coupler.batch_linesearch_active[i_b]:
                self.coupler.linesearch_state.dell_dalpha[i_b] -= dvc[i_c] * gamma[i_c]
                self.coupler.linesearch_state.d2ell_dalpha2[i_b] += dvc[i_c] ** 2 * G[i_c]


class ContactMode(IntEnum):
    STICK = 0
    SLIDE = 1
    NO_CONTACT = 2


@ti.data_oriented
class BaseContactHandler(RBC):
    """
    Base class for contact handling in SAPCoupler.

    This class provides a framework for managing contact pairs, computing gradients,
    and handling contact-related computations.

    接触处理基类。定义接触对数据结构、正则化、能量/梯度/Hessian 以及线搜索相关的通用计算流程。
    """

    def __init__(
        self,
        simulator: "Simulator",
    ) -> None:
        self.sim = simulator
        self.coupler = simulator.coupler
        self.n_contact_pairs = ti.field(gs.ti_int, shape=())
        self.sap_contact_info_type = ti.types.struct(
            k=gs.ti_float,        # 接触刚度
            phi0=gs.ti_float,     # 初始符号距离
            Rn=gs.ti_float,       # 法向正则
            Rt=gs.ti_float,       # 切向正则
            Rn_inv=gs.ti_float,   # 法向正则的逆
            Rt_inv=gs.ti_float,   # 切向正则的逆
            vn_hat=gs.ti_float,   # 法向稳定化速度
            mu=gs.ti_float,       # 摩擦系数
            mu_hat=gs.ti_float,   # 正则化后的摩擦系数
            mu_factor=gs.ti_float,# 摩擦系数因子 1/(1+mu_tilde^2)
            energy=gs.ti_float,   # 接触能量
            gamma=gs.ti_vec3,     # 冲量（t0, t1, n）
            G=gs.ti_mat3,         # Hessian（3x3）
            dvc=gs.ti_vec3,       # 线搜索用接触速度变化
        )

    @ti.func
    def compute_jacobian(self):
        """
        Compute contact Jacobian (to be implemented by subclasses).

        计算接触雅可比（由子类实现）。
        """
        pass

    @ti.func
    def update_gradient_hessian_alpha(self):
        """
        Accumulate line-search derivatives from contacts.

        将接触项对线搜索的一阶/二阶导贡献累加到全局状态。
        """
        dvc = ti.static(self.contact_pairs.sap_info.dvc)
        gamma = ti.static(self.contact_pairs.sap_info.gamma)
        G = ti.static(self.contact_pairs.sap_info.G)
        for i_p in ti.ndrange(self.n_contact_pairs[None]):
            i_b = self.contact_pairs[i_p].batch_idx
            if self.coupler.batch_linesearch_active[i_b]:
                self.coupler.linesearch_state.dell_dalpha[i_b] -= dvc[i_p].dot(gamma[i_p])
                self.coupler.linesearch_state.d2ell_dalpha2[i_b] += dvc[i_p].dot(G[i_p] @ dvc[i_p])

    @ti.func
    def compute_delassus_world_frame(self):
        """
        Compute W = J M^-1 J^T in world frame (to be implemented by subclasses).

        在世界系中计算 Delassus 矩阵 W（由子类实现）。
        """
        pass

    @ti.func
    def compute_regularization(self):
        """
        Compute contact regularization parameters for all pairs.

        为所有接触对计算正则化参数（Rn/Rt、其逆、vn_hat、mu_hat 等）。
        """
        self.compute_delassus_world_frame()
        for i_p in range(self.n_contact_pairs[None]):
            W = self.compute_delassus(i_p)
            w_rms = W.norm() / 3.0
            self.compute_contact_regularization(self.contact_pairs.sap_info, i_p, w_rms, self.sim._substep_dt)

    @ti.func
    def compute_energy_gamma_G(self):
        """
        Update gamma/G and energy for all pairs.

        为每个接触对计算接触速度 vc，更新 gamma/G，并累计能量。
        """
        for i_p in range(self.n_contact_pairs[None]):
            vc = self.compute_contact_velocity(i_p)
            self.compute_contact_energy_gamma_G(self.contact_pairs.sap_info, i_p, vc)

    @ti.func
    def compute_energy(self, energy: ti.template()):
        """
        Accumulate contact energy to total energy for active batches.

        将接触能量累加到总能量（仅对激活的 batch）。
        """
        sap_info = ti.static(self.contact_pairs.sap_info)
        for i_p in range(self.n_contact_pairs[None]):
            i_b = self.contact_pairs[i_p].batch_idx
            if self.coupler.batch_linesearch_active[i_b]:
                vc = self.compute_contact_velocity(i_p)
                self.compute_contact_energy(sap_info, i_p, vc)
                energy[i_b] += sap_info[i_p].energy

    @ti.func
    def compute_contact_gamma_G(self, sap_info, i_p, vc):
        """
        Compute contact impulse gamma and Hessian G from regularized Coulomb model.

        根据正则化库伦模型计算接触冲量 gamma 与 Hessian G，分 STICK/SLIDE/NO_CONTACT 三种模式。
        """
        y = ti.Vector([0.0, 0.0, sap_info[i_p].vn_hat]) - vc
        y[0] *= sap_info[i_p].Rt_inv
        y[1] *= sap_info[i_p].Rt_inv
        y[2] *= sap_info[i_p].Rn_inv
        yr = y[:2].norm(gs.EPS)
        yn = y[2]

        t_hat = y[:2] / yr
        contact_mode = self.compute_contact_mode(sap_info[i_p].mu, sap_info[i_p].mu_hat, yr, yn)
        sap_info[i_p].gamma.fill(0.0)
        sap_info[i_p].G.fill(0.0)
        if contact_mode == ContactMode.STICK:
            sap_info[i_p].gamma = y
            sap_info[i_p].G[0, 0] = sap_info[i_p].Rt_inv
            sap_info[i_p].G[1, 1] = sap_info[i_p].Rt_inv
            sap_info[i_p].G[2, 2] = sap_info[i_p].Rn_inv
        elif contact_mode == ContactMode.SLIDE:
            gn = (yn + sap_info[i_p].mu_hat * yr) * sap_info[i_p].mu_factor
            gt = sap_info[i_p].mu * gn * t_hat
            sap_info[i_p].gamma = ti.Vector([gt[0], gt[1], gn])
            P = t_hat.outer_product(t_hat)
            Pperp = ti.Matrix.identity(gs.ti_float, 2) - P
            dgt_dyt = sap_info[i_p].mu * (gn / yr * Pperp + sap_info[i_p].mu_hat * sap_info[i_p].mu_factor * P)
            dgt_dyn = sap_info[i_p].mu * sap_info[i_p].mu_factor * t_hat
            dgn_dyt = sap_info[i_p].mu_hat * sap_info[i_p].mu_factor * t_hat
            dgn_dyn = sap_info[i_p].mu_factor

            sap_info[i_p].G[:2, :2] = dgt_dyt * sap_info[i_p].Rt_inv
            sap_info[i_p].G[:2, 2] = dgt_dyn * sap_info[i_p].Rn_inv
            sap_info[i_p].G[2, :2] = dgn_dyt * sap_info[i_p].Rt_inv
            sap_info[i_p].G[2, 2] = dgn_dyn * sap_info[i_p].Rn_inv
        else:
            # NO_CONTACT：保持 gamma=0, G=0
            pass

    @ti.func
    def compute_contact_energy_gamma_G(self, sap_info, i_p, vc):
        """
        Update gamma and G, then compute contact energy 0.5 * gamma^T R gamma.

        先更新 gamma 与 G，再计算接触能量 0.5 * gamma^T R gamma。
        """
        self.compute_contact_gamma_G(sap_info, i_p, vc)
        R_gamma = sap_info[i_p].gamma
        R_gamma[0] *= sap_info[i_p].Rt
        R_gamma[1] *= sap_info[i_p].Rt
        R_gamma[2] *= sap_info[i_p].Rn
        sap_info[i_p].energy = 0.5 * sap_info[i_p].gamma.dot(R_gamma)

    @ti.func
    def compute_contact_energy(self, sap_info, i_p, vc):
        """
        Compute contact energy by first computing gamma for mode, then 0.5 gamma^T R gamma.

        按接触模式计算 gamma 后，求能量 0.5 * gamma^T R gamma。
        """
        y = ti.Vector([0.0, 0.0, sap_info[i_p].vn_hat]) - vc
        y[0] *= sap_info[i_p].Rt_inv
        y[1] *= sap_info[i_p].Rt_inv
        y[2] *= sap_info[i_p].Rn_inv
        yr = y[:2].norm(gs.EPS)
        yn = y[2]

        t_hat = y[:2] / yr
        contact_mode = self.compute_contact_mode(sap_info[i_p].mu, sap_info[i_p].mu_hat, yr, yn)
        sap_info[i_p].gamma.fill(0.0)
        if contact_mode == ContactMode.STICK:
            sap_info[i_p].gamma = y
        elif contact_mode == ContactMode.SLIDE:
            gn = (yn + sap_info[i_p].mu_hat * yr) * sap_info[i_p].mu_factor
            gt = sap_info[i_p].mu * gn * t_hat
            sap_info[i_p].gamma = ti.Vector([gt[0], gt[1], gn])
        else:
            pass

        R_gamma = sap_info[i_p].gamma
        R_gamma[0] *= sap_info[i_p].Rt
        R_gamma[1] *= sap_info[i_p].Rt
        R_gamma[2] *= sap_info[i_p].Rn
        sap_info[i_p].energy = 0.5 * sap_info[i_p].gamma.dot(R_gamma)

    @ti.func
    def compute_contact_mode(self, mu, mu_hat, yr, yn):
        """
        Compute the contact mode based on the friction coefficients and the relative velocities.

        基于摩擦系数与相对速度判断接触模式（粘/滑/无接触）。
        """
        result = ContactMode.NO_CONTACT
        if yr <= mu * yn:
            result = ContactMode.STICK
        elif -mu_hat * yr < yn and yn < yr / mu:
            result = ContactMode.SLIDE
        return result

    @ti.func
    def compute_contact_regularization(self, sap_info, i_p, w_rms, time_step):
        """
        Compute contact regularization parameters (Rn/Rt/vn_hat/mu_hat).

        计算接触正则化参数（法/切向正则、稳定化速度 vn_hat、正则化摩擦 mu_hat 等）。
        """
        beta_factor = self.coupler._sap_beta**2 / (4.0 * ti.math.pi**2)
        k = sap_info[i_p].k
        Rn = max(beta_factor * w_rms, 1.0 / (time_step * k * (time_step + self.coupler._sap_taud)))
        Rt = self.coupler._sap_sigma * w_rms
        vn_hat = -sap_info[i_p].phi0 / (time_step + self.coupler._sap_taud)
        sap_info[i_p].Rn = Rn
        sap_info[i_p].Rt = Rt
        sap_info[i_p].Rn_inv = 1.0 / Rn
        sap_info[i_p].Rt_inv = 1.0 / Rt
        sap_info[i_p].vn_hat = vn_hat
        sap_info[i_p].mu_hat = sap_info[i_p].mu * Rt * sap_info[i_p].Rn_inv
        sap_info[i_p].mu_factor = 1.0 / (1.0 + sap_info[i_p].mu * sap_info[i_p].mu_hat)



@ti.data_oriented
class RigidContactHandler(BaseContactHandler):
    def __init__(
        self,
        simulator: "Simulator",
    ) -> None:
        super().__init__(simulator)
        self.rigid_solver = self.sim.rigid_solver

    # FIXME 类似于 constraint_solver_decomp.py:add_collision_constraints，可重构命名并移除 while。
    @ti.func
    def compute_jacobian(self):
        """
        Assemble Jt for rigid contacts by traversing kinematic tree.

        沿运动学树向上遍历，装配刚体接触的雅可比转置 Jt。
        """
        self.Jt.fill(0.0)
        for i_p in range(self.n_contact_pairs[None]):
            link = self.contact_pairs[i_p].link_idx
            i_b = self.contact_pairs[i_p].batch_idx
            while link > -1:
                link_maybe_batch = [link, i_b] if ti.static(self.rigid_solver._options.batch_links_info) else link
                # 逆序确保每行相关 DOF 严格降序
                for i_d_ in range(self.rigid_solver.links_info.n_dofs[link_maybe_batch]):
                    i_d = self.rigid_solver.links_info.dof_end[link_maybe_batch] - 1 - i_d_

                    cdof_ang = self.rigid_solver.dofs_state.cdof_ang[i_d, i_b]
                    cdof_vel = self.rigid_solver.dofs_state.cdof_vel[i_d, i_b]

                    t_quat = gu.ti_identity_quat()
                    t_pos = self.contact_pairs[i_p].contact_pos - self.rigid_solver.links_state.root_COM[link, i_b]
                    _, vel = gu.ti_transform_motion_by_trans_quat(cdof_ang, cdof_vel, t_pos, t_quat)

                    diff = vel
                    jac = diff
                    self.Jt[i_p, i_d] = self.Jt[i_p, i_d] + jac
                link = self.rigid_solver.links_info.parent_idx[link_maybe_batch]

    @ti.func
    def compute_gradient_hessian_diag(self):
        """
        Accumulate rigid contact gradient/impulse using gamma and G.

        利用 gamma 与 G，将刚体接触的梯度与冲量累加到全局。
        """
        sap_info = ti.static(self.contact_pairs.sap_info)
        for i_p in range(self.n_contact_pairs[None]):
            vc = self.compute_contact_velocity(i_p)
            self.compute_contact_gamma_G(sap_info, i_p, vc)
            self.add_Jt_x(self.coupler.rigid_state_dof.gradient, i_p, -sap_info[i_p].gamma)
            self.add_Jt_x(self.coupler.rigid_state_dof.impulse, i_p, sap_info[i_p].gamma)

    @ti.func
    def compute_pcg_matrix_vector_product(self):
        """
        Accumulate Ap = J^T G J p for rigid contacts.

        对刚体接触计算并累加 Ap = J^T G J p。
        """
        sap_info = ti.static(self.contact_pairs.sap_info)
        for i_p in range(self.n_contact_pairs[None]):
            # Jt @ G @ J @ p
            Jp = self.compute_Jx(i_p, self.coupler.pcg_rigid_state_dof.p)
            GJp = sap_info[i_p].G @ Jp
            self.add_Jt_x(self.coupler.pcg_rigid_state_dof.Ap, i_p, GJp)

    @ti.func
    def compute_contact_velocity(self, i_p):
        """
        Compute the contact velocity in the contact frame.

        计算接触坐标系下的接触速度（t0、t1、n 分量）。
        """
        return self.compute_Jx(i_p, self.coupler.rigid_state_dof.v)

    @ti.func
    def prepare_search_direction_data(self):
        """
        Precompute dvc = J @ x for line search.

        为线搜索预计算 dvc = J @ x。
        """
        sap_info = ti.static(self.contact_pairs.sap_info)
        for i_p in ti.ndrange(self.n_contact_pairs[None]):
            i_b = self.contact_pairs[i_p].batch_idx
            if self.coupler.batch_linesearch_active[i_b]:
                sap_info[i_p].dvc = self.compute_Jx(i_p, self.coupler.pcg_rigid_state_dof.x)

    @ti.func
    def compute_delassus_world_frame(self):
        """
        Compute W = J M^-1 J^T in world frame for rigid contacts.

        在世界系中计算刚体接触的 Delassus 矩阵 W。
        """
        self.coupler.rigid_solve_jacobian(
            self.Jt, self.M_inv_Jt, self.n_contact_pairs[None], self.contact_pairs.batch_idx, 3
        )
        self.W.fill(0.0)
        for i_p, i_d, i, j in ti.ndrange(self.n_contact_pairs[None], self.rigid_solver.n_dofs, 3, 3):
            self.W[i_p][i, j] += self.M_inv_Jt[i_p, i_d][i] * self.Jt[i_p, i_d][j]

    @ti.func
    def compute_delassus(self, i_p):
        """
        Return W for pair i_p in world components.

        返回第 i_p 个接触对在世界分量下的 W。
        """
        return self.W[i_p]

    @ti.func
    def compute_Jx(self, i_p, x):
        """
        Compute J @ x for pair i_p.

        计算第 i_p 个接触对的 J @ x。
        """
        pairs = ti.static(self.contact_pairs)
        i_b = pairs[i_p].batch_idx
        Jx = ti.Vector.zero(gs.ti_float, 3)
        for i in range(self.rigid_solver.n_dofs):
            Jx = Jx + self.Jt[i_p, i] * x[i_b, i]
        return Jx

    @ti.func
    def add_Jt_x(self, y, i_p, x):
        """
        y += J^T @ x for pair i_p.

        执行 y += J^T @ x（刚体接触向 DOF 回写）。
        """
        pairs = ti.static(self.contact_pairs)
        i_b = pairs[i_p].batch_idx
        for i in range(self.rigid_solver.n_dofs):
            y[i_b, i] += self.Jt[i_p, i].dot(x)


@ti.data_oriented
class RigidRigidContactHandler(RigidContactHandler):
    def __init__(
        self,
        simulator: "Simulator",
    ) -> None:
        super().__init__(simulator)

    @ti.func
    def compute_jacobian(self):
        """
        Assemble Jt for rigid-rigid contacts (two links).

        装配刚体-刚体接触的 Jt（两条链分别正负贡献）。
        """
        self.Jt.fill(0.0)
        pairs = ti.static(self.contact_pairs)
        for i_p in range(self.n_contact_pairs[None]):
            i_b = pairs[i_p].batch_idx
            link = pairs[i_p].link_idx0
            while link > -1:
                link_maybe_batch = [link, i_b] if ti.static(self.rigid_solver._options.batch_links_info) else link
                # 逆序确保每行相关 DOF 严格降序
                for i_d_ in range(self.rigid_solver.links_info.n_dofs[link_maybe_batch]):
                    i_d = self.rigid_solver.links_info.dof_end[link_maybe_batch] - 1 - i_d_

                    cdof_ang = self.rigid_solver.dofs_state.cdof_ang[i_d, i_b]
                    cdof_vel = self.rigid_solver.dofs_state.cdof_vel[i_d, i_b]

                    t_quat = gu.ti_identity_quat()
                    t_pos = pairs[i_p].contact_pos - self.rigid_solver.links_state.root_COM[link, i_b]
                    _, vel = gu.ti_transform_motion_by_trans_quat(cdof_ang, cdof_vel, t_pos, t_quat)

                    self.Jt[i_p, i_d] = self.Jt[i_p, i_d] + vel
                link = self.rigid_solver.links_info.parent_idx[link_maybe_batch]
            link = pairs[i_p].link_idx1
            while link > -1:
                link_maybe_batch = [link, i_b] if ti.static(self.rigid_solver._options.batch_links_info) else link
                # 逆序确保每行相关 DOF 严格降序
                for i_d_ in range(self.rigid_solver.links_info.n_dofs[link_maybe_batch]):
                    i_d = self.rigid_solver.links_info.dof_end[link_maybe_batch] - 1 - i_d_

                    cdof_ang = self.rigid_solver.dofs_state.cdof_ang[i_d, i_b]
                    cdof_vel = self.rigid_solver.dofs_state.cdof_vel[i_d, i_b]

                    t_quat = gu.ti_identity_quat()
                    t_pos = pairs[i_p].contact_pos - self.rigid_solver.links_state.root_COM[link, i_b]
                    _, vel = gu.ti_transform_motion_by_trans_quat(cdof_ang, cdof_vel, t_pos, t_quat)

                    self.Jt[i_p, i_d] = self.Jt[i_p, i_d] - vel
                link = self.rigid_solver.links_info.parent_idx[link_maybe_batch]

    @ti.func
    def compute_delassus(self, i_p):
        """
        Transform W to contact frame.

        将世界系 W 投影到接触坐标系（t0/t1/n）下。
        """
        pairs = ti.static(self.contact_pairs)
        world = ti.Matrix.cols([pairs[i_p].tangent0, pairs[i_p].tangent1, pairs[i_p].normal])
        return world.transpose() @ self.W[i_p] @ world

    @ti.func
    def compute_Jx(self, i_p, x):
        """
        Compute J @ x in contact frame.

        计算接触系下的 J @ x。
        """
        pairs = ti.static(self.contact_pairs)
        i_b = pairs[i_p].batch_idx
        Jx = ti.Vector.zero(gs.ti_float, 3)
        for i in range(self.rigid_solver.n_dofs):
            Jx = Jx + self.Jt[i_p, i] * x[i_b, i]
        Jx = ti.Vector([Jx.dot(pairs[i_p].tangent0), Jx.dot(pairs[i_p].tangent1), Jx.dot(pairs[i_p].normal)])
        return Jx

    @ti.func
    def add_Jt_x(self, y, i_p, x):
        """
        y += J^T @ x with x given in contact frame.

        当 x 在接触系中给出时，执行 y += J^T @ x（内部转换到世界系）。
        """
        pairs = ti.static(self.contact_pairs)
        i_b = pairs[i_p].batch_idx
        world = ti.Matrix.cols([pairs[i_p].tangent0, pairs[i_p].tangent1, pairs[i_p].normal])
        x_ = world @ x
        for i in range(self.rigid_solver.n_dofs):
            y[i_b, i] += self.Jt[i_p, i].dot(x_)


 
@ti.data_oriented
class FEMContactHandler(BaseContactHandler):
    def __init__(
        self,
        simulator: "Simulator",
    ) -> None:
        super().__init__(simulator)
        self.fem_solver = simulator.fem_solver

    @ti.func
    def compute_gradient_hessian_diag(self):
        """
        Accumulate FEM contact gradient/impulse and diag blocks.

        FEM 接触：累加梯度、冲量，并将 J^T G J 的对角块叠加到 PCG 对角中。
        """
        sap_info = ti.static(self.contact_pairs.sap_info)
        for i_p in range(self.n_contact_pairs[None]):
            vc = self.compute_Jx(i_p, self.coupler.fem_state_v.v)
            self.compute_contact_gamma_G(sap_info, i_p, vc)
            self.add_Jt_x(self.coupler.fem_state_v.gradient, i_p, -sap_info[i_p].gamma)
            self.add_Jt_x(self.coupler.fem_state_v.impulse, i_p, sap_info[i_p].gamma)
            self.add_Jt_A_J_diag3x3(self.coupler.pcg_fem_state_v.diag3x3, i_p, sap_info[i_p].G)

    @ti.func
    def prepare_search_direction_data(self):
        """
        Precompute dvc = J @ x for FEM line search.

        为线搜索预计算 FEM 接触的 dvc = J @ x。
        """
        sap_info = ti.static(self.contact_pairs.sap_info)
        for i_p in ti.ndrange(self.n_contact_pairs[None]):
            i_b = self.contact_pairs[i_p].batch_idx
            if self.coupler.batch_linesearch_active[i_b]:
                sap_info[i_p].dvc = self.compute_Jx(i_p, self.coupler.pcg_fem_state_v.x)

    @ti.func
    def compute_pcg_matrix_vector_product(self):
        """
        Accumulate Ap = J^T G J p for FEM contacts.

        FEM 接触：计算并累加 Ap = J^T G J p。
        """
        sap_info = ti.static(self.contact_pairs.sap_info)
        for i_p in range(self.n_contact_pairs[None]):
            # Jt @ G @ J @ p
            x = self.compute_Jx(i_p, self.coupler.pcg_fem_state_v.p)
            x = sap_info[i_p].G @ x
            self.add_Jt_x(self.coupler.pcg_fem_state_v.Ap, i_p, x)

    @ti.func
    def compute_contact_velocity(self, i_p):
        """
        Compute the contact velocity in the contact frame.

        计算接触坐标系下的接触速度（t0、t1、n 分量）。
        """
        return self.compute_Jx(i_p, self.coupler.fem_state_v.v)


@ti.data_oriented
class RigidFEMContactHandler(RigidContactHandler):
    def __init__(
        self,
        simulator: "Simulator",
    ) -> None:
        super().__init__(simulator)
        self.fem_solver = simulator.fem_solver

    @ti.func
    def compute_gradient_hessian_diag(self):
        """
        Accumulate gradients/impulses for coupled rigid-FEM contact, and FEM diag.

        对刚体-FEM 耦合接触累加梯度/冲量，同时将 FEM 侧的对角叠加到 PCG 对角。
        """
        sap_info = ti.static(self.contact_pairs.sap_info)
        for i_p in range(self.n_contact_pairs[None]):
            vc = self.compute_Jx(i_p, self.coupler.fem_state_v.v, self.coupler.rigid_state_dof.v)
            self.compute_contact_gamma_G(sap_info, i_p, vc)
            self.add_Jt_x(
                self.coupler.fem_state_v.gradient, self.coupler.rigid_state_dof.gradient, i_p, -sap_info[i_p].gamma
            )
            self.add_Jt_x(
                self.coupler.fem_state_v.impulse, self.coupler.rigid_state_dof.impulse, i_p, sap_info[i_p].gamma
            )
            self.add_Jt_A_J_diag3x3(self.coupler.pcg_fem_state_v.diag3x3, i_p, sap_info[i_p].G)

    @ti.func
    def prepare_search_direction_data(self):
        """
        Precompute dvc = J @ x (FEM+Rigid) for line search.

        为线搜索预计算 dvc = J @ x（同时依赖 FEM 与 刚体的 x）。
        """
        sap_info = ti.static(self.contact_pairs.sap_info)
        for i_p in ti.ndrange(self.n_contact_pairs[None]):
            i_b = self.contact_pairs[i_p].batch_idx
            if self.coupler.batch_linesearch_active[i_b]:
                sap_info[i_p].dvc = self.compute_Jx(
                    i_p, self.coupler.pcg_fem_state_v.x, self.coupler.pcg_rigid_state_dof.x
                )

    @ti.func
    def compute_pcg_matrix_vector_product(self):
        """
        Accumulate Ap = J^T G J p for rigid-FEM contacts.

        刚体-FEM 接触：计算并累加 Ap = J^T G J p。
        """
        sap_info = ti.static(self.contact_pairs.sap_info)
        for i_p in range(self.n_contact_pairs[None]):
            # Jt @ G @ J @ p
            x = self.compute_Jx(i_p, self.coupler.pcg_fem_state_v.p, self.coupler.pcg_rigid_state_dof.p)
            x = sap_info[i_p].G @ x
            self.add_Jt_x(self.coupler.pcg_fem_state_v.Ap, self.coupler.pcg_rigid_state_dof.Ap, i_p, x)

    @ti.func
    def compute_contact_velocity(self, i_p):
        """
        Compute the contact velocity in the contact frame.

        计算接触坐标系下的接触速度（FEM+刚体贡献）。
        """
        return self.compute_Jx(i_p, self.coupler.fem_state_v.v, self.coupler.rigid_state_dof.v)


@ti.func
def accumulate_area_centroid(
    polygon_vertices, i, total_area: ti.template(), total_area_weighted_centroid: ti.template()
):
    e1 = polygon_vertices[:, i - 1] - polygon_vertices[:, 0]
    e2 = polygon_vertices[:, i] - polygon_vertices[:, 0]
    area = 0.5 * e1.cross(e2).norm()
    total_area += area
    total_area_weighted_centroid += (
        area * (polygon_vertices[:, 0] + polygon_vertices[:, i - 1] + polygon_vertices[:, i]) / 3.0
    )

@ti.func
def accumulate_area_centroid(
    polygon_vertices, i, total_area: ti.template(), total_area_weighted_centroid: ti.template()
):
    """
    Accumulate triangle area and centroid contribution for polygon fan.

    用于多边形扇形累加：叠加第 i 个三角的面积与质心贡献。
    """
    e1 = polygon_vertices[:, i - 1] - polygon_vertices[:, 0]
    e2 = polygon_vertices[:, i] - polygon_vertices[:, 0]
    area = 0.5 * e1.cross(e2).norm()
    total_area += area
    total_area_weighted_centroid += (
        area * (polygon_vertices[:, 0] + polygon_vertices[:, i - 1] + polygon_vertices[:, i]) / 3.0
    )


@ti.data_oriented
class FEMFloorTetContactHandler(FEMContactHandler):
    """
    Class for handling contact between a tetrahedral mesh and a floor in a simulation using hydroelastic model.

    This class extends the BaseContact class and provides methods for detecting contact
    between the tetrahedral elements and the floor, computing contact pairs, and managing
    contact-related computations.

    基于柔顺（压力场）模型的 FEM 四面体网格与地面接触处理器。执行平面截四面体求交、接触点/面积/压力插值等。
    """

    def __init__(
        self,
        simulator: "Simulator",
        eps: float = 1e-10,
    ) -> None:
        super().__init__(simulator)
        self.name = "FEMFloorTetContactHandler"
        self.fem_solver = self.sim.fem_solver
        self.eps = eps
        self.eps = eps
        self.contact_candidate_type = ti.types.struct(
            batch_idx=gs.ti_int,        # batch 索引
            geom_idx=gs.ti_int,         # FEM 元素索引
            intersection_code=gs.ti_int,# 截交编码（Marching Tets）
            distance=gs.ti_vec4,        # 顶点到平面的有符号距离
        )
        self.n_contact_candidates = ti.field(gs.ti_int, shape=())
        self.max_contact_candidates = self.fem_solver.n_surface_elements * self.fem_solver._B
        self.contact_candidates = self.contact_candidate_type.field(shape=(self.max_contact_candidates,))

        self.contact_pair_type = ti.types.struct(
            batch_idx=gs.ti_int,     # batch 索引
            geom_idx=gs.ti_int,      # FEM 元素索引
            barycentric=gs.ti_vec4,  # 接触点四面体重心坐标
            contact_pos=gs.ti_vec3,  # 接触位置
            sap_info=self.sap_contact_info_type,  # 接触 SAP 信息
        )
        self.max_contact_pairs = self.fem_solver.n_surface_elements * self.fem_solver._B
        self.contact_pairs = self.contact_pair_type.field(shape=(self.max_contact_pairs,))

    @ti.func
    def detection(self, f: ti.i32):
        """
        Detect tet-floor intersections and generate contact pairs.

        检测四面体与地面平面的截交，并生成接触对（质心/压力/刚度）。
        """
        overflow = False
        # 统计候选接触（目前遍历所有元素；可优化为仅表面元素）
        self.n_contact_candidates[None] = 0
        # TODO 可仅检查表面元素以加速
        for i_b, i_e in ti.ndrange(self.fem_solver._B, self.fem_solver.n_elements):
            intersection_code = ti.int32(0)
            distance = ti.Vector.zero(gs.ti_float, 4)
            for i in ti.static(range(4)):
                i_v = self.fem_solver.elements_i[i_e].el2v[i]
                pos_v = self.fem_solver.elements_v[f, i_v, i_b].pos
                distance[i] = pos_v.z - self.fem_solver.floor_height
                if distance[i] > 0.0:
                    intersection_code |= 1 << i

            # 元素被平面穿过：编码既非全内也非全外
            if intersection_code != 0 and intersection_code != 15:
                i_c = ti.atomic_add(self.n_contact_candidates[None], 1)
                if i_c < self.max_contact_candidates:
                    self.contact_candidates[i_c].batch_idx = i_b
                    self.contact_candidates[i_c].geom_idx = i_e
                    self.contact_candidates[i_c].intersection_code = intersection_code
                    self.contact_candidates[i_c].distance = distance
                else:
                    overflow = True

        sap_info = ti.static(self.contact_pairs.sap_info)
        self.n_contact_pairs[None] = 0
        # 从候选生成接触对
        result_count = ti.min(self.n_contact_candidates[None], self.max_contact_candidates)
        for i_c in range(result_count):
            candidate = self.contact_candidates[i_c]
            i_b = candidate.batch_idx
            i_e = candidate.geom_idx
            intersection_code = candidate.intersection_code
            intersected_edges = self.coupler.MarchingTetsEdgeTable[intersection_code]

            tet_vertices = ti.Matrix.zero(gs.ti_float, 3, 4)
            tet_pressures = ti.Vector.zero(gs.ti_float, 4)
            for i in ti.static(range(4)):
                i_v = self.fem_solver.elements_i[i_e].el2v[i]
                tet_vertices[:, i] = self.fem_solver.elements_v[f, i_v, i_b].pos
                tet_pressures[i] = self.coupler.fem_pressure[i_v]

            polygon_vertices = ti.Matrix.zero(gs.ti_float, 3, 4)  # 截多边形 3~4 顶点
            total_area = gs.EPS
            total_area_weighted_centroid = ti.Vector.zero(gs.ti_float, 3)
            for i in ti.static(range(4)):
                if intersected_edges[i] >= 0:
                    edge = self.coupler.TetEdges[intersected_edges[i]]
                    pos_v0 = tet_vertices[:, edge[0]]
                    pos_v1 = tet_vertices[:, edge[1]]
                    d_v0 = candidate.distance[edge[0]]
                    d_v1 = candidate.distance[edge[1]]
                    t = d_v0 / (d_v0 - d_v1)
                    polygon_vertices[:, i] = pos_v0 + t * (pos_v1 - pos_v0)

                    # 累加三角面积与质心
                    if ti.static(i >= 2):
                        accumulate_area_centroid(polygon_vertices, i, total_area, total_area_weighted_centroid)

            centroid = total_area_weighted_centroid / total_area

            # 重心坐标与压力插值
            barycentric = tet_barycentric(centroid, tet_vertices)
            pressure = barycentric.dot(tet_pressures)

            deformable_g = self.coupler._hydroelastic_stiffness
            rigid_g = self.coupler.fem_pressure_gradient[i_b, i_e].z
            # 处理退化：面积过小或梯度过小时跳过
            if total_area < self.eps or rigid_g < self.eps:
                continue
            g = 1.0 / (1.0 / deformable_g + 1.0 / rigid_g)  # 调和平均
            rigid_k = total_area * g
            rigid_phi0 = -pressure / g
            if rigid_k < self.eps or rigid_phi0 > self.eps:
                continue
            i_p = ti.atomic_add(self.n_contact_pairs[None], 1)
            if i_p < self.max_contact_pairs:
                self.contact_pairs[i_p].batch_idx = i_b
                self.contact_pairs[i_p].geom_idx = i_e
                self.contact_pairs[i_p].barycentric = barycentric
                sap_info[i_p].k = rigid_k
                sap_info[i_p].phi0 = rigid_phi0
                sap_info[i_p].mu = self.fem_solver.elements_i[i_e].friction_mu
            else:
                overflow = True

        return overflow

    @ti.func
    def compute_Jx(self, i_p, x):
        """
        Compute the contact Jacobian J times a vector x.

        计算接触雅可比 J 与向量 x 的乘积（按四面体重心系数加权）。
        """
        i_b = self.contact_pairs[i_p].batch_idx
        i_g = self.contact_pairs[i_p].geom_idx
        Jx = ti.Vector.zero(gs.ti_float, 3)
        for i in ti.static(range(4)):
            i_v = self.fem_solver.elements_i[i_g].el2v[i]
            Jx += self.contact_pairs[i_p].barycentric[i] * x[i_b, i_v]
        return Jx

    @ti.func
    def add_Jt_x(self, y, i_p, x):
        """
        y += J^T @ x for FEM tet-floor.

        将接触冲量/梯度通过 J^T 回写到顶点（考虑顶点约束屏蔽）。
        """
        i_b = self.contact_pairs[i_p].batch_idx
        i_g = self.contact_pairs[i_p].geom_idx
        for i in ti.static(range(4)):
            i_v = self.fem_solver.elements_i[i_g].el2v[i]
            if ti.static(self.fem_solver._enable_vertex_constraints):
                if not self.fem_solver.vertex_constraints.is_constrained[i_v, i_b]:
                    y[i_b, i_v] += self.contact_pairs[i_p].barycentric[i] * x
            else:
                y[i_b, i_v] += self.contact_pairs[i_p].barycentric[i] * x

    @ti.func
    def add_Jt_A_J_diag3x3(self, y, i_p, A):
        """
        Accumulate diag contribution: J^T A J (3x3) to per-vertex diag.

        将 J^T A J（3x3 对角贡献）按重心系数平方叠加到顶点对角块。
        """
        i_b = self.contact_pairs[i_p].batch_idx
        i_g = self.contact_pairs[i_p].geom_idx
        for i in ti.static(range(4)):
            i_v = self.fem_solver.elements_i[i_g].el2v[i]
            if ti.static(self.fem_solver._enable_vertex_constraints):
                if not self.fem_solver.vertex_constraints.is_constrained[i_v, i_b]:
                    y[i_b, i_v] += self.contact_pairs[i_p].barycentric[i] ** 2 * A
            else:
                y[i_b, i_v] += self.contact_pairs[i_p].barycentric[i] ** 2 * A

    @ti.func
    def compute_delassus(self, i_p):
        """
        Compute W for tet-floor using FEM prec blocks.

        利用 FEM 预条件器近似 A^-1，计算 W = J A^-1 J^T。
        """
        dt2_inv = 1.0 / self.sim._substep_dt**2
        i_b = self.contact_pairs[i_p].batch_idx
        i_g = self.contact_pairs[i_p].geom_idx
        W = ti.Matrix.zero(gs.ti_float, 3, 3)
        for i in ti.static(range(4)):
            i_v = self.fem_solver.elements_i[i_g].el2v[i]
            W += self.contact_pairs[i_p].barycentric[i] ** 2 * dt2_inv * self.fem_solver.pcg_state_v[i_b, i_v].prec
        return W



@ti.data_oriented
class FEMSelfTetContactHandler(FEMContactHandler):
    """
    Class for handling self-contact between tetrahedral elements in a simulation using hydroelastic model.

    This class extends the FEMContact class and provides methods for detecting self-contact
    between tetrahedral elements, computing contact pairs, and managing contact-related computations.

    FEM 四面体自接触处理器（柔顺模型）。通过场等值面构造接触平面并执行多边形裁剪，生成接触对。
    """

    def __init__(
        self,
        simulator: "Simulator",
        eps: float = 1e-10,
    ) -> None:
        super().__init__(simulator)
        self.name = "FEMSelfTetContactHandler"
        self.eps = eps
        self.contact_candidate_type = ti.types.struct(
            batch_idx=gs.ti_int,        # batch 索引
            geom_idx0=gs.ti_int,        # 元素 0 索引
            intersection_code0=gs.ti_int,# 元素 0 截交编码
            geom_idx1=gs.ti_int,        # 元素 1 索引
            normal=gs.ti_vec3,          # 接触平面法向
            x=gs.ti_vec3,               # 接触平面上一点
            distance0=gs.ti_vec4,       # 元素 0 顶点到平面距离
        )
        self.n_contact_candidates = ti.field(gs.ti_int, shape=())
        self.max_contact_candidates = self.fem_solver.n_surface_elements * self.fem_solver._B * 8
        self.contact_candidates = self.contact_candidate_type.field(shape=(self.max_contact_candidates,))

        self.contact_pair_type = ti.types.struct(
            batch_idx=gs.ti_int,     # batch 索引
            normal=gs.ti_vec3,       # 接触法向
            tangent0=gs.ti_vec3,     # 切向 0
            tangent1=gs.ti_vec3,     # 切向 1
            geom_idx0=gs.ti_int,     # 元素 0
            geom_idx1=gs.ti_int,     # 元素 1
            barycentric0=gs.ti_vec4, # 元素 0 内的重心
            barycentric1=gs.ti_vec4, # 元素 1 内的重心
            contact_pos=gs.ti_vec3,  # 接触点
            sap_info=self.sap_contact_info_type,  # 接触 SAP 信息
        )
        self.max_contact_pairs = self.fem_solver.n_surface_elements * self.fem_solver._B
        self.contact_pairs = self.contact_pair_type.field(shape=(self.max_contact_pairs,))

    @ti.func
    def compute_candidates(self, f: ti.i32):
        """
        Generate candidate planes from intersected surface tets.

        基于表面四面体 AABB 配对生成候选接触平面，并记录截交编码与距离。
        """
        overflow = False
        self.n_contact_candidates[None] = 0
        result_count = ti.min(
            self.coupler.fem_surface_tet_bvh.query_result_count[None],
            self.coupler.fem_surface_tet_bvh.max_query_results,
        )
        for i_r in range(result_count):
            i_b, i_sa, i_sq = self.coupler.fem_surface_tet_bvh.query_result[i_r]
            i_a = self.fem_solver.surface_elements[i_sa]
            i_q = self.fem_solver.surface_elements[i_sq]
            i_v0 = self.fem_solver.elements_i[i_a].el2v[0]
            i_v1 = self.fem_solver.elements_i[i_q].el2v[0]
            x0 = self.fem_solver.elements_v[f, i_v0, i_b].pos
            x1 = self.fem_solver.elements_v[f, i_v1, i_b].pos
            p0 = self.coupler.fem_pressure[i_v0]
            p1 = self.coupler.fem_pressure[i_v1]
            g0 = self.coupler.fem_pressure_gradient[i_b, i_a]
            g1 = self.coupler.fem_pressure_gradient[i_b, i_q]
            g0_norm = g0.norm()
            g1_norm = g1.norm()
            if g0_norm < gs.EPS or g1_norm < gs.EPS:
                continue
            # 等压面（g0 - g1 为法向，x 满足 p0 + g0·(x-x0) = p1 + g1·(x-x1)）
            normal = g0 - g1
            magnitude = normal.norm()
            if magnitude < gs.EPS:
                continue
            normal /= magnitude
            b = p1 - p0 - g1.dot(x1) + g0.dot(x0)
            x = b / magnitude * normal
            # 法向需大致沿 g0、反向 g1（同 Drake 的容差）
            threshold = ti.static(np.cos(np.pi * 5.0 / 8.0))
            if normal.dot(g0) < threshold * g0_norm or normal.dot(g1) > -threshold * g1_norm:
                continue
            intersection_code0 = ti.int32(0)
            distance0 = ti.Vector.zero(gs.ti_float, 4)
            intersection_code1 = ti.int32(0)
            distance1 = ti.Vector.zero(gs.ti_float, 4)
            for i in ti.static(range(4)):
                i_v = self.fem_solver.elements_i[i_a].el2v[i]
                pos_v = self.fem_solver.elements_v[f, i_v, i_b].pos
                distance0[i] = (pos_v - x).dot(normal)
                if distance0[i] > 0.0:
                    intersection_code0 |= 1 << i
            for i in ti.static(range(4)):
                i_v = self.fem_solver.elements_i[i_q].el2v[i]
                pos_v = self.fem_solver.elements_v[f, i_v, i_b].pos
                distance1[i] = (pos_v - x).dot(normal)
                if distance1[i] > 0.0:
                    intersection_code1 |= 1 << i
            # 快速排除：需两端均被平面穿过
            if (
                intersection_code0 == 0
                or intersection_code1 == 0
                or intersection_code0 == 15
                or intersection_code1 == 15
            ):
                continue
            i_c = ti.atomic_add(self.n_contact_candidates[None], 1)
            if i_c < self.max_contact_candidates:
                self.contact_candidates[i_c].batch_idx = i_b
                self.contact_candidates[i_c].normal = normal
                self.contact_candidates[i_c].x = x
                self.contact_candidates[i_c].geom_idx0 = i_a
                self.contact_candidates[i_c].intersection_code0 = intersection_code0
                self.contact_candidates[i_c].distance0 = distance0
                self.contact_candidates[i_c].geom_idx1 = i_q
            else:
                overflow = True
        return overflow

    @ti.func
    def compute_pairs(self, i_step: ti.i32):
        """
        Computes the FEM self contact pairs and their properties.

        Intersection code reference:
        https://github.com/RobotLocomotion/drake/blob/8c3a249184ed09f0faab3c678536d66d732809ce/geometry/proximity/field_intersection.cc#L87

        根据候选平面裁剪截多边形，计算自接触对的接触点/法切/重心/刚度等。
        """
        overflow = False
        sap_info = ti.static(self.contact_pairs.sap_info)
        normal_signs = ti.Vector([1.0, -1.0, 1.0, -1.0], dt=gs.ti_float)  # 使法向指向外侧
        self.n_contact_pairs[None] = 0
        result_count = ti.min(self.n_contact_candidates[None], self.max_contact_candidates)
        for i_c in range(result_count):
            i_b = self.contact_candidates[i_c].batch_idx
            i_e0 = self.contact_candidates[i_c].geom_idx0
            i_e1 = self.contact_candidates[i_c].geom_idx1
            intersection_code0 = self.contact_candidates[i_c].intersection_code0
            distance0 = self.contact_candidates[i_c].distance0
            intersected_edges0 = self.coupler.MarchingTetsEdgeTable[intersection_code0]

            tet_vertices0 = ti.Matrix.zero(gs.ti_float, 3, 4)
            tet_pressures0 = ti.Vector.zero(gs.ti_float, 4)
            tet_vertices1 = ti.Matrix.zero(gs.ti_float, 3, 4)
            for i in ti.static(range(4)):
                i_v = self.fem_solver.elements_i[i_e0].el2v[i]
                tet_vertices0[:, i] = self.fem_solver.elements_v[i_step, i_v, i_b].pos
                tet_pressures0[i] = self.coupler.fem_pressure[i_v]
            for i in ti.static(range(4)):
                i_v = self.fem_solver.elements_i[i_e1].el2v[i]
                tet_vertices1[:, i] = self.fem_solver.elements_v[i_step, i_v, i_b].pos

            polygon_vertices = ti.Matrix.zero(gs.ti_float, 3, 8)
            polygon_n_vertices = gs.ti_int(0)
            clipped_vertices = ti.Matrix.zero(gs.ti_float, 3, 8)
            clipped_n_vertices = gs.ti_int(0)
            for i in range(4):
                if intersected_edges0[i] >= 0:
                    edge = self.coupler.TetEdges[intersected_edges0[i]]
                    pos_v0 = tet_vertices0[:, edge[0]]
                    pos_v1 = tet_vertices0[:, edge[1]]
                    d_v0 = distance0[edge[0]]
                    d_v1 = distance0[edge[1]]
                    t = d_v0 / (d_v0 - d_v1)
                    polygon_vertices[:, polygon_n_vertices] = pos_v0 + t * (pos_v1 - pos_v0)
                    polygon_n_vertices += 1
            # 与元素1的四个面半空间依次裁剪
            for face in range(4):
                clipped_n_vertices = 0
                x = tet_vertices1[:, (face + 1) % 4]
                normal = (tet_vertices1[:, (face + 2) % 4] - x).cross(
                    tet_vertices1[:, (face + 3) % 4] - x
                ) * normal_signs[face]
                normal /= normal.norm()

                distances = ti.Vector.zero(gs.ti_float, 8)
                for i in range(polygon_n_vertices):
                    distances[i] = (polygon_vertices[:, i] - x).dot(normal)

                for i in range(polygon_n_vertices):
                    j = (i + 1) % polygon_n_vertices
                    if distances[i] <= 0.0:
                        clipped_vertices[:, clipped_n_vertices] = polygon_vertices[:, i]
                        clipped_n_vertices += 1
                        if distances[j] > 0.0:
                            wa = distances[j] / (distances[j] - distances[i])
                            wb = 1.0 - wa
                            clipped_vertices[:, clipped_n_vertices] = (
                                wa * polygon_vertices[:, i] + wb * polygon_vertices[:, j]
                            )
                            clipped_n_vertices += 1
                    elif distances[j] <= 0.0:
                        wa = distances[j] / (distances[j] - distances[i])
                        wb = 1.0 - wa
                        clipped_vertices[:, clipped_n_vertices] = (
                            wa * polygon_vertices[:, i] + wb * polygon_vertices[:, j]
                        )
                        clipped_n_vertices += 1
                polygon_n_vertices = clipped_n_vertices
                polygon_vertices = clipped_vertices

                if polygon_n_vertices < 3:
                    # 少于 3 顶点，不构成有效接触多边形
                    break

            if polygon_n_vertices < 3:
                continue

            # 计算多边形面积与质心
            total_area = 0.0
            total_area_weighted_centroid = ti.Vector.zero(gs.ti_float, 3)
            for i in range(2, polygon_n_vertices):
                accumulate_area_centroid(polygon_vertices, i, total_area, total_area_weighted_centroid)

            if total_area < self.eps:
                continue
            centroid = total_area_weighted_centroid / total_area
            barycentric0 = tet_barycentric(centroid, tet_vertices0)
            barycentric1 = tet_barycentric(centroid, tet_vertices1)
            tangent0 = polygon_vertices[:, 0] - centroid
            tangent0 /= tangent0.norm()
            tangent1 = self.contact_candidates[i_c].normal.cross(tangent0)

            pressure = barycentric0.dot(tet_pressures0)
            g0 = self.coupler.fem_pressure_gradient[i_b, i_e0].dot(self.contact_candidates[i_c].normal)
            g1 = -self.coupler.fem_pressure_gradient[i_b, i_e1].dot(self.contact_candidates[i_c].normal)
            # 近似距离（与 Drake 略有不同）
            deformable_phi0 = -pressure / g0 - pressure / g1

            if deformable_phi0 > gs.EPS:
                continue

            i_p = ti.atomic_add(self.n_contact_pairs[None], 1)
            if i_p < self.max_contact_pairs:
                self.contact_pairs[i_p].batch_idx = i_b
                self.contact_pairs[i_p].normal = self.contact_candidates[i_c].normal
                self.contact_pairs[i_p].tangent0 = tangent0
                self.contact_pairs[i_p].tangent1 = tangent1
                self.contact_pairs[i_p].geom_idx0 = i_e0
                self.contact_pairs[i_p].geom_idx1 = i_e1
                self.contact_pairs[i_p].barycentric0 = barycentric0
                self.contact_pairs[i_p].barycentric1 = barycentric1

                deformable_g = self.coupler._hydroelastic_stiffness
                deformable_k = total_area * deformable_g
                sap_info[i_p].k = deformable_k
                sap_info[i_p].phi0 = deformable_phi0
                sap_info[i_p].mu = ti.sqrt(
                    self.fem_solver.elements_i[i_e0].friction_mu * self.fem_solver.elements_i[i_e1].friction_mu
                )
            else:
                overflow = True
        return overflow

    @ti.func
    def detection(self, f: ti.i32):
        """
        Run BVH query, generate candidates and pairs.

        执行 BVH 查询，生成候选与最终接触对。
        """
        overflow = False
        overflow |= self.coupler.fem_surface_tet_bvh.query(self.coupler.fem_surface_tet_aabb.aabbs)
        overflow |= self.compute_candidates(f)
        overflow |= self.compute_pairs(f)
        return overflow

    @ti.func
    def compute_Jx(self, i_p, x):
        """
        Compute the contact Jacobian J times a vector x.

        计算接触雅可比 J @ x，并投影到接触系（t0/t1/n）。
        """
        i_b = self.contact_pairs[i_p].batch_idx
        i_g0 = self.contact_pairs[i_p].geom_idx0
        i_g1 = self.contact_pairs[i_p].geom_idx1
        Jx = ti.Vector.zero(gs.ti_float, 3)
        for i in ti.static(range(4)):
            i_v = self.fem_solver.elements_i[i_g0].el2v[i]
            Jx += self.contact_pairs[i_p].barycentric0[i] * x[i_b, i_v]
        for i in ti.static(range(4)):
            i_v = self.fem_solver.elements_i[i_g1].el2v[i]
            Jx -= self.contact_pairs[i_p].barycentric1[i] * x[i_b, i_v]
        return ti.Vector(
            [
                Jx.dot(self.contact_pairs[i_p].tangent0),
                Jx.dot(self.contact_pairs[i_p].tangent1),
                Jx.dot(self.contact_pairs[i_p].normal),
            ]
        )

    @ti.func
    def add_Jt_x(self, y, i_p, x):
        """
        y += J^T @ x with x in contact frame.

        当 x 在接触系中给出时，执行 y += J^T @ x（转换到世界系后回写顶点）。
        """
        i_b = self.contact_pairs[i_p].batch_idx
        i_g0 = self.contact_pairs[i_p].geom_idx0
        i_g1 = self.contact_pairs[i_p].geom_idx1
        world = ti.Matrix.cols(
            [self.contact_pairs[i_p].tangent0, self.contact_pairs[i_p].tangent1, self.contact_pairs[i_p].normal]
        )
        x_ = world @ x
        for i in ti.static(range(4)):
            i_v = self.fem_solver.elements_i[i_g0].el2v[i]
            if ti.static(self.fem_solver._enable_vertex_constraints):
                if not self.fem_solver.vertex_constraints.is_constrained[i_v, i_b]:
                    y[i_b, i_v] += self.contact_pairs[i_p].barycentric0[i] * x_
            else:
                y[i_b, i_v] += self.contact_pairs[i_p].barycentric0[i] * x_
        for i in ti.static(range(4)):
            i_v = self.fem_solver.elements_i[i_g1].el2v[i]
            if ti.static(self.fem_solver._enable_vertex_constraints):
                if not self.fem_solver.vertex_constraints.is_constrained[i_v, i_b]:
                    y[i_b, i_v] -= self.contact_pairs[i_p].barycentric1[i] * x_
            else:
                y[i_b, i_v] -= self.contact_pairs[i_p].barycentric1[i] * x_

    @ti.func
    def add_Jt_A_J_diag3x3(self, y, i_p, A):
        """
        Accumulate diag J^T A J in world frame for both tets.

        对两个四面体，将世界系的 J^T A J 对角贡献叠加到对应顶点。
        """
        i_b = self.contact_pairs[i_p].batch_idx
        i_g0 = self.contact_pairs[i_p].geom_idx0
        i_g1 = self.contact_pairs[i_p].geom_idx1
        world = ti.Matrix.cols(
            [self.contact_pairs[i_p].tangent0, self.contact_pairs[i_p].tangent1, self.contact_pairs[i_p].normal]
        )
        B_ = world @ A @ world.transpose()
        for i in ti.static(range(4)):
            i_v = self.fem_solver.elements_i[i_g0].el2v[i]
            if ti.static(self.fem_solver._enable_vertex_constraints):
                if not self.fem_solver.vertex_constraints.is_constrained[i_v, i_b]:
                    y[i_b, i_v] += self.contact_pairs[i_p].barycentric0[i] ** 2 * B_
            else:
                y[i_b, i_v] += self.contact_pairs[i_p].barycentric0[i] ** 2 * B_
        for i in ti.static(range(4)):
            i_v = self.fem_solver.elements_i[i_g1].el2v[i]
            if ti.static(self.fem_solver._enable_vertex_constraints):
                if not self.fem_solver.vertex_constraints.is_constrained[i_v, i_b]:
                    y[i_b, i_v] += self.contact_pairs[i_p].barycentric1[i] ** 2 * B_
            else:
                y[i_b, i_v] += self.contact_pairs[i_p].barycentric1[i] ** 2 * B_

    @ti.func
    def compute_delassus(self, i_p):
        """
        Compute W in contact frame using FEM prec.

        利用 FEM 预条件器在接触系下计算 W。
        """
        dt2_inv = 1.0 / self.sim._substep_dt**2
        i_b = self.contact_pairs[i_p].batch_idx
        i_g0 = self.contact_pairs[i_p].geom_idx0
        i_g1 = self.contact_pairs[i_p].geom_idx1
        world = ti.Matrix.cols(
            [self.contact_pairs[i_p].tangent0, self.contact_pairs[i_p].tangent1, self.contact_pairs[i_p].normal]
        )
        W = ti.Matrix.zero(gs.ti_float, 3, 3)
        for i in ti.static(range(4)):
            i_v = self.fem_solver.elements_i[i_g0].el2v[i]
            W += self.contact_pairs[i_p].barycentric0[i] ** 2 * dt2_inv * self.fem_solver.pcg_state_v[i_b, i_v].prec
        for i in ti.static(range(4)):
            i_v = self.fem_solver.elements_i[i_g1].el2v[i]
            W += self.contact_pairs[i_p].barycentric1[i] ** 2 * dt2_inv * self.fem_solver.pcg_state_v[i_b, i_v].prec
        W = world.transpose() @ W @ world
        return W

@ti.data_oriented
class FEMFloorVertContactHandler(FEMContactHandler):
    """
    Class for handling contact between tetrahedral elements and a floor in a simulation using point contact model.

    This class extends the FEMContact class and provides methods for detecting contact
    between the tetrahedral elements and the floor, computing contact pairs, and managing
    contact-related computations.

    基于点接触模型的 FEM-地面接触处理器。直接对表面顶点与平面做距离测试并生成接触对。
    """

    def __init__(
        self,
        simulator: "Simulator",
    ) -> None:
        super().__init__(simulator)
        self.name = "FEMFloorVertContactHandler"
        self.fem_solver = self.sim.fem_solver

        self.contact_pair_type = ti.types.struct(
            batch_idx=gs.ti_int,     # batch 索引
            geom_idx=gs.ti_int,      # 顶点索引
            contact_pos=gs.ti_vec3,  # 接触位置
            sap_info=self.sap_contact_info_type,  # 接触 SAP 信息
        )
        self.max_contact_pairs = self.fem_solver.n_surface_elements * self.fem_solver._B
        self.contact_pairs = self.contact_pair_type.field(shape=(self.max_contact_pairs,))

    @ti.func
    def detection(self, f: ti.i32):
        """
        Detect point-floor contacts for surface vertices.

        检测表面顶点与地面的点接触，生成接触对与参数。
        """
        overflow = False
        sap_info = ti.static(self.contact_pairs.sap_info)
        # 统计接触对
        self.n_contact_pairs[None] = 0
        for i_b, i_sv in ti.ndrange(self.fem_solver._B, self.fem_solver.n_surface_vertices):
            i_v = self.fem_solver.surface_vertices[i_sv]
            pos_v = self.fem_solver.elements_v[f, i_v, i_b].pos
            distance = pos_v.z - self.fem_solver.floor_height
            if distance > 0.0:
                continue
            i_p = ti.atomic_add(self.n_contact_pairs[None], 1)
            if i_p < self.max_contact_pairs:
                self.contact_pairs[i_p].batch_idx = i_b
                self.contact_pairs[i_p].geom_idx = i_v
                sap_info[i_p].k = self.coupler._point_contact_stiffness * self.fem_solver.surface_vert_mass[i_v]
                sap_info[i_p].phi0 = distance
                sap_info[i_p].mu = self.fem_solver.elements_v_info[i_v].friction_mu
            else:
                overflow = True
        return overflow

    @ti.func
    def compute_Jx(self, i_p, x):
        """
        Compute the contact Jacobian J times a vector x.

        点接触下 J 为单位映射（直接取该顶点的 x）。
        """
        i_b = self.contact_pairs[i_p].batch_idx
        i_g = self.contact_pairs[i_p].geom_idx
        Jx = x[i_b, i_g]
        return Jx

    @ti.func
    def add_Jt_x(self, y, i_p, x):
        """
        y += J^T @ x for point contact.

        点接触的 J^T 回写（考虑顶点约束屏蔽）。
        """
        i_b = self.contact_pairs[i_p].batch_idx
        i_g = self.contact_pairs[i_p].geom_idx
        if ti.static(self.fem_solver._enable_vertex_constraints):
            if not self.fem_solver.vertex_constraints.is_constrained[i_g, i_b]:
                y[i_b, i_g] += x
        else:
            y[i_b, i_g] += x

    @ti.func
    def add_Jt_A_J_diag3x3(self, y, i_p, A):
        """
        Accumulate diag J^T A J for point contact.

        点接触：将 A 直接叠加到该顶点的 3x3 对角块（考虑约束屏蔽）。
        """
        i_b = self.contact_pairs[i_p].batch_idx
        i_g = self.contact_pairs[i_p].geom_idx
        if ti.static(self.fem_solver._enable_vertex_constraints):
            if not self.fem_solver.vertex_constraints.is_constrained[i_g, i_b]:
                y[i_b, i_g] += A
        else:
            y[i_b, i_g] += A

    @ti.func
    def compute_delassus(self, i_p):
        """
        Compute W using FEM prec for point contact.

        点接触下的 W，直接取 FEM 预条件器并按 dt^2 缩放。
        """
        dt2_inv = 1.0 / self.sim._substep_dt**2
        i_b = self.contact_pairs[i_p].batch_idx
        i_g = self.contact_pairs[i_p].geom_idx
        W = self.fem_solver.pcg_state_v[i_b, i_g].prec * dt2_inv
        return W

@ti.data_oriented
class RigidFemTriTetContactHandler(RigidFEMContactHandler):
    """
    Class for handling self-contact between tetrahedral elements in a simulation using hydroelastic model.

    This class extends the FEMContact class and provides methods for detecting self-contact
    between tetrahedral elements, computing contact pairs, and managing contact-related computations.

    刚体三角形与 FEM 四面体的混合接触处理器（Hydroelastic 模型）。
    - 通过刚体三角形与 FEM 表面四面体的几何裁剪，构造接触多边形与接触点；
    - 插值 FEM 顶点压力场，结合法向上的梯度，计算等效刚度 k 与穿透距离 phi0；
    - 装配刚体与 FEM 的雅可比，支持 SAP 管线的能量/梯度/Hessian 及线搜索。
    """

    def __init__(
        self,
        simulator: "Simulator",
        eps: float = 1e-10,
    ) -> None:
        super().__init__(simulator)
        self.name = "RigidFemTriTetContactHandler"
        self.eps = eps
        self.contact_candidate_type = ti.types.struct(
            batch_idx=gs.ti_int,  # batch index
            # batch 索引
            geom_idx0=gs.ti_int,  # index of the FEM element
            # FEM 元素（四面体）索引
            geom_idx1=gs.ti_int,  # index of the Rigid Triangle
            # 刚体三角形索引
            vert_idx1=gs.ti_ivec3,  # vertex indices of the rigid triangle
            # 刚体三角形的三顶点全局索引
            normal=gs.ti_vec3,  # contact plane normal
            # 接触平面法向（由三角形确定）
            x=gs.ti_vec3,  # a point on the contact plane
            # 接触平面上一点（取三角形上任一点）
        )
        self.n_contact_candidates = ti.field(gs.ti_int, shape=())
        self.max_contact_candidates = (
            max(self.fem_solver.n_surface_elements, self.rigid_solver.n_faces) * self.fem_solver._B * 8
        )
        self.contact_candidates = self.contact_candidate_type.field(shape=(self.max_contact_candidates,))
        self.contact_pair_type = ti.types.struct(
            batch_idx=gs.ti_int,  # batch index
            # batch 索引
            normal=gs.ti_vec3,  # contact plane normal
            # 接触法向（世界系）
            tangent0=gs.ti_vec3,  # contact plane tangent0
            # 接触切向 0（世界系）
            tangent1=gs.ti_vec3,  # contact plane tangent1
            # 接触切向 1（世界系）
            geom_idx0=gs.ti_int,  # index of the FEM element
            # FEM 元素（四面体）索引
            barycentric0=gs.ti_vec4,  # barycentric coordinates of the contact point in tet
            # 接触点在四面体中的重心坐标
            link_idx=gs.ti_int,  # index of the link
            # 刚体所在的 link 索引
            contact_pos=gs.ti_vec3,  # contact position
            # 接触点位置（世界系）
            sap_info=self.sap_contact_info_type,  # contact info
            # SAP 接触信息（刚度、phi0、摩擦等）
        )
        self.max_contact_pairs = max(self.fem_solver.n_surface_elements, self.rigid_solver.n_faces) * self.fem_solver._B
        self.contact_pairs = self.contact_pair_type.field(shape=(self.max_contact_pairs,))
        self.Jt = ti.field(gs.ti_vec3, shape=(self.max_contact_pairs, self.rigid_solver.n_dofs))
        self.M_inv_Jt = ti.field(gs.ti_vec3, shape=(self.max_contact_pairs, self.rigid_solver.n_dofs))
        self.W = ti.field(gs.ti_mat3, shape=(self.max_contact_pairs,))

    @ti.func
    def compute_candidates(self, f: ti.i32):
        """
        Generate triangle–tet intersection candidates via BVH broad-phase and plane-side tests.

        通过 BVH 宽相配对与平面侧判定，生成三角形-四面体的候选相交对：
        - 取刚体三角形的平面法向与顶点；
        - 要求法向与 FEM 元素的压力梯度大致同向（防止法向退化）；
        - 根据四面体四顶点相对于三角形平面的符号距离构造截交编码。
        """
        self.n_contact_candidates[None] = 0
        overflow = False
        result_count = ti.min(
            self.coupler.rigid_tri_bvh.query_result_count[None], self.coupler.rigid_tri_bvh.max_query_results
        )
        for i_r in range(result_count):
            i_b, i_a, i_sq = self.coupler.rigid_tri_bvh.query_result[i_r]
            i_q = self.fem_solver.surface_elements[i_sq]

            vert_idx1 = ti.Vector.zero(gs.ti_int, 3)
            tri_vertices = ti.Matrix.zero(gs.ti_float, 3, 3)
            for i in ti.static(range(3)):
                i_v = self.rigid_solver.faces_info.verts_idx[i_a][i]
                i_fv = self.rigid_solver.verts_info.verts_state_idx[i_v]
                if self.rigid_solver.verts_info.is_fixed[i_v]:
                    tri_vertices[:, i] = self.rigid_solver.fixed_verts_state.pos[i_fv]
                else:
                    tri_vertices[:, i] = self.rigid_solver.free_verts_state.pos[i_fv, i_b]
                vert_idx1[i] = i_v
            pos_v0, pos_v1, pos_v2 = tri_vertices[:, 0], tri_vertices[:, 1], tri_vertices[:, 2]

            normal = (pos_v1 - pos_v0).cross(pos_v2 - pos_v0)
            magnitude_sqr = normal.norm_sqr()
            if magnitude_sqr < gs.EPS:
                # 退化三角形，跳过
                continue
            normal *= ti.rsqrt(magnitude_sqr)
            g0 = self.coupler.fem_pressure_gradient[i_b, i_q]
            if g0.dot(normal) < gs.EPS:
                # 法向与压力梯度近乎垂直，接触法向不可靠，跳过
                continue

            intersection_code = ti.int32(0)
            for i in ti.static(range(4)):
                i_v = self.fem_solver.elements_i[i_q].el2v[i]
                pos_v = self.fem_solver.elements_v[f, i_v, i_b].pos
                distance = (pos_v - pos_v0).dot(normal)  # signed distance
                # 四面体顶点到三角形平面的有符号距离
                if distance > 0.0:
                    intersection_code |= 1 << i
            if intersection_code == 0 or intersection_code == 15:
                # 全在平面同侧或完全相离，跳过
                continue

            i_c = ti.atomic_add(self.n_contact_candidates[None], 1)
            if i_c < self.max_contact_candidates:
                self.contact_candidates[i_c].batch_idx = i_b
                self.contact_candidates[i_c].normal = normal
                self.contact_candidates[i_c].x = pos_v0
                self.contact_candidates[i_c].geom_idx0 = i_q
                self.contact_candidates[i_c].geom_idx1 = i_a
                self.contact_candidates[i_c].vert_idx1 = vert_idx1
            else:
                overflow = True
        return overflow

    @ti.func
    def compute_pairs(self, f: ti.i32):
        """
        Computes the tet triangle intersection pair and their properties.

        Intersection code reference:
        https://github.com/RobotLocomotion/drake/blob/49ab120ec6f5981484918daa821fc7101e10ebc6/geometry/proximity/mesh_intersection.cc

        基于候选对，执行“三角形被四面体四个半空间依次裁剪”，得到接触多边形并计算：
        - 面积与质心；
        - 质心在四面体中的重心坐标；
        - 接触切向与法向基；
        - 由 FEM 压力场与梯度估计等效刚度 k 与 phi0。
        """
        sap_info = ti.static(self.contact_pairs.sap_info)
        overflow = False
        normal_signs = ti.Vector([1.0, -1.0, 1.0, -1.0])  # make normal point outward
        # 按四面体面序修正法向指向性，使其朝外
        self.n_contact_pairs[None] = 0
        result_count = ti.min(self.n_contact_candidates[None], self.max_contact_candidates)
        for i_c in range(result_count):
            i_b = self.contact_candidates[i_c].batch_idx
            i_e = self.contact_candidates[i_c].geom_idx0
            i_f = self.contact_candidates[i_c].geom_idx1

            tri_vertices = ti.Matrix.zero(gs.ti_float, 3, 3)  # 3 vertices of the triangle
            # 三角形三个顶点
            tet_vertices = ti.Matrix.zero(gs.ti_float, 3, 4)  # 4 vertices of tet 0
            # 四面体四个顶点
            tet_pressures = ti.Vector.zero(gs.ti_float, 4)  # pressures at the vertices of tet 0
            # 四面体顶点压力（静态场）
            for i in ti.static(range(3)):
                i_v = self.contact_candidates[i_c].vert_idx1[i]
                i_fv = self.rigid_solver.verts_info.verts_state_idx[i_v]
                if self.rigid_solver.verts_info.is_fixed[i_v]:
                    tri_vertices[:, i] = self.rigid_solver.fixed_verts_state.pos[i_fv]
                else:
                    tri_vertices[:, i] = self.rigid_solver.free_verts_state.pos[i_fv, i_b]
            for i in ti.static(range(4)):
                i_v = self.fem_solver.elements_i[i_e].el2v[i]
                tet_vertices[:, i] = self.fem_solver.elements_v[f, i_v, i_b].pos
                tet_pressures[i] = self.coupler.fem_pressure[i_v]

            polygon_vertices = ti.Matrix.zero(gs.ti_float, 3, 7)  # maximum 7 vertices
            # 裁剪后的多边形最多 7 个顶点
            polygon_n_vertices = 3
            for i in ti.static(range(3)):
                polygon_vertices[:, i] = tri_vertices[:, i]
            clipped_vertices = ti.Matrix.zero(gs.ti_float, 3, 7)  # maximum 7 vertices
            clipped_n_vertices = 0
            distances = ti.Vector.zero(gs.ti_float, 7)
            # Sutherland–Hodgman 多边形裁剪，依次与四面体四个面的半空间相交
            for face in range(4):
                clipped_n_vertices = 0
                x = tet_vertices[:, (face + 1) % 4]
                normal = (tet_vertices[:, (face + 2) % 4] - x).cross(
                    tet_vertices[:, (face + 3) % 4] - x
                ) * normal_signs[face]
                normal /= normal.norm()

                for i in range(polygon_n_vertices):
                    distances[i] = (polygon_vertices[:, i] - x).dot(normal)

                for i in range(polygon_n_vertices):
                    j = (i + 1) % polygon_n_vertices
                    if distances[i] <= 0.0:
                        clipped_vertices[:, clipped_n_vertices] = polygon_vertices[:, i]
                        clipped_n_vertices += 1
                    if distances[i] * distances[j] < 0.0:
                        # 边跨越平面，插值出交点
                        wa = distances[j] / (distances[j] - distances[i])
                        wb = 1.0 - wa
                        clipped_vertices[:, clipped_n_vertices] = (
                            wa * polygon_vertices[:, i] + wb * polygon_vertices[:, j]
                        )
                        clipped_n_vertices += 1
                polygon_n_vertices = clipped_n_vertices
                polygon_vertices = clipped_vertices

                if polygon_n_vertices < 3:
                    # If the polygon has less than 3 vertices, it is not a valid contact
                    # 少于 3 顶点，不构成有效接触多边形
                    break

            if polygon_n_vertices < 3:
                continue

            total_area = 0.0
            total_area_weighted_centroid = ti.Vector.zero(gs.ti_float, 3)
            for i in range(2, polygon_n_vertices):
                e1 = polygon_vertices[:, i - 1] - polygon_vertices[:, 0]
                e2 = polygon_vertices[:, i] - polygon_vertices[:, 0]
                area = 0.5 * e1.cross(e2).norm()
                total_area += area
                total_area_weighted_centroid += (
                    area * (polygon_vertices[:, 0] + polygon_vertices[:, i - 1] + polygon_vertices[:, i]) / 3.0
                )

            centroid = total_area_weighted_centroid / total_area
            barycentric0 = tet_barycentric(centroid, tet_vertices)
            tangent0 = (polygon_vertices[:, 0] - centroid).normalized()
            tangent1 = self.contact_candidates[i_c].normal.cross(tangent0)
            deformable_g = self.coupler._hydroelastic_stiffness
            rigid_g = self.coupler.fem_pressure_gradient[i_b, i_e].dot(self.contact_candidates[i_c].normal)
            pressure = barycentric0.dot(tet_pressures)
            if total_area < self.eps or rigid_g < self.eps:
                # 接触多边形退化或法向压力梯度过小，跳过
                continue
            g = rigid_g * deformable_g / (deformable_g + rigid_g)  # harmonic average
            # Deformable 与 Rigid 的等效刚度（调和平均）
            rigid_k = total_area * g
            rigid_phi0 = -pressure / g
            i_g = self.rigid_solver.verts_info.geom_idx[self.contact_candidates[i_c].vert_idx1[0]]
            i_l = self.rigid_solver.geoms_info.link_idx[i_g]
            i_p = ti.atomic_add(self.n_contact_pairs[None], 1)
            if i_p < self.max_contact_pairs:
                self.contact_pairs[i_p].batch_idx = i_b
                self.contact_pairs[i_p].normal = self.contact_candidates[i_c].normal
                self.contact_pairs[i_p].tangent0 = tangent0
                self.contact_pairs[i_p].tangent1 = tangent1
                self.contact_pairs[i_p].geom_idx0 = i_e
                self.contact_pairs[i_p].barycentric0 = barycentric0
                self.contact_pairs[i_p].link_idx = i_l
                self.contact_pairs[i_p].contact_pos = centroid
                sap_info[i_p].k = rigid_k
                sap_info[i_p].phi0 = rigid_phi0
                sap_info[i_p].mu = ti.sqrt(
                    self.fem_solver.elements_i[i_e].friction_mu * self.rigid_solver.geoms_info.coup_friction[i_g]
                )
            else:
                overflow = True

        return overflow

    @ti.func
    def detection(self, f: ti.i32):
        """
        Run BVH query and pipeline to produce final contact pairs.

        执行刚体三角形 vs FEM 表面四面体的 BVH 查询与候选/配对管线，生成最终接触对。
        """
        overflow = False
        overflow |= self.coupler.rigid_tri_bvh.query(self.coupler.fem_surface_tet_aabb.aabbs)
        overflow |= self.compute_candidates(f)
        overflow |= self.compute_pairs(f)
        return overflow

    @ti.func
    def compute_delassus_world_frame(self):
        """
        Assemble Delassus W = J M^-1 J^T + J A_fem^{-1} J^T in world frame.

        在世界系组装 Delassus 矩阵：
        - 刚体部分：通过质量矩阵分解求 M^-1 J^T，并累加 J M^-1 J^T；
        - FEM 部分：用 PCG 对角近似作为 A_fem^{-1}，按重心权叠加 J A^{-1} J^T。
        """
        dt2_inv = 1.0 / self.sim._substep_dt**2
        # rigid
        self.coupler.rigid_solve_jacobian(
            self.Jt, self.M_inv_Jt, self.n_contact_pairs[None], self.contact_pairs.batch_idx, 3
        )
        self.W.fill(0.0)
        for i_p, i_d, i, j in ti.ndrange(self.n_contact_pairs[None], self.rigid_solver.n_dofs, 3, 3):
            self.W[i_p][i, j] += self.M_inv_Jt[i_p, i_d][i] * self.Jt[i_p, i_d][j]

        # fem
        barycentric0 = ti.static(self.contact_pairs.barycentric0)
        for i_p in range(self.n_contact_pairs[None]):
            i_g0 = self.contact_pairs[i_p].geom_idx0
            i_b = self.contact_pairs[i_p].batch_idx
            for i in ti.static(range(4)):
                i_v = self.fem_solver.elements_i[i_g0].el2v[i]
                self.W[i_p] += barycentric0[i_p][i] ** 2 * dt2_inv * self.fem_solver.pcg_state_v[i_b, i_v].prec

    @ti.func
    def compute_delassus(self, i_p):
        """
        Project world-frame W to contact frame (tangent0, tangent1, normal).

        将世界系 W 投影到接触坐标系（t0, t1, n）。
        """
        world = ti.Matrix.cols(
            [self.contact_pairs[i_p].tangent0, self.contact_pairs[i_p].tangent1, self.contact_pairs[i_p].normal]
        )
        return world.transpose() @ self.W[i_p] @ world

    @ti.func
    def compute_Jx(self, i_p, x0, x1):
        """
        Compute the contact Jacobian J times a vector x.

        计算接触雅可比 J @ x，其中 x0 为 FEM 顶点速度/方向，x1 为刚体 DOF 速度/方向。
        最后将结果投影到接触系（t0, t1, n）。
        """
        i_b = self.contact_pairs[i_p].batch_idx
        i_g0 = self.contact_pairs[i_p].geom_idx0
        Jx = ti.Vector.zero(gs.ti_float, 3)

        # fem: 加权聚合四面体四顶点（重心权）
        for i in ti.static(range(4)):
            i_v = self.fem_solver.elements_i[i_g0].el2v[i]
            Jx = Jx + self.contact_pairs[i_p].barycentric0[i] * x0[i_b, i_v]

        # rigid: 减去刚体部分的 J @ x
        for i in range(self.rigid_solver.n_dofs):
            Jx = Jx - self.Jt[i_p, i] * x1[i_b, i]
        return ti.Vector(
            [
                Jx.dot(self.contact_pairs[i_p].tangent0),
                Jx.dot(self.contact_pairs[i_p].tangent1),
                Jx.dot(self.contact_pairs[i_p].normal),
            ]
        )

    @ti.func
    def add_Jt_x(self, y0, y1, i_p, x):
        """
        y0/y1 += J^T @ x with x in contact frame.

        当 x 在接触系给出时，先转换到世界系，然后回写：
        - FEM 顶点：按重心权分配到 4 个顶点；
        - 刚体 DOF：沿运动学树的 Jt 行向量做点积累加。
        """
        i_b = self.contact_pairs[i_p].batch_idx
        i_g0 = self.contact_pairs[i_p].geom_idx0
        world = ti.Matrix.cols(
            [self.contact_pairs[i_p].tangent0, self.contact_pairs[i_p].tangent1, self.contact_pairs[i_p].normal]
        )
        x_ = world @ x

        # fem
        for i in ti.static(range(4)):
            i_v = self.fem_solver.elements_i[i_g0].el2v[i]
            y0[i_b, i_v] += self.contact_pairs[i_p].barycentric0[i] * x_

        # rigid
        for i in range(self.rigid_solver.n_dofs):
            y1[i_b, i] -= self.Jt[i_p, i].dot(x_)

    @ti.func
    def add_Jt_A_J_diag3x3(self, y, i_p, A):
        """
        Accumulate FEM diag contribution: J^T A J in world frame.

        仅对 FEM 侧对角叠加：将接触系 A 变换到世界系后，按重心权平方累加到 4 顶点的 3x3 对角块。
        """
        i_b = self.contact_pairs[i_p].batch_idx
        i_g0 = self.contact_pairs[i_p].geom_idx0
        world = ti.Matrix.cols(
            [self.contact_pairs[i_p].tangent0, self.contact_pairs[i_p].tangent1, self.contact_pairs[i_p].normal]
        )
        B_ = world @ A @ world.transpose()
        for i in ti.static(range(4)):
            i_v = self.fem_solver.elements_i[i_g0].el2v[i]
            if i_v < self.fem_solver.n_vertices:
                y[i_b, i_v] += self.contact_pairs[i_p].barycentric0[i] ** 2 * B_


@ti.data_oriented
class RigidRigidTetContactHandler(RigidRigidContactHandler):
    """
    Class for handling contact between Rigid bodies using hydroelastic model.

    This class extends the RigidContact class and provides methods for detecting contact
    between tetrahedral elements, computing contact pairs, and managing contact-related computations.

    刚体-刚体（体四面体）Hydroelastic 接触处理器。
    - 通过刚体体四面体之间的等压面裁剪构造接触多边形；
    - 由两侧压力梯度计算等效刚度与 phi0；
    - 装配两条刚体链的 Jt，用于 SAP 求解。
    """

    def __init__(
        self,
        simulator: "Simulator",
        eps: float = 1e-10,
    ) -> None:
        super().__init__(simulator)
        self.coupler = simulator.coupler
        self.name = "RigidRigidTetContactHandler"
        self.eps = eps
        self.contact_candidate_type = ti.types.struct(
            batch_idx=gs.ti_int,  # batch index
            # batch 索引
            geom_idx0=gs.ti_int,  # index of the element
            # 刚体体四面体 0 的索引
            geom_idx1=gs.ti_int,  # index of the other element
            # 刚体体四面体 1 的索引
            intersection_code0=gs.ti_int,  # intersection code for element0
            # 四面体 0 的截交编码（Marching Tets）
            normal=gs.ti_vec3,  # contact plane normal
            # 等压面法向
            x=gs.ti_vec3,  # a point on the contact plane
            # 等压面上一点
            distance0=gs.ti_vec4,  # distance vector for element0
            # 四面体 0 的顶点到平面的距离
        )
        self.n_contact_candidates = ti.field(gs.ti_int, shape=())
        self.max_contact_candidates = self.coupler.rigid_volume_elems.shape[0] * self.sim._B * 8
        self.contact_candidates = self.contact_candidate_type.field(shape=(self.max_contact_candidates,))

        self.contact_pair_type = ti.types.struct(
            batch_idx=gs.ti_int,  # batch index
            # batch 索引
            normal=gs.ti_vec3,  # contact plane normal
            # 接触法向
            tangent0=gs.ti_vec3,  # contact plane tangent0
            # 接触切向 0
            tangent1=gs.ti_vec3,  # contact plane tangent1
            # 接触切向 1
            link_idx0=gs.ti_int,  # index of the link
            # 刚体 0 的 link 索引
            link_idx1=gs.ti_int,  # index of the other link
            # 刚体 1 的 link 索引
            contact_pos=gs.ti_vec3,  # contact position
            # 接触点位置
            sap_info=self.sap_contact_info_type,  # contact info
            # SAP 接触参数
        )
        self.max_contact_pairs = self.coupler.rigid_volume_elems.shape[0] * self.sim._B
        self.contact_pairs = self.contact_pair_type.field(shape=(self.max_contact_pairs,))
        self.Jt = ti.field(gs.ti_vec3, shape=(self.max_contact_pairs, self.rigid_solver.n_dofs))
        self.M_inv_Jt = ti.field(gs.ti_vec3, shape=(self.max_contact_pairs, self.rigid_solver.n_dofs))
        self.W = ti.field(gs.ti_mat3, shape=(self.max_contact_pairs,))

    @ti.func
    def compute_candidates(self, f: ti.i32):
        """
        Build rigid-rigid tet intersection candidates using isobaric plane.

        利用两刚体四面体的压力场在等压面上构造候选平面，并做平面侧判定生成候选对：
        - 法向 normal = g0 - g1，点 x 由等压条件求解；
        - 要求 normal 与 g0 同向且与 g1 反向（带容差）；
        - 两个四面体均需被平面穿过。
        """
        overflow = False
        candidates = ti.static(self.contact_candidates)
        self.n_contact_candidates[None] = 0
        result_count = ti.min(
            self.coupler.rigid_tet_bvh.query_result_count[None],
            self.coupler.rigid_tet_bvh.max_query_results,
        )
        for i_r in range(result_count):
            i_b, i_a, i_q = self.coupler.rigid_tet_bvh.query_result[i_r]
            i_v0 = self.coupler.rigid_volume_elems[i_a][0]
            i_v1 = self.coupler.rigid_volume_elems[i_q][1]
            x0 = self.coupler.rigid_volume_verts[i_b, i_v0]
            x1 = self.coupler.rigid_volume_verts[i_b, i_v1]
            p0 = self.coupler.rigid_pressure_field[i_v0]
            p1 = self.coupler.rigid_pressure_field[i_v1]
            g0 = self.coupler.rigid_pressure_gradient[i_b, i_a]
            g1 = self.coupler.rigid_pressure_gradient[i_b, i_q]
            g0_norm = g0.norm()
            g1_norm = g1.norm()
            if g0_norm < gs.EPS or g1_norm < gs.EPS:
                continue
            # Calculate the isosurface, i.e. equal pressure plane defined by x and normal
            # Solve for p0 + g0.dot(x - x0) = p1 + g1.dot(x - x1)
            # 等压面：p0 + g0·(x-x0) = p1 + g1·(x-x1)
            normal = g0 - g1
            magnitude = normal.norm()
            if magnitude < gs.EPS:
                continue
            normal /= magnitude
            b = p1 - p0 - g1.dot(x1) + g0.dot(x0)
            x = b / magnitude * normal
            # Check that the normal is pointing along g0 and against g1, some allowance as used in Drake
            # 法向需大致沿 g0 且反向 g1
            if normal.dot(g0) < self.eps or normal.dot(g1) > -self.eps:
                continue

            intersection_code0 = ti.int32(0)
            distance0 = ti.Vector([0.0, 0.0, 0.0, 0.0])
            intersection_code1 = ti.int32(0)
            distance1 = ti.Vector([0.0, 0.0, 0.0, 0.0])
            for i in ti.static(range(4)):
                i_v = self.coupler.rigid_volume_elems[i_a][i]
                pos_v = self.coupler.rigid_volume_verts[i_b, i_v]
                distance0[i] = (pos_v - x).dot(normal)  # signed distance
                if distance0[i] > 0:
                    intersection_code0 |= 1 << i
            for i in ti.static(range(4)):
                i_v = self.coupler.rigid_volume_elems[i_q][i]
                pos_v = self.coupler.rigid_volume_verts[i_b, i_v]
                distance1[i] = (pos_v - x).dot(normal)
                if distance1[i] > 0:
                    intersection_code1 |= 1 << i
            # Fast check for whether both tets intersect with the plane
            # 两端都需被平面穿过
            if (
                intersection_code0 == 0
                or intersection_code1 == 0
                or intersection_code0 == 15
                or intersection_code1 == 15
            ):
                continue
            i_c = ti.atomic_add(self.n_contact_candidates[None], 1)
            if i_c < self.max_contact_candidates:
                candidates[i_c].batch_idx = i_b
                candidates[i_c].normal = normal
                candidates[i_c].x = x
                candidates[i_c].geom_idx0 = i_a
                candidates[i_c].intersection_code0 = intersection_code0
                candidates[i_c].distance0 = distance0
                candidates[i_c].geom_idx1 = i_q
            else:
                overflow = True
        return overflow

    @ti.func
    def compute_pairs(self, i_step: ti.i32):
        """
        Clip polygon by four faces of tet1, compute centroid/area, and build contact pairs.

        用候选平面裁剪四面体 0 的截多边形后，与四面体 1 的四个面半空间依次裁剪：
        - 若得到至少三顶点的多边形，则计算面积与质心；
        - 插值四面体 0 顶点压力得到压力值；
        - 利用两侧梯度 g0/g1 得到等效刚度 g 与 phi0，构建接触对。
        """
        overflow = False
        candidates = ti.static(self.contact_candidates)
        pairs = ti.static(self.contact_pairs)
        sap_info = ti.static(pairs.sap_info)
        normal_signs = ti.Vector([1.0, -1.0, 1.0, -1.0])  # make normal point outward
        # 修正各面的法向朝外
        self.n_contact_pairs[None] = 0
        result_count = ti.min(self.n_contact_candidates[None], self.max_contact_candidates)
        for i_c in range(result_count):
            i_b = candidates[i_c].batch_idx
            i_e0 = candidates[i_c].geom_idx0
            i_e1 = candidates[i_c].geom_idx1
            intersection_code0 = candidates[i_c].intersection_code0
            distance0 = candidates[i_c].distance0
            intersected_edges0 = self.coupler.MarchingTetsEdgeTable[intersection_code0]
            tet_vertices0 = ti.Matrix.zero(gs.ti_float, 3, 4)  # 4 vertices of tet 0
            tet_pressures0 = ti.Vector.zero(gs.ti_float, 4)  # pressures at the vertices of tet 0
            tet_vertices1 = ti.Matrix.zero(gs.ti_float, 3, 4)  # 4 vertices of tet 1

            for i in ti.static(range(4)):
                i_v = self.coupler.rigid_volume_elems[i_e0][i]
                tet_vertices0[:, i] = self.coupler.rigid_volume_verts[i_b, i_v]
                tet_pressures0[i] = self.coupler.rigid_pressure_field[i_v]

            for i in ti.static(range(4)):
                i_v = self.coupler.rigid_volume_elems[i_e1][i]
                tet_vertices1[:, i] = self.coupler.rigid_volume_verts[i_b, i_v]

            polygon_vertices = ti.Matrix.zero(gs.ti_float, 3, 8)  # maximum 8 vertices
            # 裁剪多边形最多 8 顶点
            polygon_n_vertices = gs.ti_int(0)
            clipped_vertices = ti.Matrix.zero(gs.ti_float, 3, 8)  # maximum 8 vertices
            clipped_n_vertices = gs.ti_int(0)
            # 先由四面体 0 与平面求交得到 3~4 顶点的多边形
            for i in range(4):
                if intersected_edges0[i] >= 0:
                    edge = self.coupler.TetEdges[intersected_edges0[i]]
                    pos_v0 = tet_vertices0[:, edge[0]]
                    pos_v1 = tet_vertices0[:, edge[1]]
                    d_v0 = distance0[edge[0]]
                    d_v1 = distance0[edge[1]]
                    t = d_v0 / (d_v0 - d_v1)
                    polygon_vertices[:, polygon_n_vertices] = pos_v0 + t * (pos_v1 - pos_v0)
                    polygon_n_vertices += 1
            # Intersects the polygon with the four halfspaces of the four triangles
            # of the tetrahedral element1.
            # 将该多边形依次与四面体 1 的四个面半空间裁剪
            for face in range(4):
                clipped_n_vertices = 0
                x = tet_vertices1[:, (face + 1) % 4]
                normal = (tet_vertices1[:, (face + 2) % 4] - x).cross(
                    tet_vertices1[:, (face + 3) % 4] - x
                ) * normal_signs[face]
                normal /= normal.norm()

                distances = ti.Vector.zero(gs.ti_float, 8)
                for i in range(polygon_n_vertices):
                    distances[i] = (polygon_vertices[:, i] - x).dot(normal)

                for i in range(polygon_n_vertices):
                    j = (i + 1) % polygon_n_vertices
                    if distances[i] <= 0.0:
                        clipped_vertices[:, clipped_n_vertices] = polygon_vertices[:, i]
                        clipped_n_vertices += 1
                        if distances[j] > 0.0:
                            # 边跨越平面，插值出交点
                            wa = distances[j] / (distances[j] - distances[i])
                            wb = 1.0 - wa
                            clipped_vertices[:, clipped_n_vertices] = (
                                wa * polygon_vertices[:, i] + wb * polygon_vertices[:, j]
                            )
                            clipped_n_vertices += 1
                    elif distances[j] <= 0.0:
                        wa = distances[j] / (distances[j] - distances[i])
                        wb = 1.0 - wa
                        clipped_vertices[:, clipped_n_vertices] = (
                            wa * polygon_vertices[:, i] + wb * polygon_vertices[:, j]
                        )
                        clipped_n_vertices += 1
                polygon_n_vertices = clipped_n_vertices
                polygon_vertices = clipped_vertices

                if polygon_n_vertices < 3:
                    # If the polygon has less than 3 vertices, it is not a valid contact
                    # 裁剪后顶点少于 3，不构成接触
                    break

            if polygon_n_vertices < 3:
                continue

            # compute centroid and area of the polygon
            # 计算裁剪多边形的面积与质心
            total_area = 0.0  # avoid division by zero
            total_area_weighted_centroid = ti.Vector.zero(gs.ti_float, 3)
            for i in range(2, polygon_n_vertices):
                e1 = polygon_vertices[:, i - 1] - polygon_vertices[:, 0]
                e2 = polygon_vertices[:, i] - polygon_vertices[:, 0]
                area = 0.5 * e1.cross(e2).norm()
                total_area += area
                total_area_weighted_centroid += (
                    area * (polygon_vertices[:, 0] + polygon_vertices[:, i - 1] + polygon_vertices[:, i]) / 3.0
                )

            if total_area < self.eps:
                continue
            centroid = total_area_weighted_centroid / total_area
            tangent0 = polygon_vertices[:, 0] - centroid
            tangent0 /= tangent0.norm()
            tangent1 = candidates[i_c].normal.cross(tangent0)
            g0 = self.coupler.rigid_pressure_gradient[i_b, i_e0].dot(candidates[i_c].normal)
            g1 = -self.coupler.rigid_pressure_gradient[i_b, i_e1].dot(candidates[i_c].normal)
            g = 1.0 / (1.0 / g0 + 1.0 / g1)  # harmonic average, can handle infinity
            # 两侧梯度的调和平均（允许一侧无穷大）
            rigid_k = total_area * g
            barycentric0 = tet_barycentric(centroid, tet_vertices0)
            pressure = (
                barycentric0[0] * tet_pressures0[0]
                + barycentric0[1] * tet_pressures0[1]
                + barycentric0[2] * tet_pressures0[2]
                + barycentric0[3] * tet_pressures0[3]
            )
            rigid_phi0 = -pressure / g
            if rigid_phi0 > self.eps:
                continue
            i_p = ti.atomic_add(self.n_contact_pairs[None], 1)
            if i_p < self.max_contact_pairs:
                pairs[i_p].batch_idx = i_b
                pairs[i_p].normal = candidates[i_c].normal
                pairs[i_p].tangent0 = tangent0
                pairs[i_p].tangent1 = tangent1
                pairs[i_p].contact_pos = centroid
                i_g0 = self.coupler.rigid_volume_elems_geom_idx[i_e0]
                i_g1 = self.coupler.rigid_volume_elems_geom_idx[i_e1]
                i_l0 = self.rigid_solver.geoms_info.link_idx[i_g0]
                i_l1 = self.rigid_solver.geoms_info.link_idx[i_g1]
                pairs[i_p].link_idx0 = i_l0
                pairs[i_p].link_idx1 = i_l1
                sap_info[i_p].k = rigid_k
                sap_info[i_p].phi0 = rigid_phi0
                sap_info[i_p].mu = ti.sqrt(
                    self.rigid_solver.geoms_info.friction[i_g0] * self.rigid_solver.geoms_info.friction[i_g1]
                )
            else:
                overflow = True
        return overflow

    @ti.func
    def detection(self, f: ti.i32):
        """
        Broad phase, candidate building and narrow phase to produce rigid-rigid pairs.

        执行刚体体四面体 BVH 查询、候选生成与裁剪配对，得到刚体-刚体接触对。
        """
        overflow = False
        overflow |= self.coupler.rigid_tet_bvh.query(self.coupler.rigid_tet_aabb.aabbs)
        overflow |= self.compute_candidates(f)
        overflow |= self.compute_pairs(f)
        return overflow