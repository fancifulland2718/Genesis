"""
PBD 求解器（Position-Based Dynamics）

概述
- 面向布料/弹性体/液体/自由粒子的统一 PBD 求解器，支持多环境 batch。
- 时间步管线（子步）
  1) 预测阶段：存初始位置 + 外力积分预测位置
  2) 拓扑约束：拉伸/弯曲/体积（无邻域查找）
  3) 空间哈希：重排粒子以加速邻域查询
  4) 空间约束：密度（不可压缩）/粘度（XSPH）
  5) 粒子间碰撞：分离 + 静/动摩擦
  6) 速度更新：v = (x - x0) / dt
  7) 后处理：从重排数组回写 + 边界碰撞
"""

import math

import numpy as np
from numpy.typing import NDArray
import gstaichi as ti

import genesis as gs
import genesis.utils.geom as gu
from genesis.engine.boundaries import CubeBoundary
from genesis.engine.entities import (
    PBD2DEntity,
    PBD3DEntity,
    PBDFreeParticleEntity,
    PBDParticleEntity,
)
from genesis.engine.states.solvers import PBDSolverState
from genesis.utils.array_class import LinksState
from genesis.utils.geom import SpatialHasher

from .base_solver import Solver


@ti.data_oriented
class PBDSolver(Solver):
    # ------------------------------------------------------------------------------------
    # --------------------------------- Initialization -----------------------------------
    # ------------------------------------------------------------------------------------

    class MATERIAL(gs.IntEnum):
        """
        材料类型枚举：用于决定参与何种约束/碰撞逻辑
        - CLOTH: 布料（2D 拓扑，拉伸/弯曲）
        - ELASTIC: 弹性体（3D 拓扑，体积约束）
        - LIQUID: 液体（PBF，密度/粘度约束）
        - PARTICLE: 自由粒子（仅外力与边界，忽略内部约束/碰撞）
        """
        CLOTH = 0
        ELASTIC = 1
        LIQUID = 2
        PARTICLE = 3  # non-physics particles

    def __init__(self, scene, sim, options):
        """
        读取选项并初始化求解器：边界、粒子尺寸、各类约束的迭代次数与参数、空间哈希等。
        """
        super().__init__(scene, sim, options)

        # options
        self._upper_bound = np.array(options.upper_bound)
        self._lower_bound = np.array(options.lower_bound)
        self._particle_size = options.particle_size
        self._max_stretch_solver_iterations = options.max_stretch_solver_iterations
        self._max_bending_solver_iterations = options.max_bending_solver_iterations
        self._max_volume_solver_iterations = options.max_volume_solver_iterations
        self._max_density_solver_iterations = options.max_density_solver_iterations
        self._max_viscosity_solver_iterations = options.max_viscosity_solver_iterations

        self._n_vvert_supports = self.scene.vis_options.n_support_neighbors

        # 邻域核函数尺度（与半径相关），用于 SPH 风格核函数
        self.dist_scale = self.particle_radius / 0.4  # @Zhenjia: 有待确认比例
        self.h = 1.0
        self.h_2 = self.h**2
        self.h_6 = self.h**6
        self.h_9 = self.h**9

        # Poly6 核
        self.poly6_Coe = 315.0 / (64 * math.pi)

        # Spiky 核（梯度）
        self.spiky_Coe = -45.0 / math.pi

        # λ 的数值稳定项（PBF/XPBD）
        self.lambda_epsilon = 100.0

        # 密度约束中的 S_Corr（表面张力修正）
        self.S_Corr_delta_q = 0.3
        self.S_Corr_k = 0.0001

        # 梯度近似差分步长（曾用于涡量，当前注释）
        self.g_del = 0.01

        self.vorti_epsilon = 0.01

        # 空间哈希：用于邻域搜索和粒子重排，提高局部遍历效率
        self.sh = SpatialHasher(
            cell_size=options.hash_grid_cell_size,
            grid_res=options._hash_grid_res,
        )

        # boundary
        self.setup_boundary()

    def _batch_shape(self, shape=None, first_dim=False, B=None):
        """
        工具：返回带 batch 维度的 shape
        - first_dim=True 时将 batch 放前
        """
        if B is None:
            B = self._B

        if shape is None:
            return (B,)
        elif isinstance(shape, (list, tuple)):
            return (B,) + shape if first_dim else shape + (B,)
        else:
            return (B, shape) if first_dim else (shape, B)

    def setup_boundary(self):
        """
        设置立方体边界，用于边界碰撞与位置/速度修正。
        """
        self.boundary = CubeBoundary(
            lower=self._lower_bound,
            upper=self._upper_bound,
        )

    def init_vvert_fields(self):
        """
        初始化可视化顶点数据（vverts）：
        - 每个可视化顶点由若干物理粒子加权组合，用于稠密渲染。
        """
        struct_vvert_info = ti.types.struct(
            support_idxs=ti.types.vector(self._n_vvert_supports, gs.ti_int),
            support_weights=ti.types.vector(self._n_vvert_supports, gs.ti_float),
        )
        self.vverts_info = struct_vvert_info.field(shape=max(1, self._n_vverts), layout=ti.Layout.SOA)

        struct_vvert_state_render = ti.types.struct(
            pos=gs.ti_vec3,
            active=gs.ti_bool,
        )
        self.vverts_render = struct_vvert_state_render.field(
            shape=self._batch_shape(shape=max(1, self._n_vverts)), layout=ti.Layout.SOA
        )

    def init_particle_fields(self):
        """
        初始化粒子场：
        - info: 静态参数（质量、静止位置、材料/摩擦、液体参数等）
        - 动态状态：自由标志/位置/速度/增量/密度/λ 等
        - 渲染用状态与重排版本（加速邻居访问）
        """
        # particles information (static)
        struct_particle_info = ti.types.struct(
            mass=gs.ti_float,
            pos_rest=gs.ti_vec3,
            rho_rest=gs.ti_float,
            material_type=gs.ti_int,
            mu_s=gs.ti_float,
            mu_k=gs.ti_float,
            air_resistance=gs.ti_float,
            density_relaxation=gs.ti_float,
            viscosity_relaxation=gs.ti_float,
        )
        # particles state (dynamic)
        struct_particle_state = ti.types.struct(
            free=gs.ti_bool,  # 若非 free，粒子不受内部约束，仅受外部控制
            pos=gs.ti_vec3,  # 当前位置
            ipos=gs.ti_vec3,  # 子步初始位置（预测/修正参考）
            dpos=gs.ti_vec3,  # 位置增量（约束/碰撞修正累计）
            vel=gs.ti_vec3,  # 速度
            lam=gs.ti_float,
            rho=gs.ti_float,
        )

        # dynamic particle state without gradient
        struct_particle_state_ng = ti.types.struct(
            reordered_idx=gs.ti_int,
            active=gs.ti_bool,
        )

        # single frame particle state for rendering
        struct_particle_state_render = ti.types.struct(
            pos=gs.ti_vec3,
            vel=gs.ti_vec3,
            active=gs.ti_bool,
        )

        shared_shape = self._n_particles
        batched_shape = self._batch_shape(shared_shape)

        self.particles_info = struct_particle_info.field(shape=shared_shape, layout=ti.Layout.SOA)
        self.particles_info_reordered = struct_particle_info.field(shape=batched_shape, layout=ti.Layout.SOA)

        self.particles = struct_particle_state.field(shape=batched_shape, layout=ti.Layout.SOA)
        self.particles_reordered = struct_particle_state.field(shape=batched_shape, layout=ti.Layout.SOA)

        self.particles_ng = struct_particle_state_ng.field(shape=batched_shape, layout=ti.Layout.SOA)
        self.particles_ng_reordered = struct_particle_state_ng.field(shape=batched_shape, layout=ti.Layout.SOA)

        self.particles_render = struct_particle_state_render.field(shape=batched_shape, layout=ti.Layout.SOA)

    def init_edge_fields(self):
        """
        初始化边数据：
        - edges_info：用于拉伸约束（边长保持）
        - inner_edges_info：用于弯曲约束（二面角）
        """
        # edges information for stretch. edge: (v1, v2)
        struct_edge_info = ti.types.struct(
            len_rest=gs.ti_float,
            stretch_compliance=gs.ti_float,
            stretch_relaxation=gs.ti_float,
            v1=gs.ti_int,
            v2=gs.ti_int,
        )
        self.edges_info = struct_edge_info.field(shape=max(1, self._n_edges), layout=ti.Layout.SOA)

        # inner edges information for bending. edge: (v1, v2), adjacent faces: (v1, v2, v3) and (v1, v2, v4)
        struct_inner_edge_info = ti.types.struct(
            len_rest=gs.ti_float,
            bending_compliance=gs.ti_float,
            bending_relaxation=gs.ti_float,
            v1=gs.ti_int,
            v2=gs.ti_int,
            v3=gs.ti_int,
            v4=gs.ti_int,
        )
        self.inner_edges_info = struct_inner_edge_info.field(shape=max(1, self._n_inner_edges), layout=ti.Layout.SOA)

    def init_elem_fields(self):
        """
        初始化体单元数据（四面体）：用于体积约束（保持体积接近静止体积）。
        """
        struct_elem_info = ti.types.struct(
            vol_rest=gs.ti_float,
            volume_compliance=gs.ti_float,
            volume_relaxation=gs.ti_float,
            v1=gs.ti_int,
            v2=gs.ti_int,
            v3=gs.ti_int,
            v4=gs.ti_int,
        )
        self.elems_info = struct_elem_info.field(shape=max(1, self._n_elems), layout=ti.Layout.SOA)

    def init_ckpt(self):
        "初始化检查点容器。"
        self._ckpt = dict()

    def reset_grad(self):
        "当前求解器未接入自动微分，空实现。"
        pass

    def build(self):
        """
        构建：分配与计数相关的 field，构建空间哈希，并将实体加入求解器。
        """
        super().build()
        self._B = self._sim._B
        self._n_particles = self.n_particles
        self._n_fluid_particles = self.n_fluid_particles
        self._n_edges = self.n_edges
        self._n_inner_edges = self.n_inner_edges
        self._n_elems = self.n_elems
        self._n_vverts = self.n_vverts
        self._n_vfaces = self.n_vfaces

        if self.is_active():
            self.sh.build(self._B)

            self.init_particle_fields()
            self.init_edge_fields()
            self.init_elem_fields()
            self.init_vvert_fields()

            self.init_ckpt()

            for entity in self._entities:
                entity._add_to_solver()

    # ------------------------------------------------------------------------------------
    # -------------------------------------- misc ----------------------------------------
    # ------------------------------------------------------------------------------------

    def add_entity(self, idx, material, morph, surface):
        """
        添加实体：根据材料类型构造对应的 PBD 实体，并记录其在全局数组的起始下标。
        """
        if isinstance(material, gs.materials.PBD.Cloth):
            entity = PBD2DEntity(
                scene=self.scene,
                solver=self,
                material=material,
                morph=morph,
                surface=surface,
                particle_size=self._particle_size,
                idx=idx,
                particle_start=self.n_particles,
                edge_start=self.n_edges,
                inner_edge_start=self.n_inner_edges,
                vvert_start=self.n_vverts,
                vface_start=self.n_vfaces,
            )

        elif isinstance(material, gs.materials.PBD.Elastic):
            entity = PBD3DEntity(
                scene=self.scene,
                solver=self,
                material=material,
                morph=morph,
                surface=surface,
                particle_size=self._particle_size,
                idx=idx,
                particle_start=self.n_particles,
                edge_start=self.n_edges,
                elem_start=self.n_elems,
                vvert_start=self.n_vverts,
                vface_start=self.n_vfaces,
            )

        elif isinstance(material, gs.materials.PBD.Liquid):
            entity = PBDParticleEntity(
                scene=self.scene,
                solver=self,
                material=material,
                morph=morph,
                surface=surface,
                particle_size=self._particle_size,
                idx=idx,
                particle_start=self.n_particles,
            )

        elif isinstance(material, gs.materials.PBD.Particle):
            entity = PBDFreeParticleEntity(
                scene=self.scene,
                solver=self,
                material=material,
                morph=morph,
                surface=surface,
                particle_size=self._particle_size,
                idx=idx,
                particle_start=self.n_particles,
            )

        else:
            raise NotImplementedError()

        self._entities.append(entity)

        return entity

    def is_active(self):
        "是否有粒子参与仿真。"
        return self._n_particles > 0

    # ------------------------------------------------------------------------------------
    # ------------------------------------- utils ----------------------------------------
    # ------------------------------------------------------------------------------------

    @ti.func
    def poly6(self, dist):
        """
        Poly6 核（向量版本）：W(r) = 315/(64πh^9) * (h^2 - |r|^2)^3, 0 < |r| < h
        用途：密度/粘度计算
        """
        result = gs.ti_float(0.0)
        d = dist.norm() / self.dist_scale
        if 0 < d < self.h:
            rhs = (self.h_2 - d * d) * (self.h_2 - d * d) * (self.h_2 - d * d)
            result = self.poly6_Coe * rhs / self.h_9
        return result

    @ti.func
    def poly6_scalar(self, dist):
        "Poly6 核（标量版本）。"
        result = gs.ti_float(0.0)
        d = dist
        if 0 < d < self.h:
            rhs = (self.h_2 - d * d) * (self.h_2 - d * d) * (self.h_2 - d * d)
            result = self.poly6_Coe * rhs / self.h_9
        return result

    @ti.func
    def spiky(self, dist):
        """
        Spiky 核的梯度项（返回向量）：∇W(r) = -45/(πh^6)*(h-|r|)^2/|r| * r_hat, 0<|r|<h
        用途：PBF 中 ∇C、涡量等。
        """
        result = ti.Vector.zero(gs.ti_float, 3)
        d = dist.norm() / self.dist_scale
        if 0 < d < self.h:
            m = (self.h - d) * (self.h - d)
            result = (self.spiky_Coe * m / (self.h_6 * d)) * dist / self.dist_scale
        return result

    @ti.func
    def S_Corr(self, dist):
        """
        PBF 中的表面张力修正：s_corr = -k * (W(r)/W(δq))^4
        缓解粒子聚集造成的数值伪振荡。
        """
        upper = self.poly6(dist)
        lower = self.poly6_scalar(self.S_Corr_delta_q)
        m = upper / lower
        return -1.0 * self.S_Corr_k * m * m * m * m

    # ------------------------------------------------------------------------------------
    # ----------------------------------- simulation -------------------------------------
    # ------------------------------------------------------------------------------------
    @ti.kernel
    def _kernel_store_initial_pos(self, f: ti.i32):
        "存储子步初始位置 ipos ← pos，供后续速度计算与碰撞使用。"
        for i_p, i_b in ti.ndrange(self._n_particles, self._B):
            self.particles[i_p, i_b].ipos = self.particles[i_p, i_b].pos

    @ti.kernel
    def _kernel_reorder_particles(self, f: ti.i32):
        """
        重排粒子：
        - 使用空间哈希为每个粒子计算 slot 索引并得到重排下标
        - 将状态/静态信息按重排次序拷贝到 reordered 缓冲，提高邻域遍历局部性
        """
        self.sh.compute_reordered_idx(
            self._n_particles, self.particles.pos, self.particles_ng.active, self.particles_ng.reordered_idx
        )

        # copy to reordered
        self.particles_ng_reordered.active.fill(False)
        for i_p, i_b in ti.ndrange(self._n_particles, self._B):
            if self.particles_ng[i_p, i_b].active:
                reordered_idx = self.particles_ng[i_p, i_b].reordered_idx

                self.particles_reordered[reordered_idx, i_b] = self.particles[i_p, i_b]
                self.particles_info_reordered[reordered_idx, i_b] = self.particles_info[i_p]
                self.particles_ng_reordered[reordered_idx, i_b].active = self.particles_ng[i_p, i_b].active

    @ti.kernel
    def _kernel_apply_external_force(self, f: ti.i32, t: ti.f32):
        """
        外力与预测：
        - 对自由粒子应用重力与外部力场（可选空气阻力）
        - 位置预测：pos += vel * dt
        """
        for i_p, i_b in ti.ndrange(self._n_particles, self._B):
            if self.particles[i_p, i_b].free:
                # gravity
                self.particles[i_p, i_b].vel = self.particles[i_p, i_b].vel + self._gravity[i_b] * self._substep_dt

                # external force fields
                acc = ti.Vector.zero(gs.ti_float, 3)
                for i_ff in ti.static(range(len(self._ffs))):
                    acc += self._ffs[i_ff].get_acc(self.particles[i_p, i_b].pos, self.particles[i_p, i_b].vel, t, i_p)
                self.particles[i_p, i_b].vel = self.particles[i_p, i_b].vel + acc * self._substep_dt

                if self.particles_info[i_p].material_type == self.MATERIAL.CLOTH:
                    # 简单空气阻力：与速度幅值与方向相关
                    f_air_resistance = (
                        self.particles_info[i_p].air_resistance
                        * self.particles[i_p, i_b].vel.norm()
                        * self.particles[i_p, i_b].vel
                    )
                    self.particles[i_p, i_b].vel = (
                        self.particles[i_p, i_b].vel
                        - f_air_resistance / self.particles_info[i_p].mass * self._substep_dt
                    )

            # 即便不是 free（被外部控制），也需要更新位置以跟随外部约束
            self.particles[i_p, i_b].pos = (
                self.particles[i_p, i_b].pos + self.particles[i_p, i_b].vel * self._substep_dt
            )

    @ti.kernel
    def _kernel_solve_stretch(self, f: ti.i32):
        """
        拉伸约束（边长保持，XPBD）：
        - 约束: C = |p1 - p2| - L0
        - 修正: dp = -C / (w1 + w2 + α) * n * relaxation
        - 迭代 self._max_stretch_solver_iterations 次
        """
        for _ in ti.static(range(self._max_stretch_solver_iterations)):
            for i_e, i_b in ti.ndrange(self._n_edges, self._B):
                v1 = self.edges_info[i_e].v1
                v2 = self.edges_info[i_e].v2

                w1 = self.particles[v1, i_b].free / self.particles_info[v1].mass
                w2 = self.particles[v2, i_b].free / self.particles_info[v2].mass
                n = self.particles[v1, i_b].pos - self.particles[v2, i_b].pos
                C = n.norm() - self.edges_info[i_e].len_rest
                alpha = self.edges_info[i_e].stretch_compliance / (self._substep_dt**2)
                dp = -C / (w1 + w2 + alpha) * n / n.norm(gs.EPS) * self.edges_info[i_e].stretch_relaxation
                self.particles[v1, i_b].dpos += dp * w1
                self.particles[v2, i_b].dpos -= dp * w2

            # 应用累计修正并清零 dpos
            for i_p, i_b in ti.ndrange(self._n_particles, self._B):
                if self.particles[i_p, i_b].free and self.particles_info[i_p].material_type != self.MATERIAL.PARTICLE:
                    self.particles[i_p, i_b].pos = self.particles[i_p, i_b].pos + self.particles[i_p, i_b].dpos
                    self.particles[i_p, i_b].dpos.fill(0)

    @ti.kernel
    def _kernel_solve_bending(self, f: ti.i32):
        """
        弯曲约束（二面角，XPBD）：
        - 基于 Position Based Dynamics 附录 A（Bending Constraint Projection）
        - 通过四点（共用边的两个三角形）计算法向并构造约束
        """
        for _ in ti.static(range(self._max_bending_solver_iterations)):
            for i_ie, i_b in ti.ndrange(self._n_inner_edges, self._B):  # 140 - 142
                v1 = self.inner_edges_info[i_ie].v1
                v2 = self.inner_edges_info[i_ie].v2
                v3 = self.inner_edges_info[i_ie].v3
                v4 = self.inner_edges_info[i_ie].v4

                w1 = self.particles[v1, i_b].free / self.particles_info[v1].mass
                w2 = self.particles[v2, i_b].free / self.particles_info[v2].mass
                w3 = self.particles[v3, i_b].free / self.particles_info[v3].mass
                w4 = self.particles[v4, i_b].free / self.particles_info[v4].mass

                if w1 + w2 + w3 + w4 > 0.0:
                    # https://matthias-research.github.io/pages/publications/posBasedDyn.pdf
                    # Appendix A: Bending Constraint Projection
                    p2 = self.particles[v2, i_b].pos - self.particles[v1, i_b].pos
                    p3 = self.particles[v3, i_b].pos - self.particles[v1, i_b].pos
                    p4 = self.particles[v4, i_b].pos - self.particles[v1, i_b].pos
                    l23 = p2.cross(p3).norm()
                    l24 = p2.cross(p4).norm()
                    n1 = p2.cross(p3) / l23
                    n2 = p2.cross(p4) / l24
                    d = ti.math.clamp(n1.dot(n2), -1.0, 1.0)

                    q3 = (p2.cross(n2) + n1.cross(p2) * d) / l23  # eq. (25)
                    q4 = (p2.cross(n1) + n2.cross(p2) * d) / l24  # eq. (26)
                    q2 = -(p3.cross(n2) + n1.cross(p3) * d) / l23 - (p4.cross(n1) + n2.cross(p4) * d) / l24  # eq. (27)
                    q1 = -q2 - q3 - q4
                    # eq. (29)
                    sum_wq = w1 * q1.norm_sqr() + w2 * q2.norm_sqr() + w3 * q3.norm_sqr() + w4 * q4.norm_sqr()
                    constraint = ti.acos(d) - ti.acos(-1.0)

                    # XPBD
                    alpha = self.inner_edges_info[i_ie].bending_compliance / (self._substep_dt**2)
                    constraint = (
                        -ti.sqrt(1 - d**2)
                        * constraint
                        / (sum_wq + alpha)
                        * self.inner_edges_info[i_ie].bending_relaxation
                    )

                    self.particles[v1, i_b].dpos += w1 * constraint * q1
                    self.particles[v2, i_b].dpos += w2 * constraint * q2
                    self.particles[v3, i_b].dpos += w3 * constraint * q3
                    self.particles[v4, i_b].dpos += w4 * constraint * q4

            for i_p, i_b in ti.ndrange(self._n_particles, self._B):
                if self.particles[i_p, i_b].free and self.particles_info[i_p].material_type != self.MATERIAL.PARTICLE:
                    self.particles[i_p, i_b].pos = self.particles[i_p, i_b].pos + self.particles[i_p, i_b].dpos
                    self.particles[i_p, i_b].dpos.fill(0)

    @ti.kernel
    def _kernel_solve_volume(self, f: ti.i32):
        """
        体积约束（四面体，XPBD）：
        - C = vol - vol_rest，vol = det(...) / 6
        - 梯度为每个顶点对应的面法向/6
        """
        for _ in ti.static(range(self._max_volume_solver_iterations)):
            for i_el, i_b in ti.ndrange(self._n_elems, self._B):
                v1 = self.elems_info[i_el].v1
                v2 = self.elems_info[i_el].v2
                v3 = self.elems_info[i_el].v3
                v4 = self.elems_info[i_el].v4

                p1 = self.particles[v1, i_b].pos
                p2 = self.particles[v2, i_b].pos
                p3 = self.particles[v3, i_b].pos
                p4 = self.particles[v4, i_b].pos

                grad1 = (p4 - p2).cross(p3 - p2) / 6.0
                grad2 = (p3 - p1).cross(p4 - p1) / 6.0
                grad3 = (p4 - p1).cross(p2 - p1) / 6.0
                grad4 = (p2 - p1).cross(p3 - p1) / 6.0

                w1 = self.particles[v1, i_b].free / self.particles_info[v1].mass * grad1.norm_sqr()
                w2 = self.particles[v2, i_b].free / self.particles_info[v2].mass * grad2.norm_sqr()
                w3 = self.particles[v3, i_b].free / self.particles_info[v3].mass * grad3.norm_sqr()
                w4 = self.particles[v4, i_b].free / self.particles_info[v4].mass * grad4.norm_sqr()

                if w1 + w2 + w3 + w4 > 0.0:
                    vol = gu.ti_tet_vol(p1, p2, p3, p4)
                    C = vol - self.elems_info[i_el].vol_rest
                    alpha = self.elems_info[i_el].volume_compliance / (self._substep_dt**2)
                    s = -C / (w1 + w2 + w3 + w4 + alpha) * self.elems_info[i_el].volume_relaxation

                    self.particles[v1, i_b].dpos += s * w1 * grad1
                    self.particles[v2, i_b].dpos += s * w2 * grad2
                    self.particles[v3, i_b].dpos += s * w3 * grad3
                    self.particles[v4, i_b].dpos += s * w4 * grad4

            for i_p, i_b in ti.ndrange(self._n_particles, self._B):
                if self.particles[i_p, i_b].free and self.particles_info[i_p].material_type != self.MATERIAL.PARTICLE:
                    self.particles[i_p, i_b].pos = self.particles[i_p, i_b].pos + self.particles[i_p, i_b].dpos
                    self.particles[i_p, i_b].dpos.fill(0)

    @ti.func
    def _func_solve_collision(self, i, j, i_b):
        """
        处理粒子 i 与 j 的碰撞（j -> i）：
        - 当 curr_dist < target_dist 且 rest_dist > target_dist 时，进行分离
        - 分离量按质量-可动权重分配，附加静/动摩擦修正
        """
        cur_dist = (self.particles_reordered[i, i_b].pos - self.particles_reordered[j, i_b].pos).norm(gs.EPS)
        rest_dist = (
            self.particles_info_reordered[i, i_b].pos_rest - self.particles_info_reordered[j, i_b].pos_rest
        ).norm(gs.EPS)
        target_dist = self._particle_size  # target particle distance is 2 * particle radius, i.e. particle_size
        if cur_dist < target_dist and rest_dist > target_dist:
            wi = self.particles_reordered[i, i_b].free / self.particles_info_reordered[i, i_b].mass
            wj = self.particles_reordered[j, i_b].free / self.particles_info_reordered[j, i_b].mass
            n = (self.particles_reordered[i, i_b].pos - self.particles_reordered[j, i_b].pos) / cur_dist

            # 分离碰撞
            self.particles_reordered[i, i_b].dpos += wi / (wi + wj) * (target_dist - cur_dist) * n

            # 摩擦（参见 Macklin, "Unified Particle Physics" 弹性-摩擦模型）
            # equation (23)
            dv = (self.particles_reordered[i, i_b].pos - self.particles_reordered[i, i_b].ipos) - (
                self.particles_reordered[j, i_b].pos - self.particles_reordered[j, i_b].ipos
            )
            dpos = -(dv - n * n.dot(dv))
            # equation (24)
            d = target_dist - cur_dist
            mu_s = ti.max(self.particles_info_reordered[i, i_b].mu_s, self.particles_info_reordered[j, i_b].mu_s)
            mu_k = ti.max(self.particles_info_reordered[i, i_b].mu_k, self.particles_info_reordered[j, i_b].mu_k)
            if dpos.norm() < mu_s * d:
                self.particles_reordered[i, i_b].dpos += wi / (wi + wj) * dpos
            else:
                self.particles_reordered[i, i_b].dpos += (
                    wi / (wi + wj) * dpos * ti.min(1.0, mu_k * d / dpos.norm(gs.EPS))
                )

    @ti.kernel
    def _kernel_solve_collision(self, f: ti.i32):
        """
        粒子间碰撞：
        - 使用空间哈希在 3x3x3 邻域中遍历候选对
        - 对至少一方为 free 且非液-液 的粒子对进行碰撞处理
        - 应用累计位移增量并清零
        """
        for i_p, i_b in ti.ndrange(self._n_particles, self._B):
            if self.particles_info_reordered[i_p, i_b].material_type != self.MATERIAL.PARTICLE:
                base = self.sh.pos_to_grid(self.particles_reordered[i_p, i_b].pos)
                for offset in ti.grouped(ti.ndrange((-1, 2), (-1, 2), (-1, 2))):
                    slot_idx = self.sh.grid_to_slot(base + offset)
                    for j in range(
                        self.sh.slot_start[slot_idx, i_b],
                        self.sh.slot_size[slot_idx, i_b] + self.sh.slot_start[slot_idx, i_b],
                    ):
                        if (
                            i_p != j
                            and (self.particles_reordered[i_p, i_b].free or self.particles_reordered[j, i_b].free)
                            and not (
                                self.particles_info_reordered[i_p, i_b].material_type == self.MATERIAL.LIQUID
                                and self.particles_info_reordered[j, i_b].material_type == self.MATERIAL.LIQUID
                            )
                        ):
                            self._func_solve_collision(i_p, j, i_b)

        for i_p, i_b in ti.ndrange(self._n_particles, self._B):
            if (
                self.particles_reordered[i_p, i_b].free
                and self.particles_info_reordered[i_p, i_b].material_type != self.MATERIAL.PARTICLE
            ):
                self.particles_reordered[i_p, i_b].pos = (
                    self.particles_reordered[i_p, i_b].pos + self.particles_reordered[i_p, i_b].dpos
                )
                self.particles_reordered[i_p, i_b].dpos.fill(0)

    @ti.kernel
    def _kernel_solve_boundary_collision(self, f: ti.i32):
        """
        边界碰撞：对所有粒子施加边界条件（无论是否 free）。
        - 使用边界对象的 impose_pos_vel 进行位置/速度修正
        """
        for i_p, i_b in ti.ndrange(self._n_particles, self._B):
            # boundary is enforced regardless of whether free
            pos_new, vel_new = self.boundary.impose_pos_vel(self.particles[i_p, i_b].pos, self.particles[i_p, i_b].vel)
            self.particles[i_p, i_b].pos = pos_new
            self.particles[i_p, i_b].vel = vel_new

    @ti.kernel
    def _kernel_solve_density(self, f: ti.i32):
        """
        密度约束（PBF, Position-Based Fluids）：
        - 迭代两阶段：
          1) 计算 λ：λ_i = -C_i / (∑|∇C_i|^2 + ε)，C_i = ρ_i/ρ0 - 1
          2) 计算位置修正：Δp_i = (1/ρ0)∑_j (λ_i+λ_j+s_corr)∇W(r_ij) * density_relaxation
        """
        for _ in ti.static(range(self._max_density_solver_iterations)):
            # ---Calculate lambdas---
            for i_p, i_b in ti.ndrange(self._n_particles, self._B):
                if self.particles_info_reordered[i_p, i_b].material_type == self.MATERIAL.LIQUID:
                    pos_i = self.particles_reordered[i_p, i_b].pos
                    base = self.sh.pos_to_grid(pos_i)
                    lower_sum = gs.ti_float(0.0)
                    rho = gs.ti_float(0.0)
                    spiky_i = ti.Vector.zero(gs.ti_float, 3)
                    for offset in ti.grouped(ti.ndrange((-1, 2), (-1, 2), (-1, 2))):
                        slot_idx = self.sh.grid_to_slot(base + offset)
                        for j in range(
                            self.sh.slot_start[slot_idx, i_b],
                            self.sh.slot_size[slot_idx, i_b] + self.sh.slot_start[slot_idx, i_b],
                        ):
                            pos_j = self.particles_reordered[j, i_b].pos
                            # ---Poly6--- 密度累加
                            rho += self.poly6(pos_i - pos_j) * self.particles_info_reordered[j, i_b].mass
                            # ---Spiky--- 梯度项累加
                            s = self.spiky(pos_i - pos_j) / self.particles_info_reordered[i_p, i_b].rho_rest
                            spiky_i += s
                            lower_sum += s.dot(s)
                    constraint = (rho / self.particles_info_reordered[i_p, i_b].rho_rest) - 1.0
                    lower_sum += spiky_i.dot(spiky_i)
                    self.particles_reordered[i_p, i_b].lam = -1.0 * (constraint / (lower_sum + self.lambda_epsilon))

            # ---Calculate delta pos---
            for i_p, i_b in ti.ndrange(self._n_particles, self._B):
                if self.particles_info_reordered[i_p, i_b].material_type == self.MATERIAL.LIQUID:
                    pos_i = self.particles_reordered[i_p, i_b].pos
                    base = self.sh.pos_to_grid(pos_i)
                    for offset in ti.grouped(ti.ndrange((-1, 2), (-1, 2), (-1, 2))):
                        slot_idx = self.sh.grid_to_slot(base + offset)
                        for j in range(
                            self.sh.slot_start[slot_idx, i_b],
                            self.sh.slot_size[slot_idx, i_b] + self.sh.slot_start[slot_idx, i_b],
                        ):
                            if i_p != j:
                                pos_j = self.particles_reordered[j, i_b].pos
                                # ---S_Corr--- 表面张力修正
                                scorr = self.S_Corr(pos_i - pos_j)
                                left = (
                                    self.particles_reordered[i_p, i_b].lam
                                    + self.particles_reordered[j, i_b].lam
                                    + scorr
                                )
                                right = self.spiky(pos_i - pos_j)
                                self.particles_reordered[i_p, i_b].dpos = (
                                    self.particles_reordered[i_p, i_b].dpos
                                    + left
                                    * right
                                    / self.particles_info_reordered[i_p, i_b].rho_rest
                                    * self.dist_scale
                                    * self.particles_info_reordered[i_p, i_b].density_relaxation
                                )

            for i_p, i_b in ti.ndrange(self._n_particles, self._B):
                if (
                    self.particles_info_reordered[i_p, i_b].material_type == self.MATERIAL.LIQUID
                    and self.particles_reordered[i_p, i_b].free
                ):
                    self.particles_reordered[i_p, i_b].pos = (
                        self.particles_reordered[i_p, i_b].pos + self.particles_reordered[i_p, i_b].dpos
                    )
                    self.particles_reordered[i_p, i_b].dpos.fill(0)

    @ti.kernel
    def _kernel_solve_viscosity(self, f: ti.i32):
        """
        粘度（XSPH 形式）：
        - Δp_i = ∑_j W(r_ij) * v_ij * viscosity_relaxation
        - 使相邻粒子速度更一致，平滑速度场
        - 注：曾加入涡量与数值梯度近似，已注释
        """
        for _ in ti.static(range(self._max_viscosity_solver_iterations)):
            for i_p, i_b in ti.ndrange(self._n_particles, self._B):
                if self.particles_info_reordered[i_p, i_b].material_type == self.MATERIAL.LIQUID:
                    pos_i = self.particles_reordered[i_p, i_b].pos
                    base = self.sh.pos_to_grid(pos_i)
                    xsph_sum = ti.Vector.zero(gs.ti_float, 3)
                    omega_sum = ti.Vector.zero(gs.ti_float, 3)
                    # -For Gradient Approx.-
                    dx_sum = ti.Vector.zero(gs.ti_float, 3)
                    dy_sum = ti.Vector.zero(gs.ti_float, 3)
                    dz_sum = ti.Vector.zero(gs.ti_float, 3)
                    n_dx_sum = ti.Vector.zero(gs.ti_float, 3)
                    n_dy_sum = ti.Vector.zero(gs.ti_float, 3)
                    n_dz_sum = ti.Vector.zero(gs.ti_float, 3)
                    dx = ti.Vector([self.g_del, 0.0, 0.0], dt=gs.ti_float)
                    dy = ti.Vector([0.0, self.g_del, 0.0], dt=gs.ti_float)
                    dz = ti.Vector([0.0, 0.0, self.g_del], dt=gs.ti_float)

                    for offset in ti.grouped(ti.ndrange((-1, 2), (-1, 2), (-1, 2))):
                        slot_idx = self.sh.grid_to_slot(base + offset)
                        for j in range(
                            self.sh.slot_start[slot_idx, i_b],
                            self.sh.slot_size[slot_idx, i_b] + self.sh.slot_start[slot_idx, i_b],
                        ):
                            pos_j = self.particles_reordered[j, i_b].pos
                            v_ij = (self.particles_reordered[j, i_b].pos - self.particles_reordered[j, i_b].ipos) - (
                                self.particles_reordered[i_p, i_b].pos - self.particles_reordered[i_p, i_b].ipos
                            )

                            dist = pos_i - pos_j
                            # ---Vorticity---（保留计算通路，但未生效）
                            omega_sum += v_ij.cross(self.spiky(dist))
                            # -Gradient Approx.- 数值梯度近似（已不启用）
                            dx_sum += v_ij.cross(self.spiky(dist + dx))
                            dy_sum += v_ij.cross(self.spiky(dist + dy))
                            dz_sum += v_ij.cross(self.spiky(dist + dz))
                            n_dx_sum += v_ij.cross(self.spiky(dist - dx))
                            n_dy_sum += v_ij.cross(self.spiky(dist - dy))
                            n_dz_sum += v_ij.cross(self.spiky(dist - dz))
                            # ---Viscosity--- XSPH
                            poly = self.poly6(dist)
                            xsph_sum += poly * v_ij

                    # 位置增量用于平滑速度差
                    self.particles_reordered[i_p, i_b].dpos = (
                        self.particles_reordered[i_p, i_b].dpos
                        + xsph_sum * self.particles_info_reordered[i_p, i_b].viscosity_relaxation
                    )

            for i_p, i_b in ti.ndrange(self._n_particles, self._B):
                if (
                    self.particles_info_reordered[i_p, i_b].material_type == self.MATERIAL.LIQUID
                    and self.particles_reordered[i_p, i_b].free
                ):
                    self.particles_reordered[i_p, i_b].pos = (
                        self.particles_reordered[i_p, i_b].pos + self.particles_reordered[i_p, i_b].dpos
                    )
                    self.particles_reordered[i_p, i_b].dpos.fill(0)

    @ti.kernel
    def _kernel_compute_velocity(self, f: ti.i32):
        "速度更新：v = (pos - ipos) / dt，用约束后位置反推有效速度。"
        for i_p, i_b in ti.ndrange(self._n_particles, self._B):
            self.particles_reordered[i_p, i_b].vel = (
                self.particles_reordered[i_p, i_b].pos - self.particles_reordered[i_p, i_b].ipos
            ) / self._substep_dt

    @ti.kernel
    def _kernel_copy_from_reordered(self, f: ti.i32):
        "将重排数组的结果拷贝回原粒子数组。"
        for i_p, i_b in ti.ndrange(self._n_particles, self._B):
            if self.particles_ng[i_p, i_b].active:
                reordered_idx = self.particles_ng[i_p, i_b].reordered_idx
                self.particles[i_p, i_b] = self.particles_reordered[reordered_idx, i_b]

    # ------------------------------------------------------------------------------------
    # ------------------------------------ stepping --------------------------------------
    # ------------------------------------------------------------------------------------

    def process_input(self, in_backward=False):
        "转发实体输入处理（前向）。"
        for entity in self._entities:
            entity.process_input(in_backward=in_backward)

    def process_input_grad(self):
        "梯度流程暂未接入。"
        pass

    def substep_pre_coupling(self, f):
        """
        子步（耦合前）主管线：
        1) 存初始位置 + 外力预测
        2) 拓扑约束：拉伸/弯曲/体积（可选）
        3) 空间哈希重排
        4) 空间约束：密度/粘度（液体）
        5) 粒子间碰撞
        6) 速度更新（从位置差分）
        """
        if self.is_active():
            self._kernel_store_initial_pos(f)
            self._kernel_apply_external_force(f, self._sim.cur_t)

            # topology constraints (doesn't require spatial hashing)
            if self._n_edges > 0:
                self._kernel_solve_stretch(f)

            if self._n_inner_edges > 0:
                self._kernel_solve_bending(f)

            if self._n_elems > 0:
                self._kernel_solve_volume(f)

            # perform spatial hashing
            self._kernel_reorder_particles(f)

            # spatial constraints
            if self._n_particles > 0:
                self._kernel_solve_density(f)
                self._kernel_solve_viscosity(f)

            self._kernel_solve_collision(f)

            # compute effective velocity
            self._kernel_compute_velocity(f)

    def substep_pre_coupling_grad(self, f):
        "梯度流程暂未接入。"
        pass

    def substep_post_coupling(self, f):
        """
        子步（耦合后）：
        1) 从重排数组回写（便于外部模块访问）
        2) 边界碰撞（统一修正）
        """
        if self.is_active():
            self._kernel_copy_from_reordered(f)

            # boundary collision
            self._kernel_solve_boundary_collision(f)

    def substep_post_coupling_grad(self, f):
        "梯度流程暂未接入。"
        pass

    # ------------------------------------------------------------------------------------
    # ------------------------------------ gradient --------------------------------------
    # ------------------------------------------------------------------------------------

    def collect_output_grads(self):
        "未集成梯度回传。"
        pass

    def add_grad_from_state(self, state):
        "未集成梯度回传。"
        pass

    # ------------------------------------------------------------------------------------
    # --------------------------------------- io -----------------------------------------
    # ------------------------------------------------------------------------------------

    def save_ckpt(self, ckpt_name):
        "检查点（未实现）。"
        pass

    def load_ckpt(self, ckpt_name):
        "检查点恢复（未实现）。"
        pass

    def set_state(self, f, state, envs_idx=None):
        "设置当前帧的粒子状态（位置/速度/free）。"
        if self.is_active():
            self._kernel_set_state(f, state.pos, state.vel, state.free)

    @ti.kernel
    def _kernel_set_state(
        self,
        f: ti.i32,
        pos: ti.types.ndarray(),  # shape [B, _n_particles, 3]
        vel: ti.types.ndarray(),  # shape [B, _n_particles, 3]
        free: ti.types.ndarray(),  # shape [B, _n_particles]
    ):
        "将外部传入的 pos/vel/free 写入到粒子场。"
        for i_p, i_b in ti.ndrange(self._n_particles, self._B):
            for j in ti.static(range(3)):
                self.particles[i_p, i_b].pos[j] = pos[i_b, i_p, j]
                self.particles[i_p, i_b].vel[j] = vel[i_b, i_p, j]
            self.particles[i_p, i_b].free = free[i_b, i_p]

    def get_state(self, f):
        "导出当前帧的状态为 PBDSolverState。"
        if self.is_active():
            state = PBDSolverState(self.scene)
            self._kernel_get_state(f, state.pos, state.vel, state.free)
        else:
            state = None
        return state

    @ti.kernel
    def _kernel_get_state(
        self,
        f: ti.i32,
        pos: ti.types.ndarray(),  # shape [B, _n_particles, 3]
        vel: ti.types.ndarray(),  # shape [B, _n_particles, 3]
        free: ti.types.ndarray(),  # shape [B, _n_particles]
    ):
        "读取粒子 pos/vel/free 到外部数组。"
        for i_p, i_b in ti.ndrange(self._n_particles, self._B):
            for j in ti.static(range(3)):
                pos[i_b, i_p, j] = self.particles[i_p, i_b].pos[j]
                vel[i_b, i_p, j] = self.particles[i_p, i_b].vel[j]
            free[i_b, i_p] = ti.cast(self.particles[i_p, i_b].free, gs.ti_bool)

    def update_render_fields(self):
        "更新渲染缓冲：将粒子与可视化顶点写入渲染结构。"
        self._kernel_update_render_fields(self.sim.cur_substep_local)

    @ti.kernel
    def _kernel_update_render_fields(self, f: ti.i32):
        "渲染字段更新：粒子可见性与可视化顶点加权位置。"
        for i_p, i_b in ti.ndrange(self._n_particles, self._B):
            if self.particles_ng[i_p, i_b].active:
                self.particles_render[i_p, i_b].pos = self.particles[i_p, i_b].pos
                self.particles_render[i_p, i_b].vel = self.particles[i_p, i_b].vel
            else:
                self.particles_render[i_p, i_b].pos = gu.ti_nowhere()
            self.particles_render[i_p, i_b].active = self.particles_ng[i_p, i_b].active

        for i_v, i_b in ti.ndrange(self._n_vverts, self._B):
            vvert_pos = ti.Vector.zero(gs.ti_float, 3)
            for j in range(self._n_vvert_supports):
                vvert_pos += (
                    self.particles[self.vverts_info.support_idxs[i_v][j], i_b].pos
                    * self.vverts_info.support_weights[i_v][j]
                )
            self.vverts_render[i_v, i_b].pos = vvert_pos
            self.vverts_render[i_v, i_b].active = self.particles_render[
                self.vverts_info.support_idxs[i_v][0], i_b
            ].active

    @ti.kernel
    def _kernel_set_particles_pos(
        self,
        particles_idx: ti.types.ndarray(),
        envs_idx: ti.types.ndarray(),
        poss: ti.types.ndarray(),
    ):
        "批量设置部分粒子的位置，并清零其速度。"
        for i_p_, i_b_ in ti.ndrange(particles_idx.shape[1], envs_idx.shape[0]):
            i_p = particles_idx[i_b_, i_p_]
            i_b = envs_idx[i_b_]
            for i in ti.static(range(3)):
                self.particles[i_p, i_b].pos[i] = poss[i_b_, i_p_, i]
            self.particles[i_p, i_b].vel.fill(0.0)

    @ti.kernel
    def _kernel_get_particles_pos(
        self,
        particle_start: ti.i32,
        n_particles: ti.i32,
        envs_idx: ti.types.ndarray(),
        poss: ti.types.ndarray(),
    ):
        "批量读取部分粒子的位置。"
        for i_p_, i_b_ in ti.ndrange(n_particles, envs_idx.shape[0]):
            i_p = i_p_ + particle_start
            i_b = envs_idx[i_b_]
            for i in ti.static(range(3)):
                poss[i_b_, i_p_, i] = self.particles[i_p, i_b].pos[i]

    @ti.kernel
    def _kernel_set_particles_vel(
        self,
        particles_idx: ti.types.ndarray(),
        envs_idx: ti.types.ndarray(),
        vels: ti.types.ndarray(),
    ):
        "批量设置部分粒子的速度。"
        for i_p_, i_b_ in ti.ndrange(particles_idx.shape[1], envs_idx.shape[0]):
            i_p = particles_idx[i_b_, i_p_]
            i_b = envs_idx[i_b_]
            for i in ti.static(range(3)):
                self.particles[i_p, i_b].vel[i] = vels[i_b_, i_p_, i]

    @gs.assert_built
    def set_animate_particles_by_link(
        self,
        particles_idx: NDArray[np.int32],
        link_idx: int,
        links_state: LinksState,
        envs_idx: NDArray[np.int32] | None = None,
    ) -> None:
        """
        将一组粒子绑定到某个刚体 link，用于动画/驱动。
        - 内部通过耦合器 kernel_attach_pbd_to_rigid_link 实现
        """
        envs_idx = self._scene._sanitize_envs_idx(envs_idx)
        self._sim._coupler.kernel_attach_pbd_to_rigid_link(particles_idx, envs_idx, link_idx, links_state)

    @ti.kernel
    def _kernel_get_particles_vel(
        self,
        particle_start: ti.i32,
        n_particles: ti.i32,
        envs_idx: ti.types.ndarray(),
        vels: ti.types.ndarray(),
    ):
        "批量读取部分粒子的速度。"
        for i_p_, i_b_ in ti.ndrange(n_particles, envs_idx.shape[0]):
            i_p = i_p_ + particle_start
            i_b = envs_idx[i_b_]
            for i in ti.static(range(3)):
                vels[i_b_, i_p_, i] = self.particles[i_p, i_b].vel[i]

    @ti.kernel
    def _kernel_set_particles_active(
        self,
        particles_idx: ti.types.ndarray(),
        envs_idx: ti.types.ndarray(),
        actives: ti.types.ndarray(),  # shape [B, n_particles]
    ):
        "批量设置粒子激活状态。"
        for i_p_, i_b_ in ti.ndrange(particles_idx.shape[1], envs_idx.shape[0]):
            i_p = particles_idx[i_b_, i_p_]
            i_b = envs_idx[i_b_]
            self.particles_ng[i_p, i_b].active = ti.cast(actives[i_b_, i_p_], gs.ti_bool)

    @ti.kernel
    def _kernel_get_particles_active(
        self,
        particle_start: ti.i32,
        n_particles: ti.i32,
        envs_idx: ti.types.ndarray(),
        actives: ti.types.ndarray(),  # shape [B, n_particles]
    ):
        "批量读取粒子激活状态。"
        for i_p_, i_b_ in ti.ndrange(n_particles, envs_idx.shape[0]):
            i_p = i_p_ + particle_start
            i_b = envs_idx[i_b_]
            actives[i_b_, i_p_] = self.particles_ng[i_p, i_b].active

    @ti.kernel
    def _kernel_fix_particles(self, particles_idx: ti.types.ndarray(), envs_idx: ti.types.ndarray()):
        "批量固定粒子（free=False）。"
        for i_p_, i_b_ in ti.ndrange(particles_idx.shape[1], envs_idx.shape[0]):
            i_p = particles_idx[i_b_, i_p_]
            i_b = envs_idx[i_b_]
            self.particles[i_p, i_b].free = False

    @ti.kernel
    def _kernel_release_particle(self, particles_idx: ti.types.ndarray(), envs_idx: ti.types.ndarray()):
        "批量释放粒子（free=True）。"
        for i_p_, i_b_ in ti.ndrange(particles_idx.shape[1], envs_idx.shape[0]):
            i_p = particles_idx[i_b_, i_p_]
            i_b = envs_idx[i_b_]
            self.particles[i_p, i_b].free = True

    @ti.kernel
    def _kernel_get_mass(
        self, particle_start: ti.i32, n_particles: ti.i32, mass: ti.types.ndarray(), envs_idx: ti.types.ndarray()
    ):
        "计算一段粒子的总质量（对所有 env 相同）。"
        total_mass = gs.ti_float(0.0)
        for i_p_ in range(n_particles):
            i_p = i_p_ + particle_start
            total_mass += self.particles_info[i_p].mass
        for i_b_ in range(envs_idx.shape[0]):
            mass[i_b_] = total_mass

    # ------------------------------------------------------------------------------------
    # ----------------------------------- properties -------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def n_particles(self):
        "粒子总数。"
        if self.is_built:
            return self._n_particles
        else:
            return sum([entity.n_particles for entity in self._entities])

    @property
    def n_fluid_particles(self):
        "液体粒子数量。"
        if self.is_built:
            return self._n_fluid_particles
        else:
            return sum(
                [entity.n_fluid_particles if hasattr(entity, "n_fluid_particles") else 0 for entity in self._entities]
            )

    @property
    def n_edges(self):
        "边数量（用于拉伸约束）。"
        if self.is_built:
            return self._n_edges
        else:
            return sum([entity.n_edges if hasattr(entity, "n_edges") else 0 for entity in self._entities])

    @property
    def n_inner_edges(self):
        "内边数量（用于弯曲约束）。"
        if self.is_built:
            return self._n_inner_edges
        else:
            return sum([entity.n_inner_edges if hasattr(entity, "n_inner_edges") else 0 for entity in self._entities])

    @property
    def n_elems(self):
        "体单元数量（用于体积约束）。"
        if self.is_built:
            return self._n_elems
        else:
            return sum([entity.n_elems if hasattr(entity, "n_elems") else 0 for entity in self._entities])

    @property
    def n_vverts(self):
        "可视化顶点数量。"
        if self.is_built:
            return self._n_vverts
        else:
            return sum([entity.n_vverts if hasattr(entity, "n_vverts") else 0 for entity in self._entities])

    @property
    def n_vfaces(self):
        "可视化面数量。"
        if self.is_built:
            return self._n_vfaces
        else:
            return sum([entity.n_vfaces if hasattr(entity, "n_vfaces") else 0 for entity in self._entities])

    @property
    def particle_size(self):
        "粒子直径。"
        return self._particle_size

    @property
    def particle_radius(self):
        "粒子半径。"
        return self._particle_size / 2.0

    @property
    def hash_grid_res(self):
        "空间哈希的网格分辨率。"
        return self.sh.grid_res

    @property
    def hash_grid_cell_size(self):
        "空间哈希单元尺寸。"
        return self.sh.cell_size

    @property
    def upper_bound(self):
        "边界上界。"
        return self._upper_bound

    @property
    def lower_bound(self):
        "边界下界。"
        return self._lower_bound