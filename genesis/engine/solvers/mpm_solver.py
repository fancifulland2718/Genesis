from typing import TYPE_CHECKING

import numpy as np
import gstaichi as ti
import torch

import genesis as gs
import genesis.utils.geom as gu
import genesis.utils.sdf_decomp as sdf_decomp
from genesis.engine.boundaries import CubeBoundary
from genesis.engine.entities import MPMEntity
from genesis.engine.states.solvers import MPMSolverState
from genesis.options.solvers import MPMOptions
from genesis.utils.misc import DeprecationError

from .base_solver import Solver

if TYPE_CHECKING:
    from genesis.engine.scene import Scene
    from genesis.engine.solvers.base_solver import Solver
    from genesis.engine.simulator import Simulator


@ti.data_oriented
class MPMSolver(Solver):
    """
    MPM（Material Point Method）求解器。
    作用：
    - 管理 MPM 粒子/网格的数据结构与生命周期（构建、状态更新、渲染字段）
    - 实现每个子步的核心仿真管线：
      1) compute_F_tmp：用仿射速度场 C 更新形变梯度的临时值 F_tmp
      2) svd：对 F_tmp 做 SVD，得到 U/S/V
      3) p2g：粒子到网格（P2G）投影，计算应力并累加至网格质量/动量
      4) （耦合/边界/外力在网格上处理，更新 vel_out）
      5) g2p：网格到粒子（G2P）回传，更新粒子速度/位置/仿射场
    - 支持与刚体求解器的耦合（包括 CPIC 分离策略）
    - 支持可微：提供梯度收集与回传（svd_grad、p2g.grad、g2p.grad 等）

    复杂点说明：
    - 粒子体积缩放 _particle_volume_scale：数值上放大粒子体积，用于避免质量过小引发的不稳定；
      MPM 物理本身不依赖粒子体积的绝对量纲，但质量会依赖该值，跨求解器耦合时需成对考虑缩放。
    - CPIC（Convective Particle-In-Cell）模式：在薄物体两侧的单元与粒子间分离时，避免将动量错误投影到“另一侧”；
      通过粒子与网格单元中心的 SDF 法线点积判断是否被同一几何体隔开，标记在 coupler.cpic_flag，
      进而在 G2P 用碰撞速度替代网格速度，保证物理一致性（当前仅非可微模式支持）。
    """

    # ------------------------------------------------------------------------------------
    # --------------------------------- Initialization -----------------------------------
    # ------------------------------------------------------------------------------------

    def __init__(self, scene: "Scene", sim: "Simulator", options: "MPMOptions"):
        """
        参数：
        - scene：场景对象
        - sim：仿真器（提供批大小 B、子步数等）
        - options：MPMOptions，包含网格密度、边界、粒子尺寸、是否开启 CPIC 等
        """
        super().__init__(scene, sim, options)

        # options
        self._grid_density = options.grid_density
        self._particle_size = options.particle_size
        self._upper_bound = np.array(options.upper_bound)
        self._lower_bound = np.array(options.lower_bound)
        self._enable_CPIC = options.enable_CPIC

        self._n_vvert_supports = self.scene.vis_options.n_support_neighbors

        # `_particle_volume_scale` 用于避免质量过小导致的不稳定。
        # 注意：粒子体积的绝对大小不会影响 MPM 本体，但质量 = 体积 * 密度，会受其影响。
        # 在耦合（与刚体等）时，相关动量/冲量计算需要考虑该缩放。
        self._particle_volume_real = float(self._particle_size**3)
        self._particle_volume_scale = 1e3
        self._particle_volume = self._particle_volume_real * self._particle_volume_scale

        # 派生网格参数
        self._dx = float(1.0 / self._grid_density)
        self._inv_dx = float(self._grid_density)
        self._lower_bound_cell = np.round(self._grid_density * self._lower_bound).astype(gs.np_int)
        self._upper_bound_cell = np.round(self._grid_density * self._upper_bound).astype(gs.np_int)
        self._grid_res = self._upper_bound_cell - self._lower_bound_cell + 1  # +1 保含两侧端点
        self._grid_offset = ti.Vector(self._lower_bound_cell)
        if np.prod(self._grid_res) > 1e9:
            gs.raise_exception(
                "Grid size larger than 1e9 not supported by MPM solver. Please reduce 'grid_density', or set tighter "
                "boundaries via 'lower_bound' / 'upper_bound'."
            )

        # 材料（按“引用去重”注册），保存其更新函数
        self._materials = list()
        self._materials_idx = list()
        self._materials_update_F_S_Jp = list()
        self._materials_update_stress = list()

        # 边界
        self.setup_boundary()

    def _batch_shape(self, shape=None, first_dim=False, B=None):
        """
        构造带批次维度 B 的形状工具。
        - shape 为 None：返回 (B,)
        - shape 为序列：根据 first_dim 决定 B 在前或在后
        - shape 为整数：同上
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
        设置 MPM 场景的边界（立方体），并添加安全 padding，防止数值爆炸时访问越界单元。
        """
        # safety padding
        self.boundary_padding = 3 * self._dx
        self.boundary = CubeBoundary(
            lower=self._lower_bound + self.boundary_padding,
            upper=self._upper_bound - self.boundary_padding,
        )

    def init_particle_fields(self):
        """
        初始化粒子相关字段：
        - particles（可导）：动态状态 pos/vel/C/F/F_tmp/U/V/S/actu/Jp
        - particles_ng（不可导）：active 掩码
        - particles_info（静态信息）：材料索引、质量、默认 Jp、是否自由粒子、肌肉参数
        - particles_render：渲染用单帧状态（pos/vel/active）
        """
        # dynamic particle state
        struct_particle_state = ti.types.struct(
            pos=gs.ti_vec3,  # 位置
            vel=gs.ti_vec3,  # 速度
            C=gs.ti_mat3,  # 仿射速度场（APIC）
            F=gs.ti_mat3,  # 形变梯度
            F_tmp=gs.ti_mat3,  # 用于 SVD 的临时 F
            U=gs.ti_mat3,  # SVD
            V=gs.ti_mat3,  # SVD
            S=gs.ti_mat3,  # SVD
            actu=gs.ti_float,  # 肌肉激活
            Jp=gs.ti_float,  # 体积比（塑性）
        )

        # dynamic particle state without gradient
        struct_particle_state_ng = ti.types.struct(
            active=gs.ti_bool,  # 活跃标志
        )

        # static particle info
        struct_particle_info = ti.types.struct(
            material_idx=gs.ti_int,
            mass=gs.ti_float,
            default_Jp=gs.ti_float,
            free=gs.ti_bool,
            # for muscle
            muscle_group=gs.ti_int,
            muscle_direction=gs.ti_vec3,
        )

        # single frame particle state for rendering
        struct_particle_state_render = ti.types.struct(
            pos=gs.ti_vec3,
            vel=gs.ti_vec3,
            active=gs.ti_bool,
        )

        # 构造字段（注意：时间维度使用 substeps_local+1 帧）
        self.particles = struct_particle_state.field(
            shape=self._batch_shape((self._sim.substeps_local + 1, self._n_particles)),
            needs_grad=True,
            layout=ti.Layout.SOA,
        )
        self.particles_ng = struct_particle_state_ng.field(
            shape=self._batch_shape((self._sim.substeps_local + 1, self._n_particles)),
            needs_grad=False,
            layout=ti.Layout.SOA,
        )
        self.particles_info = struct_particle_info.field(
            shape=self._n_particles, needs_grad=False, layout=ti.Layout.SOA
        )
        self.particles_render = struct_particle_state_render.field(
            shape=self._batch_shape(self._n_particles), needs_grad=False, layout=ti.Layout.SOA
        )

    def init_grid_fields(self):
        """
        初始化网格字段：
        - mass：质量累积
        - vel_in：P2G 后的输入动量/速度
        - vel_out：网格阶段（边界/耦合/外力）处理后的输出速度
        """
        grid_cell_state = ti.types.struct(
            mass=gs.ti_float,  # 质量
            vel_in=gs.ti_vec3,  # 输入动量/速度
            vel_out=gs.ti_vec3,  # 输出动量/速度
        )
        self.grid = grid_cell_state.field(
            shape=self._batch_shape((self._sim.substeps_local + 1, *self._grid_res)),
            needs_grad=True,
            layout=ti.Layout.SOA,
        )

    def init_vvert_fields(self):
        """
        初始化可视化顶点（vverts）辅助字段：
        - vverts_info：每个可视化顶点由若干粒子支持及其权重线性插值得到
        - vverts_render：渲染用 vvert 的位置与活跃标志
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
            shape=self._batch_shape(max(1, self._n_vverts)), layout=ti.Layout.SOA
        )

    def init_ckpt(self):
        """初始化检查点缓存（仅在可微模式下使用）。"""
        self._ckpt = dict()

    def reset_grad(self):
        """清零本求解器自身的梯度，并递归清零实体梯度。"""
        self.particles.grad.fill(0.0)
        self.grid.grad.fill(0.0)

        for entity in self._entities:
            entity.reset_grad()

    def build(self):
        """
        构建阶段：
        - 统计总粒子/可视化顶点/面数
        - 初始化字段与检查点
        - 将实体加入求解器
        - 给出稳定性建议（substep_dt 与 dx 的关系）
        - CPIC 模式提示与限制（当前不可微）
        """
        super().build()

        # particles and entities
        self._B = self._sim._B
        self._n_particles = self.n_particles
        self._n_vverts = self.n_vverts
        self._n_vfaces = self.n_vfaces

        self._coupler = self.sim._coupler

        if self.is_active():
            if self._enable_CPIC:
                gs.logger.warning(
                    "Kernel compilation takes longer when running MPM solver in CPIC mode. Please be patient."
                )
                if self._sim.requires_grad:
                    gs.raise_exception(
                        "CPIC is not supported in differentiable mode yet. Submit a feature request if you need it."
                    )

            self.init_particle_fields()
            self.init_grid_fields()
            self.init_vvert_fields()
            self.init_ckpt()

            for entity in self._entities:
                entity._add_to_solver()

            # 经验建议的 dt（参照 taichi elements）
            suggested_dt = 2e-2 * self._dx
            if self.substep_dt > suggested_dt:
                gs.logger.warning(
                    f"Current `substep_dt` ({self.substep_dt:.6g}) is greater than suggested_dt ({suggested_dt:.6g}, "
                    "calculated based on `grid_density`). Simulation might be unstable."
                )

    # ------------------------------------------------------------------------------------
    # -------------------------------------- misc ----------------------------------------
    # ------------------------------------------------------------------------------------

    def add_entity(self, idx, material, morph, surface):
        """
        添加一个 MPM 实体，并注册其材料。
        返回：创建的 MPMEntity 实例。
        """
        self.add_material(material)

        # create entity
        entity = MPMEntity(
            scene=self._scene,
            solver=self,
            material=material,
            morph=morph,
            surface=surface,
            particle_size=self._particle_size,
            idx=idx,
            particle_start=self.n_particles,
            vvert_start=self.n_vverts,
            vface_start=self.n_vfaces,
        )
        self._entities.append(entity)

        return entity

    def add_material(self, material):
        """
        注册材料（去重）：
        - 若已注册，则复用其 _idx
        - 否则追加 material 并登记 update_F_S_Jp 与 update_stress 回调
        """
        # Register material update methods if and only if the provided material is not already registered
        for material_i in self._materials:
            if material == material_i:
                material._idx = material_i._idx
                break
        else:
            material._idx = len(self._materials_idx)
            self._materials_idx.append(material._idx)
            self._materials_update_F_S_Jp.append(material.update_F_S_Jp)
            self._materials_update_stress.append(material.update_stress)
        self._materials.append(material)

    def is_active(self):
        """是否有粒子（至少一个）"""
        return self.n_particles > 0

    @ti.func
    def stencil_range(self):
        """3x3x3 邻域的偏移范围（用于 P2G/G2P 权重遍历）。"""
        return ti.ndrange(3, 3, 3)

    # ------------------------------------------------------------------------------------
    # ----------------------------------- simulation -------------------------------------
    # ------------------------------------------------------------------------------------

    @ti.kernel
    def compute_F_tmp(self, f: ti.i32):
        """计算 F_tmp = (I + dt*C) @ F，为后续 SVD 做准备。"""
        for i_p, i_b in ti.ndrange(self._n_particles, self._B):
            if self.particles_ng[f, i_p, i_b].active:
                self.particles[f, i_p, i_b].F_tmp = (
                    ti.Matrix.identity(gs.ti_float, 3) + self.substep_dt * self.particles[f, i_p, i_b].C
                ) @ self.particles[f, i_p, i_b].F

    @ti.kernel
    def svd(self, f: ti.i32):
        """对 F_tmp 做奇异值分解，得到 U S V。"""
        for i_p, i_b in ti.ndrange(self._n_particles, self._B):
            if self.particles_ng[f, i_p, i_b].active:
                self.particles[f, i_p, i_b].U, self.particles[f, i_p, i_b].S, self.particles[f, i_p, i_b].V = ti.svd(
                    self.particles[f, i_p, i_b].F_tmp, gs.ti_float
                )

    @ti.kernel
    def svd_grad(self, f: ti.i32):
        """SVD 的反向传播：将 U/S/V 的梯度回传给 F_tmp。"""
        for i_p, i_b in ti.ndrange(self._n_particles, self._B):
            if self.particles_ng[f, i_p, i_b].active:
                self.particles.grad[f, i_p, i_b].F_tmp += backward_svd(
                    self.particles.grad[f, i_p, i_b].U,
                    self.particles.grad[f, i_p, i_b].S,
                    self.particles.grad[f, i_p, i_b].V,
                    self.particles[f, i_p, i_b].U,
                    self.particles[f, i_p, i_b].S,
                    self.particles[f, i_p, i_b].V,
                )

    @ti.kernel
    def p2g(self, f: ti.i32):
        """
        粒子到网格（P2G）投影：
        管线（逐粒子）：
        A) 更新材料相关状态：
           - 由材料回调更新形变梯度 F_new、奇异值 S_new 与 Jp_new（体积比/塑性）
           - 将 F_new/Jp_new 写至下一帧 f+1（显式推进）
        B) 计算应力：
           - 对弹性/塑性材料一般使用 F_new；
             但对粘性液体（mu>0）理论上应基于 F_tmp（重置为单位阵前的形变），本实现保留两者以适配不同材料
           - 由材料回调给出 Piola-Kirchhoff 应力并转为网格上的等效力（乘子包含 dt、体积、dx 等系数）
           - 构造仿射项 affine = stress + m*C（APIC）
        C) 投影到网格（3x3x3 B-spline 权重）：
           - 计算 base 与 fx（三次 B-spline 权重）
           - 对每个 offset 计算 dpos/weight，累加至网格 vel_in 与 mass
           - CPIC 分离：若粒子与单元中心被某薄物体分隔（法线点积<0），则跳过该单元的投影，避免穿透能量泄漏
           - 非自由粒子（free=False）作为边界条件，强制 vel_in=0
        """
        for i_p, i_b in ti.ndrange(self._n_particles, self._B):
            if self.particles_ng[f, i_p, i_b].active:
                # A. update F (deformation gradient), S (Sigma from SVD(F), essentially represents volume) and Jp
                # (volume compression ratio) based on material type
                J = self.particles[f, i_p, i_b].S.determinant()
                F_new = ti.Matrix.zero(gs.ti_float, 3, 3)
                S_new = ti.Matrix.zero(gs.ti_float, 3, 3)
                Jp_new = gs.ti_float(1.0)
                for material_idx in ti.static(self._materials_idx):
                    if self.particles_info[i_p].material_idx == material_idx:
                        F_new, S_new, Jp_new = self._materials_update_F_S_Jp[material_idx](
                            J=J,
                            F_tmp=self.particles[f, i_p, i_b].F_tmp,
                            U=self.particles[f, i_p, i_b].U,
                            S=self.particles[f, i_p, i_b].S,
                            V=self.particles[f, i_p, i_b].V,
                            Jp=self.particles[f, i_p, i_b].Jp,
                        )
                self.particles[f + 1, i_p, i_b].F = F_new
                self.particles[f + 1, i_p, i_b].Jp = Jp_new

                # B. compute stress
                # NOTE:
                # 1. Here we pass in both F_tmp and the updated F_new because in the official taichi example, F_new is
                # used for stress computation. However, although this works for both elastic and elasto-plastic
                # materials, it is mathematically incorrect for liquid material with non-zero viscosity (mu). In the
                # latter case, stress computation needs to be based on the F_tmp (deformation gradient before resetting
                # to identity).
                # 2. Jp is only used by Snow material, and it uses Jp from the previous frame, not the updated one.
                stress = ti.Matrix.zero(gs.ti_float, 3, 3)
                for material_idx in ti.static(self._materials_idx):
                    if self.particles_info[i_p].material_idx == material_idx:
                        stress = self._materials_update_stress[material_idx](
                            U=self.particles[f, i_p, i_b].U,
                            S=S_new,
                            V=self.particles[f, i_p, i_b].V,
                            F_tmp=self.particles[f, i_p, i_b].F_tmp,
                            F_new=F_new,
                            J=J,
                            Jp=self.particles[f, i_p, i_b].Jp,
                            actu=self.particles[f, i_p, i_b].actu,
                            m_dir=self.particles_info[i_p].muscle_direction,
                        )
                # 将应力转化为等效对网格的贡献；APIC 仿射动量项
                stress = (-self.substep_dt * self._particle_volume * 4 * self._inv_dx * self._inv_dx) * stress
                affine = stress + self.particles_info[i_p].mass * self.particles[f, i_p, i_b].C

                # C. project onto grid
                base = ti.floor(self.particles[f, i_p, i_b].pos * self._inv_dx - 0.5).cast(gs.ti_int)
                fx = self.particles[f, i_p, i_b].pos * self._inv_dx - base.cast(gs.ti_float)
                w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
                for offset in ti.static(ti.grouped(self.stencil_range())):
                    dpos = (offset.cast(gs.ti_float) - fx) * self._dx
                    weight = gs.ti_float(1.0)
                    for d in ti.static(range(3)):
                        weight *= w[offset[d]][d]

                    sep_geom_idx = -1
                    if ti.static(self._enable_CPIC and self.sim.rigid_solver.is_active()):
                        # 若被薄物体分隔：粒子和单元中心处的 SDF 法线点积 < 0
                        cell_pos = (base + offset) * self._dx

                        for i_g in range(self.sim.rigid_solver.n_geoms):
                            if self.sim.rigid_solver.geoms_info.needs_coup[i_g]:
                                sdf_normal_particle = self._coupler.mpm_rigid_normal[i_p, i_g, i_b]
                                sdf_normal_cell = sdf_decomp.sdf_func_normal_world(
                                    geoms_state=self.sim.rigid_solver.geoms_state,
                                    geoms_info=self.sim.rigid_solver.geoms_info,
                                    collider_static_config=self.sim.rigid_solver.collider._collider_static_config,
                                    sdf_info=self.sim.rigid_solver.sdf._sdf_info,
                                    pos_world=cell_pos,
                                    geom_idx=i_g,
                                    batch_idx=i_b,
                                )

                                if sdf_normal_particle.dot(sdf_normal_cell) < 0:  # separated by geom i_g
                                    sep_geom_idx = i_g
                                    break
                        self._coupler.cpic_flag[i_p, offset[0], offset[1], offset[2], i_b] = sep_geom_idx
                    if sep_geom_idx == -1:
                        self.grid[f, base - self._grid_offset + offset, i_b].vel_in += weight * (
                            self.particles_info[i_p].mass * self.particles[f, i_p, i_b].vel + affine @ dpos
                        )
                        self.grid[f, base - self._grid_offset + offset, i_b].mass += (
                            weight * self.particles_info[i_p].mass
                        )

                    # 非自由粒子作为边界条件：强制单元速度为零
                    if not self.particles_info[i_p].free:  # non-free particles behave as boundary conditions
                        self.grid[f, base - self._grid_offset + offset, i_b].vel_in = ti.Vector.zero(gs.ti_float, 3)

    @ti.kernel
    def g2p(self, f: ti.i32):
        """
        网格到粒子（G2P）回传：
        - 基于 vel_out 回传速度与仿射场 C，并更新粒子位置
        - 安全边界：对 pos/vel 进行边界约束，防止越界访问
        - CPIC：若该 offset 与粒子被薄体分隔，则用耦合器的碰撞速度替代网格速度
        """
        for i_p, i_b in ti.ndrange(self._n_particles, self._B):
            if self.particles_ng[f, i_p, i_b].active:
                base = ti.floor(self.particles[f, i_p, i_b].pos * self._inv_dx - 0.5).cast(gs.ti_int)
                fx = self.particles[f, i_p, i_b].pos * self._inv_dx - base.cast(gs.ti_float)
                w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
                new_vel = ti.Vector.zero(gs.ti_float, 3)
                new_C = ti.Matrix.zero(gs.ti_float, 3, 3)
                for offset in ti.static(ti.grouped(self.stencil_range())):
                    dpos = offset.cast(gs.ti_float) - fx
                    grid_vel = self.grid[f, base - self._grid_offset + offset, i_b].vel_out
                    weight = gs.ti_float(1.0)
                    for d in ti.static(range(3)):
                        weight *= w[offset[d]][d]

                    if ti.static(self._enable_CPIC and self.sim.rigid_solver.is_active()):
                        sep_geom_idx = self._coupler.cpic_flag[i_p, offset[0], offset[1], offset[2], i_b]
                        if sep_geom_idx != -1:
                            # 若被分隔，使用刚体碰撞速度（动量守恒、避免穿透）
                            grid_vel = self.sim.coupler._func_collide_in_rigid_geom(
                                self.particles[f, i_p, i_b].pos,
                                self.particles[f, i_p, i_b].vel,
                                self.particles_info[i_p].mass * weight / self._particle_volume_scale,
                                self._coupler.mpm_rigid_normal[i_p, sep_geom_idx, i_b],
                                1.0,
                                sep_geom_idx,
                                i_b,
                            )

                    new_vel += weight * grid_vel
                    new_C += 4 * self._inv_dx * weight * grid_vel.outer_product(dpos)

                # 位置更新（显式欧拉），并强制边界
                new_pos = self.particles[f, i_p, i_b].pos + self.substep_dt * new_vel
                new_pos, new_vel = self.boundary.impose_pos_vel(new_pos, new_vel)

                # 写入下一帧
                self.particles[f + 1, i_p, i_b].vel = new_vel
                self.particles[f + 1, i_p, i_b].C = new_C
                self.particles[f + 1, i_p, i_b].pos = new_pos

            else:
                # 非活跃粒子：直接拷贝上一帧（避免未初始化）
                self.particles[f + 1, i_p, i_b].vel = self.particles[f, i_p, i_b].vel
                self.particles[f + 1, i_p, i_b].pos = self.particles[f, i_p, i_b].pos
                self.particles[f + 1, i_p, i_b].C = self.particles[f, i_p, i_b].C
                self.particles[f + 1, i_p, i_b].F = self.particles[f, i_p, i_b].F
                self.particles[f + 1, i_p, i_b].Jp = self.particles[f, i_p, i_b].Jp

            self.particles_ng[f + 1, i_p, i_b].active = self.particles_ng[f, i_p, i_b].active

    # ------------------------------------------------------------------------------------
    # ------------------------------------ stepping --------------------------------------
    # ------------------------------------------------------------------------------------

    def process_input(self, in_backward=False):
        """转发输入处理到每个实体；in_backward 表示是否处于反向流程。"""
        for entity in self._entities:
            entity.process_input(in_backward=in_backward)

    def process_input_grad(self):
        """输入梯度处理（反向），按实体倒序遍历以保证依赖顺序。"""
        for entity in self._entities[::-1]:
            entity.process_input_grad()

    def substep_pre_coupling(self, f):
        """
        子步的前耦合阶段（Forward）：
        - 重置网格与梯度（当前帧）
        - 计算 F_tmp、SVD
        - P2G 投影（得到 vel_in/mass，随后在网格上进行耦合/边界处理输出 vel_out）
        """
        self.reset_grid_and_grad(f)
        self.compute_F_tmp(f)
        self.svd(f)
        self.p2g(f)

    def substep_pre_coupling_grad(self, f):
        """
        子步的前耦合阶段（Backward）：
        - 依次回传 p2g、svd、compute_F_tmp 的梯度
        """
        self.p2g.grad(f)
        self.svd_grad(f)
        self.compute_F_tmp.grad(f)

    def substep_post_coupling(self, f):
        """
        子步的后耦合阶段（Forward）：
        - 在网格上完成耦合/边界处理（由外部完成 vel_out）后，执行 G2P 回传到粒子。
        """
        self.g2p(f)

    def substep_post_coupling_grad(self, f):
        """子步的后耦合阶段（Backward）：对 G2P 进行梯度回传。"""
        self.g2p.grad(f)

    @ti.kernel
    def copy_frame(self, source: ti.i32, target: ti.i32):
        """拷贝指定两帧的粒子状态（pos/vel/F/C/Jp/active）。"""
        for i_p, i_b in ti.ndrange(self._n_particles, self._B):
            self.particles[target, i_p, i_b].pos = self.particles[source, i_p, i_b].pos
            self.particles[target, i_p, i_b].vel = self.particles[source, i_p, i_b].vel
            self.particles[target, i_p, i_b].F = self.particles[source, i_p, i_b].F
            self.particles[target, i_p, i_b].C = self.particles[source, i_p, i_b].C
            self.particles[target, i_p, i_b].Jp = self.particles[source, i_p, i_b].Jp

            self.particles_ng[target, i_p, i_b].active = self.particles_ng[source, i_p, i_b].active

    @ti.kernel
    def copy_grad(self, source: ti.i32, target: ti.i32):
        """拷贝指定两帧的粒子梯度（pos/vel/F/C/Jp/active）。"""
        for i_p, i_b in ti.ndrange(self._n_particles, self._B):
            self.particles.grad[target, i_p, i_b].pos = self.particles.grad[source, i_p, i_b].pos
            self.particles.grad[target, i_p, i_b].vel = self.particles.grad[source, i_p, i_b].vel
            self.particles.grad[target, i_p, i_b].F = self.particles.grad[source, i_p, i_b].F
            self.particles.grad[target, i_p, i_b].C = self.particles.grad[source, i_p, i_b].C
            self.particles.grad[target, i_p, i_b].Jp = self.particles.grad[source, i_p, i_b].Jp
            self.particles_ng[target, i_p, i_b].active = self.particles_ng[source, i_p, i_b].active

    @ti.kernel
    def reset_grid_and_grad(self, f: ti.i32):
        """将当前帧 f 的网格（及其梯度）全部清零，避免跨帧污染。"""
        # Zero out the grid at frame f for *all* grid cells and *all* batch indices
        for i, j, k, i_b in ti.ndrange(*self._grid_res, self._B):
            self.grid[f, i, j, k, i_b].vel_in = ti.Vector.zero(gs.ti_float, 3)
            self.grid[f, i, j, k, i_b].mass = gs.ti_float(0.0)
            self.grid[f, i, j, k, i_b].vel_out = ti.Vector.zero(gs.ti_float, 3)

            self.grid.grad[f, i, j, k, i_b].vel_in = ti.Vector.zero(gs.ti_float, 3)
            self.grid.grad[f, i, j, k, i_b].mass = gs.ti_float(0.0)
            self.grid.grad[f, i, j, k, i_b].vel_out = ti.Vector.zero(gs.ti_float, 3)

    @ti.kernel
    def reset_grad_till_frame(self, f: ti.i32):
        """将 [0, f-1] 帧的粒子梯度清零（用于从某个时间点重启反传）。"""
        # Zero out particle grads in frames [0, f-1], for all particles, all batch indices
        for i_f, i_p, i_b in ti.ndrange(f, self._n_particles, self._B):
            self.particles.grad[i_f, i_p, i_b].pos = ti.Vector.zero(gs.ti_float, 3)
            self.particles.grad[i_f, i_p, i_b].vel = ti.Vector.zero(gs.ti_float, 3)
            self.particles.grad[i_f, i_p, i_b].C = ti.Matrix.zero(gs.ti_float, 3, 3)
            self.particles.grad[i_f, i_p, i_b].F = ti.Matrix.zero(gs.ti_float, 3, 3)
            self.particles.grad[i_f, i_p, i_b].F_tmp = ti.Matrix.zero(gs.ti_float, 3, 3)
            self.particles.grad[i_f, i_p, i_b].Jp = gs.ti_float(0.0)
            self.particles.grad[i_f, i_p, i_b].U = ti.Matrix.zero(gs.ti_float, 3, 3)
            self.particles.grad[i_f, i_p, i_b].V = ti.Matrix.zero(gs.ti_float, 3, 3)
            self.particles.grad[i_f, i_p, i_b].S = ti.Matrix.zero(gs.ti_float, 3, 3)

    # ------------------------------------------------------------------------------------
    # ------------------------------------ gradient --------------------------------------
    # ------------------------------------------------------------------------------------

    def collect_output_grads(self):
        """收集由下游查询的状态所回传的梯度，并分发给实体。"""
        for entity in self._entities:
            entity.collect_output_grads()

    def add_grad_from_state(self, state):
        """
        从外部状态对象累加梯度到当前帧粒子梯度。
        要求 state 张量为连续内存（assert_contiguous）。
        """
        if self.is_active():
            if state.pos.grad is not None:
                state.pos.assert_contiguous()
                self.add_grad_from_pos(self._sim.cur_substep_local, state.pos.grad)

            if state.vel.grad is not None:
                state.vel.assert_contiguous()
                self.add_grad_from_vel(self._sim.cur_substep_local, state.vel.grad)

            if state.C.grad is not None:
                state.C.assert_contiguous()
                self.add_grad_from_C(self._sim.cur_substep_local, state.C.grad)

            if state.F.grad is not None:
                state.F.assert_contiguous()
                self.add_grad_from_F(self._sim.cur_substep_local, state.F.grad)

            if state.Jp.grad is not None:
                state.Jp.assert_contiguous()
                self.add_grad_from_Jp(self._sim.cur_substep_local, state.Jp.grad)

    @ti.kernel
    def add_grad_from_pos(self, f: ti.i32, pos_grad: ti.types.ndarray()):
        """将外部 pos 的梯度写回到第 f 帧粒子 pos.grad（形状 [B, n_particles, 3]）。"""
        # pos_grad shape: [B, n_particles, 3]
        for i_p, i_b in ti.ndrange(self._n_particles, self._B):
            for j in ti.static(range(3)):
                self.particles.grad[f, i_p, i_b].pos[j] += pos_grad[i_b, i_p, j]

    @ti.kernel
    def add_grad_from_vel(self, f: ti.i32, vel_grad: ti.types.ndarray()):
        """将外部 vel 的梯度写回到第 f 帧粒子 vel.grad（形状 [B, n_particles, 3]）。"""
        # vel_grad shape: [B, n_particles, 3]
        for i_p, i_b in ti.ndrange(self._n_particles, self._B):
            for j in ti.static(range(3)):
                self.particles.grad[f, i_p, i_b].vel[j] += vel_grad[i_b, i_p, j]

    @ti.kernel
    def add_grad_from_C(self, f: ti.i32, C_grad: ti.types.ndarray()):
        """将外部 C 的梯度写回到第 f 帧粒子 C.grad（形状 [B, n_particles, 3, 3]）。"""
        # C_grad shape: [B, n_particles, 3, 3]
        for i_p, i_b in ti.ndrange(self._n_particles, self._B):
            for j in ti.static(range(3)):
                for k in ti.static(range(3)):
                    self.particles.grad[f, i_p, i_b].C[j, k] += C_grad[i_b, i_p, j, k]

    @ti.kernel
    def add_grad_from_F(self, f: ti.i32, F_grad: ti.types.ndarray()):
        """将外部 F 的梯度写回到第 f 帧粒子 F.grad（形状 [B, n_particles, 3, 3]）。"""
        # F_grad shape: [B, n_particles, 3, 3]
        for i_p, i_b in ti.ndrange(self._n_particles, self._B):
            for j in ti.static(range(3)):
                for k in ti.static(range(3)):
                    self.particles.grad[f, i_p, i_b].F[j, k] += F_grad[i_b, i_p, j, k]

    @ti.kernel
    def add_grad_from_Jp(self, f: ti.i32, Jp_grad: ti.types.ndarray()):
        """将外部 Jp 的梯度写回到第 f 帧粒子 Jp.grad（形状 [B, n_particles]）。"""
        # Jp_grad shape: [B, n_particles]
        for i_p, i_b in ti.ndrange(self._n_particles, self._B):
            self.particles.grad[f, i_p, i_b].Jp += Jp_grad[i_b, i_p]

    # ------------------------------------------------------------------------------------
    # --------------------------------------- io -----------------------------------------
    # ------------------------------------------------------------------------------------

    def save_ckpt(self, ckpt_name):
        """
        保存检查点（仅可微模式下）：
        - 将第 0 帧的状态写入 ckpt 字典的张量缓存
        - 对每个实体调用 save_ckpt
        - 内存中的时间帧回到 0（copy last->0）
        """
        if self._sim.requires_grad:
            if ckpt_name not in self._ckpt:
                self._ckpt[ckpt_name] = dict()
                self._ckpt[ckpt_name]["pos"] = torch.zeros((self._B, self._n_particles, 3), dtype=gs.tc_float)
                self._ckpt[ckpt_name]["vel"] = torch.zeros((self._B, self._n_particles, 3), dtype=gs.tc_float)
                self._ckpt[ckpt_name]["C"] = torch.zeros((self._B, self._n_particles, 3, 3), dtype=gs.tc_float)
                self._ckpt[ckpt_name]["F"] = torch.zeros((self._B, self._n_particles, 3, 3), dtype=gs.tc_float)
                self._ckpt[ckpt_name]["Jp"] = torch.zeros((self._B, self._n_particles), dtype=gs.tc_float)
                self._ckpt[ckpt_name]["active"] = torch.zeros((self._B, self._n_particles), dtype=gs.tc_bool)

            self._kernel_get_state(
                0,
                self._ckpt[ckpt_name]["pos"],
                self._ckpt[ckpt_name]["vel"],
                self._ckpt[ckpt_name]["C"],
                self._ckpt[ckpt_name]["F"],
                self._ckpt[ckpt_name]["Jp"],
                self._ckpt[ckpt_name]["active"],
            )

            for entity in self._entities:
                entity.save_ckpt(ckpt_name)

        # restart from frame 0 in memory
        self.copy_frame(self._sim.substeps_local, 0)

    def load_ckpt(self, ckpt_name):
        """
        加载检查点：
        - 将帧 0 的状态拷回最后一帧（用于继续仿真）
        - 在可微模式下重置 0~last 的梯度，并从 ckpt 张量恢复到帧 0
        - 对每个实体调用 load_ckpt
        """
        self.copy_frame(0, self._sim.substeps_local)
        self.copy_grad(0, self._sim.substeps_local)

        if self._sim.requires_grad:
            self.reset_grad_till_frame(self._sim.substeps_local)

            self._kernel_set_state(
                0,
                self._ckpt[ckpt_name]["pos"],
                self._ckpt[ckpt_name]["vel"],
                self._ckpt[ckpt_name]["C"],
                self._ckpt[ckpt_name]["F"],
                self._ckpt[ckpt_name]["Jp"],
                self._ckpt[ckpt_name]["active"],
            )

            for entity in self._entities:
                entity.load_ckpt(ckpt_name=ckpt_name)

    def set_state(self, f, state, envs_idx=None):
        """将外部 MPMSolverState 写入到第 f 帧粒子字段。"""
        if self.is_active():
            self._kernel_set_state(f, state.pos, state.vel, state.C, state.F, state.Jp, state.active)

    @ti.kernel
    def _kernel_set_state(
        self,
        f: ti.i32,
        pos: ti.types.ndarray(),  # shape [B, n_particles, 3]
        vel: ti.types.ndarray(),  # shape [B, n_particles, 3]
        C: ti.types.ndarray(),  # shape [B, n_particles, 3, 3]
        F: ti.types.ndarray(),  # shape [B, n_particles, 3, 3]
        Jp: ti.types.ndarray(),  # shape [B, n_particles]
        active: ti.types.ndarray(),  # shape [B, n_particles]
    ):
        """将 numpy/torch 张量状态拷入 Taichi 粒子字段（第 f 帧）。"""
        for i_p, i_b in ti.ndrange(self._n_particles, self._B):
            # Write pos, vel
            for j in ti.static(range(3)):
                self.particles[f, i_p, i_b].pos[j] = pos[i_b, i_p, j]
                self.particles[f, i_p, i_b].vel[j] = vel[i_b, i_p, j]
                # Write C, F
                for k in ti.static(range(3)):
                    self.particles[f, i_p, i_b].C[j, k] = C[i_b, i_p, j, k]
                    self.particles[f, i_p, i_b].F[j, k] = F[i_b, i_p, j, k]
            # Write Jp, active
            self.particles[f, i_p, i_b].Jp = Jp[i_b, i_p]
            self.particles_ng[f, i_p, i_b].active = active[i_b, i_p]

    def get_state(self, f):
        """读取第 f 帧的粒子状态到 MPMSolverState（若未激活返回 None）。"""
        if not self.is_active():
            return None

        state = MPMSolverState(self._scene)
        self._kernel_get_state(f, state.pos, state.vel, state.C, state.F, state.Jp, state.active)
        return state

    @ti.kernel
    def _kernel_get_state(
        self,
        f: ti.i32,
        pos: ti.types.ndarray(),  # shape [B, n_particles, 3]
        vel: ti.types.ndarray(),  # shape [B, n_particles, 3]
        C: ti.types.ndarray(),  # shape [B, n_particles, 3, 3]
        F: ti.types.ndarray(),  # shape [B, n_particles, 3, 3]
        Jp: ti.types.ndarray(),  # shape [B, n_particles]
        active: ti.types.ndarray(),  # shape [B, n_particles]
    ):
        """将第 f 帧 Taichi 粒子字段拷出到 numpy/torch 张量。"""
        for i_p, i_b in ti.ndrange(self._n_particles, self._B):
            for j in ti.static(range(3)):
                pos[i_b, i_p, j] = self.particles[f, i_p, i_b].pos[j]
                vel[i_b, i_p, j] = self.particles[f, i_p, i_b].vel[j]
                for k in ti.static(range(3)):
                    C[i_b, i_p, j, k] = self.particles[f, i_p, i_b].C[j, k]
                    F[i_b, i_p, j, k] = self.particles[f, i_p, i_b].F[j, k]
            Jp[i_b, i_p] = self.particles[f, i_p, i_b].Jp
            active[i_b, i_p] = ti.cast(self.particles_ng[f, i_p, i_b].active, gs.ti_bool)

    def update_render_fields(self):
        """更新渲染字段（当前子步），包括粒子与可视化顶点位置/活跃标志。"""
        self._kernel_update_render_fields(self.sim.cur_substep_local)

    @ti.kernel
    def _kernel_update_render_fields(self, f: ti.i32):
        """从第 f 帧的粒子状态生成渲染用的粒子与 vvert 位置/活跃标志。"""
        for i_p, i_b in ti.ndrange(self._n_particles, self._B):
            if self.particles_ng[f, i_p, i_b].active:
                self.particles_render[i_p, i_b].pos = self.particles[f, i_p, i_b].pos
                self.particles_render[i_p, i_b].vel = self.particles[f, i_p, i_b].vel
            else:
                self.particles_render[i_p, i_b].pos = gu.ti_nowhere()
            self.particles_render[i_p, i_b].active = self.particles_ng[f, i_p, i_b].active

        for i_v, i_b in ti.ndrange(self._n_vverts, self._B):
            vvert_pos = ti.Vector.zero(gs.ti_float, 3)
            for j in range(self._n_vvert_supports):
                vvert_pos += (
                    self.particles[f, self.vverts_info.support_idxs[i_v][j], i_b].pos
                    * self.vverts_info.support_weights[i_v][j]
                )
            self.vverts_render[i_v, i_b].pos = vvert_pos
            self.vverts_render[i_v, i_b].active = self.particles_render[
                self.vverts_info.support_idxs[i_v][0], i_b
            ].active

    @ti.kernel
    def _kernel_add_particles(
        self,
        f: ti.i32,
        active: ti.i32,
        particle_start: ti.i32,
        n_particles: ti.i32,
        material_idx: ti.i32,
        mat_default_Jp: ti.f32,
        mat_rho: ti.f32,
        pos: ti.types.ndarray(),  # shape [n_particles, 3]
    ):
        """
        批量添加粒子：
        - 写入 particles_info（材料索引、质量=体积*密度、默认 Jp、肌肉信息、free）
        - 写入第 f 帧的 pos/vel/F/C/Jp/actu/active 初值
        """
        for i_p_ in range(n_particles):
            i_p = i_p_ + particle_start

            self.particles_info[i_p].material_idx = material_idx
            self.particles_info[i_p].default_Jp = mat_default_Jp
            self.particles_info[i_p].mass = self._particle_volume * mat_rho
            self.particles_info[i_p].free = True
            self.particles_info[i_p].muscle_group = 0
            self.particles_info[i_p].muscle_direction = ti.Vector([0.0, 0.0, 1.0], dt=gs.ti_float)

        for i_p_, i_b in ti.ndrange(n_particles, self._B):
            i_p = i_p_ + particle_start

            self.particles_ng[f, i_p, i_b].active = ti.cast(active, gs.ti_bool)
            for i in ti.static(range(3)):
                self.particles[f, i_p, i_b].pos[i] = pos[i_p_, i]

            self.particles[f, i_p, i_b].vel = ti.Vector.zero(gs.ti_float, 3)
            self.particles[f, i_p, i_b].F = ti.Matrix.identity(gs.ti_float, 3)
            self.particles[f, i_p, i_b].C = ti.Matrix.zero(gs.ti_float, 3, 3)
            self.particles[f, i_p, i_b].Jp = mat_default_Jp
            self.particles[f, i_p, i_b].actu = gs.ti_float(0.0)

    @ti.kernel
    def _kernel_set_particles_pos(
        self,
        f: ti.i32,
        particles_idx: ti.types.ndarray(),
        envs_idx: ti.types.ndarray(),
        poss: ti.types.ndarray(),
    ):
        """
        覆写指定粒子的第 f 帧位置（按环境索引批量）；并重置 vel/F/C/Jp 到一致初值。
        """
        for i_p_, i_b_ in ti.ndrange(particles_idx.shape[1], envs_idx.shape[0]):
            i_p = particles_idx[i_b_, i_p_]
            i_b = envs_idx[i_b_]

            for i in ti.static(range(3)):
                self.particles[f, i_p, i_b].pos[i] = poss[i_b_, i_p_, i]

            # Reset these attributes whenever overwritting particle positions manually
            self.particles[f, i_p, i_b].vel.fill(0.0)
            self.particles[f, i_p, i_b].F = ti.Matrix.identity(gs.ti_float, 3)
            self.particles[f, i_p, i_b].C.fill(0.0)
            self.particles[f, i_p, i_b].Jp = self.particles_info[i_p].default_Jp

    @ti.kernel
    def _kernel_set_particles_pos_grad(
        self,
        f: ti.i32,
        particle_start: ti.i32,
        n_particles: ti.i32,
        poss_grad: ti.types.ndarray(),  # shape [B, n_particles, 3]
    ):
        """导出第 f 帧粒子 pos 的梯度到 poss_grad（按连续的粒子区间）。"""
        for i_p_, i_b in ti.ndrange(n_particles, self._B):
            i_p = i_p_ + particle_start
            for i in ti.static(range(3)):
                poss_grad[i_b, i_p_, i] = self.particles.grad[f, i_p, i_b].pos[i]

    @ti.kernel
    def _kernel_get_particles_pos(
        self,
        f: ti.i32,
        particle_start: ti.i32,
        n_particles: ti.i32,
        envs_idx: ti.types.ndarray(),
        poss: ti.types.ndarray(),
    ):
        """导出第 f 帧粒子 pos 到 poss（按环境索引与连续的粒子区间）。"""
        for i_p_, i_b_ in ti.ndrange(n_particles, envs_idx.shape[0]):
            i_p = i_p_ + particle_start
            i_b = envs_idx[i_b_]
            for i in ti.static(range(3)):
                poss[i_b_, i_p_, i] = self.particles[f, i_p, i_b].pos[i]

    @ti.kernel
    def _kernel_set_particles_vel(
        self,
        f: ti.i32,
        particles_idx: ti.types.ndarray(),
        envs_idx: ti.types.ndarray(),
        vels: ti.types.ndarray(),  # shape [B, n_particles, 3]
    ):
        """覆写第 f 帧粒子速度（按环境与粒子索引）。"""
        for i_p_, i_b_ in ti.ndrange(particles_idx.shape[1], envs_idx.shape[0]):
            i_p = particles_idx[i_b_, i_p_]
            i_b = envs_idx[i_b_]
            for i in ti.static(range(3)):
                self.particles[f, i_p, i_b].vel[i] = vels[i_b_, i_p_, i]

    @ti.kernel
    def _kernel_set_particles_vel_grad(
        self,
        f: ti.i32,
        particle_start: ti.i32,
        n_particles: ti.i32,
        vels_grad: ti.types.ndarray(),  # shape [B, n_particles, 3]
    ):
        """导出第 f 帧粒子 vel 的梯度到 vels_grad。"""
        for i_p_, i_b in ti.ndrange(n_particles, self._B):
            i_p = i_p_ + particle_start
            for i in ti.static(range(3)):
                vels_grad[i_b, i_p_, i] = self.particles.grad[f, i_p, i_b].vel[i]

    @ti.kernel
    def _kernel_get_particles_vel(
        self,
        f: ti.i32,
        particle_start: ti.i32,
        n_particles: ti.i32,
        envs_idx: ti.types.ndarray(),
        vels: ti.types.ndarray(),
    ):
        """导出第 f 帧粒子 vel 到 vels。"""
        for i_p_, i_b_ in ti.ndrange(n_particles, envs_idx.shape[0]):
            i_p = i_p_ + particle_start
            i_b = envs_idx[i_b_]
            for i in ti.static(range(3)):
                vels[i_b_, i_p_, i] = self.particles[f, i_p, i_b].vel[i]

    @ti.kernel
    def _kernel_set_particles_active(
        self,
        f: ti.i32,
        particles_idx: ti.types.ndarray(),
        envs_idx: ti.types.ndarray(),
        actives: ti.types.ndarray(),  # shape [B, n_particles]
    ):
        """覆写第 f 帧粒子 active 标志。"""
        for i_p_, i_b_ in ti.ndrange(particles_idx.shape[1], envs_idx.shape[0]):
            i_p = particles_idx[i_b_, i_p_]
            i_b = envs_idx[i_b_]
            self.particles_ng[f, i_p, i_b].active = ti.cast(actives[i_b_, i_p_], gs.ti_bool)

    @ti.kernel
    def _kernel_get_particles_active(
        self,
        f: ti.i32,
        particle_start: ti.i32,
        n_particles: ti.i32,
        envs_idx: ti.types.ndarray(),
        actives: ti.types.ndarray(),  # shape [B, n_particles]
    ):
        """导出第 f 帧粒子 active 到 actives。"""
        for i_p_, i_b_ in ti.ndrange(n_particles, envs_idx.shape[0]):
            i_p = i_p_ + particle_start
            i_b = envs_idx[i_b_]
            actives[i_b_, i_p_] = self.particles_ng[f, i_p, i_b].active

    @ti.kernel
    def _kernel_set_particles_actu(
        self,
        f: ti.i32,
        n_groups: ti.i32,
        particles_idx: ti.types.ndarray(),
        envs_idx: ti.types.ndarray(),
        actus: ti.types.ndarray(),  # shape [B, n_particles, n_groups]
    ):
        """设置第 f 帧粒子的肌肉激活值（按分组写入）。"""
        for i_p_, i_g, i_b_ in ti.ndrange(particles_idx.shape[1], n_groups, envs_idx.shape[0]):
            i_p = particles_idx[i_b_, i_p_]
            i_b = envs_idx[i_b_]
            if self.particles_info[i_p].muscle_group == i_g:
                self.particles[f, i_p, i_b].actu = actus[i_b_, i_p_, i_g]

    @ti.kernel
    def _kernel_set_particles_actu_grad(
        self,
        f: ti.i32,
        particle_start: ti.i32,
        n_particles: ti.i32,
        envs_idx: ti.types.ndarray(),
        actus_grad: ti.types.ndarray(),  # shape [B, n_particles]
    ):
        """导出第 f 帧粒子肌肉激活的梯度。"""
        for i_p_, i_g, i_b_ in ti.ndrange(n_particles, envs_idx.shape[0]):
            i_p = i_p_ + particle_start
            i_b = envs_idx[i_b_]
            actus_grad[i_b_, i_p_] = self.particles.grad[f, i_p, i_b].actu

    @ti.kernel
    def _kernel_get_particles_actu(
        self,
        f: ti.i32,
        particle_start: ti.i32,
        n_particles: ti.i32,
        envs_idx: ti.types.ndarray(),
        actus: ti.types.ndarray(),  # shape [B, n_particles]
    ):
        """导出第 f 帧粒子肌肉激活。"""
        for i_p_, i_b_ in ti.ndrange(n_particles, envs_idx.shape[0]):
            i_p = i_p_ + particle_start
            i_b = envs_idx[i_b_]
            actus[i_b_, i_p_] = self.particles[f, i_p, i_b].actu

    @ti.kernel
    def _kernel_set_particles_muscle_group(self, particles_idx: ti.types.ndarray(), muscle_group: ti.types.ndarray()):
        """设置粒子肌肉分组。"""
        for i_p_ in range(particles_idx.shape[0]):
            i_p = particles_idx[i_p_]
            self.particles_info[i_p].muscle_group = muscle_group[i_p_]

    @ti.kernel
    def _kernel_get_particles_muscle_group(
        self, particle_start: ti.i32, n_particles: ti.i32, muscle_group: ti.types.ndarray()
    ):
        """导出粒子肌肉分组。"""
        for i_p_ in range(n_particles):
            i_p = i_p_ + particle_start
            muscle_group[i_p_] = self.particles_info[i_p].muscle_group

    @ti.kernel
    def _kernel_set_particles_muscle_direction(
        self, particles_idx: ti.types.ndarray(), muscle_direction: ti.types.ndarray()
    ):
        """设置粒子肌肉方向（逐分量）。"""
        for i_p_ in range(particles_idx.shape[0]):
            i_p = particles_idx[i_p_]
            for i in ti.static(range(3)):
                self.particles_info[i_p].muscle_direction[i] = muscle_direction[i_p_, i]

    @ti.kernel
    def _kernel_set_particles_free(self, particles_idx: ti.types.ndarray(), free: ti.types.ndarray()):
        """设置粒子是否为自由（free）——非自由可作为边界条件。"""
        for i_p_ in range(particles_idx.shape[0]):
            i_p = particles_idx[i_p_]
            self.particles_info[i_p].free = free[i_p_]

    @ti.kernel
    def _kernel_get_particles_free(self, particle_start: ti.i32, n_particles: ti.i32, free: ti.types.ndarray()):
        """导出粒子 free 标志。"""
        for i_p_ in range(n_particles):
            i_p = i_p_ + particle_start
            free[i_p_] = self.particles_info[i_p].free

    @ti.kernel
    def _kernel_get_mass(
        self, particle_start: ti.i32, n_particles: ti.i32, mass: ti.types.ndarray(), envs_idx: ti.types.ndarray()
    ):
        """导出（所选粒子区间的）总质量（按体积缩放还原）。"""
        total_mass = gs.ti_float(0.0)
        for i_p_ in range(n_particles):
            i_p = i_p_ + particle_start
            total_mass += self.particles_info[i_p].mass
        total_mass = total_mass / self._particle_volume_scale
        for i_b_ in range(envs_idx.shape[0]):
            mass[i_b_] = total_mass

    # ------------------------------------------------------------------------------------
    # ----------------------------------- properties -------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def n_particles(self):
        """粒子总数（构建后使用缓存，构建前聚合实体统计）。"""
        if self.is_built:
            return self._n_particles
        return sum(entity.n_particles for entity in self._entities)

    @property
    def n_vverts(self):
        """可视化顶点（vverts）总数。"""
        if self.is_built:
            return self._n_vverts
        return sum(entity.n_vverts for entity in self._entities)

    @property
    def n_vfaces(self):
        """可视化面（vfaces）总数。"""
        if self.is_built:
            return self._n_vfaces
        return sum(entity.n_vfaces for entity in self._entities)

    @property
    def grid_density(self):
        return self._grid_density

    @property
    def particle_size(self):
        return self._particle_size

    @property
    def particle_radius(self):
        return self._particle_size / 2.0

    @property
    def upper_bound(self):
        return self._upper_bound

    @property
    def lower_bound(self):
        return self._lower_bound

    @property
    def leaf_block_size(self):
        raise DeprecationError("This property has been removed.")

    @property
    def use_sparse_grid(self):
        return DeprecationError("This property has been removed.")

    @property
    def dx(self):
        return self._dx

    @property
    def inv_dx(self):
        return self._inv_dx

    @property
    def particle_volume_real(self):
        return self._particle_volume_real

    @property
    def particle_volume(self):
        return self._particle_volume

    @property
    def particle_volume_scale(self):
        return self._particle_volume_scale

    @property
    def is_built(self):
        return self._scene._is_built

    @property
    def lower_bound_cell(self):
        return self._lower_bound_cell

    @property
    def upper_bound_cell(self):
        return self._upper_bound_cell

    @property
    def grid_res(self):
        return self._grid_res

    @property
    def grid_offset(self):
        return self._grid_offset

    @property
    def enable_CPIC(self):
        return self._enable_CPIC


@ti.func
def signmax(a, eps):
    """返回带符号的 max(|a|, eps)，用于避免奇异值相等时的数值不稳定。"""
    sign = ti.select(a >= 0, 1.0, -1.0)
    return sign * ti.max(ti.abs(a), eps)


@ti.func
def backward_svd(grad_U, grad_S, grad_V, U, S, V):
    """
    SVD 反向传播近似（参考 PyTorch 实现模板）：
    - 给定 U/S/V 及其梯度，求对 F 的梯度（通过对角差分矩阵 F 的构造避免奇异值重复的数值问题）。
    注：此处为 3x3 情况的特化实现。
    """
    # https://github.com/pytorch/pytorch/blob/ab0a04dc9c8b84d4a03412f1c21a6c4a2cefd36c/tools/autograd/templates/Functions.cpp
    vt = V.transpose()
    ut = U.transpose()
    S_term = U @ grad_S @ vt

    s = ti.Vector.zero(gs.ti_float, 3)
    s = ti.Vector([S[0, 0], S[1, 1], S[2, 2]]) ** 2
    F = ti.Matrix.zero(gs.ti_float, 3, 3)
    for i, j in ti.static(ti.ndrange(3, 3)):
        if i == j:
            F[i, j] = 0.0
        else:
            F[i, j] = 1.0 / signmax(s[j] - s[i], 1e-6)
    u_term = U @ ((F * (ut @ grad_U - grad_U.transpose() @ U)) @ S) @ vt
    v_term = U @ (S @ ((F * (vt @ grad_V - grad_V.transpose() @ V)) @ vt))
    return u_term + v_term + S_term