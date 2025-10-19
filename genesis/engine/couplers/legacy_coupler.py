from typing import TYPE_CHECKING

import numpy as np
import gstaichi as ti

import genesis as gs
import genesis.utils.sdf_decomp as sdf_decomp

from genesis.options.solvers import LegacyCouplerOptions
from genesis.repr_base import RBC
from genesis.utils.array_class import LinksState
from genesis.utils.geom import ti_inv_transform_by_trans_quat, ti_transform_by_trans_quat

if TYPE_CHECKING:
    from genesis.engine.simulator import Simulator

CLAMPED_INV_DT = 50.0  # 纠正速度时的最大反时间步（防止数值抖动）


@ti.data_oriented
class LegacyCoupler(RBC):
    """
    传统耦合器：处理不同求解器之间的耦合交互（将被逐步废弃）。
    """

    def __init__(
        self,
        simulator: "Simulator",
        options: "LegacyCouplerOptions",
    ) -> None:
        self.sim = simulator
        self.options = options

        # 引用各求解器实例
        self.tool_solver = self.sim.tool_solver
        self.rigid_solver = self.sim.rigid_solver
        self.avatar_solver = self.sim.avatar_solver
        self.mpm_solver = self.sim.mpm_solver
        self.sph_solver = self.sim.sph_solver
        self.pbd_solver = self.sim.pbd_solver
        self.fem_solver = self.sim.fem_solver
        self.sf_solver = self.sim.sf_solver

    def build(self) -> None:
        # 各类耦合开关（仅当双方都激活且选项允许时生效）
        self._rigid_mpm = self.rigid_solver.is_active() and self.mpm_solver.is_active() and self.options.rigid_mpm
        self._rigid_sph = self.rigid_solver.is_active() and self.sph_solver.is_active() and self.options.rigid_sph
        self._rigid_pbd = self.rigid_solver.is_active() and self.pbd_solver.is_active() and self.options.rigid_pbd
        self._rigid_fem = self.rigid_solver.is_active() and self.fem_solver.is_active() and self.options.rigid_fem
        self._mpm_sph = self.mpm_solver.is_active() and self.sph_solver.is_active() and self.options.mpm_sph
        self._mpm_pbd = self.mpm_solver.is_active() and self.pbd_solver.is_active() and self.options.mpm_pbd
        self._fem_mpm = self.fem_solver.is_active() and self.mpm_solver.is_active() and self.options.fem_mpm
        self._fem_sph = self.fem_solver.is_active() and self.sph_solver.is_active() and self.options.fem_sph

        # MPM 与刚体（CPIC 模式）需要的中间存储
        if self._rigid_mpm and self.mpm_solver.enable_CPIC:
            # 存储薄壳几何对粒子与其周围网格单元的分割标记
            self.cpic_flag = ti.field(gs.ti_int, shape=(self.mpm_solver.n_particles, 3, 3, 3, self.mpm_solver._B))
            # 粒子对应几何的法向缓存
            self.mpm_rigid_normal = ti.Vector.field(
                3,
                dtype=gs.ti_float,
                shape=(self.mpm_solver.n_particles, self.rigid_solver.n_geoms_, self.mpm_solver._B),
            )

        # SPH 与刚体法向缓存
        if self._rigid_sph:
            self.sph_rigid_normal = ti.Vector.field(
                3,
                dtype=gs.ti_float,
                shape=(self.sph_solver.n_particles, self.rigid_solver.n_geoms_, self.sph_solver._B),
            )
            self.sph_rigid_normal_reordered = ti.Vector.field(
                3,
                dtype=gs.ti_float,
                shape=(self.sph_solver.n_particles, self.rigid_solver.n_geoms_, self.sph_solver._B),
            )

        # PBD 与刚体：法向缓存 + 绑定信息
        if self._rigid_pbd:
            self.pbd_rigid_normal_reordered = ti.Vector.field(
                3, dtype=gs.ti_float, shape=(self.pbd_solver.n_particles, self.pbd_solver._B, self.rigid_solver.n_geoms)
            )

            struct_particle_attach_info = ti.types.struct(
                link_idx=gs.ti_int,      # 绑定的刚体链节索引
                local_pos=gs.ti_vec3,    # 粒子在链节局部坐标系的位置
            )
            pbd_batch_shape = self.pbd_solver._batch_shape(self.pbd_solver._n_particles)

            self.particle_attach_info = struct_particle_attach_info.field(shape=pbd_batch_shape, layout=ti.Layout.SOA)
            self.particle_attach_info.link_idx.fill(-1)
            self.particle_attach_info.local_pos.fill(gs.ti_vec3(0.0, 0.0, 0.0))

        # MPM 与 SPH 的空间邻域半径（以网格大小比值估算）
        if self._mpm_sph:
            self.mpm_sph_stencil_size = int(np.floor(self.mpm_solver.dx / self.sph_solver.hash_grid_cell_size) + 2)

        # MPM 与 PBD 的空间邻域半径
        if self._mpm_pbd:
            self.mpm_pbd_stencil_size = int(np.floor(self.mpm_solver.dx / self.pbd_solver.hash_grid_cell_size) + 2)

        # 调试用参数
        self._dx = 1 / 1024
        self._stencil_size = int(np.floor(self._dx / self.sph_solver.hash_grid_cell_size) + 2)

        self.reset(envs_idx=self.sim.scene._envs_idx)

    def reset(self, envs_idx=None) -> None:
        # 清理法向缓存
        if self._rigid_mpm and self.mpm_solver.enable_CPIC:
            if envs_idx is None:
                self.mpm_rigid_normal.fill(0)
            else:
                self._kernel_reset_mpm(envs_idx)

        if self._rigid_sph:
            if envs_idx is None:
                self.sph_rigid_normal.fill(0)
            else:
                self._kernel_reset_sph(envs_idx)

    @ti.kernel
    def _kernel_reset_mpm(self, envs_idx: ti.types.ndarray()):
        for i_p, i_g, i_b_ in ti.ndrange(self.mpm_solver.n_particles, self.rigid_solver.n_geoms, envs_idx.shape[0]):
            self.mpm_rigid_normal[i_p, i_g, envs_idx[i_b_]] = 0.0

    @ti.kernel
    def _kernel_reset_sph(self, envs_idx: ti.types.ndarray()):
        for i_p, i_g, i_b_ in ti.ndrange(self.sph_solver.n_particles, self.rigid_solver.n_geoms, envs_idx.shape[0]):
            self.sph_rigid_normal[i_p, i_g, envs_idx[i_b_]] = 0.0

    @ti.func
    def _func_collide_with_rigid(self, f, pos_world, vel, mass, i_b):
        # 针对需要耦合的刚体几何依次处理
        for i_g in range(self.rigid_solver.n_geoms):
            if self.rigid_solver.geoms_info.needs_coup[i_g]:
                vel = self._func_collide_with_rigid_geom(pos_world, vel, mass, i_g, i_b)
        return vel

    @ti.func
    def _func_collide_with_rigid_geom(self, pos_world, vel, mass, geom_idx, batch_idx):
        # SDF 距离
        signed_dist = sdf_decomp.sdf_func_world(
            geoms_state=self.rigid_solver.geoms_state,
            geoms_info=self.rigid_solver.geoms_info,
            sdf_info=self.rigid_solver.sdf._sdf_info,
            pos_world=pos_world,
            geom_idx=geom_idx,
            batch_idx=batch_idx,
        )
        # 耦合影响范围（越软传播越远）
        influence = ti.min(ti.exp(-signed_dist / max(1e-10, self.rigid_solver.geoms_info.coup_softness[geom_idx])), 1)

        if influence > 0.1:
            normal_rigid = sdf_decomp.sdf_func_normal_world(
                geoms_state=self.rigid_solver.geoms_state,
                geoms_info=self.rigid_solver.geoms_info,
                collider_static_config=self.rigid_solver.collider._collider_static_config,
                sdf_info=self.rigid_solver.sdf._sdf_info,
                pos_world=pos_world,
                geom_idx=geom_idx,
                batch_idx=batch_idx,
            )
            vel = self._func_collide_in_rigid_geom(pos_world, vel, mass, normal_rigid, influence, geom_idx, batch_idx)

        return vel

    @ti.func
    def _func_collide_with_rigid_geom_robust(self, pos_world, vel, mass, normal_prev, geom_idx, batch_idx):
        """
        更鲁棒的刚体碰撞处理（可处理部分穿透导致的法向翻转）。
        """
        signed_dist = sdf_decomp.sdf_func_world(
            geoms_state=self.rigid_solver.geoms_state,
            geoms_info=self.rigid_solver.geoms_info,
            sdf_info=self.rigid_solver.sdf._sdf_info,
            pos_world=pos_world,
            geom_idx=geom_idx,
            batch_idx=batch_idx,
        )
        normal_rigid = sdf_decomp.sdf_func_normal_world(
            geoms_state=self.rigid_solver.geoms_state,
            geoms_info=self.rigid_solver.geoms_info,
            collider_static_config=self.rigid_solver.collider._collider_static_config,
            sdf_info=self.rigid_solver.sdf._sdf_info,
            pos_world=pos_world,
            geom_idx=geom_idx,
            batch_idx=batch_idx,
        )
        influence = ti.min(ti.exp(-signed_dist / max(1e-10, self.rigid_solver.geoms_info.coup_softness[geom_idx])), 1)

        # 仅在影响足够大时处理
        if influence > 0.1:
            vel = self._func_collide_in_rigid_geom(pos_world, vel, mass, normal_rigid, influence, geom_idx, batch_idx)
        return vel, normal_rigid

    @ti.func
    def _func_collide_in_rigid_geom(self, pos_world, vel, mass, normal_rigid, influence, geom_idx, batch_idx):
        """
        已确认发生碰撞时的速度修正与动量传递。
        """
        # 刚体参考点速度
        vel_rigid = self.rigid_solver._func_vel_at_point(
            pos_world=pos_world,
            link_idx=self.rigid_solver.geoms_info.link_idx[geom_idx],
            i_b=batch_idx,
            links_state=self.rigid_solver.links_state,
        )

        # 相对速度
        rvel = vel - vel_rigid
        rvel_normal_magnitude = rvel.dot(normal_rigid)

        if rvel_normal_magnitude < 0:  # 指向内侧才处理
            # 切向分量
            rvel_tan = rvel - rvel_normal_magnitude * normal_rigid
            rvel_tan_norm = rvel_tan.norm(gs.EPS)

            # 摩擦后的切向
            rvel_tan = (
                rvel_tan
                / rvel_tan_norm
                * ti.max(
                    0, rvel_tan_norm + rvel_normal_magnitude * self.rigid_solver.geoms_info.coup_friction[geom_idx]
                )
            )

            # 法向恢复（弹性系数）
            rvel_normal = (
                -normal_rigid * rvel_normal_magnitude * self.rigid_solver.geoms_info.coup_restitution[geom_idx]
            )

            # 合成新相对速度
            rvel_new = rvel_tan + rvel_normal

            # 影响插值
            vel_old = vel
            vel = vel_rigid + rvel_new * influence + rvel * (1 - influence)

            # 反作用力传递给刚体
            delta_mv = mass * (vel - vel_old)
            force = -delta_mv / self.rigid_solver.substep_dt
            self.rigid_solver._func_apply_external_force(
                pos_world,
                force,
                self.rigid_solver.geoms_info.link_idx[geom_idx],
                batch_idx,
                self.rigid_solver.links_state,
            )

        return vel

    @ti.func
    def _func_mpm_tool(self, f, pos_world, vel, i_b):
        # MPM 与工具的逐实体碰撞
        for entity in ti.static(self.tool_solver.entities):
            if ti.static(entity.material.collision):
                vel = entity.collide(f, pos_world, vel, i_b)
        return vel

    @ti.kernel
    def mpm_grid_op(self, f: ti.i32, t: ti.f32):
        """
        MPM 网格操作与耦合统一执行（避免为每个耦合对重复网格遍历）。
        """
        for ii, jj, kk, i_b in ti.ndrange(*self.mpm_solver.grid_res, self.mpm_solver._B):
            I = (ii, jj, kk)
            if self.mpm_solver.grid[f, I, i_b].mass > gs.EPS:
                # 动量转速度
                vel_mpm = (1 / self.mpm_solver.grid[f, I, i_b].mass) * self.mpm_solver.grid[f, I, i_b].vel_in
                # 重力
                vel_mpm += self.mpm_solver.substep_dt * self.mpm_solver._gravity[i_b]

                pos = (I + self.mpm_solver.grid_offset) * self.mpm_solver.dx
                mass_mpm = self.mpm_solver.grid[f, I, i_b].mass / self.mpm_solver._particle_volume_scale

                # 外力场
                for i_ff in ti.static(range(len(self.mpm_solver._ffs))):
                    vel_mpm += self.mpm_solver._ffs[i_ff].get_acc(pos, vel_mpm, t, -1) * self.mpm_solver.substep_dt

                # MPM 与工具
                if ti.static(self.tool_solver.is_active()):
                    vel_mpm = self._func_mpm_tool(f, pos, vel_mpm, i_b)

                # MPM 与刚体
                if ti.static(self._rigid_mpm and self.rigid_solver.is_active()):
                    vel_mpm = self._func_collide_with_rigid(f, pos, vel_mpm, mass_mpm, i_b)

                # MPM 与 SPH
                if ti.static(self._mpm_sph):
                    base = self.sph_solver.sh.pos_to_grid(pos - 0.5 * self.mpm_solver.dx)
                    sph_vel = ti.Vector([0.0, 0.0, 0.0])
                    colliding_particles = 0
                    for offset in ti.grouped(
                        ti.ndrange(self.mpm_sph_stencil_size, self.mpm_sph_stencil_size, self.mpm_sph_stencil_size)
                    ):
                        slot_idx = self.sph_solver.sh.grid_to_slot(base + offset)
                        for i in range(
                            self.sph_solver.sh.slot_start[slot_idx, i_b],
                            self.sph_solver.sh.slot_start[slot_idx, i_b] + self.sph_solver.sh.slot_size[slot_idx, i_b],
                        ):
                            if (
                                ti.abs(pos - self.sph_solver.particles_reordered.pos[i, i_b]).max()
                                < self.mpm_solver.dx * 0.5
                            ):
                                sph_vel += self.sph_solver.particles_reordered.vel[i, i_b]
                                colliding_particles += 1
                    if colliding_particles > 0:
                        vel_old = vel_mpm
                        vel_mpm = sph_vel / colliding_particles
                        delta_mv = mass_mpm * (vel_mpm - vel_old)
                        for offset in ti.grouped(
                            ti.ndrange(self.mpm_sph_stencil_size, self.mpm_sph_stencil_size, self.mpm_sph_stencil_size)
                        ):
                            slot_idx = self.sph_solver.sh.grid_to_slot(base + offset)
                            for i in range(
                                self.sph_solver.sh.slot_start[slot_idx, i_b],
                                self.sph_solver.sh.slot_start[slot_idx, i_b]
                                + self.sph_solver.sh.slot_size[slot_idx, i_b],
                            ):
                                if (
                                    ti.abs(pos - self.sph_solver.particles_reordered.pos[i, i_b]).max()
                                    < self.mpm_solver.dx * 0.5
                                ):
                                    self.sph_solver.particles_reordered[i, i_b].vel = (
                                        self.sph_solver.particles_reordered[i, i_b].vel
                                        - delta_mv / self.sph_solver.particles_info_reordered[i, i_b].mass
                                    )

                # MPM 与 PBD
                if ti.static(self._mpm_pbd):
                    base = self.pbd_solver.sh.pos_to_grid(pos - 0.5 * self.mpm_solver.dx)
                    pbd_vel = ti.Vector([0.0, 0.0, 0.0])
                    colliding_particles = 0
                    for offset in ti.grouped(
                        ti.ndrange(self.mpm_pbd_stencil_size, self.mpm_pbd_stencil_size, self.mpm_pbd_stencil_size)
                    ):
                        slot_idx = self.pbd_solver.sh.grid_to_slot(base + offset)
                        for i in range(
                            self.pbd_solver.sh.slot_start[slot_idx, i_b],
                            self.pbd_solver.sh.slot_start[slot_idx, i_b] + self.pbd_solver.sh.slot_size[slot_idx, i_b],
                        ):
                            if (
                                ti.abs(pos - self.pbd_solver.particles_reordered.pos[i, i_b]).max()
                                < self.mpm_solver.dx * 0.5
                            ):
                                pbd_vel += self.pbd_solver.particles_reordered.vel[i, i_b]
                                colliding_particles += 1
                    if colliding_particles > 0:
                        vel_old = vel_mpm
                        vel_mpm = pbd_vel / colliding_particles
                        delta_mv = mass_mpm * (vel_mpm - vel_old)
                        for offset in ti.grouped(
                            ti.ndrange(self.mpm_pbd_stencil_size, self.mpm_pbd_stencil_size, self.mpm_pbd_stencil_size)
                        ):
                            slot_idx = self.pbd_solver.sh.grid_to_slot(base + offset)
                            for i in range(
                                self.pbd_solver.sh.slot_start[slot_idx, i_b],
                                self.pbd_solver.sh.slot_start[slot_idx, i_b]
                                + self.pbd_solver.sh.slot_size[slot_idx, i_b],
                            ):
                                if (
                                    ti.abs(pos - self.pbd_solver.particles_reordered.pos[i, i_b]).max()
                                    < self.mpm_solver.dx * 0.5
                                ):
                                    if self.pbd_solver.particles_reordered[i, i_b].free:
                                        self.pbd_solver.particles_reordered[i, i_b].vel = (
                                            self.pbd_solver.particles_reordered[i, i_b].vel
                                            - delta_mv / self.pbd_solver.particles_info_reordered[i, i_b].mass
                                        )

                # MPM 边界
                _, self.mpm_solver.grid[f, I, i_b].vel_out = self.mpm_solver.boundary.impose_pos_vel(pos, vel_mpm)

    @ti.kernel
    def mpm_surface_to_particle(self, f: ti.i32):
        # 将 MPM 粒子邻近刚体表面的法向写入缓存（用于 CPIC）
        for i_p, i_b in ti.ndrange(self.mpm_solver.n_particles, self.mpm_solver._B):
            if self.mpm_solver.particles_ng[f, i_p, i_b].active:
                for i_g in range(self.rigid_solver.n_geoms):
                    if self.rigid_solver.geoms_info.needs_coup[i_g]:
                        sdf_normal = sdf_decomp.sdf_func_normal_world(
                            geoms_state=self.rigid_solver.geoms_state,
                            geoms_info=self.rigid_solver.geoms_info,
                            collider_static_config=self.rigid_solver.collider._collider_static_config,
                            sdf_info=self.rigid_solver.sdf._sdf_info,
                            pos_world=self.mpm_solver.particles[f, i_p, i_b].pos,
                            geom_idx=i_g,
                            batch_idx=i_b,
                        )
                        # 仅在没有反向穿插时更新
                        if sdf_normal.dot(self.mpm_rigid_normal[i_p, i_g, i_b]) >= 0:
                            self.mpm_rigid_normal[i_p, i_g, i_b] = sdf_normal

    def fem_rigid_link_constraints(self):
        # 更新 FEM 与刚体链节绑定约束
        if self.fem_solver._constraints_initialized and self.rigid_solver.is_active():
            links_pos = self.rigid_solver.links_state.pos
            links_quat = self.rigid_solver.links_state.quat
            self.fem_solver._kernel_update_linked_vertex_constraints(links_pos, links_quat)

    @ti.kernel
    def fem_surface_force(self, f: ti.i32):
        # 遍历 FEM 表面三角面做耦合交互
        for i_s, i_b in ti.ndrange(self.fem_solver.n_surfaces, self.fem_solver._B):
            if self.fem_solver.surface[i_s].active:
                dt = self.fem_solver.substep_dt
                iel = self.fem_solver.surface[i_s].tri2el
                mass = self.fem_solver.elements_i[iel].mass_scaled / self.fem_solver.vol_scale

                p1 = self.fem_solver.elements_v[f, self.fem_solver.surface[i_s].tri2v[0], i_b].pos
                p2 = self.fem_solver.elements_v[f, self.fem_solver.surface[i_s].tri2v[1], i_b].pos
                p3 = self.fem_solver.elements_v[f, self.fem_solver.surface[i_s].tri2v[2], i_b].pos
                u = p2 - p1
                v = p3 - p1
                surface_normal = ti.math.cross(u, v)
                surface_normal = surface_normal / surface_normal.norm(gs.EPS)

                # FEM 与刚体（仅顶点级）
                if ti.static(self._rigid_fem):
                    for j in ti.static(range(3)):
                        iv = self.fem_solver.surface[i_s].tri2v[j]
                        vel_fem_sv = self._func_collide_with_rigid(
                            f,
                            self.fem_solver.elements_v[f, iv, i_b].pos,
                            self.fem_solver.elements_v[f + 1, iv, i_b].vel,
                            mass / 3.0,
                            i_b,
                        )
                        self.fem_solver.elements_v[f + 1, iv, i_b].vel = vel_fem_sv

                # FEM 与 MPM（通过网格近似）
                if ti.static(self._fem_mpm):
                    for j in ti.static(range(3)):
                        iv = self.fem_solver.surface[i_s].tri2v[j]
                        pos = self.fem_solver.elements_v[f, iv, i_b].pos
                        vel_fem_sv = self.fem_solver.elements_v[f + 1, iv, i_b].vel
                        mass_fem_sv = mass / 4.0

                        vel_mpm = ti.Vector([0.0, 0.0, 0.0])
                        mass_mpm = 0.0
                        mpm_base = ti.floor(pos * self.mpm_solver.inv_dx - 0.5).cast(gs.ti_int)
                        mpm_fx = pos * self.mpm_solver.inv_dx - mpm_base.cast(gs.ti_float)
                        mpm_w = [0.5 * (1.5 - mpm_fx) ** 2, 0.75 - (mpm_fx - 1.0) ** 2, 0.5 * (mpm_fx - 0.5) ** 2]
                        new_vel_fem_sv = vel_fem_sv
                        for mpm_offset in ti.static(ti.grouped(self.mpm_solver.stencil_range())):
                            mpm_grid_I = mpm_base - self.mpm_solver.grid_offset + mpm_offset
                            mpm_grid_mass = (
                                self.mpm_solver.grid[f, mpm_grid_I, i_b].mass / self.mpm_solver.particle_volume_scale
                            )

                            mpm_weight = gs.ti_float(1.0)
                            for d in ti.static(range(3)):
                                mpm_weight *= mpm_w[mpm_offset[d]][d]

                            mpm_grid_pos = (mpm_grid_I + self.mpm_solver.grid_offset) * self.mpm_solver.dx
                            signed_dist = (mpm_grid_pos - pos).dot(surface_normal)
                            if signed_dist <= self.mpm_solver.dx:
                                vel_mpm_at_cell = mpm_weight * self.mpm_solver.grid[f, mpm_grid_I, i_b].vel_out
                                mass_mpm_at_cell = mpm_weight * mpm_grid_mass

                                vel_mpm += vel_mpm_at_cell
                                mass_mpm += mass_mpm_at_cell

                                if mass_mpm_at_cell > gs.EPS:
                                    delta_mpm_vel_at_cell_unmul = (
                                        vel_fem_sv * mpm_weight - self.mpm_solver.grid[f, mpm_grid_I, i_b].vel_out
                                    )
                                    mass_mul_at_cell = mpm_grid_mass / mass_fem_sv
                                    delta_mpm_vel_at_cell = delta_mpm_vel_at_cell_unmul * mass_mul_at_cell
                                    self.mpm_solver.grid[f, mpm_grid_I, i_b].vel_out += delta_mpm_vel_at_cell
                                    new_vel_fem_sv -= delta_mpm_vel_at_cell * mass_mpm_at_cell / mass_fem_sv

                        if mass_mpm > gs.EPS:
                            self.fem_solver.elements_v[f + 1, iv, i_b].vel = new_vel_fem_sv

                # FEM 与 SPH（实验性，效果较差）
                if ti.static(self._fem_sph):
                    for j in ti.static(range(3)):
                        iv = self.fem_solver.surface[i_s].tri2v[j]
                        pos = self.fem_solver.elements_v[f, iv, i_b].pos
                        vel_fem_sv = self.fem_solver.elements_v[f + 1, iv, i_b].vel
                        mass_fem_sv = mass / 4.0

                        dx = self.sph_solver.hash_grid_cell_size
                        stencil_size = 2
                        base = self.sph_solver.sh.pos_to_grid(pos - 0.5 * dx)

                        sph_vel = ti.Vector([0.0, 0.0, 0.0])
                        colliding_particles = 0
                        for offset in ti.grouped(ti.ndrange(stencil_size, stencil_size, stencil_size)):
                            slot_idx = self.sph_solver.sh.grid_to_slot(base + offset)
                            for k in range(
                                self.sph_solver.sh.slot_start[slot_idx, i_b],
                                self.sph_solver.sh.slot_start[slot_idx, i_b]
                                + self.sph_solver.sh.slot_size[slot_idx, i_b],
                            ):
                                if ti.abs(pos - self.sph_solver.particles_reordered.pos[k, i_b]).max() < dx * 0.5:
                                    sph_vel += self.sph_solver.particles_reordered.vel[k, i_b]
                                    colliding_particles += 1

                        if colliding_particles > 0:
                            vel_old = vel_fem_sv
                            vel_fem_sv_unprojected = sph_vel / colliding_particles
                            vel_fem_sv = vel_fem_sv_unprojected.dot(surface_normal) * surface_normal

                            delta_mv = mass_fem_sv * (vel_fem_sv - vel_old)
                            for offset in ti.grouped(ti.ndrange(stencil_size, stencil_size, stencil_size)):
                                slot_idx = self.sph_solver.sh.grid_to_slot(base + offset)
                                for k in range(
                                    self.sph_solver.sh.slot_start[slot_idx, i_b],
                                    self.sph_solver.sh.slot_start[slot_idx, i_b]
                                    + self.sph_solver.sh.slot_size[slot_idx, i_b],
                                ):
                                    if ti.abs(pos - self.sph_solver.particles_reordered.pos[k, i_b]).max() < dx * 0.5:
                                        self.sph_solver.particles_reordered[k, i_b].vel = (
                                            self.sph_solver.particles_reordered[k, i_b].vel
                                            - delta_mv / self.sph_solver.particles_info_reordered[k, i_b].mass
                                        )

                            self.fem_solver.elements_v[f + 1, iv, i_b].vel = vel_fem_sv

                # 边界速度约束
                for j in ti.static(range(3)):
                    iv = self.fem_solver.surface[i_s].tri2v[j]
                    _, self.fem_solver.elements_v[f + 1, iv, i_b].vel = self.fem_solver.boundary.impose_pos_vel(
                        self.fem_solver.elements_v[f, iv, i_b].pos, self.fem_solver.elements_v[f + 1, iv, i_b].vel
                    )

    def fem_hydroelastic(self, f: ti.i32):
        # 地面接触（流固/柔体水弹特征检测）
        self.fem_solver.floor_hydroelastic_detection(f)

    @ti.kernel
    def sph_rigid(self, f: ti.i32):
        # SPH 与刚体碰撞（鲁棒法向）
        for i_p, i_b in ti.ndrange(self.sph_solver._n_particles, self.sph_solver._B):
            if self.sph_solver.particles_ng_reordered[i_p, i_b].active:
                for i_g in range(self.rigid_solver.n_geoms):
                    if self.rigid_solver.geoms_info.needs_coup[i_g]:
                        (
                            self.sph_solver.particles_reordered[i_p, i_b].vel,
                            self.sph_rigid_normal_reordered[i_p, i_g, i_b],
                        ) = self._func_collide_with_rigid_geom_robust(
                            self.sph_solver.particles_reordered[i_p, i_b].pos,
                            self.sph_solver.particles_reordered[i_p, i_b].vel,
                            self.sph_solver.particles_info_reordered[i_p, i_b].mass,
                            self.sph_rigid_normal_reordered[i_p, i_g, i_b],
                            i_g,
                            i_b,
                        )

    @ti.kernel
    def kernel_pbd_rigid_collide(self):
        # PBD 与刚体的简单位置-速度校正
        for i_p, i_b in ti.ndrange(self.pbd_solver._n_particles, self.sph_solver._B):
            if self.pbd_solver.particles_ng_reordered[i_p, i_b].active:
                for i_g in range(self.rigid_solver.n_geoms):
                    if self.rigid_solver.geoms_info.needs_coup[i_g]:
                        (
                            self.pbd_solver.particles_reordered[i_p, i_b].pos,
                            self.pbd_solver.particles_reordered[i_p, i_b].vel,
                            self.pbd_rigid_normal_reordered[i_p, i_b, i_g],
                        ) = self._func_pbd_collide_with_rigid_geom(
                            i_p,
                            self.pbd_solver.particles_reordered[i_p, i_b].pos,
                            self.pbd_solver.particles_reordered[i_p, i_b].vel,
                            self.pbd_solver.particles_info_reordered[i_p, i_b].mass,
                            self.pbd_rigid_normal_reordered[i_p, i_b, i_g],
                            i_g,
                            i_b,
                        )

    @ti.kernel
    def kernel_attach_pbd_to_rigid_link(
        self,
        particles_idx: ti.types.ndarray(),
        envs_idx: ti.types.ndarray(),
        link_idx: ti.i32,
        links_state: LinksState,
    ) -> None:
        """
        绑定指定环境中给定粒子到某链节（记录其相对局部位置用于后续动画驱动）。
        """
        pdb = self.pbd_solver

        for i_p_, i_b_ in ti.ndrange(particles_idx.shape[1], envs_idx.shape[0]):
            i_p = particles_idx[i_b_, i_p_]
            i_b = envs_idx[i_b_]
            link_pos = links_state.pos[link_idx, i_b]
            link_quat = links_state.quat[link_idx, i_b]

            world_pos = pdb.particles[i_p, i_b].pos
            local_pos = ti_inv_transform_by_trans_quat(world_pos, link_pos, link_quat)

            pdb.particles[i_p, i_b].free = False
            self.particle_attach_info[i_p, i_b].link_idx = link_idx
            self.particle_attach_info[i_p, i_b].local_pos = local_pos

    @ti.kernel
    def kernel_pbd_rigid_clear_animate_particles_by_link(
        self,
        particles_idx: ti.types.ndarray(),
        envs_idx: ti.types.ndarray(),
    ) -> None:
        """
        解除粒子与链节绑定，恢复自由模拟。
        """
        pdb = self.pbd_solver
        for i_p_, i_b_ in ti.ndrange(particles_idx.shape[1], envs_idx.shape[0]):
            i_p = particles_idx[i_b_, i_p_]
            i_b = envs_idx[i_b_]
            pdb.particles[i_p, i_b].free = True
            self.particle_attach_info[i_p, i_b].link_idx = -1
            self.particle_attach_info[i_p, i_b].local_pos = ti.math.vec3([0.0, 0.0, 0.0])

    @ti.kernel
    def kernel_pbd_rigid_solve_animate_particles_by_link(self, clamped_inv_dt: ti.f32, links_state: LinksState):
        """
        根据链节最新状态为绑定粒子施加“纠正速度”，使其追随链节运动（位置滞后一帧）。
        """
        pdb = self.pbd_solver
        for i_p, i_env in ti.ndrange(pdb._n_particles, pdb._B):
            if self.particle_attach_info[i_p, i_env].link_idx >= 0:
                link_idx = self.particle_attach_info[i_p, i_env].link_idx
                link_pos = links_state.pos[link_idx, i_env]
                link_quat = links_state.quat[link_idx, i_env]

                link_lin_vel = links_state.cd_vel[link_idx, i_env]
                link_ang_vel = links_state.cd_ang[link_idx, i_env]
                link_com_in_world = links_state.root_COM[link_idx, i_env] + links_state.i_pos[link_idx, i_env]

                local_pos = self.particle_attach_info[i_p, i_env].local_pos
                target_world_pos = ti_transform_by_trans_quat(local_pos, link_pos, link_quat)

                world_arm = target_world_pos - link_com_in_world
                target_world_vel = link_lin_vel + link_ang_vel.cross(world_arm)

                i_rp = pdb.particles_ng[i_p, i_env].reordered_idx
                particle_pos = pdb.particles_reordered[i_rp, i_env].pos
                pos_correction = target_world_pos - particle_pos
                corrective_vel = pos_correction * clamped_inv_dt
                old_vel = pdb.particles_reordered[i_rp, i_env].vel
                pdb.particles_reordered[i_rp, i_env].vel = corrective_vel + target_world_vel

    @ti.func
    def _func_pbd_collide_with_rigid_geom(self, i, pos_world, vel, mass, normal_prev, geom_idx, batch_idx):
        """
        PBD 粒子与刚体的穿透修正（位置校正 + 作用力回传）。
        """
        signed_dist = sdf_decomp.sdf_func_world(
            geoms_state=self.rigid_solver.geoms_state,
            geoms_info=self.rigid_solver.geoms_info,
            sdf_info=self.rigid_solver.sdf._sdf_info,
            pos_world=pos_world,
            geom_idx=geom_idx,
            batch_idx=batch_idx,
        )
        vel_rigid = self.rigid_solver._func_vel_at_point(
            pos_world=pos_world,
            link_idx=self.rigid_solver.geoms_info.link_idx[geom_idx],
            i_b=batch_idx,
            links_state=self.rigid_solver.links_state,
        )
        contact_normal = sdf_decomp.sdf_func_normal_world(
            geoms_state=self.rigid_solver.geoms_state,
            geoms_info=self.rigid_solver.geoms_info,
            collider_static_config=self.rigid_solver.collider._collider_static_config,
            sdf_info=self.rigid_solver.sdf._sdf_info,
            pos_world=pos_world,
            geom_idx=geom_idx,
            batch_idx=batch_idx,
        )
        new_pos = pos_world
        new_vel = vel
        if signed_dist < self.pbd_solver.particle_size / 2:
            stiffness = 1.0
            energy_loss = 0.0
            new_pos = pos_world + stiffness * contact_normal * (self.pbd_solver.particle_size / 2 - signed_dist)
            prev_pos = self.pbd_solver.particles_reordered[i, batch_idx].ipos
            new_vel = (new_pos - prev_pos) / self.pbd_solver._substep_dt

            delta_mv = mass * (new_vel - vel)
            force = (-delta_mv / self.rigid_solver._substep_dt) * (1 - energy_loss)

            self.rigid_solver._func_apply_external_force(
                pos_world,
                force,
                self.rigid_solver.geoms_info.link_idx[geom_idx],
                batch_idx,
                self.rigid_solver.links_state,
            )

        return new_pos, new_vel, contact_normal

    def preprocess(self, f):
        # 预处理：MPM CPIC 下将法向写入缓冲
        if self.mpm_solver.is_active() and self.rigid_solver.is_active() and self.mpm_solver.enable_CPIC:
            self.mpm_surface_to_particle(f)

    def couple(self, f):
        # 统一调度各耦合步骤
        if self.mpm_solver.is_active():
            self.mpm_grid_op(f, self.sim.cur_t)

        if self._rigid_sph and self.rigid_solver.is_active():
            self.sph_rigid(f)

        if self._rigid_pbd and self.rigid_solver.is_active():
            self.kernel_pbd_rigid_collide()
            full_step_inv_dt = 1.0 / self.pbd_solver._dt
            clamped_inv_dt = min(full_step_inv_dt, CLAMPED_INV_DT)
            self.kernel_pbd_rigid_solve_animate_particles_by_link(clamped_inv_dt, self.rigid_solver.links_state)

        if self.fem_solver.is_active():
            self.fem_surface_force(f)
            self.fem_rigid_link_constraints()

    def couple_grad(self, f):
        # 反向传播版本（目前仅对少数核启用）
        if self.mpm_solver.is_active():
            self.mpm_grid_op.grad(f, self.sim.cur_t)

        if self.fem_solver.is_active():
            self.fem_surface_force.grad(f)

    @property
    def active_solvers(self):
        """当前激活的求解器集合（来自上层模拟器）。"""
        return self.sim.active_solvers