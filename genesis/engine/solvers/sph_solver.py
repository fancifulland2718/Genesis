import numpy as np
import gstaichi as ti

import genesis as gs
import genesis.utils.geom as gu
from genesis.engine.boundaries import CubeBoundary
from genesis.engine.entities import SPHEntity
from genesis.engine.states.solvers import SPHSolverState

from .base_solver import Solver


@ti.data_oriented
class SPHSolver(Solver):
    """
    基于 SPH（Smoothed Particle Hydrodynamics）/ DFSPH 的流体求解器。
    主要职责：
    - 维护 SPH 粒子（动态/静态信息）与哈希邻域搜索数据结构。
    - 在每个子步中执行邻域查询、密度/压力与非压力力的计算、时间推进、边界约束。
    - 支持两种压力解法：
      - WCSPH（弱可压）：状态方程直接给出压力，按对称形式累加压力力；
      - DFSPH（Divergence-free SPH）：通过迭代（散度解 + 密度解）强制速度场近似无散、密度接近静态密度。

    子步前后阶段管线：
    - pre_coupling（WCSPH）:
      1) 重新排序粒子（基于空间哈希，加速邻域查询）
      2) 计算密度 _kernel_compute_rho（核函数求和）
      3) 计算非压力力（重力/粘性/表面张力/外力场）_kernel_compute_non_pressure_forces
      4) 计算压力与压力力 _kernel_compute_pressure_forces（状态方程 + 对称压力项）
      5) 速度推进 _kernel_advect_velocity
    - pre_coupling（DFSPH）:
      1) 重新排序
      2) 计算密度
      3) 计算 DFSPH 因子（压力刚度分母）_kernel_compute_DFSPH_factor
      4) 散度解 _divergence_solve（迭代修正速度，使密度时间导数≈0）
      5) 速度推进（非压力项）
      6) 预测速度 _kernel_predict_velocity
      7) 密度解 _density_solve（迭代修正速度，使密度≈ρ0）
    - post_coupling：
      1) 位置推进 + 边界投影 _kernel_advect_position
      2) 从重排数组拷回原布局 _kernel_copy_from_reordered
    """

    # ------------------------------------------------------------------------------------
    # --------------------------------- Initialization -----------------------------------
    # ------------------------------------------------------------------------------------

    def __init__(self, scene, sim, options):
        """
        参数
        - scene: 场景
        - sim:   仿真器
        - options: 包含粒子尺寸、支持半径、压力解法、边界、DFSPH 迭代阈值等
        """
        super().__init__(scene, sim, options)

        # options
        self._particle_size = options.particle_size
        self._support_radius = options._support_radius
        self._pressure_solver = options.pressure_solver

        # DFSPH 参数
        self._df_max_error_div = options.max_divergence_error          # 允许的最大散度误差（百分比）
        self._df_max_error_den = options.max_density_error_percent     # 允许的最大密度误差（百分比）
        self._df_max_div_iters = options.max_divergence_solver_iterations
        self._df_max_den_iters = options.max_density_solver_iterations
        self._df_eps = 1e-5                                            # 数值稳定阈值

        self._upper_bound = np.array(options.upper_bound)
        self._lower_bound = np.array(options.lower_bound)

        # 经验粒子体积（影响质量/密度估计），0.8 为经验系数
        self._particle_volume = 0.8 * self._particle_size**3  # 0.8 is an empirical value

        # 空间哈希（邻域搜索）
        self.sh = gu.SpatialHasher(
            cell_size=options.hash_grid_cell_size,
            grid_res=options._hash_grid_res,
        )
        # 边界
        self.setup_boundary()

    def _batch_shape(self, shape=None, first_dim=False, B=None):
        """
        构造带批次维（B）的 shape 工具。
        - first_dim=True 时将 B 放前，否则放后；
        - shape 为 None/序列/整数时分别处理。
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
        """设置立方体边界（可在 impose_pos_vel 中进行位置/速度投影）。"""
        self.boundary = CubeBoundary(
            lower=self._lower_bound,
            upper=self._upper_bound,
            # restitution=0.5,
        )

    def init_particle_fields(self):
        """
        初始化各类粒子字段：
        - 动态状态（particles）：pos/vel/acc/rho/p/dfsph_factor/drho
        - 动态标志（particles_ng）：active、重排后的索引 reordered_idx
        - 静态信息（particles_info）：静态密度、质量、状态方程参数（stiffness/exponent）、粘性 mu、表面张力 gamma
        - 重排缓冲（*_reordered）：同上三类字段的重排版本
        - 渲染字段（particles_render）：当前帧渲染所需 pos/vel/active
        """
        # dynamic particle state
        struct_particle_state = ti.types.struct(
            pos=gs.ti_vec3,  # position
            vel=gs.ti_vec3,  # velocity
            acc=gs.ti_vec3,  # acceleration
            rho=gs.ti_float,  # density
            p=gs.ti_float,  # pressure
            dfsph_factor=gs.ti_float,  # DFSPH use: Factor for Divergence and density solver
            drho=gs.ti_float,  # density deritivate
        )

        # dynamic particle state without gradient
        struct_particle_state_ng = ti.types.struct(
            reordered_idx=gs.ti_int,  # 重排后的索引（哈希栅格排序）
            active=gs.ti_bool,        # 活跃标志
        )

        # static particle info
        struct_particle_info = ti.types.struct(
            rho=gs.ti_float,      # rest density（静态密度 ρ0）
            mass=gs.ti_float,     # 质量（≈ ρ0 * 体积）
            stiffness=gs.ti_float,# 状态方程刚度
            exponent=gs.ti_float, # 状态方程指数（Tait 方程）
            mu=gs.ti_float,       # 粘性系数
            gamma=gs.ti_float,    # 表面张力系数
        )

        # single frame particle state for rendering
        struct_particle_state_render = ti.types.struct(
            pos=gs.ti_vec3,
            vel=gs.ti_vec3,
            active=gs.ti_bool,
        )

        # 构造字段
        self.particles = struct_particle_state.field(
            shape=self._batch_shape((self._n_particles,)), needs_grad=False, layout=ti.Layout.SOA
        )
        self.particles_ng = struct_particle_state_ng.field(
            shape=self._batch_shape((self._n_particles,)), needs_grad=False, layout=ti.Layout.SOA
        )
        self.particles_info = struct_particle_info.field(
            shape=(self._n_particles,), needs_grad=False, layout=ti.Layout.SOA
        )
        self.particles_reordered = struct_particle_state.field(
            shape=self._batch_shape((self._n_particles,)), needs_grad=False, layout=ti.Layout.SOA
        )
        self.particles_ng_reordered = struct_particle_state_ng.field(
            shape=self._batch_shape((self._n_particles,)), needs_grad=False, layout=ti.Layout.SOA
        )
        self.particles_info_reordered = struct_particle_info.field(
            shape=self._batch_shape((self._n_particles,)), needs_grad=False, layout=ti.Layout.SOA
        )

        self.particles_render = struct_particle_state_render.field(
            shape=self._batch_shape((self._n_particles,)), needs_grad=False, layout=ti.Layout.SOA
        )

    def init_ckpt(self):
        """检查点缓存（当前未用）。"""
        self._ckpt = dict()

    def reset_grad(self):
        """SPH 求解器当前不支持可微，留空。"""
        pass

    def build(self):
        """
        构建阶段：
        - 初始化批大小 B、粒子总数、耦合器引用；
        - 构建空间哈希，初始化各字段、检查点；
        - 将实体注册到求解器；
        - 读取静态密度（暂按每粒子相同处理）。
        """
        super().build()
        self._B = self._sim._B

        # particles and entities
        self._n_particles = self.n_particles

        self._coupler = self.sim._coupler

        if self.is_active():
            self.sh.build(self._B)
            self.init_particle_fields()
            self.init_ckpt()

            for entity in self.entities:
                entity._add_to_solver()

            # TODO: 支持每粒子独立密度
            self._density0 = self.particles_info[0].rho

    # ------------------------------------------------------------------------------------
    # -------------------------------------- misc ----------------------------------------
    # ------------------------------------------------------------------------------------

    def add_entity(self, idx, material, morph, surface):
        """
        添加一个 SPH 实体并返回。
        - 记录粒子起始下标（连续区间）
        - 注：Solver 基类中通常维护 entities/_entities
        """
        entity = SPHEntity(
            scene=self.scene,
            solver=self,
            material=material,
            morph=morph,
            surface=surface,
            particle_size=self._particle_size,
            idx=idx,
            particle_start=self.n_particles,
        )

        self.entities.append(entity)
        return entity

    def is_active(self):
        """是否有粒子（至少 1 个）"""
        return self.n_particles > 0

    # ------------------------------------------------------------------------------------
    # ----------------------------------- simulation -------------------------------------
    # ------------------------------------------------------------------------------------

    @ti.kernel
    def _kernel_reorder_particles(self, f: ti.i32):
        """
        基于空间哈希对粒子进行重排（沿 Morton/栅格顺序），以提升邻域访问局部性。
        - 先计算每粒子的 reordered_idx；
        - 将动态状态/静态信息拷贝到重排缓冲；
        - 若与刚体耦合（rigid_sph），同步重排后的法线缓存。
        """
        self.sh.compute_reordered_idx(
            self._n_particles, self.particles.pos, self.particles_ng.active, self.particles_ng.reordered_idx
        )

        # 初始化重排缓冲的 active 状态
        self.particles_ng_reordered.active.fill(False)

        # 拷贝到重排缓冲
        for i_p, i_b in ti.ndrange(self._n_particles, self._B):
            if self.particles_ng[i_p, i_b].active:
                reordered_idx = self.particles_ng[i_p, i_b].reordered_idx

                self.particles_reordered[reordered_idx, i_b] = self.particles[i_p, i_b]
                self.particles_info_reordered[reordered_idx, i_b] = self.particles_info[i_p]
                self.particles_ng_reordered[reordered_idx, i_b].active = self.particles_ng[i_p, i_b].active

        # 耦合数据重排（若开启刚体-SPH 耦合）
        if ti.static(self._coupler._rigid_sph):
            for i_p, i_g, i_b in ti.ndrange(self._n_particles, self._coupler.rigid_solver.n_geoms, self._B):
                if self.particles_ng[i_p, i_b].active:
                    self._coupler.sph_rigid_normal_reordered[self.particles_ng[i_p, i_b].reordered_idx, i_g, i_b] = (
                        self._coupler.sph_rigid_normal[i_p, i_g, i_b]
                    )

    @ti.kernel
    def _kernel_copy_from_reordered(self, f: ti.i32):
        """
        从重排缓冲拷贝回原布局（仅动态状态），并同步耦合法线。
        """
        for i_p, i_b in ti.ndrange(self._n_particles, self._B):
            if self.particles_ng[i_p, i_b].active:
                # 仅需拷贝动态状态
                self.particles[i_p, i_b] = self.particles_reordered[self.particles_ng[i_p, i_b].reordered_idx, i_b]

        if ti.static(self._coupler._rigid_sph):
            for i_p, i_g, i_b in ti.ndrange(self._n_particles, self._coupler.rigid_solver.n_geoms, self._B):
                if self.particles_ng[i_p, i_b].active:
                    self._coupler.sph_rigid_normal[i_p, i_g, i_b] = self._coupler.sph_rigid_normal_reordered[
                        self.particles_ng[i_p, i_b].reordered_idx, i_g, i_b
                    ]

    @ti.func
    def _task_compute_rho(self, i, j, ret: ti.template(), i_b):
        """
        密度积累的邻域任务：
        ρ_i += V * W(|x_i - x_j|)
        """
        ret += self._particle_volume * self.cubic_kernel(
            (self.particles_reordered[i, i_b].pos - self.particles_reordered[j, i_b].pos).norm()
        )

    @ti.kernel
    def _kernel_compute_rho(self, f: ti.i32):
        """
        计算粒子密度 ρ：核函数在邻域内求和。
        - 先加上自身核值（r=0）；
        - 遍历邻域累加；
        - 乘以静态密度 ρ0（若使用相对核密度）。
        """
        for i_p, i_b in ti.ndrange(self._n_particles, self._B):
            if self.particles_ng_reordered[i_p, i_b].active:
                # 自身核（r=0）
                self.particles_reordered[i_p, i_b].rho = self._particle_volume * self.cubic_kernel(0.0)

                den = gs.ti_float(0.0)
                self.sh.for_all_neighbors(
                    i_p, self.particles_reordered.pos, self._support_radius, den, self._task_compute_rho, i_b
                )
                self.particles_reordered[i_p, i_b].rho += den

                # 若 ρ 存储为“相对密度”，此处乘以 ρ0 转物理密度
                self.particles_reordered[i_p, i_b].rho *= self.particles_info_reordered[i_p, i_b].rho

    @ti.func
    def _task_compute_non_pressure_forces(self, i, j, ret: ti.template(), i_b: ti.i32):
        """
        非压力力邻域项：包含
        - 表面张力（简化为核函数权重沿相对位移方向）；
        - 粘性力（速度差沿连线方向的粘附项，带核梯度）。
        返回累积加速度增量。
        """
        d_ij = self.particles_reordered[i, i_b].pos - self.particles_reordered[j, i_b].pos
        dist = d_ij.norm()

        gamma_i = self.particles_info_reordered[i, i_b].gamma
        mass_i = self.particles_info_reordered[i, i_b].mass
        mu_i = self.particles_info_reordered[i, i_b].mu

        mass_j = self.particles_info_reordered[j, i_b].mass

        # -----------------------------
        # Surface Tension term（表面张力）
        # -----------------------------
        # 小距离下夹紧有效距离，避免过强
        effective_dist = dist if dist > self._particle_size else self._particle_size
        ret -= gamma_i / mass_i * mass_j * d_ij * self.cubic_kernel(effective_dist)

        # -----------------------------
        # Viscosity Force（粘性）
        # -----------------------------
        v_ij = (self.particles_reordered[i, i_b].vel - self.particles_reordered[j, i_b].vel).dot(d_ij)

        # 粘性公式中的常数（与维度相关，此处经验设定）
        d = 2 * (3 + 2)

        # 邻居 j 的密度
        rho_j = self.particles_reordered[j, i_b].rho

        f_v = (
            d
            * mu_i
            * (mass_j / rho_j)
            * v_ij
            / (dist**2 + 0.01 * self._support_radius**2)
            * self.cubic_kernel_derivative(d_ij)
        )
        ret += f_v

    @ti.kernel
    def _kernel_compute_non_pressure_forces(self, f: ti.i32, t: ti.f32):
        """
        计算非压力力加速度：重力 + 表面张力 + 粘性 + 外部力场。
        """
        for i_p, i_b in ti.ndrange(self._n_particles, self._B):
            if self.particles_ng_reordered[i_p, i_b].active:
                acc = self._gravity[i_b]
                self.sh.for_all_neighbors(
                    i_p,
                    self.particles_reordered.pos,
                    self._support_radius,
                    acc,
                    self._task_compute_non_pressure_forces,
                    i_b,
                )

                # 外部力场（逐力场采样加速度）
                for i_ff in ti.static(range(len(self._ffs))):
                    acc += self._ffs[i_ff].get_acc(
                        self.particles_reordered[i_p, i_b].pos, self.particles_reordered[i_p, i_b].vel, t, i_p
                    )
                self.particles_reordered[i_p, i_b].acc = acc

    @ti.func
    def _task_compute_pressure_forces(self, i, j, ret: ti.template(), i_b):
        """
        压力力邻域项（对称形式）：
        f_i += -ρ0 * V * (p_i/ρ_i^2 + p_j/ρ_j^2) ∇W(x_i - x_j)
        """
        dp_i = self.particles_reordered[i, i_b].p / self.particles_reordered[i, i_b].rho ** 2
        rho_j = (
            self.particles_reordered[j, i_b].rho
            * self.particles_info_reordered[j, i_b].rho
            / self.particles_info_reordered[j, i_b].rho
        )
        dp_j = self.particles_reordered[j, i_b].p / rho_j**2

        ret += (
            -self.particles_info_reordered[j, i_b].rho
            * self._particle_volume
            * (dp_i + dp_j)
            * self.cubic_kernel_derivative(self.particles_reordered[i, i_b].pos - self.particles_reordered[j, i_b].pos)
        )

    @ti.kernel
    def _kernel_compute_pressure_forces(self, f: ti.i32):
        """
        WCSPH 压力与压力力：
        - 用状态方程 p = k[(ρ/ρ0)^n - 1] 计算压力（ρ >= ρ0）；
        - 用对称形式累加压力力到 acc。
        """
        # 状态方程计算压力
        for i_p, i_b in ti.ndrange(self._n_particles, self._B):
            if self.particles_ng_reordered[i_p, i_b].active:
                rho0 = self.particles_info_reordered[i_p, i_b].rho
                stiff = self.particles_info_reordered[i_p, i_b].stiffness
                expnt = self.particles_info_reordered[i_p, i_b].exponent

                self.particles_reordered[i_p, i_b].rho = ti.max(self.particles_reordered[i_p, i_b].rho, rho0)

                self.particles_reordered[i_p, i_b].p = stiff * (
                    ti.pow(self.particles_reordered[i_p, i_b].rho / rho0, expnt) - 1.0
                )

        # 压力力累加
        for i_p, i_b in ti.ndrange(self._n_particles, self._B):
            if self.particles_ng_reordered[i_p, i_b].active:
                acc = ti.Vector.zero(gs.ti_float, 3)

                self.sh.for_all_neighbors(
                    i_p,
                    self.particles_reordered.pos,  # shape [n_particles, B, 3] or similar
                    self._support_radius,
                    acc,
                    self._task_compute_pressure_forces,
                    i_b,
                )
                self.particles_reordered[i_p, i_b].acc += acc

    @ti.kernel
    def _kernel_advect_velocity(self, f: ti.i32):
        """速度推进：v += dt * a（仅依靠当前 acc）。"""
        for i_p, i_b in ti.ndrange(self._n_particles, self._B):
            if self.particles_ng_reordered[i_p, i_b].active:
                self.particles_reordered[i_p, i_b].vel += self._substep_dt * self.particles_reordered[i_p, i_b].acc

    @ti.kernel
    def _kernel_advect_position(self, f: ti.i32):
        """
        位置推进与边界投影：
        - x_new = x + dt * v
        - impose_pos_vel 对位置/速度进行边界约束，返回修正后的结果。
        """
        for i_p, i_b in ti.ndrange(self._n_particles, self._B):
            if self.particles_ng_reordered[i_p, i_b].active:
                pos = self.particles_reordered[i_p, i_b].pos
                new_vel = self.particles_reordered[i_p, i_b].vel

                # advect
                new_pos = pos + self._substep_dt * new_vel

                # impose boundary (pass b if needed)
                corrected_pos, corrected_vel = self.boundary.impose_pos_vel(new_pos, new_vel)

                # update
                self.particles_reordered[i_p, i_b].vel = corrected_vel
                self.particles_reordered[i_p, i_b].pos = corrected_pos

    # ------------------------------------------------------------------------------------
    # ------------------------------------- DFSPH ----------------------------------------
    # ------------------------------------------------------------------------------------
    @ti.func
    def _task_compute_DFSPH_factor(self, i, j, ret: ti.template(), i_b):
        """
        DFSPH 因子计算的邻域任务：
        - 汇总∑|∇W|^2 与 ∑∇W，用于构造压强“刚度”的分母项：factor = -1/(∑|∇p_k|^2 + |∑∇p_i|^2)
        """
        # Fluid neighbors
        grad_j = -self._particle_volume * self.cubic_kernel_derivative(
            self.particles_reordered[i, i_b].pos - self.particles_reordered[j, i_b].pos
        )
        ret[3] += grad_j.norm_sqr()  # sum_grad_p_k
        for ii in ti.static(range(3)):  # grad_p_i
            ret[ii] -= grad_j[ii]

    @ti.kernel
    def _kernel_compute_DFSPH_factor(self, f: ti.i32):
        """
        计算 DFSPH 因子（压力刚度的分母）并写入 dfsph_factor：
        factor_i = -1 / (∑_k |∇W_ik|^2 + |∑_k ∇W_ik|^2)
        """
        for i_p, i_b in ti.ndrange(self._n_particles, self._B):
            if self.particles_ng_reordered[i_p, i_b].active:
                sum_grad_p_k = gs.ti_float(0.0)
                grad_p_i = ti.Vector.zero(gs.ti_float, 3)

                # ret 拼接了 grad_p_i（前三分量）与 sum_grad_p_k（最后一分量）
                ret = ti.Vector.zero(gs.ti_float, 4)

                # 注意：此处调用应传 i_p（源码使用 i 可能为笔误），不改动逻辑，仅注释说明。
                self.sh.for_all_neighbors(
                    i, self.particles_reordered.pos, self._support_radius, ret, self._task_compute_DFSPH_factor
                )

                sum_grad_p_k = ret[3]
                for ii in ti.static(range(3)):
                    grad_p_i[ii] = ret[ii]
                sum_grad_p_k += grad_p_i.norm_sqr()

                # 计算分母项
                factor = gs.ti_float(0.0)
                if sum_grad_p_k > 1e-6:
                    factor = -1.0 / sum_grad_p_k
                else:
                    factor = 0.0
                self.particles_reordered[i_p, i_b].dfsph_factor = factor

    @ti.func
    def _task_compute_density_time_derivative(self, i, j, ret: ti.template(), i_b):
        """
        计算密度时间导数的邻域任务（用于散度解）：
        drho_i/dt += V (v_i - v_j) · ∇W(x_i - x_j)
        同时统计邻居数量，避免粒子稀疏时不稳定。
        """
        v_i = self.particles_reordered[i, i_b].vel
        v_j = self.particles_reordered[j, i_b].vel

        x_i = self.particles_reordered[i, i_b].pos
        x_j = self.particles_reordered[j, i_b].pos

        # Fluid neighbors
        ret.drho += self._particle_volume * (v_i - v_j).dot(self.cubic_kernel_derivative(x_i - x_j))
        ret.num_neighbors += 1

    @ti.kernel
    def _kernel_compute_density_time_derivative(self):
        """
        计算 drho（密度时间导数，取正值），稀疏邻居（<20）时强制为 0，避免散度解失真。
        """
        for i_p, i_b in ti.ndrange(self._n_particles, self._B):
            if self.particles_ng_reordered[i_p, i_b].active:
                ret = ti.Struct(drho=0.0, num_neighbors=0)
                self.sh.for_all_neighbors(
                    i_p,
                    self.particles_reordered.pos,
                    self._support_radius,
                    ret,
                    self._task_compute_density_time_derivative,
                    i_b,
                )

                # 仅修正正散度
                drho = ti.max(ret.drho, 0.0)
                num_neighbors = ret.num_neighbors

                # 粒子缺失时跳过散度修正
                if num_neighbors < 20:
                    drho = 0.0

                self.particles_reordered[i_p, i_b].drho = drho

    @ti.func
    def _task_divergence_solver_iteration(self, i, j, ret: ti.template(), i_b):
        """
        散度解迭代的邻域任务（Jacobi 迭代）：
        - 利用 k_i = b_i * factor_i，k_j 同理；
        - 累加速度增量 dv -= (k_i + k_j) ∇W_ij
        注：k_* 已包含密度的逆，因此无需再除密度（多相流另议）。
        """
        # Fluid neighbors
        b_j = self.particles_reordered[j, i_b].drho
        k_j = b_j * self.particles_reordered[j, i_b].dfsph_factor
        k_sum = (
            self._density0 / self._density0 * ret.k_i + k_j
        )  # TODO: 多相流时使用不同静态密度
        if ti.abs(k_sum) > self._df_eps:
            grad_p_j = -self._particle_volume * self.cubic_kernel_derivative(
                self.particles_reordered.pos[i, i_b] - self.particles_reordered.pos[j, i_b]
            )
            ret.dv -= (
                k_sum * grad_p_j
            )  # ki, kj 已含密度逆
    @ti.kernel
    def _kernel_divergence_solver_iteration(self):
        """
        散度解一次 Jacobi 迭代：
        - 构造 k_i = b_i * factor_i；
        - 邻域累加 dv；
        - 速度增量回写：v += dv。
        """
        # Perform Jacobi iteration
        for i_p, i_b in ti.ndrange(self._n_particles, self._B):
            if self.particles_ng_reordered[i_p, i_b].active:
                # evaluate rhs
                b_i = self.particles_reordered[i_p, i_b].drho
                k_i = b_i * self.particles_reordered[i_p, i_b].dfsph_factor
                ret = ti.Struct(dv=ti.Vector.zero(gs.ti_float, 3), k_i=k_i)
                # TODO: 若需 warm start，可在此加入历史项
                self.sh.for_all_neighbors(
                    i_p, self.particles_reordered.pos, self._support_radius, ret, self._task_divergence_solver_iteration
                )
                self.particles_reordered.vel[i_p, i_b] = self.particles_reordered.vel[i_p, i_b] + ret.dv

    @ti.kernel
    def _kernel_compute_density_error(self, offset: float) -> float:
        """
        统计密度误差（按 drho 聚合），offset=0 用于散度解，offset=ρ0 用于密度解。
        返回总和（将被平均）。
        """
        density_error = gs.ti_float(0.0)
        for i_p, i_b in ti.ndrange(self._n_particles, self._B):
            if self.particles_ng_reordered[i_p, i_b].active:
                density_error += self._density0 * self.particles_reordered[i_p, i_b].drho - offset
        return density_error

    def _divergence_solver_iteration(self):
        """
        散度解子迭代（python 容器封装）：
        - 执行一次 Jacobi；
        - 更新 drho；
        - 计算平均密度误差（偏移 0）。
        """
        self._kernel_divergence_solver_iteration()
        self._kernel_compute_density_time_derivative()
        density_err = self._kernel_compute_density_error(0.0)
        return density_err / self._n_particles

    def _divergence_solve(self, f: ti.i32):
        """
        DFSPH 散度解主循环：
        - 预计算 drho（密度时间导数）；
        - 按迭代上限与误差阈值停止；
        - 输出日志（迭代轮数与平均误差）。
        """
        # TODO: warm start
        # Compute velocity of density change
        self._kernel_compute_density_time_derivative()
        inv_dt = 1 / self._substep_dt
        # self._kernel_multiply_time_step(self.ps.dfsph_factor, inv_dt)

        # Start solver
        iteration = gs.ti_int(0)
        avg_density_err = gs.ti_float(0.0)
        while iteration < self._df_max_div_iters:

            avg_density_err = self._divergence_solver_iteration()
            # 允许的最大散度误差（量纲约为 s^-1：用 η = (max_divergence_error% * ρ0) / dt）
            eta = inv_dt * self._df_max_error_div * 0.01 * self._density0
            if avg_density_err <= eta:
                break
            iteration += 1

        gs.logger.debug(f"DFSPH - iteration V: {iteration} Avg divergence err: {avg_density_err / self._density0:.4f}")

        # 注：若使用 warm start，此处可处理时间步缩放等操作。

    @ti.kernel
    def _kernel_predict_velocity(self, f: ti.i32):
        """仅凭非压力加速度对速度进行一次预测（DFSPH 密度解前的预测步）。"""
        # compute new velocities only considering non-pressure forces
        for i_p, i_b in ti.ndrange(self._n_particles, self._B):
            if self.particles_ng_reordered[i_p, i_b].active:
                self.particles_reordered[i_p, i_b].vel += self._substep_dt * self.particles_reordered[i_p, i_b].acc

    @ti.func
    def _task_compute_density_star(self, i, j, ret: ti.template(), i_b):
        """
        密度解中 ρ* 的邻域项：
        ρ* ≈ ρ/ρ0 + dt ∑ V (v_i - v_j)·∇W
        """
        v_i = self.particles_reordered[i, i_b].vel
        v_j = self.particles_reordered[j, i_b].vel
        x_i = self.particles_reordered[i, i_b].pos
        x_j = self.particles_reordered[j, i_b].pos
        ret += self._particle_volume * (v_i - v_j).dot(self.cubic_kernel_derivative(x_i - x_j))

    @ti.kernel
    def _kernel_compute_density_star(self):
        """
        计算 ρ* 并写入 drho（临时存储）：
        drho_i = max(ρ*/1, 1.0)
        """
        for i_p, i_b in ti.ndrange(self._n_particles, self._B):
            if self.particles_ng_reordered[i_p, i_b].active:
                delta = gs.ti_float(0.0)
                self.sh.for_all_neighbors(
                    i_p, self.particles_reordered.pos, self._support_radius, delta, self._task_compute_density_star, i_b
                )
                drho = self.particles_reordered[i_p, i_b].rho / self._density0 + self._substep_dt * delta
                self.particles_reordered[i_p, i_b].drho = ti.max(drho, 1.0)  # - 1.0

    @ti.func
    def density_solve_iteration_task(self, i, j, ret: ti.template(), i_b):
        """
        密度解迭代邻域任务：
        - 使用 b = ρ* - 1，k = b * factor；
        - dv -= dt * (k_i + k_j) ∇W_ij
        """
        # Fluid neighbors
        b_j = self.particles_reordered[j, i_b].drho - 1.0
        k_j = b_j * self.particles_reordered[j, i_b].dfsph_factor
        k_sum = (
            self._density0 / self._density0 * ret.k_i + k_j
        )  # TODO: 多相流时使用不同静态密度
        if ti.abs(k_sum) > self._df_eps:
            grad_p_j = -self._particle_volume * self.cubic_kernel_derivative(
                self.particles_reordered[i, i_b].pos - self.particles_reordered[j, i_b].pos
            )
            # 直接更新速度（不存压力加速度）
            ret.dv -= (
                self._substep_dt * k_sum * grad_p_j
            )  # ki, kj 已含密度逆

    @ti.kernel
    def _kernel_density_solve_iteration(self):
        """
        密度解一次迭代：
        - 构造 k_i = (ρ* - 1) * factor_i；
        - 邻域累加 dv 并回写 v += dv。
        """
        # Compute pressure forces
        for i_p, i_b in ti.ndrange(self._n_particles, self._B):
            if self.particles_ng_reordered[i_p, i_b].active:
                # Evaluate rhs
                b_i = self.particles_reordered[i_p, i_b].drho - 1.0
                k_i = b_i * self.particles_reordered[i_p, i_b].dfsph_factor

                ret = ti.Struct(dv=ti.Vector.zero(gs.ti_float, 3), k_i=k_i)

                # TODO: warmstart 时可引入历史项
                self.sh.for_all_neighbors(
                    i_p, self.particles_reordered.pos, self._support_radius, ret, self.density_solve_iteration_task, i_b
                )
                self.particles_reordered[i_p, i_b].vel = self.particles_reordered[i_p, i_b].vel + ret.dv

    def _density_solve_iteration(self):
        """
        密度解子迭代：
        - 执行一次密度解迭代；
        - 计算 ρ*；
        - 统计平均密度误差（偏移 ρ0）。
        """
        self._kernel_density_solve_iteration()
        self._kernel_compute_density_star()
        density_err = self._kernel_compute_density_error(self._density0)
        return density_err / self._n_particles

    @ti.kernel
    def _kernel_multiply_time_step(self, field: ti.template(), time_step: float):
        """将某字段整体乘以时间步（供 DFSPH 一些系数缩放用，当前未使用）。"""
        for i_p, i_b in ti.ndrange(self._n_particles, self._B):
            if self.particles_ng_reordered[i_p, i_b].active:
                field[i_p, i_b] *= time_step

    def _density_solve(self, f: ti.i32):
        """
        DFSPH 密度解主循环：
        - 计算 ρ*；
        - 将 factor 乘以 1/dt^2（常见缩放）；
        - 迭代至误差阈值或迭代上限；
        - 输出日志。
        """
        inv_dt2 = 1.0 / self._substep_dt**2

        # Compute density star
        self._kernel_compute_density_star()

        self._kernel_multiply_time_step(self.particles_reordered.dfsph_factor, inv_dt2)

        # Start solver
        iteration = gs.ti_int(0)
        avg_density_err = gs.ti_float(0.0)
        while iteration < self._df_max_den_iters:
            avg_density_err = self._density_solve_iteration()
            # 最大允许密度波动（绝对值）
            eta = self._df_max_error_den * 0.01 * self._density0
            if avg_density_err <= eta:
                break
            iteration += 1

        gs.logger.debug(f"DFSPH - iterations: {iteration} Avg density err: {avg_density_err:.4f} kg/m^3")

    # ------------------------------------------------------------------------------------
    # ------------------------------------- utils ----------------------------------------
    # ------------------------------------------------------------------------------------

    @ti.func
    def cubic_kernel(self, r_norm):
        """
        立方样条平滑核（Cubic spline smoothing kernel）。
        W(r, h) = k * piecewise(q=r/h)，3D 常量 k = 8/(π h^3)
        """
        res = gs.ti_float(0.0)
        h = self._support_radius
        k = 8.0 / np.pi / h**3
        q = r_norm / h
        if q <= 1.0:
            if q <= 0.5:
                q2 = q**2
                q3 = q2 * q
                res = k * (6.0 * q3 - 6.0 * q2 + 1.0)
            else:
                res = 2 * k * (1.0 - q) ** 3
        return res

    @ti.func
    def cubic_kernel_derivative(self, r):
        """
        立方样条核的梯度：∇W(r) = dW/dq * dq/dr，q = |r|/h，dq/dr = r/(|r|h)
        """
        res = ti.Vector.zero(gs.ti_float, 3)

        r_norm = r.norm()
        h = self._support_radius
        k = 8.0 / np.pi / h**3
        q = r_norm / h
        if r_norm > 1e-5 and q <= 1.0:
            grad_q = r / (r_norm * h)
            if q <= 0.5:
                res = 6.0 * k * q * (3.0 * q - 2.0) * grad_q
            else:
                res = -6.0 * k * (1.0 - q) ** 2 * grad_q
        return res

    # ------------------------------------------------------------------------------------
    # ------------------------------------ stepping --------------------------------------
    # ------------------------------------------------------------------------------------

    def process_input(self, in_backward=False):
        """转发输入处理至各实体。"""
        for entity in self.entities:
            entity.process_input(in_backward=in_backward)

    def process_input_grad(self):
        """反向流程输入梯度处理（若支持可微）。"""
        for entity in self.entities[::-1]:
            entity.process_input_grad()

    def substep_pre_coupling(self, f):
        """
        子步前耦合阶段：根据 pressure_solver 选择 WCSPH 或 DFSPH 的预处理管线。
        详见类注释中的“pre_coupling 管线”。
        """
        if self.is_active():
            self._kernel_reorder_particles(f)
            if self._pressure_solver == "WCSPH":
                self._kernel_compute_rho(f)
                self._kernel_compute_non_pressure_forces(f, self._sim.cur_t)
                self._kernel_compute_pressure_forces(f)
                self._kernel_advect_velocity(f)
            elif self._pressure_solver == "DFSPH":
                self._kernel_compute_rho(f)
                self._kernel_compute_DFSPH_factor(f)
                self._divergence_solve(f)
                self._kernel_advect_velocity(f)
                self._kernel_compute_non_pressure_forces(f, self._sim.cur_t)
                self._kernel_predict_velocity(f)
                self._density_solve(f)

    def substep_pre_coupling_grad(self, f):
        """当前不支持可微，留空。"""
        pass

    def substep_post_coupling(self, f):
        """
        子步后耦合阶段：位置推进 + 边界 + 从重排缓冲拷回原布局。
        """
        if self.is_active():
            self._kernel_advect_position(f)
            self._kernel_copy_from_reordered(f)

    def substep_post_coupling_grad(self, f):
        """当前不支持可微，留空。"""
        pass

    # ------------------------------------------------------------------------------------
    # ------------------------------------ gradient --------------------------------------
    # ------------------------------------------------------------------------------------

    def collect_output_grads(self):
        """
        从下游查询的状态收集梯度（当前不支持可微，留空）。
        """
        pass

    def add_grad_from_state(self, state):
        """从外部状态累加梯度（当前不支持可微，留空）。"""
        pass

    # ------------------------------------------------------------------------------------
    # --------------------------------------- io -----------------------------------------
    # ------------------------------------------------------------------------------------

    def save_ckpt(self, ckpt_name):
        """保存检查点（当前未实现）。"""
        pass

    def load_ckpt(self, ckpt_name):
        """加载检查点（当前未实现）。"""
        pass

    def set_state(self, f, state, envs_idx=None):
        """将外部状态写入第 f 帧（pos/vel/active）。"""
        if self.is_active():
            self._kernel_set_state(f, state.pos, state.vel, state.active)

    @ti.kernel
    def _kernel_set_state(
        self,
        f: ti.i32,
        pos: ti.types.ndarray(),
        vel: ti.types.ndarray(),
        active: ti.types.ndarray(),
    ):
        """从 numpy/torch 张量写入粒子 pos/vel/active。"""
        for i_p, i_b in ti.ndrange(self._n_particles, self._B):
            for j in ti.static(range(3)):
                self.particles[i_p, i_b].pos[j] = pos[i_b, i_p, j]
                self.particles[i_p, i_b].vel[j] = vel[i_b, i_p, j]
            self.particles_ng[i_p, i_b].active = active[i_b, i_p]

    def get_state(self, f):
        """读取第 f 帧的粒子状态到 SPHSolverState（若未激活返回 None）。"""
        if self.is_active():
            state = SPHSolverState(self.scene)
            self._kernel_get_state(f, state.pos, state.vel, state.active)
        else:
            state = None
        return state

    @ti.kernel
    def _kernel_get_state(
        self,
        f: ti.i32,
        pos: ti.types.ndarray(),
        vel: ti.types.ndarray(),
        active: ti.types.ndarray(),
    ):
        """将第 f 帧的粒子 pos/vel/active 拷出到 numpy/torch 张量。"""
        for i_p, i_b in ti.ndrange(self._n_particles, self._B):
            for j in ti.static(range(3)):
                pos[i_b, i_p, j] = self.particles[i_p, i_b].pos[j]
                vel[i_b, i_p, j] = self.particles[i_p, i_b].vel[j]
            active[i_b, i_p] = self.particles_ng[i_p, i_b].active

    def update_render_fields(self):
        """更新渲染字段（当前子步）。"""
        self._kernel_update_render_fields(self.sim.cur_substep_local)

    @ti.kernel
    def _kernel_update_render_fields(self, f: ti.i32):
        """填充渲染用的粒子 pos/vel/active；未激活粒子放置到“不可见”位置。"""
        for i_p, i_b in ti.ndrange(self._n_particles, self._B):
            if self.particles_ng[i_p, i_b].active:
                self.particles_render[i_p, i_b].pos = self.particles[i_p, i_b].pos
                self.particles_render[i_p, i_b].vel = self.particles[i_p, i_b].vel
            else:
                self.particles_render[i_p, i_b].pos = gu.ti_nowhere()
            self.particles_render[i_p, i_b].active = self.particles_ng[i_p, i_b].active

    @ti.kernel
    def _kernel_add_particles(
        self,
        f: ti.i32,
        active: ti.i32,
        particle_start: ti.i32,
        n_particles: ti.i32,
        mat_rho: ti.f32,
        mat_stiffness: ti.f32,
        mat_exponent: ti.f32,
        mat_mu: ti.f32,
        mat_gamma: ti.f32,
        pos: ti.types.ndarray(),
    ):
        """
        批量添加粒子：
        - 写入 pos/vel/p/active 等动态量；
        - 写入静态量（ρ0/k/n/μ/γ/质量）。
        """
        for i_p_, i_b in ti.ndrange(n_particles, self._B):
            i_p = i_p_ + particle_start
            self.particles_ng[i_p, i_b].active = ti.cast(active, gs.ti_bool)
            for i in ti.static(range(3)):
                self.particles[i_p, i_b].pos[i] = pos[i_p_, i]
            self.particles[i_p, i_b].vel = ti.Vector.zero(gs.ti_float, 3)
            self.particles[i_p, i_b].p = 0

        for i_p_ in range(n_particles):
            i_p = i_p_ + particle_start
            self.particles_info[i_p].rho = mat_rho
            self.particles_info[i_p].stiffness = mat_stiffness
            self.particles_info[i_p].exponent = mat_exponent
            self.particles_info[i_p].mu = mat_mu
            self.particles_info[i_p].gamma = mat_gamma
            self.particles_info[i_p].mass = self._particle_volume * mat_rho

    # ----------------------------------------------------------------------

    @ti.kernel
    def _kernel_set_particles_pos(
        self,
        particles_idx: ti.types.ndarray(),
        envs_idx: ti.types.ndarray(),
        poss: ti.types.ndarray(),
    ):
        """覆写指定粒子的位置，并清零其速度/加速度。"""
        for i_p_, i_b_ in ti.ndrange(particles_idx.shape[1], envs_idx.shape[0]):
            i_p = particles_idx[i_b_, i_p_]
            i_b = envs_idx[i_b_]
            for i in ti.static(range(3)):
                self.particles[i_p, i_b].pos[i] = poss[i_b_, i_p_, i]
            self.particles[i_p, i_b].vel.fill(0.0)
            self.particles[i_p, i_b].acc.fill(0.0)

    @ti.kernel
    def _kernel_get_particles_pos(
        self,
        particle_start: ti.i32,
        n_particles: ti.i32,
        envs_idx: ti.types.ndarray(),
        poss: ti.types.ndarray(),
    ):
        """导出粒子位置（按区间与环境批）。"""
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
        """覆写粒子速度，并清零其加速度。"""
        for i_p_, i_b_ in ti.ndrange(particles_idx.shape[1], envs_idx.shape[0]):
            i_p = particles_idx[i_b_, i_p_]
            i_b = envs_idx[i_b_]
            for i in ti.static(range(3)):
                self.particles[i_p, i_b].vel[i] = vels[i_b_, i_p_, i]
            self.particles[i_p, i_b].acc = ti.Vector.zero(gs.ti_float, 3)

    @ti.kernel
    def _kernel_get_particles_vel(
        self,
        particle_start: ti.i32,
        n_particles: ti.i32,
        envs_idx: ti.types.ndarray(),
        vels: ti.types.ndarray(),
    ):
        """导出粒子速度（按区间与环境批）。"""
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
        """覆写粒子 active 标志。"""
        for i_p_, i_b_ in ti.ndrange(particles_idx.shape[1], envs_idx.shape[0]):
            i_p = particles_idx[i_b_, i_p_]
            i_b = envs_idx[i_b_]
            self.particles_ng[i_p, i_b].active = actives[i_b_, i_p_]

    @ti.kernel
    def _kernel_get_particles_active(
        self,
        particle_start: ti.i32,
        n_particles: ti.i32,
        envs_idx: ti.types.ndarray(),
        actives: ti.types.ndarray(),  # shape [B, n_particles]
    ):
        """导出粒子 active 标志（按区间与环境批）。"""
        for i_p_, i_b_ in ti.ndrange(n_particles, envs_idx.shape[0]):
            i_p = i_p_ + particle_start
            i_b = envs_idx[i_b_]
            actives[i_b_, i_p_] = self.particles_ng[i_p, i_b].active

    @ti.kernel
    def _kernel_get_mass(
        self, particle_start: ti.i32, n_particles: ti.i32, mass: ti.types.ndarray(), envs_idx: ti.types.ndarray()
    ):
        """
        导出总质量（当前实现使用 self.particles[i_p, i_b].m，注意：若未定义 m 字段，此处可能需改为 info.mass）。
        保持逻辑不改动，仅注释提示。
        """
        for i_p_, i_b_ in ti.ndrange(n_particles, envs_idx.shape[0]):
            i_p = i_p_ + particle_start
            i_b = envs_idx[i_b_]
            mass[i_b_] += self.particles[i_p, i_b].m

    # ------------------------------------------------------------------------------------
    # ----------------------------------- properties -------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def n_particles(self):
        """粒子总数（构建后返回缓存值，构建前聚合实体统计）。"""
        if self.is_built:
            return self._n_particles
        else:
            return sum([entity.n_particles for entity in self._entities])

    @property
    def particle_volume(self):
        """粒子体积（经验常数 * size^3）。"""
        return self._particle_volume

    @property
    def particle_size(self):
        return self._particle_size

    @property
    def particle_radius(self):
        return self._particle_size / 2.0

    @property
    def support_radius(self):
        return self._support_radius

    @property
    def hash_grid_res(self):
        return self.sh.grid_res

    @property
    def hash_grid_cell_size(self):
        return self.sh.cell_size

    @property
    def upper_bound(self):
        return self._upper_bound

    @property
    def lower_bound(self):
        return self._lower_bound
