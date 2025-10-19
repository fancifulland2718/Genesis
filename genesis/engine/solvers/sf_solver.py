# 说明：欧拉网格稳定流体（Stable Fluids）求解器
# - 网格均匀立方体，大小为 (res, res, res)，采用单位化网格间距（离散公式中常数见注释）
# - 半拉格朗日对流：基于速度场的特征线追踪（RK3）+ 三线性插值
# - 外部喷口（jets）作为动量与标量源：注入速度动量并维持/衰减标量通道 q
# - 压力投影：雅可比迭代求解泊松方程 ∇²p = div(v) 并作速度投影 v ← v - ∇p
# - 使用双缓冲（TexPair）进行压力迭代的 cur/nxt 交换

import math
import numpy as np
import gstaichi as ti

import genesis as gs
import genesis.utils.geom as gu
from genesis.engine.entities import SFParticleEntity
from genesis.engine.boundaries import CubeBoundary

from .base_solver import Solver


@ti.data_oriented
class SFSolver(Solver):
    """
    气体（欧拉）稳定流体求解器 Stable Fluids。
    - 基于标量压力场与矢量速度场的三维网格离散
    - 支持多个喷口（jet）作为源项，给速度与标量场充值/衰减
    - 管线：对流（半拉格朗日）→ 计算散度 → 压力雅可比迭代 → 速度减压梯度
    """

    # ------------------------------------------------------------------------------------
    # --------------------------------- Initialization -----------------------------------
    # ------------------------------------------------------------------------------------

    def __init__(self, scene, sim, options):
        super().__init__(scene, sim, options)

        if options is None:
            return

        # 网格分辨率与间距（单位化域，dx = 1 / res）
        self.n_grid = options.res
        self.dx = 1 / self.n_grid
        self.res = (self.n_grid, self.n_grid, self.n_grid)
        # 压力投影迭代次数（雅可比迭代步数）
        self.solver_iters = options.solver_iters
        # 标量衰减率（对每个 q 通道做线性衰减）
        self.decay = options.decay

        # 全局时间
        self.t = 0.0
        # 喷口入口强度（用于动量注入）
        self.inlet_s = options.inlet_s

        # 喷口列表（每个 jet 需提供 get_tan_dir(t), get_factor(i,j,k,dx,t) 接口）
        self.jets = []

    def setup_fields(self):
        """
        构建网格场：
        - v: 当前速度
        - v_tmp: 临时速度（对流与减压梯度之间的过渡）
        - div: 速度散度（用于压力方程右端）
        - p: 压力
        - q: 标量通道（长度 = jets 数量），表示每个喷口影响的占据/浓度
        """
        cell_state = ti.types.struct(
            v=gs.ti_vec3,
            v_tmp=gs.ti_vec3,
            div=gs.ti_float,
            p=gs.ti_float,
            q=ti.types.vector(len(self.jets), gs.ti_float),
        )

        self.grid = cell_state.field(shape=self.res, layout=ti.Layout.SOA)

        # 压力投影的双缓冲（雅可比迭代需要读旧写新）
        self.p_swap = TexPair(
            cur=ti.field(dtype=gs.ti_float, shape=self.res),
            nxt=ti.field(dtype=gs.ti_float, shape=self.res),
        )

    @ti.kernel
    def init_fields(self):
        # 初始化所有标量通道为 0（速度/压力默认 0）
        for I in ti.grouped(ti.ndrange(*self.res)):
            for q in ti.static(range(self.grid.q.n)):
                self.grid.q[I][q] = 0.0

    def reset_grad(self):
        # 本求解器当前未接入自动微分图
        pass

    def build(self):
        """
        构建求解器：若有喷口则初始化场与时间。
        """
        super().build()
        if self.is_active():
            self.t = 0.0
            self.setup_fields()
            self.init_fields()

    # ------------------------------------------------------------------------------------
    # -------------------------------------- misc ----------------------------------------
    # ------------------------------------------------------------------------------------

    def set_jets(self, jets):
        """
        设置喷口列表（外部传入带接口的对象）。
        """
        self.jets = jets

    def is_active(self):
        """
        是否启用求解器：至少存在一个喷口才进行求解。
        """
        return len(self.jets) > 0

    def reset_swap(self):
        """
        重置压力双缓冲（清零）。
        """
        self.p_swap.cur.fill(0)
        self.p_swap.nxt.fill(0)

    # ------------------------------------------------------------------------------------
    # ----------------------------------- simulation -------------------------------------
    # ------------------------------------------------------------------------------------

    @ti.kernel
    def pressure_jacobi(self, pf: ti.template(), new_pf: ti.template()):
        # 压力雅可比迭代一步：new_p = (∑邻域p - div) / 6
        # 对应离散 ∇²p = div(v)，6 点邻域（六面体）
        for I in ti.grouped(ti.ndrange(*self.res)):
            pl = pf[self.compute_location(*I, -1, 0, 0)]
            pr = pf[self.compute_location(*I, 1, 0, 0)]
            pb = pf[self.compute_location(*I, 0, -1, 0)]
            pt = pf[self.compute_location(*I, 0, 1, 0)]
            pp = pf[self.compute_location(*I, 0, 0, -1)]
            pq = pf[self.compute_location(*I, 0, 0, 1)]

            new_pf[I] = (pl + pr + pb + pt + pp + pq - self.grid[I].div) / 6.0

    @ti.kernel
    def advect_and_impulse(self, f: ti.i32, t: ti.f32):
        # 半拉格朗日对流 + 喷口动量注入 + 标量通道维护
        # 流程：
        #   1) 反向追踪（RK3）获得粒子来源位置 p_src
        #   2) 三线性插值速度 v(p_src) 作为对流后的速度
        #   3) 对每个喷口通道 q：
        #       - 计算喷口方向 imp_dir 与作用因子 factor(i,j,k,dx,t)
        #       - 注入动量 v += imp_dir * inlet_s * factor * dt
        #       - 标量 q 进行拉格朗日对流采样、插值混合、衰减并截断至 [0, +∞)
        #   4) 写入 v_tmp
        for i, j, k in ti.ndrange(*self.res):
            p = ti.Vector([i, j, k], dt=gs.ti_float) + 0.5
            p = self.backtrace(self.grid.v, p, self.dt)
            v_tmp = self.trilerp(self.grid.v, p)

            for q in ti.static(range(self.grid.q.n)):
                q_f = self.trilerp_scalar(self.grid.q, p, q)

                imp_dir = self.jets[q].get_tan_dir(t)              # 喷口切向/主方向
                factor = self.jets[q].get_factor(i, j, k, self.dx, t)  # 空间-时间作用强度 [0,1]
                momentum = (imp_dir * self.inlet_s * factor) * self.dt

                v_tmp += momentum

                # 将标量推进到当前格点：混合旧值与喷口激活
                self.grid.q[i, j, k][q] = (1 - factor) * q_f + factor
                # 衰减并截断（避免负值）
                self.grid.q[i, j, k][q] -= self.decay * self.dt
                self.grid.q[i, j, k][q] = max(0.0, self.grid.q[i, j, k][q])

            self.grid.v_tmp[i, j, k] = v_tmp

    @ti.kernel
    def divergence(self):
        # 计算速度散度 div(v_tmp)，并施加边界条件：
        # - 若邻点越界，则其法向分量取为中心点速度的相反数（等价于固壁无穿透）
        # - 差分采用中心差分形式，常数 0.5 对应单位化网格间距
        for I in ti.grouped(ti.ndrange(*self.res)):
            vl = self.grid.v_tmp[self.compute_location(*I, -1, 0, 0)]
            vr = self.grid.v_tmp[self.compute_location(*I, 1, 0, 0)]
            vb = self.grid.v_tmp[self.compute_location(*I, 0, -1, 0)]
            vt = self.grid.v_tmp[self.compute_location(*I, 0, 1, 0)]
            vp = self.grid.v_tmp[self.compute_location(*I, 0, 0, -1)]
            vq = self.grid.v_tmp[self.compute_location(*I, 0, 0, 1)]
            vc = self.grid.v_tmp[self.compute_location(*I, 0, 0, 0)]

            if not self.is_free(*I, -1, 0, 0):
                vl.x = -vc.x
            if not self.is_free(*I, 1, 0, 0):
                vr.x = -vc.x
            if not self.is_free(*I, 0, -1, 0):
                vb.y = -vc.y
            if not self.is_free(*I, 0, 1, 0):
                vt.y = -vc.y
            if not self.is_free(*I, 0, 0, -1):
                vp.z = -vc.z
            if not self.is_free(*I, 0, 0, 1):
                vq.z = -vc.z

            self.grid.div[I] = 0.5 * (vr.x - vl.x + vt.y - vb.y + vq.z - vp.z)

    @ti.kernel
    def pressure_to_swap(self):
        # 将网格压力拷贝到双缓冲当前帧
        for I in ti.grouped(ti.ndrange(*self.res)):
            self.p_swap.cur[I] = self.grid.p[I]

    @ti.kernel
    def pressure_from_swap(self):
        # 从双缓冲当前帧回写到网格压力
        for I in ti.grouped(ti.ndrange(*self.res)):
            self.grid.p[I] = self.p_swap.cur[I]

    @ti.kernel
    def subtract_gradient(self):
        # 速度投影：v ← v_tmp - ∇p（中心差分，常数 0.5 与 divergence 对称）
        for I in ti.grouped(ti.ndrange(*self.res)):
            pl = self.grid.p[self.compute_location(*I, -1, 0, 0)]
            pr = self.grid.p[self.compute_location(*I, 1, 0, 0)]
            pb = self.grid.p[self.compute_location(*I, 0, -1, 0)]
            pt = self.grid.p[self.compute_location(*I, 0, 1, 0)]
            pp = self.grid.p[self.compute_location(*I, 0, 0, -1)]
            pq = self.grid.p[self.compute_location(*I, 0, 0, 1)]

            self.grid.v[I] = self.grid.v_tmp[I] - 0.5 * ti.Vector([pr - pl, pt - pb, pq - pp], dt=gs.ti_float)

    @ti.func
    def compute_location(self, u, v, w, du, dv, dw):
        # 计算邻居网格坐标并 clamp 到边界内（避免越界访问）
        I = ti.Vector([u + du, v + dv, w + dw], dt=gs.ti_int)
        return ti.math.clamp(I, 0, self.n_grid - 1)

    @ti.func
    def is_free(self, u, v, w, du, dv, dw):
        # 判断邻居网格是否在域内（用于边界条件处理）
        I = ti.Vector([u + du, v + dv, w + dw], dt=gs.ti_int)
        return gs.ti_bool((0 <= I).all() and (I < self.n_grid).all())

    @ti.func
    def trilerp_scalar(self, qf, p, qf_idx):
        """
        三线性插值标量通道 q：
        - p: 物理位置（单位化，[0,1) 的网格坐标系）
        - qf: 带通道的场（field）
        - qf_idx: 通道下标
        """
        # 将物理位置转为网格基元索引与局部坐标
        base_I = ti.floor(p - 0.5, gs.ti_int)
        p_I = p - 0.5

        q = 0.0
        w_total = 0.0
        for offset in ti.static(ti.grouped(ti.ndrange(2, 2, 2))):
            grid_I = base_I + offset
            w_xyz = 1 - ti.abs(p_I - grid_I)  # 分离权重
            w = w_xyz[0] * w_xyz[1] * w_xyz[2]
            grid_I_ = self.compute_location(grid_I[0], grid_I[1], grid_I[2], 0, 0, 0)
            q += w * qf[grid_I_][qf_idx]
            w_total += w
        # 边界处权重和 < 1，需要归一化
        q /= w_total
        return q

    @ti.func
    def trilerp(self, qf, p):
        """
        三线性插值矢量场 qf（通常为速度）。
        - p: 物理位置（单位化，[0,1) 的网格坐标系）
        """
        # 将物理位置转为网格基元索引与局部坐标
        base_I = ti.floor(p - 0.5, gs.ti_int)
        p_I = p - 0.5

        q = ti.Vector([0.0, 0.0, 0.0], dt=gs.ti_float)
        w_total = 0.0
        for offset in ti.static(ti.grouped(ti.ndrange(2, 2, 2))):
            grid_I = base_I + offset
            w_xyz = 1 - ti.abs(p_I - grid_I)
            w = w_xyz[0] * w_xyz[1] * w_xyz[2]
            grid_I_ = self.compute_location(grid_I[0], grid_I[1], grid_I[2], 0, 0, 0)
            q += w * qf[grid_I_]
            w_total += w
        # 边界处权重和 < 1，需要归一化
        q /= w_total
        return q

    # RK3
    @ti.func
    def backtrace(self, vf, p, dt):
        """
        半拉格朗日反向特征线追踪（RK3）：
        - vf: 速度场
        - p: 当前位置（网格中心）
        - dt: 时间步长
        返回：从 p 出发反向追踪 dt 后的“源位置”
        """
        v1 = self.trilerp(vf, p)
        p1 = p - 0.5 * dt * v1
        v2 = self.trilerp(vf, p1)
        p2 = p - 0.75 * dt * v2
        v3 = self.trilerp(vf, p2)
        p -= dt * ((2 / 9) * v1 + (1 / 3) * v2 + (4 / 9) * v3)
        return p

    # ------------------------------------------------------------------------------------
    # ------------------------------------ stepping --------------------------------------
    # ------------------------------------------------------------------------------------

    def process_input(self, in_backward):
        # 当前无外部输入处理
        return None

    def substep_pre_coupling(self, f):
        # 单个子步管线：
        # 1) 对流 + 喷口动量注入（写 v_tmp；q 通道衰减/截断）
        self.advect_and_impulse(f, self.t)
        # 2) 计算 v_tmp 的散度 div
        self.divergence()

        # 3) 压力投影：雅可比迭代解 ∇²p = div
        self.reset_swap()
        self.pressure_to_swap()
        for _ in range(self.solver_iters):
            self.pressure_jacobi(self.p_swap.cur, self.p_swap.nxt)
            self.p_swap.swap()
        self.pressure_from_swap()
        self.reset_swap()

        # 4) 投影速度：v ← v_tmp - ∇p
        self.subtract_gradient()
        # 5) 时间推进
        self.t += self.dt

    def substep_post_coupling(self, f):
        # 无耦合后处理
        return

    # ------------------------------------------------------------------------------------
    # ------------------------------------ gradient --------------------------------------
    # ------------------------------------------------------------------------------------

    def collect_output_grads(self):
        # 未集成梯度回传
        pass

    def add_grad_from_state(self, state):
        # 未集成梯度回传
        pass

    # ------------------------------------------------------------------------------------
    # --------------------------------------- io -----------------------------------------
    # ------------------------------------------------------------------------------------

    def get_state(self, f):
        # TODO：如需输出可视化/导出网格场，可在此实现
        pass

    def set_state(self, f, state, envs_idx=None):
        # TODO：如需外部设置网格场状态，可在此实现
        pass

    def save_ckpt(self, ckpt_name):
        # TODO：如需检查点，可在此实现
        pass

    def load_ckpt(self, ckpt_name):
        # TODO：如需检查点恢复，可在此实现
        pass


class TexPair:
    """
    简单的双缓冲包装：用于迭代法中读写分离（cur 读，nxt 写，迭代后交换）。
    """
    def __init__(self, cur, nxt):
        self.cur = cur
        self.nxt = nxt

    def swap(self):
        self.cur, self.nxt = self.nxt, self.cur