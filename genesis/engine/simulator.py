from typing import TYPE_CHECKING
import numpy as np
import gstaichi as ti

import genesis as gs
from genesis.engine.entities.base_entity import Entity
from genesis.options.morphs import Morph
from genesis.options.solvers import (
    AvatarOptions,
    BaseCouplerOptions,
    LegacyCouplerOptions,
    SAPCouplerOptions,
    FEMOptions,
    MPMOptions,
    PBDOptions,
    RigidOptions,
    SFOptions,
    SPHOptions,
    SimOptions,
    ToolOptions,
)
from genesis.repr_base import RBC
from genesis.sensors import SensorManager

from .entities import HybridEntity
from .solvers.base_solver import Solver
from .solvers import (
    AvatarSolver,
    FEMSolver,
    MPMSolver,
    PBDSolver,
    RigidSolver,
    SFSolver,
    SPHSolver,
    ToolSolver,
)
from .couplers import LegacyCoupler, SAPCoupler
from .states.cache import QueriedStates
from .states.solvers import SimState

if TYPE_CHECKING:
    from genesis.engine.scene import Scene


@ti.data_oriented
class Simulator(RBC):
    """
    A simulator is a scene-level simulation manager, which manages all simulation-related operations in the scene, including multiple solvers and the inter-solver coupler.
    模拟器是场景级的仿真管理器，负责场景中的所有仿真相关操作，包括多个求解器以及求解器之间的耦合器。

    Parameters
    ----------
    scene : gs.Scene
        The scene object that the simulator is associated with.
        与该模拟器关联的场景对象。
    options : gs.SimOptions
        A SimOptions object that contains all simulator-level options.
        包含模拟器级配置的 SimOptions 对象。
    coupler_options : gs.CouplerOptions
        A CouplerOptions object that contains all the options for the coupler.
        耦合器配置对象。
    tool_options : gs.ToolOptions
        A ToolOptions object that contains all the options for the ToolSolver.
        工具求解器配置。
    rigid_options : gs.RigidOptions
        A RigidOptions object that contains all the options for the RigidSolver.
        刚体求解器配置。
    avatar_options : gs.AvatarOptions
        An AvatarOptions object that contains all the options for the AvatarSolver.
        Avatar 求解器配置。
    mpm_options : gs.MPMOptions
        An MPMOptions object that contains all the options for the MPMSolver.
        MPM 求解器配置。
    sph_options : gs.SPHOptions
        An SPHOptions object that contains all the options for the SPHSolver.
        SPH 求解器配置。
    fem_options : gs.FEMOptions
        An FEMOptions object that contains all the options for the FEMSolver.
        FEM 求解器配置。
    sf_options : gs.SFOptions
        An SFOptions object that contains all the options for the SFSolver.
        SF 求解器配置。
    pbd_options : gs.PBDOptions
        A PBDOptions object that contains all the options for the PBDSolver.
        PBD 求解器配置。
    """

    def __init__(
        self,
        scene: "Scene",
        options: SimOptions,
        coupler_options: BaseCouplerOptions,
        tool_options: ToolOptions,
        rigid_options: RigidOptions,
        avatar_options: AvatarOptions,
        mpm_options: MPMOptions,
        sph_options: SPHOptions,
        fem_options: FEMOptions,
        sf_options: SFOptions,
        pbd_options: PBDOptions,
    ):
        self._scene = scene

        # options
        # 配置选项集合
        self.options = options
        self.coupler_options = coupler_options
        self.tool_options = tool_options
        self.rigid_options = rigid_options
        self.avatar_options = avatar_options
        self.mpm_options = mpm_options
        self.sph_options = sph_options
        self.fem_options = fem_options
        self.sf_options = sf_options
        self.pbd_options = pbd_options

        self._dt: float = options.dt
        self._substep_dt: float = options.dt / options.substeps
        self._substeps: int = options.substeps
        self._substeps_local: int | None = options.substeps_local
        self._requires_grad: bool = options.requires_grad
        self._steps_local: int | None = options._steps_local

        self._cur_substep_global = 0
        self._gravity = np.array(options.gravity, dtype=gs.np_float)

        # solvers
        # 求解器实例集合
        self.tool_solver = ToolSolver(self.scene, self, self.tool_options)
        self.rigid_solver = RigidSolver(self.scene, self, self.rigid_options)
        self.avatar_solver = AvatarSolver(self.scene, self, self.avatar_options)
        self.mpm_solver = MPMSolver(self.scene, self, self.mpm_options)
        self.sph_solver = SPHSolver(self.scene, self, self.sph_options)
        self.pbd_solver = PBDSolver(self.scene, self, self.pbd_options)
        self.fem_solver = FEMSolver(self.scene, self, self.fem_options)
        self.sf_solver = SFSolver(self.scene, self, self.sf_options)

        self._solvers: list[Solver] = gs.List(
            [
                self.tool_solver,
                self.rigid_solver,
                self.avatar_solver,
                self.mpm_solver,
                self.sph_solver,
                self.pbd_solver,
                self.fem_solver,
                self.sf_solver,
            ]
        )

        self._active_solvers: list[Solver] = gs.List()

        # coupler
        # 耦合器（跨求解器交互）
        if isinstance(self.coupler_options, SAPCouplerOptions):
            self._coupler = SAPCoupler(self, self.coupler_options)
        elif isinstance(self.coupler_options, LegacyCouplerOptions):
            self._coupler = LegacyCoupler(self, self.coupler_options)
        else:
            gs.raise_exception(
                f"Coupler options {self.coupler_options} not supported. Please use SAPCouplerOptions or LegacyCouplerOptions."
            )

        # states
        # 已查询状态缓存（用于梯度追踪）
        self._queried_states = QueriedStates()

        # entities
        # 场景中已注册实体
        self._entities: list[Entity] = gs.List()

        # sensors
        # 传感器管理器
        self._sensor_manager = SensorManager(self)

    def _add_entity(self, morph: Morph, material, surface, visualize_contact=False):
        if isinstance(material, gs.materials.Tool):
            entity = self.tool_solver.add_entity(self.n_entities, material, morph, surface)

        elif isinstance(material, gs.materials.Avatar):
            entity = self.avatar_solver.add_entity(self.n_entities, material, morph, surface, visualize_contact)

        elif isinstance(material, gs.materials.Rigid):
            entity = self.rigid_solver.add_entity(self.n_entities, material, morph, surface, visualize_contact)

        elif isinstance(material, gs.materials.MPM.Base):
            entity = self.mpm_solver.add_entity(self.n_entities, material, morph, surface)

        elif isinstance(material, gs.materials.SPH.Base):
            entity = self.sph_solver.add_entity(self.n_entities, material, morph, surface)

        elif isinstance(material, gs.materials.PBD.Base):
            entity = self.pbd_solver.add_entity(self.n_entities, material, morph, surface)

        elif isinstance(material, gs.materials.FEM.Base):
            entity = self.fem_solver.add_entity(self.n_entities, material, morph, surface)

        elif isinstance(material, gs.materials.Hybrid):
            entity = HybridEntity(
                self.n_entities, self.scene, material, morph, surface
            )  # adding to solver is handled in the hybrid entity
            # 添加到对应求解器的逻辑在 HybridEntity 内部处理

        else:
            gs.raise_exception(f"Material not supported.: {material}")

        self._entities.append(entity)
        return entity

    def _add_force_field(self, force_field):
        for solver in self._solvers:
            solver._add_force_field(force_field)

    def build(self):
        self.n_envs = self.scene.n_envs
        self._B = self.scene._B
        self._para_level = self.scene._para_level

        # solvers
        # 构建各求解器并筛选激活的求解器
        self._rigid_only = self.rigid_solver.is_active() and not isinstance(self._coupler, SAPCoupler)
        for solver in self._solvers:
            solver.build()
            if solver.is_active():
                # 这里构建的激活求解器也包含了刚体求解器
                self._active_solvers.append(solver)
                if not isinstance(solver, RigidSolver):
                    self._rigid_only = False
        self._coupler.build()

        if self.n_envs > 0 and self.sf_solver.is_active():
            gs.raise_exception("Batching is not supported for SF solver as of now.")
            # 当前 SF 求解器不支持批量场景

        # hybrid
        # 构建混合实体
        for entity in self._entities:
            if isinstance(entity, HybridEntity):
                entity.build()

        self._sensor_manager.build()

    def reset(self, state: SimState, envs_idx=None):
        for solver, solver_state in zip(self._solvers, state):
            if solver.n_entities > 0:
                solver.set_state(0, solver_state, envs_idx)

        self.coupler.reset(envs_idx=envs_idx)

        # TODO: keeping as is for now
        # TODO：暂保持现状
        self.reset_grad()
        self._cur_substep_global = 0

        # reset sensors state
        # 重置传感器状态
        self._sensor_manager.reset(envs_idx=envs_idx)

    def reset_grad(self):
        for solver in self._active_solvers:
            solver.reset_grad()

        # clear up all queried scene states and free up memory
        # 清理已查询状态缓存释放内存
        self._queried_states.clear()

    # ------------------------------------------------------------------------------------
    # -------------------------------- step computation ----------------------------------
    # ------------------------------------------------------------------------------------
    """
    We use f to represent substep, and s to represent step.
    约定：f 表示子步 (substep)，s 表示主步 (step)。
    此处代码仅用于处理索引。
    """

    def f_global_to_f_local(self, f_global):
        f_local = f_global % self._substeps_local
        return f_local

    def f_local_to_s_local(self, f_local):
        f_local = f_local // self._substeps
        return f_local

    def f_global_to_s_local(self, f_global):
        f_local = self.f_global_to_f_local(f_global)
        s_local = self.f_local_to_s_local(f_local)
        return s_local

    def f_global_to_s_global(self, f_global):
        s_global = f_global // self._substeps
        return s_global

    # ------------------------------------------------------------------------------------
    # ------------------------------------ stepping --------------------------------------
    # ------------------------------------------------------------------------------------

    def step(self, in_backward=False):
        if self._rigid_only:  # "Only Advance!" --Thomas Wade :P
            # 仅刚体模式：直接推进子步
            for _ in range(self._substeps):
                self.rigid_solver.substep()
                self._cur_substep_global += 1
        else:
            # 使得所有求解器处理外部输入
            self.process_input(in_backward=in_backward)
            for _ in range(self._substeps):
                self.substep(self.cur_substep_local)

                self._cur_substep_global += 1
                if self.cur_substep_local == 0 and not in_backward:
                    self.save_ckpt()

        if self.rigid_solver.is_active():
            self.rigid_solver.clear_external_force()  # 清空刚体外力缓存

        self._sensor_manager.step()  # 传感器步进

    def _step_grad(self):
        for _ in range(self._substeps - 1, -1, -1):

            if self.cur_substep_local == 0:
                self.load_ckpt()
            self._cur_substep_global -= 1

            self.sub_step_grad(self.cur_substep_local)

        self.process_input_grad()

    def process_input(self, in_backward=False):
        """
        setting _tgt state using external commands
        note that external inputs are given at step level, not substep
        使用外部控制指令设置目标状态；输入作用在 step 级别而非 substep 级别。
        """
        for solver in self._active_solvers:
            solver.process_input(in_backward=in_backward)

    def process_input_grad(self):
        for solver in reversed(self._active_solvers):
            solver.process_input_grad()

    def substep(self, f):
        self._coupler.preprocess(f)
        self.substep_pre_coupling(f)
        self._coupler.couple(f)
        self.substep_post_coupling(f)

    # 上述函数的逆过程，但无需对预处理求导
    def sub_step_grad(self, f):
        self.substep_post_coupling_grad(f)
        self._coupler.couple_grad(f)
        self.substep_pre_coupling_grad(f)

    # -------------- pre coupling --------------
    # -------------- 耦合前阶段 --------------
    def substep_pre_coupling(self, f):
        for solver in self._active_solvers:
            solver.substep_pre_coupling(f)

    def substep_pre_coupling_grad(self, f):
        for solver in reversed(self._active_solvers):
            solver.substep_pre_coupling_grad(f)

    # -------------- post coupling --------------
    # -------------- 耦合后阶段 --------------
    def substep_post_coupling(self, f):
        for solver in self._active_solvers:
            solver.substep_post_coupling(f)

    def substep_post_coupling_grad(self, f):
        for solver in reversed(self._active_solvers):
            solver.substep_post_coupling_grad(f)

    # ------------------------------------------------------------------------------------
    # ------------------------------------ gradient --------------------------------------
    # ------------------------------------------------------------------------------------

    def add_grad_from_state(self, state):
        for solver, solver_state in zip(self._solvers, state):
            solver.add_grad_from_state(solver_state)

    def collect_output_grads(self):
        """
        Collect gradients from downstream queried states.
        从下游已查询的状态中回收梯度。
        """

        # simulator-level states
        # 模拟器级别状态
        if self.cur_step_global in self._queried_states:
            # one step could have multiple states
            # 一个 step 可能对应多个状态对象
            for state in self._queried_states[self.cur_step_global]:
                self.add_grad_from_state(state)

        # each solver will have their own entities, each of which stores a set of _queried_states
        # 每个求解器的实体也维护自己的查询状态集合
        for solver in self._active_solvers:
            solver.collect_output_grads()

    def save_ckpt(self):
        """
        This function refreshes the gpu memory (copy the last frame to the first frame in the local memory), and then saves the checkpoint.
        This function is called every `substeps_local` steps, which means it's called only once per step when `requires_grad` is True.
        刷新本地 GPU 帧缓存（将最后一帧复制到第一帧），随后保存检查点。
        当需要梯度时每个 step（对应 substeps_local 个子步）仅调用一次。
        """
        ckpt_start_substep = self._cur_substep_global - self._substeps_local
        ckpt_end_step = self._cur_substep_global - 1
        ckpt_name = f"{ckpt_start_substep}"

        for solver in self._active_solvers:
            solver.save_ckpt(ckpt_name)

        if self._requires_grad:
            gs.logger.debug(
                f"Forward: Saved checkpoint for global substep {ckpt_start_substep} to {ckpt_end_step}. Now starts from substep {self._cur_substep_global}."
            )

    def load_ckpt(self):
        ckpt_start_substep = self._cur_substep_global - self._substeps_local
        ckpt_end_step = self._cur_substep_global - 1
        ckpt_name = f"{ckpt_start_substep}"

        for solver in self._active_solvers:
            solver.load_ckpt(ckpt_name)

        # now that we loaded the first frame, we do a forward pass to fill up the rest
        # 载入首帧后再做一次前向填充后续帧
        self._cur_substep_global = ckpt_start_substep
        for _ in range(self._steps_local):
            self.step(in_backward=True)

        gs.logger.debug(
            f"Backward: Loaded checkpoint for global substep {ckpt_start_substep} to {ckpt_end_step}. Now starts from substep {ckpt_start_substep}."
        )

    # ------------------------------------------------------------------------------------
    # --------------------------------------- io -----------------------------------------
    # ------------------------------------------------------------------------------------

    def get_state(self):
        state = SimState(
            scene=self.scene,
            s_global=self.cur_step_global,
            f_local=self.cur_substep_local,
            solvers=self._solvers,
        )

        # store all queried states to track gradient flow
        # 记录已查询状态以追踪梯度流
        self._queried_states.append(state)

        return state

    def set_gravity(self, gravity, envs_idx=None):
        for solver in self._solvers:
            if solver.is_active():
                solver.set_gravity(gravity, envs_idx)

    # ------------------------------------------------------------------------------------
    # ----------------------------------- properties -------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def dt(self) -> float:
        """The time duration for each simulation step.
        每个仿真 step 的时间长度（秒）。
        """
        return self._dt

    @property
    def substeps(self):
        """The number of substeps per simulation step.
        每个仿真 step 内包含的子步数量。
        """
        return self._substeps

    @property
    def scene(self):
        """The scene object that the simulator is associated with.
        关联的场景对象。
        """
        return self._scene

    @property
    def gravity(self):
        """The gravity vector.
        重力向量。
        """
        return self._gravity

    @property
    def requires_grad(self):
        """Whether the simulator requires gradients.
        是否需要计算梯度。
        """
        return self._requires_grad

    @property
    def n_entities(self) -> int:
        """The number of entities in the simulator.
        当前模拟器中的实体数量。
        """
        return len(self._entities)

    @property
    def entities(self):
        """The list of entities in the simulator.
        实体列表。
        """
        return self._entities

    @property
    def substeps_local(self):
        """The number of substeps stored in local memory.
        本地缓存中保留的子步数量。
        """
        return self._substeps_local

    @property
    def cur_substep_global(self):
        """The current substep of the simulation.
        当前全局子步索引。
        """
        return self._cur_substep_global

    @property
    def cur_substep_local(self):
        """The current substep of the simulation in local memory.
        当前本地（循环缓冲区）子步索引。
        """
        return self.f_global_to_f_local(self._cur_substep_global)

    @property
    def cur_step_local(self):
        """The current step of the simulation in local memory.
        当前本地 step 索引。
        """
        return self.f_global_to_s_local(self._cur_substep_global)

    @property
    def cur_step_global(self):
        """The current step of the simulation.
        当前全局 step 索引。
        """
        return self.f_global_to_s_global(self._cur_substep_global)

    @property
    def cur_t(self):
        """The current time of the simulation.
        当前仿真时间（秒）。
        """
        return self._cur_substep_global * self._substep_dt

    @property
    def coupler(self):
        """The coupler object that manages the inter-solver coupling.
        负责跨求解器耦合的对象。
        """
        return self._coupler

    @property
    def solvers(self):
        """The list of solvers in the simulator.
        所有求解器列表。
        """
        return self._solvers

    @property
    def active_solvers(self):
        """The list of active solvers in the simulator.
        已激活（实际参与计算）的求解器列表。
        """
        return self._active_solvers