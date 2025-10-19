from typing import TYPE_CHECKING
import gstaichi as ti

from genesis.engine.boundaries import FloorBoundary
from genesis.engine.states.solvers import ToolSolverState
from genesis.engine.entities.tool_entity.tool_entity import ToolEntity
from genesis.utils.misc import *

from .base_solver import Solver


@ti.data_oriented
class ToolSolver(Solver):
    """
    工具（刚体工具）求解器（临时方案）。
    作用：管理场景中的 ToolEntity（工具实体），并提供与软体的一向耦合（rigid->soft）接口，
          以便在未完成 RigidSolver 可微前实现可微管线。
    注意：一旦 RigidSolver 支持可微，本类将被移除。

    主要职责：
    - 初始化并构建工具实体与边界（如地面）
    - 在仿真子步中按顺序驱动实体执行 pre/post-coupling（前/后耦合）阶段
    - 处理前向与反向传播（梯度）流程，保持顺序与依赖一致
    - 状态（state）读写、检查点保存/加载、以及碰撞（PBD）接口转发
    """

    # ------------------------------------------------------------------------------------
    # --------------------------------- Initialization -----------------------------------
    # ------------------------------------------------------------------------------------

    def __init__(self, scene, sim, options):
        """
        参数
        - scene: 场景对象
        - sim:   仿真器
        - options: 包含本求解器相关参数，例如地面高度等
        """
        super().__init__(scene, sim, options)

        # options
        self.floor_height = options.floor_height

        # boundary
        self.setup_boundary()

    def build(self):
        """
        构建求解器与其管理的所有工具实体。
        流程：
        1) 调用父类 build（通用分配、缓存等）
        2) 遍历实体逐个执行 entity.build()
        """
        super().build()
        for entity in self._entities:
            entity.build()

    def setup_boundary(self):
        """
        设置边界条件（如地面平面）。
        当前实现：仅创建一个 FloorBoundary，使用 options 的 floor_height。
        """
        self.boundary = FloorBoundary(height=self.floor_height)

    def add_entity(self, idx, material, morph, surface):
        """
        向本求解器添加一个工具实体并返回。
        参数
        - idx:      实体在求解器中的索引
        - material: 材质/物理属性
        - morph:    形态/几何描述
        - surface:  表面材质/纹理
        """
        entity = ToolEntity(
            scene=self._scene,
            idx=idx,
            solver=self,
            material=material,
            morph=morph,
            surface=surface,
        )
        self._entities.append(entity)
        return entity

    def reset_grad(self):
        """
        重置所有实体上的梯度累积（用于反向传播前的清零）。
        """
        for entity in self._entities:
            entity.reset_grad()

    def get_state(self, f):
        """
        读取求解器在帧/子步 f 的聚合状态（只读快照）。
        若求解器非激活（无实体），返回 None。
        说明：ToolSolverState 内仅聚合各 entity.get_state(f) 的结果。
        """
        if self.is_active():
            state = ToolSolverState(self._scene)
            for entity in self._entities:
                state.entities.append(entity.get_state(f))
        else:
            state = None
        return state

    def set_state(self, f, state, envs_idx=None):
        """
        设置求解器在帧/子步 f 的状态。
        要求 state 的实体数量与 _entities 一致，并逐一下发至对应实体。
        envs_idx 预留用于局部环境重置（当前未使用）。
        """
        if state is not None:
            assert len(state) == len(self._entities)
            for i, entity in enumerate(self._entities):
                entity.set_state(f, state[i])

    def process_input(self, in_backward=False):
        """
        处理输入（例如外部控制、命令），转发到每个实体。
        in_backward=True 时可用于反向流程下的输入处理（需要时）。
        """
        for entity in self._entities:
            entity.process_input(in_backward=in_backward)

    def process_input_grad(self):
        """
        处理输入的梯度（反向传播阶段）。
        注意反向顺序：对实体列表倒序遍历，保证梯度依赖的拓扑顺序正确。
        """
        for entity in self._entities[::-1]:
            entity.process_input_grad()

    def substep_pre_coupling(self, f):
        """
        子步前耦合阶段（forward）：
        - 在软体/其他系统交互前，先让工具实体更新自身状态（如控制指令、运动学驱动）。
        - 按顺序遍历实体执行 pre_coupling。
        """
        for entity in self._entities:
            entity.substep_pre_coupling(f)

    def substep_pre_coupling_grad(self, f):
        """
        子步前耦合阶段（backward）：
        - 与 forward 相反，按倒序遍历实体执行 pre_coupling 的梯度回传。
        """
        for entity in self._entities[::-1]:
            entity.substep_pre_coupling_grad(f)

    def substep_post_coupling(self, f):
        """
        子步后耦合阶段（forward）：
        - 在与软体/其他系统进行碰撞/约束等耦合处理后，允许工具实体做收尾更新。
        - 按顺序遍历实体执行 post_coupling。
        """
        for entity in self._entities:
            entity.substep_post_coupling(f)

    def substep_post_coupling_grad(self, f):
        """
        子步后耦合阶段（backward）：
        - 与 forward 相反，按倒序遍历实体执行 post_coupling 的梯度回传。
        """
        for entity in self._entities[::-1]:
            entity.substep_post_coupling_grad(f)

    def add_grad_from_state(self, state):
        """
        从聚合 state 回填梯度（此处无需实现）：
        说明：ToolSolver 的 state 仅是各 entity.get_state() 的聚合，
        各实体内部已缓存自身 state 并负责梯度管理，故本函数留空。
        """
        # Nothing needed here, since tool_solver state is composed of tool_entity.get_state(), which has already been cached inside each tool_entity.
        pass

    def collect_output_grads(self):
        """
        收集下游（查询到的输出状态）回传的梯度，并分发给各实体。
        仅在求解器激活时执行。
        """
        if self.is_active():
            for entity in self._entities:
                entity.collect_output_grads()

    def save_ckpt(self, ckpt_name):
        """
        保存所有实体的检查点（以 ckpt_name 命名）。
        """
        for entity in self._entities:
            entity.save_ckpt(ckpt_name)

    def load_ckpt(self, ckpt_name):
        """
        从检查点加载所有实体状态。
        """
        for entity in self._entities:
            entity.load_ckpt(ckpt_name=ckpt_name)

    def is_active(self):
        """
        求解器是否处于激活状态：当存在至少一个实体时返回 True。
        """
        return self.n_entities > 0

    @ti.func
    def pbd_collide(self, f, pos_world, thickness, dt):
        """
        PBD 碰撞处理（Taichi 内联函数）：
        管线（逐实体串联）：
        - 输入当前世界坐标 pos_world，经每个工具实体的 pbd_collide 处理，逐次更新 pos_world
          以实现与多个工具的依次碰撞/修正。
        - 返回修正后的世界坐标。
        参数
        - f:         当前帧/子步索引
        - pos_world: 世界坐标（ti.Vector(3)）
        - thickness: 几何厚度/容差，用于碰撞检测/纠正
        - dt:        时间步长
        """
        for entity in ti.static(self._entities):
            pos_world = entity.pbd_collide(f, pos_world, thickness, dt)
        return pos_world
