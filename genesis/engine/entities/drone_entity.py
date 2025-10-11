import os
import xml.etree.ElementTree as ET

import torch
import gstaichi as ti

import genesis as gs
import genesis.utils.misc as mu

from .rigid_entity import RigidEntity


@ti.data_oriented
class DroneEntity(RigidEntity):
    """
    Drone entity for simulating quadcopters and multirotors.
    用于模拟四旋翼和多旋翼无人机的实体类。
    
    该类继承自 RigidEntity，增加了无人机特有的功能：
    - 螺旋桨转速控制（RPM）
    - 推力和力矩系数（KF、KM）
    - 螺旋桨动画可视化
    - 基于物理的飞行动力学
    
    支持不同的无人机模型（如 RACE 模型），适用于无人机控制、路径规划、
    强化学习等研究场景。
    """
    def _load_scene(self, morph, surface):
        super()._load_scene(morph, surface)

        # additional drone specific attributes
        properties = ET.parse(os.path.join(mu.get_assets_dir(), morph.file)).getroot()[0].attrib
        self._KF = float(properties["kf"])
        self._KM = float(properties["km"])

        self._n_propellers = len(morph.propellers_link_name)

        propellers_link = gs.List([self.get_link(name) for name in morph.propellers_link_name])
        self._propellers_link_idxs = torch.tensor(
            [link.idx for link in propellers_link], dtype=gs.tc_int, device=gs.device
        )
        try:
            self._propellers_vgeom_idxs = torch.tensor(
                [link.vgeoms[0].idx for link in propellers_link], dtype=gs.tc_int, device=gs.device
            )
            self._animate_propellers = True
        except Exception:
            gs.logger.warning("No visual geometry found for propellers. Skipping propeller animation.")
            self._animate_propellers = False

        self._propellers_spin = torch.tensor(morph.propellers_spin, dtype=gs.tc_float, device=gs.device)
        self._model = morph.model

    def _build(self):
        super()._build()

        self._propellers_revs = torch.zeros(
            self._solver._batch_shape(self._n_propellers), dtype=gs.tc_float, device=gs.device
        )
        self._prev_prop_t = None

    def set_propellels_rpm(self, propellels_rpm):
        """
        Set the RPM (revolutions per minute) for each propeller in the drone.
        设置无人机每个螺旋桨的 RPM（每分钟转数）。

        Parameters
        ----------
        propellels_rpm : array-like or torch.Tensor
            A tensor or array of shape (n_propellers,) or (n_envs, n_propellers) specifying
            the desired RPM values for each propeller. Must be non-negative.
            形状为 (n_propellers,) 或 (n_envs, n_propellers) 的张量或数组，
            指定每个螺旋桨的期望 RPM 值。必须为非负数。

        Raises
        ------
        RuntimeError
            If the method is called more than once per simulation step, or if the input shape
            does not match the number of propellers, or contains negative values.
            如果在每个仿真步骤中多次调用该方法，或输入形状与螺旋桨数量不匹配，
            或包含负值。
        """
        if self._prev_prop_t == self.sim.cur_step_global:
            gs.raise_exception("`set_propellels_rpm` can only be called once per step.")
        self._prev_prop_t = self.sim.cur_step_global

        propellels_rpm = self.solver._process_dim(
            torch.as_tensor(propellels_rpm, dtype=gs.tc_float, device=gs.device)
        ).T.contiguous()
        if len(propellels_rpm) != len(self._propellers_link_idxs):
            gs.raise_exception("Last dimension of `propellels_rpm` does not match `entity.n_propellers`.")
        if torch.any(propellels_rpm < 0):
            gs.raise_exception("`propellels_rpm` cannot be negative.")
        self._propellers_revs = (self._propellers_revs + propellels_rpm) % (60 / self.solver.dt)

        self.solver.set_drone_rpm(
            self._n_propellers,
            self._propellers_link_idxs,
            propellels_rpm,
            self._propellers_spin,
            self.KF,
            self.KM,
            self._model == "RACE",
        )

    def update_propeller_vgeoms(self):
        """
        Update the visual geometry of the propellers for animation based on their current rotation.
        根据螺旋桨的当前旋转更新其可视化几何以实现动画效果。

        This method is a no-op if animation is disabled due to missing visual geometry.
        如果由于缺少可视化几何而禁用动画，则此方法不执行任何操作。
        """
        if self._animate_propellers:
            self.solver.update_drone_propeller_vgeoms(
                self._n_propellers, self._propellers_vgeom_idxs, self._propellers_revs, self._propellers_spin
            )

    @property
    def model(self):
        """The model type of the drone."""
        return self._model

    @property
    def KF(self):
        """The drone's thrust coefficient."""
        return self._KF

    @property
    def KM(self):
        """The drone's moment coefficient."""
        return self._KM

    @property
    def n_propellers(self):
        """The number of propellers on the drone."""
        return self._n_propellers

    @property
    def COM_link_idx(self):
        """The index of the center-of-mass (COM) link of the drone."""
        return self._COM_link_idx

    @property
    def propellers_idx(self):
        """The indices of the drone's propeller links."""
        return self._propellers_link_idxs

    @property
    def propellers_spin(self):
        """The spin direction for each propeller."""
        return self._propellers_spin
