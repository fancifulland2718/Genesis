from typing import TYPE_CHECKING

import gstaichi as ti

import genesis as gs
from genesis.repr_base import RBC

if TYPE_CHECKING:
    from genesis.engine.scene import Scene


@ti.data_oriented
class Entity(RBC):
    """
    Base class for all types of entities.
    所有实体类型的基类。
    
    实体（Entity）是仿真系统中的基本对象单元，可以是刚体、软体、流体、粒子系统等。
    该类定义了所有实体的通用接口和属性，是连接几何形状（Morph）、材料（Material）和求解器（Solver）的桥梁。
    
    使用 @ti.data_oriented 装饰器支持 Taichi 的数据导向编程和 GPU 加速。
    """

    def __init__(
        self,
        idx,
        scene,
        morph,
        solver,
        material,
        surface,
    ):
        """
        初始化实体基类。
        
        Parameters
        ----------
        idx : int
            实体在场景中的索引
        scene : Scene
            实体所属的场景对象
        morph : Morph
            实体的几何形状描述
        solver : Solver
            负责模拟该实体的求解器
        material : Material
            实体的材料属性
        surface : Surface
            实体的表面属性（如纹理、颜色等）
        """
        self._uid = gs.UID()  # 实体的全局唯一标识符
        self._idx = idx  # 实体在场景中的索引
        self._scene: "Scene" = scene  # 所属场景
        self._solver = solver  # 所属求解器
        self._material = material  # 材料属性
        self._morph = morph  # 几何形状
        self._surface = surface  # 表面属性
        self._sim = scene.sim  # 仿真器引用

        gs.logger.info(
            f"Adding ~<{self._repr_type()}>~. idx: ~<{self._idx}>~, uid: ~~~<{self._uid}>~~~, morph: ~<{morph}>~, material: ~<{self._material}>~."
        )

    # ------------------------------------------------------------------------------------
    # ----------------------------------- properties -------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def uid(self):
        """
        获取实体的全局唯一标识符。
        
        Returns
        -------
        UID
            实体的唯一标识符
        """
        return self._uid

    @property
    def idx(self):
        """
        获取实体在场景中的索引。
        
        Returns
        -------
        int
            实体索引
        """
        return self._idx

    @property
    def scene(self):
        """
        获取实体所属的场景对象。
        
        Returns
        -------
        Scene
            场景对象
        """
        return self._scene

    @property
    def sim(self):
        """
        获取仿真器对象。
        
        Returns
        -------
        Simulator
            仿真器对象
        """
        return self._sim

    @property
    def solver(self):
        """
        获取负责模拟该实体的求解器。
        
        Returns
        -------
        Solver
            求解器对象
        """
        return self._solver

    @property
    def surface(self):
        """
        获取实体的表面属性。
        
        Returns
        -------
        Surface
            表面属性对象（纹理、颜色等）
        """
        return self._surface

    @property
    def morph(self):
        """
        获取实体的几何形状描述。
        
        Returns
        -------
        Morph
            几何形状对象
        """
        return self._morph

    @property
    def material(self):
        """
        获取实体的材料属性。
        
        Returns
        -------
        Material
            材料对象
        """
        return self._material

    @property
    def is_built(self):
        """
        检查实体是否已构建完成。
        
        Returns
        -------
        bool
            如果实体已构建则返回 True，否则返回 False
        """
        return self._solver._scene._is_built
