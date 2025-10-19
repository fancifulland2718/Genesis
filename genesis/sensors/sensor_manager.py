from typing import TYPE_CHECKING, Type

import numpy as np
import torch

from genesis.utils.ring_buffer import TensorRingBuffer

if TYPE_CHECKING:
    # 仅用于类型检查时导入，避免运行时循环依赖或额外开销
    from genesis.vis.rasterizer_context import RasterizerContext

    from .base_sensor import Sensor, SensorOptions, SharedSensorMetadata


class SensorManager:
    """
    传感器管理器：
    - 负责创建、构建、调度和复位所有传感器实例
    - 统一维护不同 dtype 的共享缓存（ground-truth 与实时缓存）以及传感器延迟的环形缓冲区
    - 为传感器提供高效的数据切片访问与按步克隆的只读视图
    """

    # 将“选项类型”映射到对应的“传感器类、共享元数据类、数据结构类型（tuple）”
    # 作用：在 create_sensor 时通过 options 的具体类型选出正确的实现类与元数据类型
    SENSOR_TYPES_MAP: dict[Type["SensorOptions"], tuple[Type["Sensor"], Type["SharedSensorMetadata"], Type[tuple]]] = {}

    def __init__(self, sim):
        """
        参数：
        - sim：仿真器实例，提供批大小 B、时间步等全局信息
        """
        self._sim = sim
        # 同类（同一个 Sensor 子类）的传感器按列表分组：{Sensor子类: [sensor实例, ...]}
        self._sensors_by_type: dict[Type["Sensor"], list["Sensor"]] = {}
        # 每个传感器类型对应一份共享元数据，部分类型可能尚未构建（值为 None）
        self._sensors_metadata: dict[Type["Sensor"], SharedSensorMetadata | None] = {}
        # 每种 torch.dtype 对应一块“真值缓存”（ground-truth cache），形状：[B, sum(cache_size per dtype)]
        self._ground_truth_cache: dict[Type[torch.dtype], torch.Tensor] = {}
        # 每种 torch.dtype 对应一块“实时缓存”（可能在 ground-truth 基础上经进一步计算），形状同上
        self._cache: dict[Type[torch.dtype], torch.Tensor] = {}
        # 针对每种 dtype 的环形缓冲区（处理传感器延迟/历史），外形类似：[T(max delay), B, size_per_dtype]
        self._buffered_data: dict[Type[torch.dtype], TensorRingBuffer] = {}
        # 每个传感器类型在对应 dtype 大缓存中的切片范围（列区间）
        self._cache_slices_by_type: dict[Type["Sensor"], slice] = {}
        # 标记某传感器类型是否需要更新实时缓存（若同类全部为仅更新真值，则无需更新实时缓存）
        self._should_update_cache_by_type: dict[Type["Sensor"], bool] = {}
        # 针对 (是否真值, dtype) 记录最近一次克隆缓存的全局时间步，用于避免同一时间步重复 clone
        self._last_cache_cloned_step: dict[tuple[bool, Type[torch.dtype]], int] = {}
        # 针对 (是否真值, dtype) 存储当步的克隆缓存（B, size_per_dtype），供按需切片只读访问
        self._cloned_cache: dict[tuple[bool, Type[torch.dtype]], torch.Tensor] = {}

    def create_sensor(self, sensor_options: "SensorOptions") -> "Sensor":
        """
        根据传入的传感器选项创建传感器实例，并注册到管理器中。
        流程：
        1) 校验选项与场景兼容性
        2) 通过 SENSOR_TYPES_MAP 定位具体 Sensor 类、共享元数据类和数据布局类型
        3) 为该类型首个传感器构造共享元数据实例
        4) 创建具体传感器并登记
        """
        sensor_options.validate(self._sim.scene)
        sensor_cls, metadata_cls, data_cls = SensorManager.SENSOR_TYPES_MAP[type(sensor_options)]
        self._sensors_by_type.setdefault(sensor_cls, [])
        if sensor_cls not in self._sensors_metadata:
            self._sensors_metadata[sensor_cls] = metadata_cls()
        # 索引为在同类型列表中的位置；传入 data_cls（数据结构类型）和管理器自身
        sensor = sensor_cls(sensor_options, len(self._sensors_by_type[sensor_cls]), data_cls, self)
        self._sensors_by_type[sensor_cls].append(sensor)
        return sensor

    def build(self):
        """
        构建阶段：
        - 为每种 dtype 统计所需缓存大小（将所有该 dtype 的传感器缓存列拼接）
        - 为每个传感器计算在大缓存中的起始列偏移 _cache_idx
        - 为各 dtype 分配 ground-truth cache、实时 cache、以及环形缓冲区
        - 调用各传感器的 build 钩子完成自身构建
        """
        max_buffer_len = 0  # 所有传感器的最大延迟步数 + 1，作为环形缓冲长度
        cache_size_per_dtype = {}  # {dtype: 总列数}

        for sensor_cls, sensors in self._sensors_by_type.items():
            dtype = sensor_cls._get_cache_dtype()

            # 初始化“按步克隆缓存”的状态
            for is_ground_truth in (False, True):
                self._last_cache_cloned_step.setdefault((is_ground_truth, dtype), -1)
                self._cloned_cache.setdefault((is_ground_truth, dtype), torch.zeros(0, dtype=dtype))

            cache_size_per_dtype.setdefault(dtype, 0)
            cls_cache_start_idx = cache_size_per_dtype[dtype]  # 本类型在该 dtype 大缓存中的起始列

            # 若该类型的所有传感器均仅更新真值缓存，则实时缓存可跳过更新
            update_ground_truth_only = True
            for sensor in sensors:
                update_ground_truth_only &= sensor._options.update_ground_truth_only
                # 为每个传感器分配在 dtype 缓存中的列偏移区间
                sensor._cache_idx = cache_size_per_dtype[dtype]
                cache_size_per_dtype[dtype] += sensor._cache_size
                # 统计最大延迟（+1 因为当前帧也需要一格）
                max_buffer_len = max(max_buffer_len, sensor._delay_ts + 1)
            self._should_update_cache_by_type[sensor_cls] = not update_ground_truth_only

            cls_cache_end_idx = cache_size_per_dtype[dtype]
            # 记录该类型在 dtype 级大缓存中的切片范围
            self._cache_slices_by_type[sensor_cls] = slice(cls_cache_start_idx, cls_cache_end_idx)

        # 依据统计结果为每个 dtype 分配大缓存与环形缓冲
        for dtype in cache_size_per_dtype.keys():
            cache_shape = (self._sim._B, cache_size_per_dtype[dtype])  # B x 列数
            # 真值缓存：用于存放场景/物理引擎直接给出的无延迟“地面真值”
            self._ground_truth_cache[dtype] = torch.zeros(cache_shape, dtype=dtype)
            # 实时缓存：通常在真值基础上进一步处理（滤波、延迟、噪声等）
            self._cache[dtype] = torch.zeros(cache_shape, dtype=dtype)
            # 环形缓冲：按时间维度循环覆盖，用于实现时序延迟与历史引用
            self._buffered_data[dtype] = TensorRingBuffer(max_buffer_len, cache_shape, dtype=dtype)

        # 通知每个传感器完成自身构建
        for sensor_cls, sensors in self._sensors_by_type.items():
            dtype = sensor_cls._get_cache_dtype()
            for sensor in sensors:
                sensor.build()
                sensor._is_built = True

    def step(self):
        """
        每仿真步更新：
        1) 让传感器类型级别的静态方法将“共享真值缓存”写入（或从场景采样写入）
        2) 若需要，再基于真值缓存更新“实时缓存”和对应环形缓冲（例如引入延迟、噪声、滤波）
        注意：这里按“传感器类型”维度操作，使用的是该类型在 dtype 级大缓存中的切片视图。
        """
        for sensor_cls in self._sensors_by_type.keys():
            dtype = sensor_cls._get_cache_dtype()
            cache_slice = self._cache_slices_by_type[sensor_cls]
            # 先更新共享真值缓存（通常为本步地面真值数据）
            sensor_cls._update_shared_ground_truth_cache(
                self._sensors_metadata[sensor_cls], self._ground_truth_cache[dtype][:, cache_slice]
            )
            # 若该类型不是“仅更新真值”，则更新实时缓存与环形缓冲
            if self._should_update_cache_by_type[sensor_cls]:
                sensor_cls._update_shared_cache(
                    self._sensors_metadata[sensor_cls],
                    self._ground_truth_cache[dtype][:, cache_slice],
                    self._cache[dtype][:, cache_slice],
                    self._buffered_data[dtype][:, cache_slice],
                )

    def draw_debug(self, context: "RasterizerContext", buffer_updates: dict[str, np.ndarray]):
        """
        调试绘制：
        - 对启用 draw_debug 的传感器调用其调试绘制接口
        - buffer_updates 可传入需要在 UI 中展示的关键数组
        """
        for sensor in self.sensors:
            if sensor._options.draw_debug:
                sensor._draw_debug(context, buffer_updates)

    def reset(self, envs_idx=None):
        """
        局部/全局重置：
        - 将指定环境索引的缓存与环形缓冲清零
        - 清空按步克隆缓存的标记，使下一步强制重新克隆
        - 通知各传感器类型执行自身的 reset 钩子
        参数：
        - envs_idx: None 表示全部环境，否则为需要重置的环境下标集合/张量
        """
        envs_idx = self._sim._scene._sanitize_envs_idx(envs_idx)
        for dtype in self._buffered_data.keys():
            # 对选定环境批次维度进行就地清零
            self._ground_truth_cache[dtype][envs_idx] = 0.0
            self._cache[dtype][envs_idx] = 0.0
            self._buffered_data[dtype].buffer[:, envs_idx] = 0.0
        # 使克隆缓存失效，下一步会重新克隆
        for key in self._last_cache_cloned_step.keys():
            self._cloned_cache[key] = 0.0
            self._last_cache_cloned_step[key] = -1  # 标记为不可用
        # 通知各传感器类型执行类型级复位（可用于清理内部状态）
        for sensor_cls in self._sensors_by_type.keys():
            sensor_cls.reset(self._sensors_metadata[sensor_cls], envs_idx)

    def get_cloned_from_cache(self, sensor: "Sensor", is_ground_truth: bool = False) -> torch.Tensor:
        """
        返回指定传感器对应缓存（真值或实时）的“按步克隆副本”的切片视图。
        设计目的：
        - 避免在同一时间步多次克隆大缓存（昂贵），此处对 (is_ground_truth, dtype) 维度做一次性克隆
        - 各传感器仅切片访问属于自己的列区间，得到该步的稳定快照，不受后续写入影响
        参数：
        - sensor: 目标传感器
        - is_ground_truth: True 则从真值缓存克隆，False 则从实时缓存克隆
        返回：
        - 形状为 [B, sensor._cache_size] 的张量切片（源自当步克隆的大缓存）
        """
        dtype = sensor._get_cache_dtype()
        key = (is_ground_truth, dtype)
        # 若当前全局步尚未克隆，则克隆一次（共用）
        if self._last_cache_cloned_step[key] != self._sim.cur_step_global:
            self._last_cache_cloned_step[key] = self._sim.cur_step_global
            if is_ground_truth:
                self._cloned_cache[key] = self._ground_truth_cache[dtype].clone()
            else:
                self._cloned_cache[key] = self._cache[dtype].clone()
        # 返回传感器专属区间的切片视图
        return self._cloned_cache[key][:, sensor._cache_idx : sensor._cache_idx + sensor._cache_size]

    @property
    def sensors(self):
        """
        返回所有传感器实例的扁平元组视图（只读语义）
        """
        return tuple([sensor for sensor_list in self._sensors_by_type.values() for sensor in sensor_list])


def register_sensor(
    options_cls: Type["SensorOptions"], metadata_cls: Type["SharedSensorMetadata"], data_cls: Type[tuple]
):
    """
    传感器注册装饰器：
    - 用法：@register_sensor(OptionsCls, MetadataCls, DataTupleType) 置于 Sensor 子类定义之上
    - 效果：将 Options 类型映射到具体的 Sensor/Metadata/Data 类型，供 create_sensor 查表使用
    """
    def _impl(sensor_cls: Type["Sensor"]):
        SensorManager.SENSOR_TYPES_MAP[options_cls] = sensor_cls, metadata_cls, data_cls
        return sensor_cls

    return _impl
