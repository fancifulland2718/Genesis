import numpy as np
import gstaichi as ti

import genesis as gs
import genesis.utils.geom as gu
import genesis.utils.array_class as array_class


@ti.data_oriented
class SDF:
    """Signed Distance Field (SDF) helper for rigid solver.

    刚体求解器的 SDF 工具类。
    - 管理每个几何体的 SDF 网格及其元数据（分辨率、体素起始索引、最大值、单元尺寸等）。
    - 在构造时将 Python/Numpy 侧数据拷入 Taichi 结构化字段，供后续核函数查询。
    """

    def __init__(self, rigid_solver):
        """Initialize SDF fields from solver geoms.

        从求解器的几何体初始化 SDF 字段与信息。
        - 收集每个 geom 的 SDF 体素网格、梯度网格、最近顶点、变换矩阵等。
        - 一次性打包为 Taichi 字段，避免运行时重复组装。
        """
        self.solver = rigid_solver

        n_geoms, n_cells = self.solver.n_geoms, self.solver.n_cells
        self._sdf_info = array_class.get_sdf_info(n_geoms, n_cells)

        if self.solver.n_geoms > 0:
            geoms = self.solver.geoms
            sdf_kernel_init_geom_fields(
                geoms_T_mesh_to_sdf=np.array([geom.T_mesh_to_sdf for geom in geoms], dtype=gs.np_float),
                geoms_sdf_res=np.array([geom.sdf_res for geom in geoms], dtype=gs.np_int),
                geoms_sdf_cell_start=np.array([geom.cell_start for geom in geoms], dtype=gs.np_int),
                geoms_sdf_val=np.concatenate([geom.sdf_val_flattened for geom in geoms], dtype=gs.np_float),
                geoms_sdf_grad=np.concatenate([geom.sdf_grad_flattened for geom in geoms], dtype=gs.np_float),
                geoms_sdf_max=np.array([geom.sdf_max for geom in geoms], dtype=gs.np_float),
                geoms_sdf_cell_size=np.array([geom.sdf_cell_size for geom in geoms], dtype=gs.np_float),
                geoms_sdf_closest_vert=np.concatenate(
                    [geom.sdf_closest_vert_flattened for geom in geoms], dtype=gs.np_int
                ),
                static_rigid_sim_config=self.solver._static_rigid_sim_config,
                sdf_info=self._sdf_info,
            )


@ti.kernel
def sdf_kernel_init_geom_fields(
    geoms_T_mesh_to_sdf: ti.types.ndarray(),
    geoms_sdf_res: ti.types.ndarray(),
    geoms_sdf_cell_start: ti.types.ndarray(),
    geoms_sdf_val: ti.types.ndarray(),
    geoms_sdf_grad: ti.types.ndarray(),
    geoms_sdf_max: ti.types.ndarray(),
    geoms_sdf_cell_size: ti.types.ndarray(),
    geoms_sdf_closest_vert: ti.types.ndarray(),
    static_rigid_sim_config: ti.template(),
    sdf_info: array_class.SDFInfo,
):
    """Copy per-geom SDF metadata/volumes from NumPy into Taichi fields.

    将每个几何体的 SDF 元数据与体素数据从 NumPy 拷贝到 Taichi 字段。
    - geoms_info（每几何体一次）：变换矩阵/分辨率/单元起始索引/最大 SDF 值/单元尺寸。
    - 体素级数组（所有几何体串接）：sdf 值、梯度、最近顶点索引。
    - 使用 loop_config 控制串行/并行以减少编译与调度开销。
    """
    n_geoms = sdf_info.geoms_sdf_start.shape[0]
    n_cells = sdf_info.geoms_sdf_val.shape[0]

    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL)
    for i in range(n_geoms):
        for j, k in ti.static(ti.ndrange(4, 4)):
            sdf_info.geoms_info.T_mesh_to_sdf[i][j, k] = geoms_T_mesh_to_sdf[i, j, k]

        for j in ti.static(range(3)):
            sdf_info.geoms_info.sdf_res[i][j] = geoms_sdf_res[i, j]

        sdf_info.geoms_info.sdf_cell_start[i] = geoms_sdf_cell_start[i]
        sdf_info.geoms_info.sdf_max[i] = geoms_sdf_max[i]
        sdf_info.geoms_info.sdf_cell_size[i] = geoms_sdf_cell_size[i]

    for i in range(n_cells):
        sdf_info.geoms_sdf_val[i] = geoms_sdf_val[i]
        sdf_info.geoms_sdf_closest_vert[i] = geoms_sdf_closest_vert[i]
        for j in ti.static(range(3)):
            sdf_info.geoms_sdf_grad[i][j] = geoms_sdf_grad[i, j]


@ti.func
def sdf_func_world(
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    sdf_info: array_class.SDFInfo,
    pos_world,
    geom_idx,
    batch_idx,
):
    """Evaluate SDF value in world frame.

    在世界坐标系下评估给定点相对指定几何体的 SDF 值。
    - 根据几何类型分支计算：
      * 球：中心距离减去半径；
      * 平面：将点变换到网格坐标后做法向投影；
      * 其它网格：点 → mesh → sdf 帧，再用体素网格插值。
    """

    g_pos = geoms_state.pos[geom_idx, batch_idx]
    g_quat = geoms_state.quat[geom_idx, batch_idx]

    sd = gs.ti_float(0.0)
    if geoms_info.type[geom_idx] == gs.GEOM_TYPE.SPHERE:
        sd = (pos_world - g_pos).norm() - geoms_info.data[geom_idx][0]

    elif geoms_info.type[geom_idx] == gs.GEOM_TYPE.PLANE:
        pos_mesh = gu.ti_inv_transform_by_trans_quat(pos_world, g_pos, g_quat)
        geom_data = geoms_info.data[geom_idx]
        plane_normal = gs.ti_vec3([geom_data[0], geom_data[1], geom_data[2]])
        sd = pos_mesh.dot(plane_normal)

    else:
        pos_mesh = gu.ti_inv_transform_by_trans_quat(pos_world, g_pos, g_quat)
        pos_sdf = gu.ti_transform_by_T(pos_mesh, sdf_info.geoms_info.T_mesh_to_sdf[geom_idx])
        sd = sdf_func_sdf(sdf_info, pos_sdf, geom_idx)

    return sd


@ti.func
def sdf_func_sdf(sdf_info: array_class.SDFInfo, pos_sdf, geom_idx):
    """Evaluate SDF value in sdf-frame (voxel space).

    在 SDF 帧（体素坐标）下求 SDF 值。
    - 若采样点在 SDF 体素外，使用 proxy（到中心距离+偏置）保证值上界；
    - 否则用体素值进行三线性插值（true sdf）。
    """
    signed_dist = gs.ti_float(0.0)
    if sdf_func_is_outside_sdf_grid(sdf_info, pos_sdf, geom_idx):
        signed_dist = sdf_func_proxy_sdf(sdf_info, pos_sdf, geom_idx)
    else:
        signed_dist = sdf_func_true_sdf(sdf_info, pos_sdf, geom_idx)
    return signed_dist


@ti.func
def sdf_func_is_outside_sdf_grid(sdf_info: array_class.SDFInfo, pos_sdf, geom_idx):
    """Check if a sdf-frame point lies outside the valid voxel range.

    判断采样点是否在 SDF 网格有效范围之外（用于切换 proxy/true 路径）。
    """
    res = sdf_info.geoms_info.sdf_res[geom_idx]
    return (pos_sdf >= res - 1).any() or (pos_sdf <= 0).any()


@ti.func
def sdf_func_proxy_sdf(sdf_info: array_class.SDFInfo, pos_sdf, geom_idx):
    """Proxy SDF used outside the cube.

    网格外部的代理 SDF：以到网格中心距离为近似，并加上 sdf_max 使其严格大于网格内值。
    - 确保在最小化/比较时不误选到网格外的点。
    """
    center = (sdf_info.geoms_info.sdf_res[geom_idx] - 1) / 2.0
    sd = (pos_sdf - center).norm() / sdf_info.geoms_info.sdf_cell_size[geom_idx]
    return sd + sdf_info.geoms_info.sdf_max[geom_idx]


@ti.func
def sdf_func_true_sdf(sdf_info: array_class.SDFInfo, pos_sdf, geom_idx):
    """True SDF via trilinear interpolation over the voxel grid.

    通过体素网格对真实 SDF 进行三线性插值。
    管线：
    1) 取 base=floor(pos) 并裁剪到 [0, res-2]；
    2) 遍历 8 个体素角点，计算每一角的权重 w=(1-|dx|)(1-|dy|)(1-|dz|)；
    3) 加权求和得到 signed_dist。
    """
    geom_sdf_res = sdf_info.geoms_info.sdf_res[geom_idx]
    base = ti.min(ti.floor(pos_sdf, gs.ti_int), geom_sdf_res - 2)
    signed_dist = gs.ti_float(0.0)
    for offset in ti.grouped(ti.ndrange(2, 2, 2)):
        pos_cell = base + offset
        w_xyz = 1 - ti.abs(pos_sdf - pos_cell)
        w = w_xyz[0] * w_xyz[1] * w_xyz[2]
        signed_dist = (
            signed_dist
            + w * sdf_info.geoms_sdf_val[sdf_func_ravel_cell_idx(sdf_info, pos_cell, geom_sdf_res, geom_idx)]
        )

    return signed_dist


@ti.func
def sdf_func_ravel_cell_idx(sdf_info: array_class.SDFInfo, cell_idx, sdf_res, geom_idx):
    """Compute flattened index of a 3D voxel coordinate.

    将 3D 体素坐标按行主序压平成一维索引（带每几何体的起始偏移）。
    """
    return (
        sdf_info.geoms_info.sdf_cell_start[geom_idx]
        + cell_idx[0] * sdf_res[1] * sdf_res[2]
        + cell_idx[1] * sdf_res[2]
        + cell_idx[2]
    )


@ti.func
def sdf_func_grad_world(
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    collider_static_config: ti.template(),
    sdf_info: array_class.SDFInfo,
    pos_world,
    geom_idx,
    batch_idx,
):
    """Evaluate SDF gradient (world frame).

    计算世界坐标系下的 SDF 梯度（法向未单位化）。
    - 球/平面：可解析表达式；
    - 网格：点从 world→mesh→sdf，求 sdf 帧梯度后再旋转回 world。
    """

    g_pos = geoms_state.pos[geom_idx, batch_idx]
    g_quat = geoms_state.quat[geom_idx, batch_idx]

    grad_world = ti.Vector.zero(gs.ti_float, 3)
    if geoms_info.type[geom_idx] == gs.GEOM_TYPE.SPHERE:
        grad_world = gu.ti_normalize(pos_world - g_pos)

    elif geoms_info.type[geom_idx] == gs.GEOM_TYPE.PLANE:
        geom_data = geoms_info.data[geom_idx]
        plane_normal = gs.ti_vec3([geom_data[0], geom_data[1], geom_data[2]])
        grad_world = gu.ti_transform_by_quat(plane_normal, g_quat)

    else:
        pos_mesh = gu.ti_inv_transform_by_trans_quat(pos_world, g_pos, g_quat)
        pos_sdf = gu.ti_transform_by_T(pos_mesh, sdf_info.geoms_info.T_mesh_to_sdf[geom_idx])
        grad_sdf = sdf_func_grad(geoms_info, collider_static_config, sdf_info, pos_sdf, geom_idx)

        grad_mesh = grad_sdf  # no rotation between mesh and sdf frame
        grad_world = gu.ti_transform_by_quat(grad_mesh, g_quat)
    return grad_world


@ti.func
def sdf_func_grad(
    geoms_info: array_class.GeomsInfo,
    collider_static_config: ti.template(),
    sdf_info: array_class.SDFInfo,
    pos_sdf,
    geom_idx,
):
    """Evaluate SDF gradient in sdf-frame.

    在 SDF 帧下计算梯度：
    - 网格外：使用指向中心的单位向量作为近似方向（proxy）;
    - 网格内：
      * Terrain 类型：用有限差分（两侧差商）计算；
      * 其它类型：从预存梯度体素以三线性插值恢复。
    """

    grad_sdf = ti.Vector.zero(gs.ti_float, 3)
    if sdf_func_is_outside_sdf_grid(sdf_info, pos_sdf, geom_idx):
        grad_sdf = sdf_func_proxy_grad(sdf_info, pos_sdf, geom_idx)
    else:
        grad_sdf = sdf_func_true_grad(geoms_info, collider_static_config, sdf_info, pos_sdf, geom_idx)
    return grad_sdf


@ti.func
def sdf_func_proxy_grad(sdf_info: array_class.SDFInfo, pos_sdf, geom_idx):
    """Approximate gradient direction by vector to grid center (outside only).

    在网格外，使用“指向 SDF 网格中心的单位向量”近似梯度方向。
    """
    center = (sdf_info.geoms_info.sdf_res[geom_idx] - 1) / 2.0
    proxy_sdf_grad = gu.ti_normalize(pos_sdf - center)
    return proxy_sdf_grad


@ti.func
def sdf_func_true_grad(
    geoms_info: array_class.GeomsInfo,
    collider_static_config: ti.template(),
    sdf_info: array_class.SDFInfo,
    pos_sdf,
    geom_idx,
):
    """True gradient either by finite-difference (Terrain) or trilinear interpolation of precomputed gradients.

    真实梯度计算：
    - Terrain：对 true_sdf 做三维有限差分（步长较大以加速/稳定）;
    - 其它：对预存梯度体素做三线性插值，得到连续梯度。
    实现要点：
    1) 取 base=floor(pos) 并裁剪到 [0, res-2]；
    2) 遍历 8 个角点并累计权重加权的梯度向量。
    """

    sdf_grad_sdf = ti.Vector.zero(gs.ti_float, 3)
    if geoms_info.type[geom_idx] == gs.GEOM_TYPE.TERRAIN:  # Terrain uses finite difference
        if ti.static(collider_static_config.has_terrain):  # for speed up compilation
            # since we are in sdf frame, delta can be a relatively big value
            delta = gs.ti_float(1e-2)

            for i in ti.static(range(3)):
                inc = pos_sdf
                dec = pos_sdf
                inc[i] += delta
                dec[i] -= delta
                sdf_grad_sdf[i] = (
                    sdf_func_true_sdf(sdf_info, inc, geom_idx) - sdf_func_true_sdf(sdf_info, dec, geom_idx)
                ) / (2 * delta)

    else:
        geom_sdf_res = sdf_info.geoms_info.sdf_res[geom_idx]
        base = ti.min(ti.floor(pos_sdf, gs.ti_int), geom_sdf_res - 2)
        for offset in ti.grouped(ti.ndrange(2, 2, 2)):
            pos_cell = base + offset
            w_xyz = 1 - ti.abs(pos_sdf - pos_cell)
            w = w_xyz[0] * w_xyz[1] * w_xyz[2]
            sdf_grad_sdf = (
                sdf_grad_sdf
                + w * sdf_info.geoms_sdf_grad[sdf_func_ravel_cell_idx(sdf_info, pos_cell, geom_sdf_res, geom_idx)]
            )

    return sdf_grad_sdf


@ti.func
def sdf_func_normal_world(
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    collider_static_config: ti.template(),
    sdf_info: array_class.SDFInfo,
    pos_world,
    geom_idx,
    batch_idx,
):
    """Compute normalized SDF normal in world frame.

    返回世界系中的单位法向（对梯度归一化）。
    """
    return gu.ti_normalize(
        sdf_func_grad_world(geoms_state, geoms_info, collider_static_config, sdf_info, pos_world, geom_idx, batch_idx)
    )


@ti.func
def sdf_func_find_closest_vert(
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    sdf_info: array_class.SDFInfo,
    pos_world,
    geom_idx,
    i_b,
):
    """Return the closest mesh vertex (global index) to a world-space query.

    返回给定世界坐标点在指定几何体上的最近网格顶点（全局顶点索引）。
    管线：
    1) world→mesh→sdf 坐标变换；
    2) 将浮点体素坐标裁剪到有效范围并取最近整格；
    3) 用 ravel 后的体素索引查询“最近顶点表”，再加上 geom 的顶点起始偏移得到全局索引。
    """

    g_pos = geoms_state.pos[geom_idx, i_b]
    g_quat = geoms_state.quat[geom_idx, i_b]
    geom_sdf_res = sdf_info.geoms_info.sdf_res[geom_idx]
    pos_mesh = gu.ti_inv_transform_by_trans_quat(pos_world, g_pos, g_quat)
    pos_sdf = gu.ti_transform_by_T(pos_mesh, sdf_info.geoms_info.T_mesh_to_sdf[geom_idx])
    nearest_cell = ti.cast(ti.min(ti.max(pos_sdf, 0), geom_sdf_res - 1), gs.ti_int)
    return (
        sdf_info.geoms_sdf_closest_vert[sdf_func_ravel_cell_idx(sdf_info, nearest_cell, geom_sdf_res, geom_idx)]
        + geoms_info.vert_start[geom_idx]
    )
