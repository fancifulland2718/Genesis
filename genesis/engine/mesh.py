import os
import pickle as pkl

import fast_simplification
import numpy as np
import numpy.typing as npt
import trimesh

import genesis as gs
from genesis.options.surfaces import Surface
import genesis.utils.mesh as mu
import genesis.utils.gltf as gltf_utils
import genesis.utils.particle as pu
from genesis.repr_base import RBC


class Mesh(RBC):
    """
    Genesis 自定义的三角网格对象。
    这是对 `trimesh.Trimesh` 的轻量封装，额外包含表面材质、UV、以及用于仿真的预处理（凸包、面数精简）能力。
    内部的 `trimesh` 对象可通过 `self.trimesh` 访问。

    预处理流水线（可选）：
    - 凸包（convexify）：将输入网格凸化，便于某些物理/碰撞计算的稳定性。
    - 精简（decimate）：通过快速简化算法将面数降至目标值附近，减小计算量。
      注意：精简可能破坏凸性，因此在需要时可在精简后再次凸化。

    参数
    ----------
    surface : genesis.Surface | None
        网格的表面材质（包含纹理/粗糙度/透明度等）。若为 None，将使用默认表面。
    uvs : np.ndarray | None
        顶点 UV 坐标，若材质需要 UV 而未提供，将给出警告。
    convexify : bool
        是否对网格进行凸化处理。
    decimate : bool
        是否对网格进行面数精简。
    decimate_face_num : int
        精简后的目标面数。
    decimate_aggressiveness : int
        精简“激进程度”，范围 0~8：
        - 0 几乎无损；2 基本保持几何特征；5 可能明显改变几何；8 不惜一切代价接近目标面数。
    metadata : dict | None
        网格的元数据（例如来源路径、处理标记等）。
    """

    def __init__(
        self,
        mesh,
        surface: Surface | None = None,
        uvs: npt.NDArray | None = None,
        convexify=False,
        decimate=False,
        decimate_face_num=500,
        decimate_aggressiveness=0,
        metadata=None,
    ):
        self._uid = gs.UID()
        self._mesh = mesh
        self._surface = surface
        self._uvs = uvs
        self._metadata = metadata or {}
        self._color = np.array([1.0, 1.0, 1.0, 1.0], dtype=gs.np_float)

        # 若当前表面需要 UV（例如有颜色/粗糙度/法线贴图），但未提供 UV，则给出警告
        if self._surface.requires_uv():  # check uvs here
            if self._uvs is None:
                if "mesh_path" in metadata:
                    gs.logger.warning(
                        f"Texture given but asset missing uv info (or failed to load): {metadata['mesh_path']}"
                    )
                else:
                    gs.logger.warning("Texture given but asset missing uv info (or failed to load).")
        else:
            # 若表面不需要 UV，则丢弃已有 UV 以避免误用
            self._uvs = None

        # 可选预处理：凸化
        if convexify:
            self.convexify()

        # 可选预处理：面数精简（必要时在精简后再次凸化）
        if decimate:
            self.decimate(decimate_face_num, decimate_aggressiveness, convexify)

    def convexify(self):
        """
        将网格凸化（生成凸包）。
        - 当顶点数大于 3 时执行；否则保持原状。
        - 调用后清空视觉属性（以便由表面重新生成）。
        """
        if self._mesh.vertices.shape[0] > 3:
            # 使用 trimesh 的凸包功能生成凸网格
            self._mesh = trimesh.convex.convex_hull(self._mesh)
            self._metadata["convexified"] = True
        self.clear_visuals()

    def decimate(self, decimate_face_num, decimate_aggressiveness, convexify):
        """
        面数精简（Decimation）。
        流水线：
        1) 若点数 > 3 且当前面数大于目标面数，则尝试精简。
        2) 预处理 self._mesh（trimesh 的内部一致性检查与缓存构建）。
        3) 调用 fast_simplification.simplify 得到简化后的 (verts, faces)。
        4) 重新构建 `trimesh.Trimesh`，记录处理标记。
        5) 若需要保证凸性，则在精简后进行一次凸化（精简可能破坏凸性）。
        6) 清空视觉属性，由表面重新生成。
        """
        if self._mesh.vertices.shape[0] > 3 and len(self._mesh.faces) > decimate_face_num:
            self._mesh.process(validate=True)
            self._mesh = trimesh.Trimesh(
                *fast_simplification.simplify(
                    self._mesh.vertices,
                    self._mesh.faces,
                    target_count=decimate_face_num,
                    agg=decimate_aggressiveness,
                    lossless=(decimate_aggressiveness == 0),
                ),
            )
            self._metadata["decimated"] = True

            # 精简后可能导致非凸，这里可选再次凸化
            if convexify:
                self.convexify()

        self.clear_visuals()

    def remesh(self, edge_len_abs=None, edge_len_ratio=0.01, fix=True):
        """
        为四面体网格生成进行（各向同性）重建 Remesh。
        缓存机制：
        - 通过输入几何与参数生成唯一缓存路径（*.rm）。
        - 若缓存存在且可读，直接加载以避免昂贵计算。
        - 否则使用 pymeshlab 执行各向同性显式重建，并写入缓存。
        参数
        ----------
        edge_len_abs : float | None
            目标边长的绝对值（单位与网格一致）。若为 None，则使用 edge_len_ratio。
        edge_len_ratio : float
            边长相对系数（百分比），当 edge_len_abs 为 None 时生效。
        fix : bool
            预留开关，用于极端情况下的修复流程（例如通过 open3d/pymeshfix）；当前注释掉。
        """
        # 生成重建缓存文件路径（与参数、几何内容相关）
        rm_file_path = mu.get_remesh_path(self.verts, self.faces, edge_len_abs, edge_len_ratio, fix)

        is_cached_loaded = False
        if os.path.exists(rm_file_path):
            gs.logger.debug("Remeshed file (`.rm`) found in cache.")
            try:
                with open(rm_file_path, "rb") as file:
                    verts, faces = pkl.load(file)
                is_cached_loaded = True
            except (EOFError, ModuleNotFoundError, pkl.UnpicklingError):
                gs.logger.info("Ignoring corrupted cache.")

        if not is_cached_loaded:
            # 延迟导入：pymeshlab 较慢，且不常用
            import pymeshlab

            gs.logger.info("Remeshing for tetrahedralization...")
            ms = pymeshlab.MeshSet()
            ms.add_mesh(pymeshlab.Mesh(vertex_matrix=self.verts, face_matrix=self.faces))
            # 选择绝对长度或相对比例两种模式
            if edge_len_abs is not None:
                ms.meshing_isotropic_explicit_remeshing(targetlen=pymeshlab.PureValue(edge_len_abs))
            else:
                ms.meshing_isotropic_explicit_remeshing(targetlen=pymeshlab.PercentageValue(edge_len_ratio * 100))
            m = ms.current_mesh()
            verts, faces = m.vertex_matrix(), m.face_matrix()

            # 极端情况的修复流程（关闭状态）
            # if fix:
            #     verts, faces = pymeshfix.clean_from_arrays(verts, faces)

            # 写入缓存以便下次复用
            os.makedirs(os.path.dirname(rm_file_path), exist_ok=True)
            with open(rm_file_path, "wb") as file:
                pkl.dump((verts, faces), file)

        # 用 remesh 结果替换内部网格，并清空视觉属性
        self._mesh = trimesh.Trimesh(
            vertices=verts,
            faces=faces,
        )
        self.clear_visuals()

    def tetrahedralize(self, tet_cfg):
        """
        将表面网格四面体化（体网格化）。
        流水线：
        1) 延迟导入 pyvista 与 tetgen（较慢）。
        2) 使用 PolyData 包装当前三角网格（注意构建单元的 faces 序列：[n, i0, i1, i2]）。
        3) 创建 TetGen 对象，依据配置生成开关字符串。
        4) 调用 tetrahedralize 生成体网格顶点与四面体单元。
        返回
        ----------
        verts : np.ndarray
            四面体网格顶点坐标。
        elems : np.ndarray
            四面体单元拓扑（每单元 4 个顶点索引）。
        """
        # 延迟导入：不常用且较慢
        import pyvista as pv
        import tetgen

        # PyVista 需要 faces 为 [n, i0, i1, i2] 这样的扁平格式，这里通过 concat 组装
        pv_obj = pv.PolyData(
            self.verts, np.concatenate([np.full((self.faces.shape[0], 1), self.faces.shape[1]), self.faces], axis=1)
        )
        tet = tetgen.TetGen(pv_obj)
        switches = mu.make_tetgen_switches(tet_cfg)
        verts, elems = tet.tetrahedralize(switches=switches)
        # visualize_tet(tet, pv_obj, show_surface=False, plot_cell_qual=False)
        return verts, elems

    def particlize(
        self,
        p_size=0.01,
        sampler="random",
    ):
        """
        使用网格体积采样粒子。
        参数
        ----------
        p_size : float
            目标粒子间距/尺寸。
        sampler : str
            采样器类型；若包含 "pbs" 则使用基于 PBS 的方法，否则使用简单采样。
        """
        if "pbs" in sampler:
            return pu.trimesh_to_particles_pbs(self._mesh, p_size, sampler)
        return pu.trimesh_to_particles_simple(self._mesh, p_size, sampler)

    def clear_visuals(self):
        """
        清空网格的视觉属性：重置为默认表面并更新其纹理。
        - 注意：不会更改几何，只影响渲染相关状态。
        """
        self._surface = gs.surfaces.Default()
        self._surface.update_texture()

    def get_unique_edges(self):
        """
        获取网格的无向唯一边集合（去重）。
        实现要点：
        - 通过将每个三角形的 (v0, v1), (v1, v2), (v2, v0) 展为两列并拼接得到候选边。
        - 对每条边排序（小索引在前）以实现无向去重。
        - 使用 np.unique 去除重复边，再移除自环边。
        返回
        ----------
        edges : (E, 2) np.ndarray
            去重后的边顶点索引对。
        """
        r_face = np.roll(self.faces, 1, axis=1)
        edges = np.concatenate(np.array([self.faces, r_face]).T)

        # 第一遍去重：将每条边的两个端点排序，使 (a,b) 与 (b,a) 归一化为 (min,max)
        edges.sort(axis=1)
        edges = np.unique(edges, axis=0)
        edges = edges[edges[:, 0] != edges[:, 1]]

        return edges

    def copy(self):
        """
        深拷贝当前网格对象（包含 trimesh 的内部缓存、副本表面、UV 与元数据）。
        """
        return Mesh(
            mesh=self._mesh.copy(include_cache=True),
            surface=self._surface.copy(),
            uvs=self._uvs.copy() if self._uvs is not None else None,
            metadata=self._metadata.copy(),
        )

    @classmethod
    def from_trimesh(
        cls,
        mesh,
        scale=None,
        convexify=False,
        decimate=False,
        decimate_face_num=500,
        decimate_aggressiveness=2,
        metadata=None,
        surface=None,
    ):
        """
        从 `trimesh.Trimesh` 构造 `genesis.Mesh`。
        材质/纹理管线（重点）：
        1) 复制输入 mesh（包含缓存），优先从其 visual 中解析 UV（注意将 V 翻转到常见约定：从左下角开始）。
        2) 若定义了 visual：
           - 当为 PBR 材质：解析 baseColorTexture/baseColorFactor、roughnessFactor。
           - 当为 Simple 材质：解析 image/diffuse、glossiness（转换为粗糙度），并考虑透明度 d。
           - 否则回退为 main_color（顶点/面颜色暂未适配 luisa，故统一处理为颜色因子）。
        3) 构建颜色/透明度/粗糙度纹理贴图，并更新 surface 的纹理。
        4) 将 surface 与 uvs 写回 trimesh.visual（供后续渲染/导出）。
        5) 若提供 scale，则缩放顶点。
        6) 根据可选项执行 convexify/decimate。
        """
        if surface is None:
            surface = gs.surfaces.Default()
            surface.update_texture()
        else:
            surface = surface.copy()
        mesh = mesh.copy(include_cache=True)

        # 总是尝试解析 UV：法线/粗糙度/颜色贴图均需要
        try:
            uvs = mesh.visual.uv.copy()
            # trimesh 的 UV V 轴原点在左上，这里翻转为左下（常见渲染约定）
            uvs[:, 1] = 1.0 - uvs[:, 1]
        except:
            uvs = None

        roughness_factor = None
        color_image = None
        color_factor = None
        opacity = 1.0

        if mesh.visual.defined:
            if mesh.visual.kind == "texture":
                material = mesh.visual.material

                # 说明：从 .obj 导入通常不是 PBR；从 .glb 导入通常是 PBR
                if isinstance(material, trimesh.visual.material.PBRMaterial):
                    # 颜色贴图与颜色因子
                    if material.baseColorTexture is not None:
                        color_image = mu.PIL_to_array(material.baseColorTexture)
                    if material.baseColorFactor is not None:
                        color_factor = tuple(np.array(material.baseColorFactor, dtype=np.float32) / 255.0)

                    # 粗糙度
                    if material.roughnessFactor is not None:
                        roughness_factor = (material.roughnessFactor,)

                elif isinstance(material, trimesh.visual.material.SimpleMaterial):
                    # 颜色贴图或漫反射颜色
                    if material.image is not None:
                        color_image = mu.PIL_to_array(material.image)
                    elif material.diffuse is not None:
                        color_factor = tuple(np.array(material.diffuse, dtype=np.float32) / 255.0)

                    # 将 glossiness 转换为粗糙度的一个经验形式
                    if material.glossiness is not None:
                        roughness_factor = ((2 / (material.glossiness + 2)) ** (1.0 / 4.0),)

                    # 透明度（Wavefront MTL 的 d 参数），并融合至颜色 alpha
                    opacity = float(material.kwargs.get("d", [1.0])[0])
                    if opacity < 1.0:
                        if color_factor is None:
                            color_factor = (1.0, 1.0, 1.0, opacity)
                        else:
                            color_factor = (*color_factor[:3], color_factor[3] * opacity)
                else:
                    gs.raise_exception()

            else:
                # 非纹理类 visual：暂不直接支持顶点/面颜色到 luisa，统一映射为颜色因子
                color_factor = tuple(np.array(mesh.visual.main_color, dtype=np.float32) / 255.0)

        else:
            # 默认白色
            color_factor = (1.0, 1.0, 1.0, 1.0)

        # 生成颜色与粗糙度纹理；透明度纹理从颜色纹理中派生
        color_texture = mu.create_texture(color_image, color_factor, "srgb")
        opacity_texture = None
        if color_texture is not None:
            opacity_texture = color_texture.check_dim(3)
        roughness_texture = mu.create_texture(None, roughness_factor, "linear")

        # 更新 surface 纹理并回写 trimesh.visual（包含 UV）
        surface.update_texture(
            color_texture=color_texture,
            opacity_texture=opacity_texture,
            roughness_texture=roughness_texture,
        )
        mesh.visual = mu.surface_uvs_to_trimesh_visual(surface, uvs, len(mesh.vertices))

        # 可选缩放
        if scale is not None:
            mesh.vertices *= scale

        return cls(
            mesh=mesh,
            surface=surface,
            uvs=uvs,
            convexify=convexify,
            decimate=decimate,
            decimate_face_num=decimate_face_num,
            decimate_aggressiveness=decimate_aggressiveness,
            metadata=metadata,
        )

    @classmethod
    def from_attrs(cls, verts, faces, normals=None, surface=None, uvs=None, scale=None):
        """
        通过基础属性（顶点/面/法线/表面/UV）构造 `genesis.Mesh`。
        - 若提供 scale，则在构造时对顶点进行缩放。
        - `process=False` 保留输入拓扑与属性，不做 trimesh 的自动修复。
        """
        if surface is None:
            surface = gs.surfaces.Default()

        return cls(
            mesh=trimesh.Trimesh(
                vertices=verts * scale if scale is not None else verts,
                faces=faces,
                vertex_normals=normals,
                visual=mu.surface_uvs_to_trimesh_visual(surface, uvs, len(verts)),
                process=False,
            ),
            surface=surface,
            uvs=uvs,
        )

    @classmethod
    def from_morph_surface(cls, morph, surface=None):
        """
        通过形变（morph）与表面配置构造 `genesis.Mesh` 或列表。
        - 若 `morph` 为 `morphs.Mesh`（文件型），可能包含多个子网格，本函数返回 `list[Mesh]`。
        - 支持多种文件格式（OBJ/GLTF/GLB/USD 等）及 MeshSet。
        - 非文件型（如 Box/Cylinder/Sphere）则直接生成程序化几何。
        """
        if isinstance(morph, gs.options.morphs.Mesh):
            if morph.is_format(gs.options.morphs.MESH_FORMATS):
                meshes = mu.parse_mesh_trimesh(morph.file, morph.group_by_material, morph.scale, surface)
            elif morph.is_format(gs.options.morphs.GLTF_FORMATS):
                if morph.parse_glb_with_trimesh:
                    meshes = mu.parse_mesh_trimesh(morph.file, morph.group_by_material, morph.scale, surface)
                else:
                    meshes = gltf_utils.parse_mesh_glb(morph.file, morph.group_by_material, morph.scale, surface)
            elif morph.is_format(gs.options.morphs.USD_FORMATS):
                import genesis.utils.usda as usda_utils

                meshes = usda_utils.parse_mesh_usd(morph.file, morph.group_by_material, morph.scale, surface)
            elif isinstance(morph, gs.options.morphs.MeshSet):
                assert all(isinstance(mesh, trimesh.Trimesh) for mesh in morph.files)
                meshes = [mu.trimesh_to_mesh(mesh, morph.scale, surface) for mesh in morph.files]
            else:
                gs.raise_exception(
                    f"File type not supported (yet). Submit a feature request if you need this: {morph.file}."
                )

            return meshes

        else:
            # 程序化几何
            if isinstance(morph, gs.options.morphs.Box):
                tmesh = mu.create_box(extents=morph.size)

            elif isinstance(morph, gs.options.morphs.Cylinder):
                tmesh = mu.create_cylinder(radius=morph.radius, height=morph.height)

            elif isinstance(morph, gs.options.morphs.Sphere):
                tmesh = mu.create_sphere(radius=morph.radius)

            else:
                gs.raise_exception()

            metadata = {"mesh_path": morph.file} if isinstance(morph, gs.options.morphs.FileMorph) else {}
            return cls.from_trimesh(tmesh, surface=surface, metadata=metadata)

    def set_color(self, color):
        """
        设置网格颜色（将颜色写入 surface 的颜色纹理，透明度纹理由颜色通道派生），并更新 trimesh 的 visual。
        参数
        ----------
        color : array-like[4] or [3]
            RGBA 或 RGB。若为 RGB，将从 surface 中推导透明度或默认 1。
        """
        self._color = color
        color_texture = gs.textures.ColorTexture(color=tuple(color))
        opacity_texture = color_texture.check_dim(3)
        self._surface.update_texture(color_texture=color_texture, opacity_texture=opacity_texture, force=True)
        self.update_trimesh_visual()

    def update_trimesh_visual(self):
        """
        使用当前 surface 与 uvs 更新内部 trimesh 对象的 visual（渲染属性）。
        """
        self._mesh.visual = mu.surface_uvs_to_trimesh_visual(self.surface, self.uvs, len(self.verts))

    def apply_transform(self, T):
        """
        对网格施加 4x4 齐次变换矩阵（就地变换顶点与法线等）。
        """
        self._mesh.apply_transform(T)

    def show(self):
        """
        使用 trimesh 自带的可视化器展示网格。
        """
        return self._mesh.show()

    @property
    def uid(self):
        """
        返回网格的唯一 ID（用于资源管理/引用）。
        """
        return self._uid

    @property
    def trimesh(self):
        """
        返回内部的 `trimesh.Trimesh` 对象引用。
        """
        return self._mesh

    @property
    def is_convex(self) -> bool:
        """
        返回网格是否为凸集（由 trimesh 判定）。
        """
        return self._mesh.is_convex

    @property
    def metadata(self):
        """
        返回网格的元数据字典（来源、处理标记等）。
        """
        return self._metadata

    @property
    def verts(self):
        """
        返回网格顶点数组（N, 3）。
        """
        return self._mesh.vertices

    @verts.setter
    def verts(self, verts):
        """
        设置网格顶点数组（长度必须一致）。
        """
        assert len(verts) == len(self.verts)
        self._mesh.vertices = verts

    @property
    def faces(self):
        """
        返回网格三角面索引（M, 3）。
        """
        return self._mesh.faces

    @property
    def normals(self):
        """
        返回顶点法线（若存在/可计算）。
        """
        return self._mesh.vertex_normals

    @property
    def surface(self):
        """
        返回网格的表面材质（包含纹理贴图）。
        """
        return self._surface

    @property
    def uvs(self):
        """
        返回网格的顶点 UV 坐标（若存在）。
        """
        return self._uvs

    @property
    def area(self):
        """
        网格表面积。
        """
        return self._mesh.area

    @property
    def volume(self):
        """
        网格体积（封闭网格）。
        """
        return self._mesh.volume