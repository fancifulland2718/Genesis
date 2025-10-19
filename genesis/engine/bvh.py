import genesis as gs
import gstaichi as ti
from genesis.repr_base import RBC
import numpy as np


@ti.data_oriented
class AABB(RBC):
    """
    轴对齐包围盒（Axis-Aligned Bounding Box，AABB）批量管理类。

    用途
    - 在 GPU 上以批量形式存储与计算一组 AABB（min/max 三维坐标）。
    - 提供相交测试等基础操作（在 Taichi dataclass 内）。

    参数
    - n_batches: 批次数（环境数）
    - n_aabbs: 每个批次中的 AABB 数量

    字段
    - ti_aabb: Taichi dataclass，表示单个 AABB（含 `min` 与 `max` 向量）
    - aabbs: Taichi field，形状为 (n_batches, n_aabbs)，存放所有 AABB
    """

    def __init__(self, n_batches, n_aabbs):
        self.n_batches = n_batches
        self.n_aabbs = n_aabbs

        @ti.dataclass
        class ti_aabb:
            min: gs.ti_vec3
            max: gs.ti_vec3

            @ti.func
            def intersects(self, other) -> bool:
                """
                AABB 相交测试（闭区间判定）
                返回 True 当且仅当两个 AABB 在三个坐标轴上均有重叠。
                """
                return (
                    self.min[0] <= other.max[0]
                    and self.max[0] >= other.min[0]
                    and self.min[1] <= other.max[1]
                    and self.max[1] >= other.min[1]
                    and self.min[2] <= other.max[2]
                    and self.max[2] >= other.min[2]
                )

        self.ti_aabb = ti_aabb

        self.aabbs = ti_aabb.field(
            shape=(n_batches, n_aabbs),
            needs_grad=False,
            layout=ti.Layout.SOA,
        )


@ti.data_oriented
class LBVH(RBC):
    """
    线性 BVH（LBVH）用于加速基于 AABB 的碰撞检测 / 交叉查询。

    管线概述
    1) compute_aabb_centers_and_scales：计算每个 AABB 的中心与归一化尺度
    2) compute_morton_codes：对中心点生成 MortonCode（Z-order，空间映射到一维）
    3) radix_sort_morton_codes：按字节进行基数排序（Radix sort）
    4) build_radix_tree：依据 MortonCode 构建层次树（Karras 2012）
    5) compute_bounds：自下而上装配各节点（internal）的包围盒

    字段说明
    - aabbs: 输入的 AABB 数组，形状 (n_batches, n_aabbs)
    - morton_codes: 对每个 AABB 生成的 Morton 编码与原索引（u32 vec2）
    - nodes: BVH 节点数组（前 n_aabbs-1 为内部节点，后 n_aabbs 为叶子）
    - internal_node_active / internal_node_ready: 自底向上装配 bound 的分层推进标记
    - query_result / query_result_count: 查询返回的三元组 (batch_id, aabb_id, query_id) 与计数

    参考
    - Karras, T. "Maximizing Parallelism in the Construction of BVHs, Octrees, and k-d Trees." HPG 2012.
      https://research.nvidia.com/sites/default/files/pubs/2012-06_Maximizing-Parallelism-in/karras2012hpg_paper.pdf
    """

    def __init__(self, aabb: AABB, max_n_query_result_per_aabb: int = 8, n_radix_sort_groups: int = 256):
        if aabb.n_aabbs < 2:
            raise gs.GenesisException("The number of AABBs must be larger than 2.")
        n_radix_sort_groups = min(aabb.n_aabbs, n_radix_sort_groups)

        self.aabbs = aabb.aabbs
        self.n_aabbs = aabb.n_aabbs
        self.n_batches = aabb.n_batches

        # 最大查询结果条数（上限避免溢出）
        self.max_query_results = min(self.n_aabbs * max_n_query_result_per_aabb * self.n_batches, 0x7FFFFFFF)
        # 遍历栈最大深度（迭代 DFS）
        self.max_stack_depth = 64
        self.aabb_centers = ti.field(gs.ti_vec3, shape=(self.n_batches, self.n_aabbs))
        self.aabb_min = ti.field(gs.ti_vec3, shape=(self.n_batches))
        self.aabb_max = ti.field(gs.ti_vec3, shape=(self.n_batches))
        self.scale = ti.field(gs.ti_vec3, shape=(self.n_batches))
        # morton_codes: vec2(u32) = [编码(高 32)、原索引(低 32)]
        self.morton_codes = ti.field(ti.types.vector(2, ti.u32), shape=(self.n_batches, self.n_aabbs))

        # 基数排序直方图（单组版本，256 桶 = 8 bit）
        self.hist = ti.field(ti.u32, shape=(self.n_batches, 256))
        # 直方图前缀和（含一个 0 前缀）
        self.prefix_sum = ti.field(ti.u32, shape=(self.n_batches, 256 + 1))
        # 每个元素在对应桶里的局部偏移
        self.offset = ti.field(ti.u32, shape=(self.n_batches, self.n_aabbs))
        # 临时存放排序结果
        self.tmp_morton_codes = ti.field(ti.types.vector(2, ti.u32), shape=(self.n_batches, self.n_aabbs))

        # 分组基数排序（更大数据量时减少串行片段）
        self.n_radix_sort_groups = n_radix_sort_groups
        self.hist_group = ti.field(ti.u32, shape=(self.n_batches, self.n_radix_sort_groups, 256 + 1))
        self.prefix_sum_group = ti.field(ti.u32, shape=(self.n_batches, self.n_radix_sort_groups + 1, 256))
        self.group_size = self.n_aabbs // self.n_radix_sort_groups
        self.visited = ti.field(ti.u8, shape=(self.n_aabbs,))

        @ti.dataclass
        class Node:
            """
            BVH 节点结构
            - left/right: 子节点索引（叶子为 -1/-1）
            - parent: 父节点索引（根的 parent = -1）
            - bound: 此节点的 AABB（内节点为子包围盒并集；叶子为原 AABB）
            """

            left: ti.i32
            right: ti.i32
            parent: ti.i32
            bound: aabb.ti_aabb

        self.Node = Node

        # 节点数组：前 n_aabbs-1 为内部节点，后 n_aabbs 为叶节点
        self.nodes = self.Node.field(shape=(self.n_batches, self.n_aabbs * 2 - 1))
        # 自底向上装配 bound 的层推进状态
        self.internal_node_active = ti.field(gs.ti_bool, shape=(self.n_batches, self.n_aabbs - 1))
        self.internal_node_ready = ti.field(gs.ti_bool, shape=(self.n_batches, self.n_aabbs - 1))

        # 查询结果 (batch id, self id, query id)
        self.query_result = ti.field(gs.ti_ivec3, shape=(self.max_query_results))
        # 查询结果计数
        self.query_result_count = ti.field(ti.i32, shape=())

    def build(self):
        """
        构建 BVH（主流程）：
        1) 计算中心与尺度
        2) 生成 Morton 编码
        3) 基数排序（按字节）
        4) 基于 Morton 相邻关系构建树
        5) 自底向上装配各节点的包围盒
        """
        self.compute_aabb_centers_and_scales()
        self.compute_morton_codes()
        self.radix_sort_morton_codes()
        self.build_radix_tree()
        self.compute_bounds()

    @ti.func
    def filter(self, i_a, i_q):
        """
        查询过滤器（默认不过滤，返回 False）。

        可在子类中覆写以实现自定义过滤逻辑。
        - i_a: 命中的 AABB 索引
        - i_q: 查询的 AABB 索引
        """
        return False

    @ti.kernel
    def compute_aabb_centers_and_scales(self):
        """
        计算 AABB 中心与全局缩放（归一化到 [0,1]，避免坐标尺度差异影响 Morton 分布）。
        - 先计算中心
        - 用所有 AABB 的 min/max 计算尺度 scale = 1/max(范围, EPS)
        """
        for i_b, i_a in ti.ndrange(self.n_batches, self.n_aabbs):
            self.aabb_centers[i_b, i_a] = (self.aabbs[i_b, i_a].min + self.aabbs[i_b, i_a].max) / 2

        for i_b in ti.ndrange(self.n_batches):
            self.aabb_min[i_b] = self.aabb_centers[i_b, 0]
            self.aabb_max[i_b] = self.aabb_centers[i_b, 0]

        for i_b, i_a in ti.ndrange(self.n_batches, self.n_aabbs):
            ti.atomic_min(self.aabb_min[i_b], self.aabbs[i_b, i_a].min)
            ti.atomic_max(self.aabb_max[i_b], self.aabbs[i_b, i_a].max)

        for i_b in ti.ndrange(self.n_batches):
            scale = self.aabb_max[i_b] - self.aabb_min[i_b]
            for i in ti.static(range(3)):
                self.scale[i_b][i] = ti.select(scale[i] > gs.EPS, 1.0 / scale[i], 1.0)

    @ti.kernel
    def compute_morton_codes(self):
        """
        计算每个 AABB 的 Morton 编码（Z-order 曲线）。
        - 将归一化中心映射到 10-bit（[0,1024)），分别展开为 30-bit（插零），再交织合并成 30-bit Morton code
        - morton_codes 的第二个分量存放原始索引，便于稳定排序与回溯
        """
        for i_b, i_a in ti.ndrange(self.n_batches, self.n_aabbs):
            center = self.aabb_centers[i_b, i_a] - self.aabb_min[i_b]
            scaled_center = center * self.scale[i_b]
            morton_code_x = ti.floor(scaled_center[0] * 1023.0, dtype=ti.u32)
            morton_code_y = ti.floor(scaled_center[1] * 1023.0, dtype=ti.u32)
            morton_code_z = ti.floor(scaled_center[2] * 1023.0, dtype=ti.u32)
            morton_code_x = self.expand_bits(morton_code_x)
            morton_code_y = self.expand_bits(morton_code_y)
            morton_code_z = self.expand_bits(morton_code_z)
            morton_code = (morton_code_x << 2) | (morton_code_y << 1) | (morton_code_z)
            self.morton_codes[i_b, i_a] = ti.Vector([morton_code, i_a], dt=ti.u32)

    @ti.func
    def expand_bits(self, v: ti.u32) -> ti.u32:
        """
        将 10-bit 展开为 30-bit：每个 bit 前插入两个 0（bit interleaving 前的预处理）。
        实现基于掩码与移位的“位扩张”技巧（参考常见 Morton 编码实现）。
        """
        v = (v * ti.u32(0x00010001)) & ti.u32(0xFF0000FF)
        # 为抑制 Taichi 的溢出警告，使用等效形式（性能差异可忽略）
        # 原式：v = (v * ti.u32(0x00000101)) & ti.u32(0x0F00F00F)
        v = (v | ((v & 0x00FFFFFF) << 8)) & 0x0F00F00F
        v = (v * ti.u32(0x00000011)) & ti.u32(0xC30C30C3)
        v = (v * ti.u32(0x00000005)) & ti.u32(0x49249249)
        return v

    def radix_sort_morton_codes(self):
        """
        基数排序（高 32bit 的 MortonCode 以字节为单位进行 4 轮排序）。
        说明：morton_codes 的低 32bit 为原索引，已天然稳定，无需排序。
        """
        # 仅处理 morton_code（vec2 的第 0 分量）；按字节从低到高进行 4 轮（i=4..7）
        for i in range(4, 8):
            if self.n_radix_sort_groups == 1:
                self._kernel_radix_sort_morton_codes_one_round(i)
            else:
                self._kernel_radix_sort_morton_codes_one_round_group(i)

    @ti.kernel
    def _kernel_radix_sort_morton_codes_one_round(self, i: int):
        # 单组版本：对所有元素构建 256 桶直方图 → 前缀和 → 重新布置 → 交换临时缓冲
        self.hist.fill(0)

        # 统计直方图（此处顺序遍历，若需更高并行度可考虑分组版）
        for i_b in range(self.n_batches):
            for i_a in range(self.n_aabbs):
                code = (self.morton_codes[i_b, i_a][1 - (i // 4)] >> ((i % 4) * 8)) & 0xFF
                self.offset[i_b, i_a] = ti.atomic_add(self.hist[i_b, ti.i32(code)], 1)

        # 前缀和（顺序累加，256 桶）
        for i_b in ti.ndrange(self.n_batches):
            self.prefix_sum[i_b, 0] = 0
            for j in range(1, 256):
                self.prefix_sum[i_b, j] = self.prefix_sum[i_b, j - 1] + self.hist[i_b, j - 1]

        # 按桶与偏移重排到临时缓冲
        for i_b, i_a in ti.ndrange(self.n_batches, self.n_aabbs):
            code = ti.i32((self.morton_codes[i_b, i_a][1 - (i // 4)] >> ((i % 4) * 8)) & 0xFF)
            idx = ti.i32(self.offset[i_b, i_a] + self.prefix_sum[i_b, code])
            self.tmp_morton_codes[i_b, idx] = self.morton_codes[i_b, i_a]

        # 写回
        for i_b, i_a in ti.ndrange(self.n_batches, self.n_aabbs):
            self.morton_codes[i_b, i_a] = self.tmp_morton_codes[i_b, i_a]

    @ti.kernel
    def _kernel_radix_sort_morton_codes_one_round_group(self, i: int):
        # 分组版本：每组独立构建直方图与局部前缀，随后用全局前缀定位写回索引
        self.hist_group.fill(0)

        # 分组直方图与组内偏移
        for i_b, i_g in ti.ndrange(self.n_batches, self.n_radix_sort_groups):
            start = i_g * self.group_size
            end = ti.select(i_g == self.n_radix_sort_groups - 1, self.n_aabbs, (i_g + 1) * self.group_size)
            for i_a in range(start, end):
                code = ti.i32((self.morton_codes[i_b, i_a][1 - (i // 4)] >> ((i % 4) * 8)) & 0xFF)
                self.offset[i_b, i_a] = self.hist_group[i_b, i_g, code]
                self.hist_group[i_b, i_g, code] = self.hist_group[i_b, i_g, code] + 1

        # 组内/组间前缀和
        for i_b, i_c in ti.ndrange(self.n_batches, 256):
            self.prefix_sum_group[i_b, 0, i_c] = 0
            for i_g in range(1, self.n_radix_sort_groups + 1):
                self.prefix_sum_group[i_b, i_g, i_c] = (
                    self.prefix_sum_group[i_b, i_g - 1, i_c] + self.hist_group[i_b, i_g - 1, i_c]
                )
        for i_b in range(self.n_batches):
            self.prefix_sum[i_b, 0] = 0
            for i_c in range(1, 256 + 1):
                self.prefix_sum[i_b, i_c] = (
                    self.prefix_sum[i_b, i_c - 1] + self.prefix_sum_group[i_b, self.n_radix_sort_groups, i_c - 1]
                )

        # 重排写入临时缓冲（全局前缀 + 组内前缀 + 组内偏移）
        for i_b, i_a in ti.ndrange(self.n_batches, self.n_aabbs):
            code = ti.i32((self.morton_codes[i_b, i_a][1 - (i // 4)] >> ((i % 4) * 8)) & 0xFF)
            i_g = ti.min(i_a // self.group_size, self.n_radix_sort_groups - 1)
            idx = ti.i32(self.prefix_sum[i_b, code] + self.prefix_sum_group[i_b, i_g, code] + self.offset[i_b, i_a])
            self.tmp_morton_codes[i_b, idx] = self.morton_codes[i_b, i_a]

        # 写回
        for i_b, i_a in ti.ndrange(self.n_batches, self.n_aabbs):
            self.morton_codes[i_b, i_a] = self.tmp_morton_codes[i_b, i_a]

    @ti.kernel
    def build_radix_tree(self):
        """
        基于已排序的 MortonCode 构建 Radix 树（Karras 2012 算法）。
        核心思路：
        - 对每个内部节点 i，沿 Morton 排序序列找到其覆盖范围 [min,max] 与分裂点 gamma，
          由此确定左右孩子（可能是内节点或叶节点）。
        """
        # 根节点 parent 设为 -1
        for i_b in ti.ndrange(self.n_batches):
            self.nodes[i_b, 0].parent = -1

        # 初始化叶节点（left/right = -1）
        for i_b, i in ti.ndrange(self.n_batches, self.n_aabbs):
            self.nodes[i_b, i + self.n_aabbs - 1].left = -1
            self.nodes[i_b, i + self.n_aabbs - 1].right = -1

        # 并行构建内部节点（索引 0..n_aabbs-2）
        for i_b, i in ti.ndrange(self.n_batches, self.n_aabbs - 1):
            # 方向 d：比较 i 与相邻两侧 LCP，选择增长更快的一侧
            d = ti.select(self.delta(i, i + 1, i_b) > self.delta(i, i - 1, i_b), 1, -1)

            # 扩大范围以找到区间上界 l_max（指数扩张）
            delta_min = self.delta(i, i - d, i_b)
            l_max = ti.u32(2)
            while self.delta(i, i + ti.i32(l_max) * d, i_b) > delta_min:
                l_max *= 2

            # 二分查找精确的范围长度 l
            l = ti.u32(0)
            t = l_max // 2
            while t > 0:
                if self.delta(i, i + ti.i32(l + t) * d, i_b) > delta_min:
                    l += t
                t //= 2

            # 区间端点 j 与当前节点的 LCP
            j = i + ti.i32(l) * d
            delta_node = self.delta(i, j, i_b)

            # 在 [i, j] 内再次二分找到分裂点 gamma（左/右子树分割）
            s = ti.u32(0)
            t = (l + 1) // 2
            while t > 0:
                if self.delta(i, i + ti.i32(s + t) * d, i_b) > delta_node:
                    s += t
                t = ti.select(t > 1, (t + 1) // 2, 0)

            gamma = i + ti.i32(s) * d + ti.min(d, 0)
            left = ti.select(ti.min(i, j) == gamma, gamma + self.n_aabbs - 1, gamma)
            right = ti.select(ti.max(i, j) == gamma + 1, gamma + self.n_aabbs, gamma + 1)
            self.nodes[i_b, i].left = ti.i32(left)
            self.nodes[i_b, i].right = ti.i32(right)
            self.nodes[i_b, ti.i32(left)].parent = i
            self.nodes[i_b, ti.i32(right)].parent = i

    @ti.func
    def delta(self, i: ti.i32, j: ti.i32, i_b: ti.i32):
        """
        计算 morton_codes[i] 与 morton_codes[j] 的最长公共前缀长度（LCP，0..64）。
        - morton_codes 存储为 vec2(u32)，依次比较高 32bit、低 32bit 直到发现首个不等 bit
        - 若 j 越界，返回 -1 表示无效
        """
        result = -1
        if j >= 0 and j < self.n_aabbs:
            result = 64
            for i_bit in range(2):
                x = self.morton_codes[i_b, i][i_bit] ^ self.morton_codes[i_b, j][i_bit]
                for b in range(32):
                    if x & (ti.u32(1) << (31 - b)):
                        result = b + 32 * i_bit
                        break
                if result != 64:
                    break
        return result

    def compute_bounds(self):
        """
        自底向上装配每个 BVH 节点的 AABB：
        - 先初始化叶子节点的 bound 并激活其父节点
        - 之后一层层推进：当父节点的两个孩子都已具备 bound 时，合并得到父 bound，再激活更上一层
        """
        self._kernel_compute_bounds_init()
        is_done = False
        while not is_done:
            is_done = self._kernel_compute_bounds_one_layer()

    @ti.kernel
    def _kernel_compute_bounds_init(self):
        # 复位层推进标志
        self.internal_node_active.fill(False)
        self.internal_node_ready.fill(False)

        # 用叶子（原 AABB）初始化底层 bound，并激活其父节点
        for i_b, i in ti.ndrange(self.n_batches, self.n_aabbs):
            idx = ti.i32(self.morton_codes[i_b, i][1])
            self.nodes[i_b, i + self.n_aabbs - 1].bound.min = self.aabbs[i_b, idx].min
            self.nodes[i_b, i + self.n_aabbs - 1].bound.max = self.aabbs[i_b, idx].max
            parent_idx = self.nodes[i_b, i + self.n_aabbs - 1].parent
            if parent_idx != -1:
                self.internal_node_active[i_b, parent_idx] = True

    @ti.kernel
    def _kernel_compute_bounds_one_layer(self) -> ti.i32:
        # 对已激活的内部节点，合并左右孩子的 bound，标记其父为 ready
        for i_b, i in ti.ndrange(self.n_batches, self.n_aabbs - 1):
            if self.internal_node_active[i_b, i]:
                left_bound = self.nodes[i_b, self.nodes[i_b, i].left].bound
                right_bound = self.nodes[i_b, self.nodes[i_b, i].right].bound
                self.nodes[i_b, i].bound.min = ti.min(left_bound.min, right_bound.min)
                self.nodes[i_b, i].bound.max = ti.max(left_bound.max, right_bound.max)
                parent_idx = self.nodes[i_b, i].parent
                if parent_idx != -1:
                    self.internal_node_ready[i_b, parent_idx] = True
                self.internal_node_active[i_b, i] = False

        # 将 ready 的节点激活到下一轮，若仍有激活则返回 False 继续循环
        is_done = True
        for i_b, i in ti.ndrange(self.n_batches, self.n_aabbs - 1):
            if self.internal_node_ready[i_b, i]:
                self.internal_node_active[i_b, i] = True
                is_done = False
        self.internal_node_ready.fill(False)

        return is_done

    @ti.func
    def query(self, aabbs: ti.template()):
        """
        遍历查询：给定一组查询 AABB，返回与 BVH 命中的 (batch, aabb_id, query_id)。

        流程（每个 batch、每个查询 AABB）：
        - 使用手动栈进行 DFS 遍历（max_stack_depth）
        - 内节点：若相交则将左右子压栈
        - 叶节点：若不过滤（filter 返回 False），写入查询结果
        - 若结果超过 max_query_results，置 overflow 标志（仍继续统计但不写）
        """
        self.query_result_count[None] = 0
        overflow = False

        n_querys = aabbs.shape[1]
        for i_b, i_q in ti.ndrange(self.n_batches, n_querys):
            query_stack = ti.Vector.zero(ti.i32, 64)  # 栈容量与 max_stack_depth 一致
            stack_depth = 1

            while stack_depth > 0:
                stack_depth -= 1
                node_idx = query_stack[stack_depth]
                node = self.nodes[i_b, node_idx]
                # 相交测试
                if aabbs[i_b, i_q].intersects(node.bound):
                    if node.left == -1 and node.right == -1:
                        # 叶子节点：由节点索引反查原 AABB 索引
                        i_a = ti.i32(self.morton_codes[i_b, node_idx - (self.n_aabbs - 1)][1])
                        # 过滤（用于剔除自碰撞/同组等）
                        if self.filter(i_a, i_q):
                            continue
                        idx = ti.atomic_add(self.query_result_count[None], 1)
                        if idx < self.max_query_results:
                            self.query_result[idx] = gs.ti_ivec3(i_b, i_a, i_q)
                        else:
                            overflow = True
                    else:
                        # 内节点：将子节点入栈（右后左先，或相反，均可）
                        if node.right != -1:
                            query_stack[stack_depth] = node.right
                            stack_depth += 1
                        if node.left != -1:
                            query_stack[stack_depth] = node.left
                            stack_depth += 1

        return overflow


@ti.data_oriented
class FEMSurfaceTetLBVH(LBVH):
    """
    针对 FEM 表面四面体的 LBVH（带过滤规则）。
    过滤逻辑：
    - 剔除与查询四面体共享任一顶点的候选四面体（避免自碰撞）
    """

    def __init__(self, fem_solver, aabb: AABB, max_n_query_result_per_aabb: int = 8, n_radix_sort_groups: int = 256):
        super().__init__(aabb, max_n_query_result_per_aabb, n_radix_sort_groups)
        self.fem_solver = fem_solver

    @ti.func
    def filter(self, i_a, i_q):
        """
        过滤条件：
        - 若 i_a >= i_q，或两者四面体共享任一顶点，则返回 True（表示“过滤掉”）
        """
        result = i_a >= i_q
        i_av = self.fem_solver.elements_i[self.fem_solver.surface_elements[i_a]].el2v
        i_qv = self.fem_solver.elements_i[self.fem_solver.surface_elements[i_q]].el2v
        for i, j in ti.static(ti.ndrange(4, 4)):
            if i_av[i] == i_qv[j]:
                result = True
        return result


@ti.data_oriented
class RigidTetLBVH(LBVH):
    """
    针对刚体四面体的 LBVH（带过滤规则）。
    过滤逻辑：
    - 使用 rigid 求解器的碰撞对有效性表，剔除同 link 或不允许的碰撞对。
    """

    def __init__(self, coupler, aabb: AABB, max_n_query_result_per_aabb: int = 8, n_radix_sort_groups: int = 256):
        super().__init__(aabb, max_n_query_result_per_aabb, n_radix_sort_groups)
        self.coupler = coupler
        self.rigid_solver = coupler.rigid_solver

    @ti.func
    def filter(self, i_a, i_q):
        """
        过滤条件：
        - 查 rigid_solver.collider._collider_info.collision_pair_validity，若为 False 则过滤
        """
        i_ag = self.coupler.rigid_volume_elems_geom_idx[i_a]
        i_qg = self.coupler.rigid_volume_elems_geom_idx[i_q]
        return not self.rigid_solver.collider._collider_info.collision_pair_validity[i_ag, i_qg]