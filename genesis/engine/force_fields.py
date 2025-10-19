import numpy as np
import gstaichi as ti

import genesis as gs
from genesis.repr_base import RBC


@ti.data_oriented
class ForceField(RBC):
    """
    力场（准确来说是“加速度场”）基类：不应直接实例化使用。
    本类为所有具体力场提供统一接口与启停控制。

    说明
    ----
    名称为 `ForceField`，但在实现上它表示“加速度场”（acceleration field），
    因为力本身没有空间密度的概念，仿真中更直接的是对粒子/点施加加速度。
    """

    def __init__(self):
        # 全局激活标志（Taichi 标量布尔字段），用于在 Kernel 中快速判定是否生效
        self._active = ti.field(gs.ti_bool, shape=())
        self._active[None] = False

    def activate(self):
        """
        激活力场（使其在 Kernel 中开始生效）。
        """
        self._active[None] = True

    def deactivate(self):
        """
        关闭力场（使其在 Kernel 中不生效）。
        """
        self._active[None] = False

    @ti.func
    def get_acc(self, pos, vel, t, i):
        # 获取该力场对给定位置/速度的加速度贡献；若未激活则返回零向量
        acc = ti.Vector.zero(gs.ti_float, 3)
        if self._active[None]:
            acc = self._get_acc(pos, vel, t, i)  # 由子类实现的核心加速度计算
        return acc

    @property
    def active(self):
        """
        当前力场是否处于激活状态。
        """
        return self._active[None]


class Constant(ForceField):
    """
    常量力场：在整个空间内施加固定方向、固定大小的加速度。

    参数
    -----------
    direction: array_like, shape=(3,)
        加速度方向，将自动归一化。
    strength: float
        加速度强度（标量）。
    """

    def __init__(self, direction=(1, 0, 0), strength=1.0):
        super().__init__()

        direction = np.array(direction)
        if direction.shape != (3,):
            raise ValueError("direction must have shape (3,)")

        self._direction = direction / np.linalg.norm(direction)
        self._strength = strength
        # 预先构造 Taichi 侧的常量加速度向量，避免重复创建
        self._acc_ti = ti.Vector(self._direction * self._strength, dt=gs.ti_float)

    @ti.func
    def _get_acc(self, pos, vel, t, i):
        # 直接返回常量加速度
        return self._acc_ti

    @property
    def direction(self):
        return self._direction

    @property
    def strength(self):
        return self._strength


class Wind(ForceField):
    """
    风场：在一个沿着给定方向的“柱体区域”内施加恒定加速度，区域外为 0。

    参数
    -----------
    direction: array_like, shape=(3,)
        风的方向，将自动归一化。
    strength: float
        风力强度（加速度大小）。
    radius: float
        柱体半径。
    center: array_like, shape=(3,)
        柱体轴线穿过的中心点（以 direction 为轴）。
    """

    def __init__(self, direction=(1, 0, 0), strength=1.0, radius=1, center=(0, 0, 0)):
        super().__init__()

        direction = np.array(direction)
        if direction.shape != (3,):
            raise ValueError("direction must have shape (3,)")

        center = np.array(center)
        if center.shape != (3,):
            raise ValueError("center must have shape (3,)")

        self._center = center
        self._direction = direction / np.linalg.norm(direction)
        self._strength = strength
        self._radius = radius

        self._direction_ti = ti.Vector(self._direction, dt=gs.ti_float)
        self._center_ti = ti.Vector(self._center, dt=gs.ti_float)
        self._acc_ti = ti.Vector(self._direction * self._strength, dt=gs.ti_float)

    @ti.func
    def _get_acc(self, pos, vel, t, i):
        # 计算点到“轴线”的距离：轴线通过 center，方向为 direction
        # 利用 |(p-c) x dir| 即为点到轴线的最短距离
        dist = (pos - self._center_ti).cross(self._direction_ti).norm()
        acc = self._acc_ti
        if dist > self._radius:
            # 超出柱体半径范围，不施加风力
            acc = ti.Vector.zero(gs.ti_float, 3)
        return acc

    @property
    def direction(self):
        return self._direction

    @property
    def strength(self):
        return self._strength

    @property
    def radius(self):
        return self._radius

    @property
    def center(self):
        return self._center


class Point(ForceField):
    """
    点力场：朝向目标点（正强度）或背离目标点（负强度）施加力，支持距离衰减与“流动”项。

    参数
    -----------
    strength: float
        强度（正：吸引；负：排斥）。
    position: array_like, shape=(3,)
        目标点位置。
    flow: float
        “流动”系数（对当前速度进行一定比例的对齐/调整）。
    falloff_pow: float
        距离衰减的幂指数，越大则衰减越快。
    """

    def __init__(self, strength=1.0, position=(0, 0, 0), falloff_pow=0.0, flow=1.0):
        super().__init__()

        position = np.array(position)
        if position.shape != (3,):
            raise ValueError("position must have shape (3,)")

        self._strength = strength
        self._position = position
        self._falloff_pow = falloff_pow
        self._flow = flow

        self._position_ti = ti.Vector(self._position, dt=gs.ti_float)

    @ti.func
    def _get_acc(self, pos, vel, t, i):
        # 管线：
        # 1) 计算相对位置与半径
        relative_pos = pos - self._position_ti
        radius = relative_pos.norm(gs.EPS)  # 加上 EPS 避免除零
        # 2) 指向（或背离）方向
        direction = relative_pos / radius
        # 3) 距离衰减（1/(r+1)^pow）
        falloff = 1 / (radius + 1.0) ** self._falloff_pow
        # 4) 基本加速度
        acc = self._strength * direction
        # 5) 流动项：将加速度与速度做一定比例的“趋近”，类似阻尼/对齐
        acc += (acc - vel) * self._flow
        # 6) 应用距离衰减
        acc *= falloff

        return acc

    @property
    def strength(self):
        return self._strength

    @property
    def position(self):
        return self._position


class Drag(ForceField):
    """
    阻力场：提供与速度相反方向的阻尼力，可选线性/二次（与速度模）两部分。

    参数
    -----------
    linear: float
        线性阻力系数（~ v）。
    quadratic: float
        二次阻力系数（~ |v| v）。
    """

    def __init__(self, linear=0.0, quadratic=0.0):
        super().__init__()

        self._linear = linear
        self._quadratic = quadratic

    @ti.func
    def _get_acc(self, pos, vel, t, i):
        # a = -k1 * v - k2 * |v| * v
        return -self._linear * vel - self._quadratic * vel.norm() * vel

    @property
    def linear(self):
        return self._linear

    @property
    def quadratic(self):
        return self._quadratic


class Noise(ForceField):
    """
    噪声力场：在每个点采样独立的随机向量，范围约为 [-1, 1]，再乘以强度。
    """

    def __init__(self, strength=1.0):
        super().__init__()

        self._strength = strength

    @ti.func
    def _get_acc(self, pos, vel, t, i):
        # 逐分量均匀采样 [0,1) -> 线性变换到 [-1,1]
        noise = (
            ti.Vector(
                [
                    ti.random(gs.ti_float),
                    ti.random(gs.ti_float),
                    ti.random(gs.ti_float),
                ],
                dt=gs.ti_float,
            )
            * 2
            - 1
        )
        return noise * self._strength

    @property
    def strength(self):
        return self._strength


class Vortex(ForceField):
    """
    涡旋力场：围绕 z 轴（x-y 平面内）产生旋转与径向分量，并带有距离衰减与速度阻尼。

    参数
    -----------
    strength_perpendicular: float
        垂直方向（切向）的流强度。正为逆时针，负为顺时针。
    strength_radial: float
        径向方向的流强度。正向内（吸入），负向外（喷出）。
    center: array_like, shape=(3,)
        涡旋中心。
    falloff_pow: float
        距离衰减幂指数。
    falloff_min: float
        衰减下限距离：小于该距离时不衰减（=1）。
    falloff_max: float
        衰减上限距离：超过该距离时力为 0。
    damping: float
        速度阻尼系数（与速度成正比的减速项）。
    """

    def __init__(
        self,
        direction=(0.0, 0.0, 1.0),
        center=(0.0, 0.0, 0.0),
        strength_perpendicular=20.0,
        strength_radial=0.0,
        falloff_pow=2.0,
        falloff_min=0.01,
        falloff_max=np.inf,
        damping=0.0,
    ):
        super().__init__()

        direction = np.array(direction)
        if direction.shape != (3,):
            raise ValueError("direction must have shape (3,)")

        center = np.array(center)
        if center.shape != (3,):
            raise ValueError("center must have shape (3,)")

        self._center = center
        self._direction = direction / np.linalg.norm(direction)
        self._damping = damping

        self._strength_perpendicular = strength_perpendicular
        self._strength_radial = strength_radial

        self._falloff_pow = falloff_pow
        self._falloff_min = falloff_min
        self._falloff_max = falloff_max

        self._direction_ti = ti.Vector(self._direction, dt=gs.ti_float)
        self._center_ti = ti.Vector(self._center, dt=gs.ti_float)

    @ti.func
    def _get_acc(self, pos, vel, t, i):
        # 仅在 x-y 平面内计算相对位置与半径，等价于绕 z 轴旋转
        relative_pos = ti.Vector([pos[0] - self._center_ti[0], pos[1] - self._center_ti[1]])
        radius = relative_pos.norm()
        # 垂直（切向）方向：(-y, x)，径向方向：(-x, -y)，z 分量为 0
        perpendicular = ti.Vector([-relative_pos[1], relative_pos[0], 0.0], dt=gs.ti_float)
        radial = -ti.Vector([relative_pos[0], relative_pos[1], 0.0], dt=gs.ti_float)

        # 距离衰减：
        # - r < min: 1
        # - min <= r < max: 1 / (r - min + 1)^pow
        # - r >= max: 0
        falloff = gs.ti_float(0.0)
        if radius < self._falloff_min:
            falloff = 1.0
        elif radius < self._falloff_max:
            falloff = 1 / (radius - self._falloff_min + 1.0) ** self._falloff_pow
        else:
            falloff = 0.0

        # 组合切向与径向分量
        acceleration = falloff * (self._strength_perpendicular * perpendicular + self._strength_radial * radial)

        # 速度阻尼
        acceleration -= self._damping * vel

        return acceleration

    @property
    def direction(self):
        return self._direction

    @property
    def radius(self):
        # 注意：此属性在当前实现中未定义对应的 _radius 字段，保留原有接口不做修改。
        return self._radius

    @property
    def center(self):
        return self._center

    @property
    def strength_perpendicular(self):
        return self._strength_perpendicular

    @property
    def strength_radial(self):
        return self._strength_radial

    @property
    def falloff_pow(self):
        return self._falloff_pow

    @property
    def falloff_min(self):
        return self._falloff_min

    @property
    def falloff_max(self):
        return self._falloff_max


class Turbulence(ForceField):
    """
    湍流力场：使用 3D Perlin 噪声生成空间相关的“噪声加速度”。
    通过三组噪声场分别提供 x/y/z 三个分量，并支持“流动”项调制。

    参数
    -----------
    strength: float
        湍流强度（整体缩放）。
    frequency: float
        空间频率（控制重复模式的密度）。
    flow: float
        “流动”系数（将噪声加速度与当前速度进行一定比例的调整）。
    seed: int | None
        随机种子（若为 None 则非确定）。
    """

    def __init__(self, strength=1.0, frequency=3, flow=0.0, seed=None):
        super().__init__()

        self._strength = strength
        self._frequency = frequency
        self._flow = flow

        # 三个正交噪声场，对应加速度的三个分量
        self._perlin_x = PerlinNoiseField(frequency=self._frequency, seed=seed, seed_offset=0)
        self._perlin_y = PerlinNoiseField(frequency=self._frequency, seed=seed, seed_offset=1)
        self._perlin_z = PerlinNoiseField(frequency=self._frequency, seed=seed, seed_offset=2)

    @ti.func
    def _get_acc(self, pos, vel, t, i):
        # 1) 分别在三个噪声场采样
        acc = ti.Vector(
            [
                self._perlin_x._noise(pos[0], pos[1], pos[2]),
                self._perlin_y._noise(pos[0], pos[1], pos[2]),
                self._perlin_z._noise(pos[0], pos[1], pos[2]),
            ],
            dt=gs.ti_float,
        )
        # 2) 强度缩放
        acc *= self._strength
        # 3) 流动项（向加速度方向对齐或偏移当前速度）
        acc += (acc - vel) * self._flow
        return acc

    @property
    def strength(self):
        return self._strength

    @property
    def frequency(self):
        return self._frequency


class Custom(ForceField):
    """
    自定义力场：使用用户提供的 Taichi 函数 `f(pos, vel, t, i)` 作为加速度计算。

    参数
    -----------
    func: 可调用的 taichi 函数（用 `@ti.func` 装饰的 python 函数）
        函数签名必须为：
        `f(pos: ti.types.vector(3), vel: ti.types.vector(3), t: ti.f32, i: ti.i32) -> ti.types.vector(3)`。
    """

    def __init__(self, func):
        super().__init__()
        # 直接覆盖基类的 get_acc（注意需要满足 taichi 内联函数要求）
        self.get_acc = func


@ti.data_oriented
class PerlinNoiseField:
    """
    3D Perlin 噪声场。
    每个对象通过置乱表（permutation）生成独立的噪声分布，可设置 wrap 与频率。

    参数
    -----------
    wrap_size: int
        置乱表周期（用于坐标 wrap），一般为 256。
    frequency: float
        空间频率（对输入坐标的缩放倍数）。
    seed: int | None
        随机种子（可复现）。
    seed_offset: int
        当创建多通道噪声场时使用不同 offset 以避免通道相关性。
    """

    def __init__(self, wrap_size=256, frequency=10, seed=None, seed_offset=0):
        self._wrap_size = wrap_size
        self._permutation = np.arange(self._wrap_size, dtype=np.int32)
        self._frequency = frequency
        if seed is not None:
            # 设定随机种子并生成置乱表（双倍拼接便于索引 wrap）
            state = np.random.get_state()
            np.random.seed(seed + seed_offset)
            np.random.shuffle(self._permutation)
            np.random.set_state(state)

        self._permutation_ti = ti.field(ti.i32, shape=(self._wrap_size * 2,))
        self._permutation_ti.from_numpy(np.concatenate([self._permutation, self._permutation]))

    @ti.func
    def _fade(self, t):
        """平滑插值的缓动函数（Perlin 标准 6t^5 - 15t^4 + 10t^3 形式）。"""
        return t * t * t * (t * (t * 6 - 15) + 10)

    @ti.func
    def _lerp(self, t, a, b):
        """线性插值：a + t (b - a)。"""
        return a + t * (b - a)

    @ti.func
    def _grad(self, hash, x, y, z):
        """
        根据哈希选择局部梯度方向，与距离向量点乘得到角度相关的权重。
        这是经典 Perlin 噪声的梯度表近似实现。
        """
        h = hash & 15  # 取低 4 位
        u = x
        if h >= 8:
            u = y

        v = y
        if h >= 4:
            v = z
            if h == 12 or h == 14:
                v = x

        return (u if (h & 1) == 0 else -u) + (v if (h & 2) == 0 else -v)

    @ti.func
    def _noise(self, x, y, z):
        # 管线：
        # 1) 对输入坐标进行频率缩放
        x *= self._frequency
        y *= self._frequency
        z *= self._frequency

        # 2) 找到包含该点的单位立方体整数索引（并做 wrap）
        X = gs.ti_int(ti.floor(x)) & (self._wrap_size - 1)
        Y = gs.ti_int(ti.floor(y)) & (self._wrap_size - 1)
        Z = gs.ti_int(ti.floor(z)) & (self._wrap_size - 1)

        # 3) 计算点在该立方体内的相对坐标（[0,1)）
        x -= ti.floor(x)
        y -= ti.floor(y)
        z -= ti.floor(z)

        # 4) 对每个坐标计算平滑权重（fade）
        u = self._fade(x)
        v = self._fade(y)
        w = self._fade(z)

        # 5) 计算 8 个顶点的哈希索引（通过置乱表）
        A = self._permutation_ti[X] + Y
        AA = self._permutation_ti[A] + Z
        AB = self._permutation_ti[A + 1] + Z
        B = self._permutation_ti[X + 1] + Y
        BA = self._permutation_ti[B] + Z
        BB = self._permutation_ti[B + 1] + Z

        # 6) 对 8 个角的梯度点积进行三线性插值，得到平滑噪声值
        return self._lerp(
            w,
            self._lerp(
                v,
                self._lerp(
                    u,
                    self._grad(self._permutation_ti[AA], x, y, z),
                    self._grad(self._permutation_ti[BA], x - 1, y, z),
                ),
                self._lerp(
                    u,
                    self._grad(self._permutation_ti[AB], x, y - 1, z),
                    self._grad(self._permutation_ti[BB], x - 1, y - 1, z),
                ),
            ),
            self._lerp(
                v,
                self._lerp(
                    u,
                    self._grad(self._permutation_ti[AA + 1], x, y, z - 1),
                    self._grad(self._permutation_ti[BA + 1], x - 1, y, z - 1),
                ),
                self._lerp(
                    u,
                    self._grad(self._permutation_ti[AB + 1], x, y - 1, z - 1),
                    self._grad(self._permutation_ti[BB + 1], x - 1, y - 1, z - 1),
                ),
            ),
        )