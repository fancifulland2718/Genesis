import inspect

import genesis as gs
import genesis.utils.repr as ru
from genesis.styles import colors, formats, styless


class RBC:
    """
    REPR Base Class.
    All class that inherits this class will have an artist-level __repr__ method that prints out all the properties (decorated by @property) of the class, ordered by length of the property name.
    REPR 基类：
    任何继承该类的类都会自动获得一个“艺术化/彩色增强”的 __repr__，
    会列出所有使用 @property 装饰的属性，并按属性名长度（以及一定次级规则）排序输出。
    提供一个统一、可读、结构化、带语义增强（彩色/对齐）的对象表示层，以便在交互、调试、日志与文档探索中快速理解对象状态，而无需为每个子类重复写REPR代码。
    """

    @classmethod
    def _repr_type(cls):
        # 返回类型前缀字符串，用于头部显示，如 <gs.Simulator>
        return f"<gs.{cls.__name__}>"

    def _repr_briefer(self):
        # 更精简的概述形式（若有 id 则附加）
        repr_str = self._repr_type()
        if hasattr(self, "id"):
            repr_str += f"(id={self.id})"
        return repr_str

    def _repr_brief(self):
        # 略详细的概述：尝试输出 id / idx / morph / material 等常见属性
        repr_str = self._repr_type()
        if hasattr(self, "id"):
            repr_str += f": {self.id}"
        if hasattr(self, "idx"):
            repr_str += f", idx: {self.idx}"
        if hasattr(self, "morph"):
            repr_str += f", morph: {self.morph}"
        if hasattr(self, "material"):
            repr_str += f", material: {self.material}"
        return repr_str

    def _is_debugger(self) -> bool:
        """Detect if running under a debugger (VSCode or PyCharm).
        检测当前是否处于调试器环境（VSCode / PyCharm 等），用于在调试时避免彩色多行输出干扰。
        """
        for frame in inspect.stack():
            # 通过调用栈文件名是否包含调试模块特征来判断
            if any(module in frame.filename for module in ("debugpy", "ptvsd", "pydevd")):
                return True
        return False

    def __repr__(self):
        # 在非调试环境下返回彩色格式；调试环境下让 Python 默认机制回退（避免花哨格式影响调试 UI）
        if not self._is_debugger():
            return self.__colorized__repr__()
        # 若在调试器中，返回 None，Python 会 fallback 到 object 默认 repr（或其他机制）
        # 这里不显式 return super().__repr__() 是为了保持最小侵入

    def __colorized__repr__(self) -> str:
        # 动态收集所有属性名（dir），筛选出被 @property 修饰的属性
        all_attrs = self.__dir__()
        property_attrs = []

        for attr in all_attrs:
            # getattr(self.__class__, attr, None) 取类属性定义；判断是否为 property 对象
            if isinstance(getattr(self.__class__, attr, None), property):
                property_attrs.append(attr)

        # 计算属性名最大长度，用于后面对齐
        max_attr_len = max([len(attr) for attr in property_attrs]) if property_attrs else 0

        repr_str = ""
        # sort property attrs
        # 排序规则：
        # 1. 以第一个下划线前的段长度
        # 2. 然后按该段字典序
        # 3. 最后按整体长度
        property_attrs = sorted(property_attrs, key=lambda x: (len(x.split("_")[0]), x.split("_")[0], len(x)))

        for attr in property_attrs:
            # 左侧属性名着色（蓝色）
            formatted_str = f"{colors.BLUE}'{attr}'{formats.RESET}"

            # content example: <gs.List>(len=0, [])
            # 使用 ru.brief 对属性值做简述（避免长数据结构全量打印）
            try:
                content = ru.brief(getattr(self, attr))
            except:
                # 忽略取值或格式化异常的属性
                continue

            # 找到类型描述的 '>' 位置，用于分段着色（类型部分与内容部分区分）
            idx = content.find(">")
            # format with italic and color
            # 前半段（类型）斜体 + 颜色，后半段保持同色
            formatted_content = (
                f"{colors.MINT}{formats.ITALIC}{content[:idx + 1]}{formats.RESET}"
                f"{colors.MINT}{content[idx + 1:]}{formats.RESET}"
            )

            # 如果是多行内容，需要进行缩进对齐补偿
            if isinstance(getattr(self, attr), gs.List):
                # 针对 gs.List 给予较固定的缩进（4 = 两个引号 + 冒号 + 空格）
                offset = max_attr_len + 4
            else:
                # 其余类型根据类型头长度 + 常量补偿
                offset = max_attr_len + idx + 7

            # 为 multi-line 的后续行添加空格缩进
            formatted_content = formatted_content.replace("\n", "\n" + " " * offset)

            # 拼装最终一行：属性名右对齐 + 灰色冒号 + 内容
            repr_str += f"{formatted_str:>{max_attr_len + 17}}{colors.GRAY}:{formats.RESET} {formatted_content}\n"

        # length of the first line
        # 第一行用于构造头部框线和标题
        first_line = styless(repr_str.split("\n")[0])  # styless 去除控制符后计算真实长度
        header_len = len(first_line)
        line_len = header_len - len(self._repr_type()) - 2  # 预留左右分割线长度
        left_line_len = line_len // 2
        right_line_len = line_len - left_line_len

        # minimum length need to match the first colon
        # 确保分割线长度不短于第一行冒号位置（视觉对齐）
        min_line_len = len(first_line.split(":")[0])
        left_line_len = max(left_line_len, min_line_len)
        right_line_len = max(right_line_len, min_line_len)

        # 构建头部：左右使用 '─' 线条，中间加粗斜体类型标签
        repr_str = (
            f"{colors.CORN}{'─' * left_line_len} {formats.BOLD}{formats.ITALIC}{self._repr_type()}{formats.RESET} "
            f"{colors.CORN}{'─' * right_line_len}\n"
            + repr_str
        )

        return repr_str

    def __format__(self, format_spec):
        # 支持 format(x) 时返回简洁类型表示；忽略 format_spec（无格式分支）
        repr_str = self._repr_type()
        return repr_str