"""路径辅助模块 - 统一管理项目路径

这个模块的作用:
1. 提供可靠的项目根目录定位功能
2. 提供assets资源目录的访问路径
3. 避免因启动目录不同导致的路径错误

为什么需要这个模块?
- 机器人在不同环境(本地/Docker/服务器)运行时,启动目录可能不同
- 各个模块需要统一的方式来访问配置文件、资源文件、数据库文件
- 使用相对路径容易出错,这个模块提供绝对路径的可靠定位

使用场景:
- 启动初始化阶段:定位配置文件位置
- 资源加载:读取assets目录下的图片、模板等文件
- 数据存储:确定数据库文件、缓存文件的存放位置

关键概念(新手必读):
- Path对象: Python标准库pathlib.Path,比字符串更安全的路径表示方式
- 项目根目录: 包含pyproject.toml的那一层目录,是项目的顶层目录
- @lru_cache: 缓存装饰器,避免重复计算,提高性能
"""

from __future__ import annotations

# functools.lru_cache - 函数结果缓存装饰器
# 作用:把函数的返回值缓存起来,相同参数再次调用时直接返回缓存结果
# 好处:避免重复的文件系统访问,提升性能
from functools import lru_cache

# pathlib.Path - 面向对象的路径处理类
# 优点:跨平台兼容,支持/运算符拼接路径,API更清晰
from pathlib import Path


@lru_cache(maxsize=1)  # 缓存最多1个结果(项目根目录只有一个,缓存1个足够)
def project_root() -> Path:
    """返回项目根目录路径(以pyproject.toml文件为定位锚点)

    查找策略:
    1. 从当前文件(paths.py)的位置开始
    2. 逐级向上遍历父目录
    3. 找到包含pyproject.toml文件的目录就返回
    4. 如果找不到,按默认目录结构回退

    Returns:
        Path: 项目根目录的绝对路径对象

    Raises:
        IndexError: 当目录层级不足时,兜底逻辑可能抛出此异常(正常情况不会发生)

    Side Effects:
        - 访问文件系统,检查pyproject.toml是否存在
        - 第一次调用后结果会被缓存,后续调用直接返回缓存值

    Examples:
        >>> root = project_root()
        >>> print(root / "pyproject.toml")  # 项目配置文件路径
    """

    # __file__: 当前Python文件(paths.py)的路径
    # Path(__file__): 将字符串路径转为Path对象
    # .resolve(): 转为绝对路径并解析符号链接,确保路径稳定可靠
    here = Path(__file__).resolve()

    # 向上查找包含pyproject.toml的目录
    # here.parents: 返回所有父目录的序列,parents[0]是直接父目录,parents[1]是上一层,以此类推
    # (here, *here.parents): 将当前路径和所有父目录组成一个元组进行遍历
    for parent in (here, *here.parents):
        # parent / "pyproject.toml": 使用/运算符拼接路径(等价于os.path.join)
        # .exists(): 检查该路径是否真实存在于文件系统中
        if (parent / "pyproject.toml").exists():
            return parent  # 找到就立即返回,不再继续向上查找

    # 兜底逻辑(fallback):
    # 当找不到pyproject.toml时,按照默认目录结构回退
    # here.parents[4]: 向上回退4层目录
    # 假设目录结构: project_root/src/plugins/yuying_chameleon/paths.py
    # 则 parents[0]=yuying_chameleon, parents[1]=plugins, parents[2]=src, parents[3]=project_root
    # 注意: 这个方法依赖固定的目录结构,如果结构改变可能导致错误
    return here.parents[4]


def assets_dir() -> Path:
    """返回assets资源目录路径

    assets目录通常用于存放:
    - 静态资源文件(图片、音频、视频等)
    - 模板文件(prompt模板、配置模板等)
    - 内置数据(默认表情包、预设数据等)

    Returns:
        Path: assets目录的绝对路径对象

    Side Effects:
        - 间接调用project_root(),会访问文件系统(但有缓存)
        - 不会检查assets目录是否真实存在

    Examples:
        >>> assets = assets_dir()
        >>> stickers_path = assets / "stickers"  # 表情包目录
        >>> templates_path = assets / "templates"  # 模板目录
    """

    # project_root()已经被@lru_cache缓存,所以这个调用非常快,不会重复访问文件系统
    # / "assets": 在项目根目录下拼接assets子目录
    # 注意: 这里只是返回路径对象,不会检查目录是否存在,也不会自动创建
    return project_root() / "assets"
