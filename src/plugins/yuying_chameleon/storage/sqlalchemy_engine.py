"""SQLAlchemy异步引擎与会话管理模块

这个模块的作用:
1. 创建和配置SQLAlchemy异步数据库引擎
2. 设置SQLite的性能优化参数(WAL模式、超时等)
3. 提供数据库会话(Session)的创建和管理
4. 确保数据库文件目录存在

为什么使用异步数据库?
- NoneBot是异步框架,所有操作都是异步的
- 异步数据库可以避免阻塞事件循环,提高并发性能
- SQLAlchemy的异步API与NoneBot完美配合

SQLite优化说明:
- WAL模式(Write-Ahead Logging): 允许读写并发,提升性能
- 同步模式NORMAL: 平衡数据安全和性能
- busy_timeout: 当数据库被锁定时等待的时间
- 外键约束: 保证数据完整性

关键概念(新手必读):
- Engine(引擎): 数据库连接池的管理者,整个应用只需一个
- Session(会话): 单次数据库操作的上下文,用完就关闭
- 异步上下文管理器: 使用async with语法,自动处理资源的获取和释放
- PRAGMA: SQLite的配置命令,用于设置数据库行为

使用方式:
```python
from .sqlalchemy_engine import get_session

async def some_function():
    # 创建一个数据库会话
    async with get_session() as session:
        # 在这里执行数据库操作
        result = await session.execute(select(SomeModel))
        # 会话会自动提交或回滚
```
"""

from __future__ import annotations

# contextlib.asynccontextmanager - 异步上下文管理器装饰器
# 作用: 将async生成器函数转换为可以用async with的对象
from contextlib import asynccontextmanager

# pathlib.Path - 文件路径处理
from pathlib import Path

# typing.AsyncIterator - 异步迭代器类型提示
from typing import AsyncIterator

from nonebot import logger  # NoneBot日志记录器

# SQLAlchemy相关导入
from sqlalchemy import event  # 事件监听,用于在特定时机执行代码
from sqlalchemy.engine.url import make_url  # 解析数据库URL
from sqlalchemy.ext.asyncio import (  # SQLAlchemy异步扩展
    AsyncSession,  # 异步会话类
    async_sessionmaker,  # 异步会话工厂
    create_async_engine,  # 创建异步引擎的函数
)

from ..config import plugin_config  # 导入插件配置


def _ensure_sqlite_parent_dir(database_url: str) -> None:
    """确保SQLite数据库文件所在的目录存在

    这个函数的作用:
    - 解析数据库URL,判断是否是SQLite
    - 如果是SQLite文件数据库,确保父目录存在
    - 如果是内存数据库(:memory:),则跳过
    - 即使失败也不中断启动,只记录警告

    为什么需要这个函数?
    - SQLite是文件数据库,如果目录不存在会报错
    - 用户可能配置了不存在的目录路径
    - 自动创建目录提升用户体验

    Args:
        database_url: 数据库连接URL字符串
            格式: "sqlite+aiosqlite:///path/to/database.db"

    Side Effects:
        - 可能创建目录(如果不存在)
        - 失败时输出警告日志

    Examples:
        >>> _ensure_sqlite_parent_dir("sqlite+aiosqlite:///data/yuying.db")
        # 会创建data目录(如果不存在)
    """

    try:
        # make_url(): 将URL字符串解析为URL对象
        # URL对象包含: drivername(驱动名), database(数据库路径)等属性
        url = make_url(database_url)
    except Exception:
        # 如果URL格式错误,直接返回,不影响启动
        return

    # 检查是否是SQLite数据库
    # drivername可能是: "sqlite", "sqlite+aiosqlite", "sqlite+pysqlite"等
    # .startswith("sqlite"): 判断是否以"sqlite"开头
    if not url.drivername.startswith("sqlite"):
        return  # 不是SQLite,无需处理目录

    # 获取数据库文件路径
    # url.database: 数据库路径部分,如 "/data/yuying.db"
    db_path = url.database
    if not db_path or db_path == ":memory:":
        # 如果是内存数据库或路径为空,无需创建目录
        return

    try:
        # Path(db_path): 将字符串路径转为Path对象
        # .expanduser(): 展开~符号为用户目录
        # .resolve(): 转为绝对路径
        # .parent: 获取父目录
        # .mkdir(): 创建目录
        #   parents=True: 递归创建多级目录
        #   exist_ok=True: 如果目录已存在不报错
        Path(db_path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        # 创建目录失败,记录警告但不中断启动
        # 可能的原因: 权限不足、磁盘满、路径非法等
        logger.warning(f"创建 SQLite 目录失败,将继续尝试启动:{exc}")


# 模块加载时立即执行: 确保数据库目录存在
# 这样在创建引擎之前就准备好了目录
_ensure_sqlite_parent_dir(plugin_config.yuying_database_url)

# ==================== 创建异步数据库引擎 ====================

# create_async_engine(): 创建SQLAlchemy异步引擎
# 引擎是整个应用的数据库连接池管理者
# 一个应用通常只需要一个引擎实例
engine = create_async_engine(
    plugin_config.yuying_database_url,  # 数据库连接URL
    echo=False,  # 是否打印SQL语句(False=不打印,避免日志刷屏)
    future=True,  # 使用SQLAlchemy 2.0的新API风格
)

# ==================== 配置SQLite性能参数 ====================

# @event.listens_for(): SQLAlchemy事件监听装饰器
# 监听引擎的"connect"事件 = 每次创建新的数据库连接时触发
# engine.sync_engine: 获取同步引擎对象(因为PRAGMA设置需要同步API)
@event.listens_for(engine.sync_engine, "connect")
def set_sqlite_pragma(dbapi_connection, _connection_record) -> None:
    """在每次数据库连接创建时设置SQLite的PRAGMA参数

    PRAGMA是SQLite的配置命令,用于优化性能和行为:

    1. journal_mode=WAL (Write-Ahead Logging)
       - 作用: 启用写前日志模式
       - 好处: 允许读操作和写操作同时进行,大幅提升并发性能
       - 原理: 写操作先记录到WAL文件,不阻塞读操作
       - 适用: 适合读多写少的场景(机器人就是这种场景)

    2. synchronous=NORMAL
       - 作用: 设置数据同步级别为NORMAL
       - 好处: 平衡性能和数据安全
       - FULL: 最安全但最慢(每次写入都等待磁盘完成)
       - NORMAL: 在关键时刻同步,多数情况下快速
       - OFF: 最快但可能丢数据(不推荐)

    3. busy_timeout=<毫秒>
       - 作用: 当数据库被锁定时,等待多久再放弃
       - 好处: 避免并发写入时立即失败,给其他操作完成的时间
       - 默认: 从配置文件读取(通常是3000毫秒=3秒)

    4. foreign_keys=ON
       - 作用: 启用外键约束检查
       - 好处: SQLite默认不检查外键,开启后保证数据完整性
       - 示例: 删除用户时,相关的消息记录也会被级联删除

    Args:
        dbapi_connection: 底层数据库连接对象(DBAPI层面)
        _connection_record: 连接记录对象(未使用,用_前缀表示)

    Side Effects:
        - 修改数据库连接的配置
        - 这些设置只对当前连接生效,每个连接都需要重新设置
    """

    # 获取数据库游标(cursor),用于执行SQL命令
    cursor = dbapi_connection.cursor()

    # 执行PRAGMA命令设置各项参数
    # cursor.execute(): 执行SQL语句
    cursor.execute("PRAGMA journal_mode=WAL")  # 启用WAL模式
    cursor.execute("PRAGMA synchronous=NORMAL")  # 设置同步级别
    cursor.execute(
        f"PRAGMA busy_timeout={int(plugin_config.yuying_sqlite_busy_timeout_ms)}"
    )  # 设置锁等待超时(从配置读取)
    cursor.execute("PRAGMA foreign_keys=ON")  # 启用外键约束

    # 关闭游标,释放资源
    cursor.close()


# ==================== 创建会话工厂 ====================

# async_sessionmaker(): 创建异步会话工厂
# 会话工厂是一个可调用对象,每次调用会返回一个新的Session实例
AsyncSessionLocal = async_sessionmaker(
    engine,  # 绑定到上面创建的引擎
    class_=AsyncSession,  # 使用AsyncSession类(异步会话)
    expire_on_commit=False,  # 提交后不自动过期对象(保持对象可用)
    autoflush=False,  # 不自动flush(手动控制何时flush)
)

# 关于expire_on_commit=False的说明:
# - 默认情况下,commit后所有对象都会过期(访问属性需要重新查询)
# - 设为False后,commit后对象仍然可用,不需要重新查询
# - 适合: 提交后还要继续使用对象的场景

# 关于autoflush=False的说明:
# - flush: 将内存中的修改写入数据库(但不提交事务)
# - 自动flush会在每次查询前执行,可能有性能开销
# - 手动控制flush可以批量操作,提高性能


@asynccontextmanager  # 异步上下文管理器装饰器
async def get_session() -> AsyncIterator[AsyncSession]:
    """获取一个异步数据库会话(使用async with语法)

    这个函数返回的是一个异步上下文管理器,用法:
    ```python
    async with get_session() as session:
        # 在这里执行数据库操作
        result = await session.execute(select(User))
        await session.commit()  # 手动提交
    # 离开with块时自动关闭session
    ```

    会话的生命周期:
    1. 进入async with时,创建新的Session实例
    2. 在with块内执行数据库操作
    3. 如果发生异常,自动回滚事务
    4. 离开with块时,自动关闭Session

    异常处理:
    - 如果with块内抛出任何异常
    - 会自动调用session.rollback()回滚所有未提交的修改
    - 然后重新抛出异常,让调用方处理

    Yields:
        AsyncSession: SQLAlchemy异步会话对象

    Side Effects:
        - 从连接池获取一个数据库连接
        - 离开上下文时归还连接到连接池
        - 异常时回滚事务

    Examples:
        >>> async with get_session() as session:
        ...     user = User(name="Alice")
        ...     session.add(user)
        ...     await session.commit()  # 提交修改

        >>> # 异常会自动回滚
        >>> async with get_session() as session:
        ...     user = User(name="Bob")
        ...     session.add(user)
        ...     raise ValueError("Something wrong")
        # 这里会自动回滚,Bob不会被保存
    """

    # AsyncSessionLocal(): 调用会话工厂,创建新的Session实例
    # async with AsyncSessionLocal() as session:
    #   - 进入with块时,Session被创建
    #   - 离开with块时,Session被关闭
    async with AsyncSessionLocal() as session:
        try:
            # yield session: 将session对象"产出"给调用方
            # 调用方可以在async with块中使用这个session
            yield session
        except Exception:
            # 捕获with块内的任何异常
            # await session.rollback(): 回滚所有未提交的修改
            await session.rollback()
            # raise: 重新抛出异常,让调用方知道发生了错误
            raise
