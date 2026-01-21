"""DBWriter：将非关键写入串行化，降低 SQLite 并发写争用。

注意：
- `asyncio.PriorityQueue` 在优先级相同的情况下会继续比较下一个元素；
  如果队列元素是不可比较的对象，会触发 TypeError。
- 因此这里加入自增序号作为稳定的次级排序键。
"""

from __future__ import annotations

import asyncio
from itertools import count
from typing import Optional, Protocol

from nonebot import logger

class DBWriteJob(Protocol):
    """写入任务协议：任务内部自行管理 Session 与事务。"""

    async def execute(self) -> object:
        """执行写入任务。"""
        ...

class DBWriter:
    """全局单例写入队列（优先级越小越优先）。"""

    _instance: Optional["DBWriter"] = None
    q: asyncio.PriorityQueue[
        tuple[int, int, DBWriteJob, Optional[asyncio.Future[object]]]
    ]
    _seq: count[int]
    _running: bool

    def __new__(cls) -> DBWriter:
        """创建/获取单例实例。"""

        if cls._instance is None:
            cls._instance = super(DBWriter, cls).__new__(cls)
            cls._instance.q = asyncio.PriorityQueue()
            cls._instance._running = False
            cls._instance._seq = count()
        return cls._instance

    def __init__(self) -> None:
        """单例初始化在 `__new__` 中完成。"""
        pass

    async def submit(self, job: DBWriteJob, priority: int = 5) -> None:
        """提交一个写入任务。

        参数：
            job: 实现 `execute()` 的写入任务对象。
            priority: 优先级，数字越小优先级越高，默认 5。
        """

        await self.q.put((priority, next(self._seq), job, None))

    async def submit_and_wait(self, job: DBWriteJob, priority: int = 5) -> object:
        """提交一个写入任务并等待其执行完成，返回 execute() 的结果。"""

        loop = asyncio.get_running_loop()
        fut: asyncio.Future[object] = loop.create_future()
        await self.q.put((priority, next(self._seq), job, fut))
        return await fut

    async def run_forever(self) -> None:
        """持续运行队列消费循环（应在 startup 时以 task 方式启动）。"""

        if self._running:
            return
        self._running = True
        logger.info("DBWriter 已启动。")
        while True:
            _priority, _seq, job, fut = await self.q.get()
            try:
                result = await job.execute()
                if fut is not None and not fut.done():
                    fut.set_result(result)
            except Exception as e:
                if fut is not None and not fut.done():
                    fut.set_exception(e)
                logger.error(f"DBWriter 任务执行失败：{e}")
            finally:
                self.q.task_done()

# 全局实例
db_writer = DBWriter()
