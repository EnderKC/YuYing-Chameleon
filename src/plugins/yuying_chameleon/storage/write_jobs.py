"""DBWriter 写入任务：将非关键写入串行化。"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Awaitable, Callable

from .models import IndexJob
from .repositories.index_jobs_repo import IndexJobRepository


@dataclass(frozen=True, slots=True)
class AddIndexJobJob:
    """新增一条 IndexJob。"""

    job: IndexJob

    async def execute(self) -> object:
        return await IndexJobRepository.add(self.job)


@dataclass(frozen=True, slots=True)
class AsyncCallableJob:
    """将任意 async 写入函数封装为 DBWriter 任务。"""

    func: Callable[..., Awaitable[object]]
    args: tuple[object, ...] = ()
    kwargs: dict[str, object] = field(default_factory=dict)

    async def execute(self) -> object:
        return await self.func(*self.args, **self.kwargs)
