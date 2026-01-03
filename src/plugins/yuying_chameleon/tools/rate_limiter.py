"""会话级速率限制器：防止 AI 过度写入记忆。

设计原理：
- 会话定义：连续对话，空闲超时（默认 10 分钟）后视为新会话
- 短期限制：每个会话最多 N 条（默认 3）
- 长期限制：每天最多 M 条（默认 25）
- 场景区分：群聊和私聊可配置不同限制
- TTL 缓存：自动过期，无需定时清理

实现策略：
1. 使用 dict 存储每个用户的会话状态（内存缓存）
2. 每次检查时判断是否超时重置
3. 使用 UTC timestamp 避免时区问题
4. 只统计成功写入的记忆（去重后）
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Optional

from nonebot import logger

from ..config import plugin_config


@dataclass
class SessionState:
    """单个用户的会话状态。"""

    # 当前会话计数
    session_count: int = 0

    # 最后一条消息时间（UTC timestamp）
    last_msg_time: float = field(default_factory=time.time)

    # 今日计数（用于每日限额）
    daily_count: int = 0

    # 今日开始时间（UTC timestamp，用于检测跨天）
    daily_start: float = field(default_factory=time.time)


class SessionRateLimiter:
    """会话级速率限制器（内存实现，进程重启后重置）。"""

    def __init__(
        self,
        *,
        session_limit_group: int = 3,
        session_limit_private: int = 5,
        daily_limit_group: int = 25,
        daily_limit_private: int = 40,
        session_idle_timeout: float = 600.0,  # 10 分钟
    ) -> None:
        """初始化速率限制器。

        Args:
            session_limit_group: 群聊每会话限额
            session_limit_private: 私聊每会话限额
            daily_limit_group: 群聊每日限额
            daily_limit_private: 私聊每日限额
            session_idle_timeout: 会话空闲超时（秒）
        """
        self.session_limit_group = session_limit_group
        self.session_limit_private = session_limit_private
        self.daily_limit_group = daily_limit_group
        self.daily_limit_private = daily_limit_private
        self.session_idle_timeout = session_idle_timeout

        # 会话状态缓存：key = f"{qq_id}:{scene_type}"
        self._sessions: Dict[str, SessionState] = {}

    def _get_limits(self, scene_type: str) -> tuple[int, int]:
        """根据场景类型返回（会话限额，每日限额）。"""
        if scene_type == "group":
            return self.session_limit_group, self.daily_limit_group
        else:  # private
            return self.session_limit_private, self.daily_limit_private

    def _is_new_day(self, state: SessionState) -> bool:
        """判断是否跨天（UTC 天级别）。"""
        import datetime

        now_utc = datetime.datetime.utcnow()
        daily_start_utc = datetime.datetime.utcfromtimestamp(state.daily_start)
        return now_utc.date() > daily_start_utc.date()

    def check_and_increment(
        self,
        qq_id: str,
        scene_type: str,
        *,
        dry_run: bool = False,
    ) -> tuple[bool, str]:
        """检查是否允许写入，并增加计数（如果允许且非 dry_run）。

        Args:
            qq_id: 用户 QQ 号
            scene_type: 场景类型（group/private）
            dry_run: 是否为模拟检查（不实际增加计数）

        Returns:
            (是否允许, 原因说明)
            - (True, "ok"): 允许写入
            - (False, "session_limit_exceeded"): 会话限额已满
            - (False, "daily_limit_exceeded"): 每日限额已满
        """
        key = f"{qq_id}:{scene_type}"
        now = time.time()

        # 获取或创建状态
        if key not in self._sessions:
            self._sessions[key] = SessionState()

        state = self._sessions[key]
        session_limit, daily_limit = self._get_limits(scene_type)

        # 检查是否跨天：如果是新的一天，重置每日计数
        if self._is_new_day(state):
            state.daily_count = 0
            state.daily_start = now

        # 检查会话超时：超时则重置会话计数
        if now - state.last_msg_time > self.session_idle_timeout:
            state.session_count = 0

        # 更新最后消息时间
        state.last_msg_time = now

        # 检查每日限额
        if state.daily_count >= daily_limit:
            logger.debug(
                f"速率限制：用户 {qq_id} ({scene_type}) 今日记忆写入已达上限 {daily_limit}"
            )
            return False, "daily_limit_exceeded"

        # 检查会话限额
        if state.session_count >= session_limit:
            logger.debug(
                f"速率限制：用户 {qq_id} ({scene_type}) 本会话记忆写入已达上限 {session_limit}"
            )
            return False, "session_limit_exceeded"

        # 允许写入：如果不是 dry_run，增加计数
        if not dry_run:
            state.session_count += 1
            state.daily_count += 1
            logger.debug(
                f"速率限制：用户 {qq_id} ({scene_type}) 记忆写入计数 "
                f"[会话: {state.session_count}/{session_limit}, "
                f"今日: {state.daily_count}/{daily_limit}]"
            )

        return True, "ok"

    def reset_session(self, qq_id: str, scene_type: str) -> None:
        """手动重置指定用户的会话计数（用于测试或管理命令）。"""
        key = f"{qq_id}:{scene_type}"
        if key in self._sessions:
            self._sessions[key].session_count = 0
            self._sessions[key].last_msg_time = time.time()

    def reset_daily(self, qq_id: str, scene_type: str) -> None:
        """手动重置指定用户的每日计数（用于测试或管理命令）。"""
        key = f"{qq_id}:{scene_type}"
        if key in self._sessions:
            self._sessions[key].daily_count = 0
            self._sessions[key].daily_start = time.time()

    def get_status(self, qq_id: str, scene_type: str) -> Dict[str, int]:
        """获取指定用户的当前速率限制状态（用于调试/监控）。"""
        key = f"{qq_id}:{scene_type}"
        session_limit, daily_limit = self._get_limits(scene_type)

        if key not in self._sessions:
            return {
                "session_count": 0,
                "session_limit": session_limit,
                "daily_count": 0,
                "daily_limit": daily_limit,
            }

        state = self._sessions[key]
        now = time.time()

        # 检查是否跨天
        if self._is_new_day(state):
            daily_count = 0
        else:
            daily_count = state.daily_count

        # 检查会话超时
        if now - state.last_msg_time > self.session_idle_timeout:
            session_count = 0
        else:
            session_count = state.session_count

        return {
            "session_count": session_count,
            "session_limit": session_limit,
            "daily_count": daily_count,
            "daily_limit": daily_limit,
        }


# 全局单例（进程级别）
_rate_limiter: Optional[SessionRateLimiter] = None


def get_rate_limiter() -> SessionRateLimiter:
    """获取全局速率限制器实例（懒初始化）。"""
    global _rate_limiter
    if _rate_limiter is None:
        # 从配置读取参数（如果配置中没有，使用默认值）
        _rate_limiter = SessionRateLimiter(
            session_limit_group=int(
                getattr(plugin_config, "yuying_memory_session_limit_group", 3)
            ),
            session_limit_private=int(
                getattr(plugin_config, "yuying_memory_session_limit_private", 5)
            ),
            daily_limit_group=int(
                getattr(plugin_config, "yuying_memory_daily_limit_group", 25)
            ),
            daily_limit_private=int(
                getattr(plugin_config, "yuying_memory_daily_limit_private", 40)
            ),
            session_idle_timeout=float(
                getattr(plugin_config, "yuying_memory_session_idle_timeout", 600.0)
            ),
        )
    return _rate_limiter
