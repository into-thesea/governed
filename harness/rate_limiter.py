"""harness.rate_limiter —— 工具调用限流（滑动窗口）。

抽象出 RateLimiter ABC，让限流状态可以在进程内（单机默认）和 Redis（多副本共享）
之间切换。ToolBroker 只依赖 ABC，不关心后端实现。

为什么是滑动窗口而非固定窗口：
- 固定窗口（INCR+EXPIRE）在窗口边界会放行 2× 配额（前窗口末尾 + 后窗口开头）；
- 滑动窗口精确统计"过去 60 秒内"的调用数，边界无突刺。

为什么 Redis 版用 Lua：
- 限流的"判定 + 记账"必须原子，否则并发 check-then-act 会超放。
- zset + Lua 是滑动窗口的标准原子实现（移除过期 → 计数 → 超阈值拒绝 / 否则写入）。
"""

from __future__ import annotations

import abc
import threading
import time
import uuid
from typing import Any

_RATE_LIMIT_WINDOW_SECONDS = 60  # 与 ToolDef.rate_limit_per_min 对应（每分钟）


class RateLimiter(abc.ABC):
    """限流准入抽象：判定 + 记账原子完成。"""

    @abc.abstractmethod
    def acquire(self, tool_name: str, max_per_min: int) -> tuple[bool, str]:
        """尝试获取一次调用配额。

        Returns:
            (allowed, reason)：allowed=True 时 reason 为空串；
            allowed=False 时 reason 是给 LLM/调用方看的拒绝说明。
        """
        ...

    def current_window_count(self, tool_name: str) -> int:
        """当前滑动窗口内的调用次数（供 stats/监控，非热路径）。"""
        return 0

    def reset(self, tool_name: str) -> None:
        """清除某工具的限流窗口记录（工具注销时调用）。默认空操作。"""

    @property
    def scope(self) -> str:
        """限流作用域：``process``（单机）或 ``redis``（跨副本共享）。"""
        return "unknown"

    def close(self) -> None:
        """释放后端连接（进程退出时调用）。默认空操作。"""


class ProcessRateLimiter(RateLimiter):
    """进程内滑动窗口限流（单机默认）。

    用 dict[tool] -> list[timestamp] + 线程锁实现。多副本时各副本各算各的，
    实际放行量 = 副本数 × 配额——要跨副本一致用 RedisRateLimiter。
    """

    def __init__(self) -> None:
        self._call_log: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def acquire(self, tool_name: str, max_per_min: int) -> tuple[bool, str]:
        now = time.time()
        window_start = now - _RATE_LIMIT_WINDOW_SECONDS

        with self._lock:
            recent = [t for t in self._call_log.get(tool_name, []) if t > window_start]
            if len(recent) >= max_per_min:
                # 被拒绝的调用不计入窗口，否则持续重试会把窗口越撑越满。
                self._call_log[tool_name] = recent
                return False, (
                    f"工具 '{tool_name}' 每分钟最多调用 {max_per_min} 次，"
                    f"当前已调用 {len(recent)} 次"
                )
            recent.append(now)
            self._call_log[tool_name] = recent
        return True, ""

    def current_window_count(self, tool_name: str) -> int:
        now = time.time()
        window_start = now - _RATE_LIMIT_WINDOW_SECONDS
        with self._lock:
            return len([t for t in self._call_log.get(tool_name, []) if t > window_start])

    def reset(self, tool_name: str) -> None:
        with self._lock:
            self._call_log.pop(tool_name, None)

    @property
    def scope(self) -> str:
        return "process"


class NoopRateLimiter(RateLimiter):
    """无限流（总是放行）。用于显式关闭限流的场景（测试/调试）。

    注意：这会让 rate_limit_per_min 完全失效，仅用于明确知道自己在做什么的场景。
    """

    def acquire(self, tool_name: str, max_per_min: int) -> tuple[bool, str]:
        return True, ""

    def current_window_count(self, tool_name: str) -> int:
        return 0

    def reset(self, tool_name: str) -> None:
        pass

    @property
    def scope(self) -> str:
        return "none"


# Redis 滑动窗口 Lua 脚本：
# KEYS[1] = rate_limit:{tool_name}
# ARGV[1] = window_start（过期时间戳，小于此值的成员被移除）
# ARGV[2] = max_per_min（阈值）
# ARGV[3] = member（当前调用的唯一标识，score=now）
# ARGV[4] = now（当前时间戳，作为 zset score）
# 返回：1 = 放行，0 = 拒绝
_REDIS_SLIDING_WINDOW_LUA = """
redis.call('ZREMRANGEBYSCORE', KEYS[1], 0, tonumber(ARGV[1]))
local count = redis.call('ZCARD', KEYS[1])
if count >= tonumber(ARGV[2]) then
    return 0
end
redis.call('ZADD', KEYS[1], tonumber(ARGV[4]), ARGV[3])
redis.call('EXPIRE', KEYS[1], 120)
return 1
"""


class RedisRateLimiter(RateLimiter):
    """Redis 滑动窗口限流（多副本共享状态）。

    用 zset + Lua 脚本实现原子的"移除过期 → 计数 → 超阈值拒绝 / 否则写入"。
    每个工具一个 key（``rate_limit:{tool_name}``），成员是 ``{timestamp}:{uuid}``
    保证唯一，score 是时间戳。EXPIRE 120 秒（窗口 60 秒的 2 倍，留余量）。

    连接由调用方注入（``redis.Redis`` 或 ``redis.asyncio.Redis`` 的同步版），
    本类不负责建连——连接池/重连策略由部署方决定。
    """

    def __init__(self, redis_client: Any, key_prefix: str = "rate_limit") -> None:
        """
        Args:
            redis_client: 已连接的 redis.Redis 实例（同步）。
            key_prefix: zset key 前缀，默认 ``rate_limit``。
        """
        self._redis = redis_client
        self._key_prefix = key_prefix
        self._lua = self._redis.register_script(_REDIS_SLIDING_WINDOW_LUA)

    def acquire(self, tool_name: str, max_per_min: int) -> tuple[bool, str]:
        now = time.time()
        window_start = now - _RATE_LIMIT_WINDOW_SECONDS
        key = f"{self._key_prefix}:{tool_name}"
        member = f"{now}:{uuid.uuid4().hex}"

        try:
            result = self._lua(
                keys=[key],
                args=[window_start, max_per_min, member, now],
            )
        except Exception as exc:
            # Redis 不可用时 fail closed：拒绝调用并说明原因。
            # 这比静默放行安全——限流失效时宁可拒绝也不超放。
            return False, f"限流后端不可用（Redis）：{exc}"

        if result == 0:
            # 被拒绝时需要拿到当前计数给提示——再查一次 zcard（此时已移除过期）。
            try:
                current = self._redis.zcard(key)
            except Exception:
                current = max_per_min  # 查不到就按满配额报
            return False, (
                f"工具 '{tool_name}' 每分钟最多调用 {max_per_min} 次，"
                f"当前已调用 {current} 次"
            )
        return True, ""

    def current_window_count(self, tool_name: str) -> int:
        key = f"{self._key_prefix}:{tool_name}"
        now = time.time()
        window_start = now - _RATE_LIMIT_WINDOW_SECONDS
        try:
            self._redis.zremrangebyscore(key, 0, window_start)
            return int(self._redis.zcard(key))
        except Exception:
            return 0

    def reset(self, tool_name: str) -> None:
        key = f"{self._key_prefix}:{tool_name}"
        try:
            self._redis.delete(key)
        except Exception:
            pass

    @property
    def scope(self) -> str:
        return "redis"

    def close(self) -> None:
        try:
            self._redis.close()
        except Exception:
            pass


def build_rate_limiter(backend: str = "process", **kwargs: Any) -> RateLimiter:
    """按配置构造 RateLimiter。

    Args:
        backend: ``process``（默认，单机）或 ``redis``（多副本共享）。
        **kwargs: 传给对应构造函数（redis 后端需 ``redis_client``）。

    Raises:
        ValueError: 未知 backend —— 显式报错，不静默回退到 process（静默回退会
            让"以为配了分布式限流"的部署实际跑在单机上，超放了都不知道）。
    """
    backend = (backend or "").strip().lower()
    if backend == "process":
        return ProcessRateLimiter()
    if backend == "none":
        return NoopRateLimiter()
    if backend == "redis":
        client = kwargs.get("redis_client")
        if client is None:
            raise ValueError("rate_limiter backend='redis' 需要提供 redis_client")
        return RedisRateLimiter(client, key_prefix=kwargs.get("key_prefix", "rate_limit"))
    raise ValueError(f"未知的 rate_limiter backend: {backend!r}（支持 process | redis）")


__all__ = [
    "RateLimiter",
    "ProcessRateLimiter",
    "NoopRateLimiter",
    "RedisRateLimiter",
    "build_rate_limiter",
]
