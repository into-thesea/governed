"""tests.test_rate_limiter —— RateLimiter ABC 及各实现的单元测试。

覆盖：
- ProcessRateLimiter：滑动窗口语义、并发原子性、被拒不占配额、reset
- NoopRateLimiter：总是放行
- RedisRateLimiter：用 mock redis 验证 Lua 脚本调用、fail closed、reset
- build_rate_limiter 工厂：合法 backend、未知 backend 报错、redis 缺 client 报错
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from harness.rate_limiter import (
    NoopRateLimiter,
    ProcessRateLimiter,
    RateLimiter,
    RedisRateLimiter,
    build_rate_limiter,
)


# ======================================================================
# ProcessRateLimiter
# ======================================================================
class TestProcessRateLimiter:
    def test_acquire_allows_up_to_limit(self) -> None:
        rl = ProcessRateLimiter()
        for _ in range(3):
            ok, _ = rl.acquire("tool", 3)
            assert ok is True
        ok, reason = rl.acquire("tool", 3)
        assert ok is False
        assert "每分钟最多调用 3 次" in reason

    def test_rejected_does_not_consume_quota(self) -> None:
        rl = ProcessRateLimiter()
        rl.acquire("tool", 1)
        for _ in range(5):
            ok, _ = rl.acquire("tool", 1)
            assert ok is False
        assert rl.current_window_count("tool") == 1

    def test_window_slides(self, monkeypatch) -> None:
        clock = {"now": 1000.0}
        monkeypatch.setattr(
            "harness.rate_limiter.time", SimpleNamespace(time=lambda: clock["now"])
        )
        rl = ProcessRateLimiter()
        assert rl.acquire("tool", 1)[0] is True
        assert rl.acquire("tool", 1)[0] is False
        clock["now"] += 61.0
        assert rl.acquire("tool", 1)[0] is True

    def test_concurrent_never_exceeds_limit(self) -> None:
        rl = ProcessRateLimiter()
        limit, callers = 5, 20
        barrier = threading.Barrier(callers)
        admitted: list[bool] = []
        lock = threading.Lock()

        def worker() -> None:
            barrier.wait()
            ok, _ = rl.acquire("tool", limit)
            with lock:
                admitted.append(ok)

        threads = [threading.Thread(target=worker) for _ in range(callers)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert sum(admitted) == limit

    def test_reset_clears_window(self) -> None:
        rl = ProcessRateLimiter()
        rl.acquire("tool", 10)
        assert rl.current_window_count("tool") == 1
        rl.reset("tool")
        assert rl.current_window_count("tool") == 0

    def test_scope(self) -> None:
        assert ProcessRateLimiter().scope == "process"

    def test_independent_tools(self) -> None:
        rl = ProcessRateLimiter()
        rl.acquire("a", 1)
        assert rl.acquire("a", 1)[0] is False
        assert rl.acquire("b", 1)[0] is True


# ======================================================================
# NoopRateLimiter
# ======================================================================
class TestNoopRateLimiter:
    def test_always_allows(self) -> None:
        rl = NoopRateLimiter()
        for _ in range(100):
            ok, reason = rl.acquire("tool", 1)
            assert ok is True
            assert reason == ""

    def test_current_count_zero(self) -> None:
        rl = NoopRateLimiter()
        rl.acquire("tool", 1)
        assert rl.current_window_count("tool") == 0

    def test_scope(self) -> None:
        assert NoopRateLimiter().scope == "none"

    def test_reset_noop(self) -> None:
        rl = NoopRateLimiter()
        rl.reset("tool")  # 不抛异常


# ======================================================================
# RedisRateLimiter (mock)
# ======================================================================
class TestRedisRateLimiter:
    def _make_mock_redis(self, lua_result: int = 1) -> MagicMock:
        mock_redis = MagicMock()
        mock_lua = MagicMock(return_value=lua_result)
        mock_redis.register_script.return_value = mock_lua
        mock_redis.zcard.return_value = 5
        return mock_redis

    def test_acquire_allowed_calls_lua(self) -> None:
        mock_redis = self._make_mock_redis(lua_result=1)
        rl = RedisRateLimiter(mock_redis)
        ok, reason = rl.acquire("tool", 10)
        assert ok is True
        assert reason == ""
        mock_redis.register_script.assert_called_once()
        mock_redis.register_script.return_value.assert_called_once()

    def test_acquire_denied_calls_zcard(self) -> None:
        mock_redis = self._make_mock_redis(lua_result=0)
        rl = RedisRateLimiter(mock_redis)
        ok, reason = rl.acquire("tool", 10)
        assert ok is False
        assert "当前已调用 5 次" in reason
        mock_redis.zcard.assert_called_once()

    def test_redis_unavailable_fail_closed(self) -> None:
        mock_redis = MagicMock()
        mock_redis.register_script.return_value.side_effect = ConnectionError("redis down")
        rl = RedisRateLimiter(mock_redis)
        ok, reason = rl.acquire("tool", 10)
        assert ok is False
        assert "限流后端不可用" in reason

    def test_reset_deletes_key(self) -> None:
        mock_redis = self._make_mock_redis()
        rl = RedisRateLimiter(mock_redis)
        rl.reset("tool")
        mock_redis.delete.assert_called_once_with("rate_limit:tool")

    def test_current_window_count(self) -> None:
        mock_redis = self._make_mock_redis()
        mock_redis.zcard.return_value = 3
        rl = RedisRateLimiter(mock_redis)
        assert rl.current_window_count("tool") == 3
        mock_redis.zremrangebyscore.assert_called_once()
        mock_redis.zcard.assert_called_once()

    def test_scope(self) -> None:
        mock_redis = self._make_mock_redis()
        assert RedisRateLimiter(mock_redis).scope == "redis"

    def test_custom_key_prefix(self) -> None:
        mock_redis = self._make_mock_redis()
        rl = RedisRateLimiter(mock_redis, key_prefix="custom")
        rl.reset("tool")
        mock_redis.delete.assert_called_once_with("custom:tool")


# ======================================================================
# build_rate_limiter 工厂
# ======================================================================
class TestBuildRateLimiter:
    def test_process(self) -> None:
        rl = build_rate_limiter("process")
        assert isinstance(rl, ProcessRateLimiter)

    def test_none(self) -> None:
        rl = build_rate_limiter("none")
        assert isinstance(rl, NoopRateLimiter)

    def test_redis_with_client(self) -> None:
        mock_client = MagicMock()
        mock_client.register_script.return_value = MagicMock()
        rl = build_rate_limiter("redis", redis_client=mock_client)
        assert isinstance(rl, RedisRateLimiter)

    def test_redis_missing_client_raises(self) -> None:
        with pytest.raises(ValueError, match="需要提供 redis_client"):
            build_rate_limiter("redis")

    def test_unknown_backend_raises(self) -> None:
        with pytest.raises(ValueError, match="未知的 rate_limiter backend"):
            build_rate_limiter("mongodb")

    def test_default_is_process(self) -> None:
        rl = build_rate_limiter()
        assert isinstance(rl, ProcessRateLimiter)


# ======================================================================
# ABC 契约
# ======================================================================
class TestABCCantract:
    def test_cannot_instantiate_abc(self) -> None:
        with pytest.raises(TypeError):
            RateLimiter()  # type: ignore[abstract]

    def test_default_current_window_count(self) -> None:
        class Minimal(RateLimiter):
            def acquire(self, tool_name, max_per_min):
                return True, ""
        assert Minimal().current_window_count("x") == 0

    def test_default_scope_unknown(self) -> None:
        class Minimal(RateLimiter):
            def acquire(self, tool_name, max_per_min):
                return True, ""
        assert Minimal().scope == "unknown"
