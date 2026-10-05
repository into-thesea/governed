"""tests.test_rate_limit —— Tool Broker 限流准入的原子性与窗口语义。

限流把「判定」与「记账」放在同一次加锁内完成。若拆成两步、中间隔着工具执行，
并发调用会同时通过判定再各自记账（check-then-act 竞态），实际放行量会超过
``rate_limit_per_min``。本文件锁住两条不变式：

1. 并发下放行数**严格等于**上限，与该工具耗时无关；
2. 窗口按 60 秒滑动，被拒绝的调用不占用配额。

限流窗口是进程内的（见 ``ToolBroker._rate_lock`` 上的说明），因此全部用例
离线可跑，不需要任何外部服务。
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

from harness.models import ToolDef
from harness.tool_broker import ToolBroker


# ======================================================================
# 测试脚手架
# ======================================================================
def _ok_handler(args: dict, context: dict) -> tuple[bool, str, dict]:
    return True, "ok", {}


class _RecordingAudit:
    """记录调用参数的审计替身（只实现 Broker 用到的那一个方法）。"""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def record_tool_call(self, **kwargs: object) -> None:
        self.calls.append(kwargs)


def _make_broker(
    rate_limit_per_min: int,
    handler=_ok_handler,
    audit_logger=None,
) -> ToolBroker:
    """构造只注册了一个限流工具的 Broker。

    ``sandbox_executor=False`` 是 Broker 自带的显式关闭开关；用例里的工具都没标
    ``run_in_sandbox``，不会碰到沙箱分支。
    """
    broker = ToolBroker(sandbox_executor=False, audit_logger=audit_logger)
    broker.register(
        ToolDef(
            name="limited",
            description="限流测试用工具",
            parameters={},
            rate_limit_per_min=rate_limit_per_min,
        ),
        handler,
    )
    return broker


# ======================================================================
# 原子性
# ======================================================================
class TestRateLimitAtomicity:
    def test_concurrent_calls_never_exceed_limit(self) -> None:
        """并发调用放行的数量恰好等于上限。

        这个用例是对 check-then-act 竞态的回归守卫：判定与记账一旦被拆开、记账
        落在工具执行之后，超出判定的调用就会一起放行，放行数大于 limit。为了让
        竞态窗口稳定复现，handler 里加了 5ms 的等待——它只影响耗时，不影响放行数。
        """
        limit, callers = 5, 20
        broker = _make_broker(limit, handler=_slow_handler)

        barrier = threading.Barrier(callers)
        admitted: list[bool] = []
        collect_lock = threading.Lock()

        def worker() -> None:
            barrier.wait()  # 所有线程尽量同时进入 invoke
            ok, _, _ = broker.invoke("limited", {}, {})
            with collect_lock:
                admitted.append(ok)

        threads = [threading.Thread(target=worker) for _ in range(callers)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(admitted) == callers
        assert sum(admitted) == limit

    def test_sequential_calls_stop_at_limit(self) -> None:
        """顺序调用：前 limit 次放行，第 limit+1 次被拒并给出可读原因。"""
        broker = _make_broker(3)
        assert [broker.invoke("limited", {}, {})[0] for _ in range(3)] == [True] * 3

        ok, text, _ = broker.invoke("limited", {}, {})
        assert ok is False
        assert "每分钟最多调用 3 次" in text

    def test_rejected_calls_do_not_consume_quota(self) -> None:
        """被拒的调用不计入窗口，否则持续重试会把窗口越撑越满。"""
        broker = _make_broker(2)
        broker.invoke("limited", {}, {})
        broker.invoke("limited", {}, {})

        for _ in range(5):
            assert broker.invoke("limited", {}, {})[0] is False

        assert broker.get_stats()["tools"][0]["recent_calls_1min"] == 2


# ======================================================================
# 窗口语义
# ======================================================================
class TestRateLimitWindow:
    def test_window_slides_after_60s(self, monkeypatch) -> None:
        """跨过 60 秒窗口后旧记录被清理，配额恢复。"""
        clock = {"now": 1000.0}
        monkeypatch.setattr(
            "harness.rate_limiter.time", SimpleNamespace(time=lambda: clock["now"])
        )

        broker = _make_broker(1)
        assert broker.invoke("limited", {}, {})[0] is True
        assert broker.invoke("limited", {}, {})[0] is False

        clock["now"] += 61.0
        assert broker.invoke("limited", {}, {})[0] is True

    def test_failed_execution_still_consumes_quota(self) -> None:
        """准入即记账：执行失败的调用同样占用配额。

        限流管的是「发起频率」而不是「成功次数」—— 否则一个持续失败的工具
        反而永远不会触发限流，与限流的目的相反。
        """

        def boom(args: dict, context: dict) -> tuple[bool, str, dict]:
            raise RuntimeError("handler 故意抛错")

        broker = _make_broker(1, handler=boom)
        ok, text, _ = broker.invoke("limited", {}, {})
        assert ok is False and "工具执行异常" in text

        # 唯一的配额已被这次失败的调用占用
        assert broker.invoke("limited", {}, {})[0] is False


# ======================================================================
# 与其它组件的交互
# ======================================================================
class TestRateLimitIntegration:
    def test_scoped_view_shares_underlying_quota(self) -> None:
        """受限视图共用底层窗口：换个视图调用绕不过限流。"""
        broker = _make_broker(1)
        scoped = broker.scoped(["limited"])

        assert scoped.invoke("limited", {}, {})[0] is True
        assert scoped.invoke("limited", {}, {})[0] is False
        assert broker.invoke("limited", {}, {})[0] is False

    def test_unregister_resets_counter(self) -> None:
        """注销会清掉该工具的窗口记录（重新注册后配额是干净的）。"""
        broker = _make_broker(1)
        assert broker.invoke("limited", {}, {})[0] is True
        assert broker.invoke("limited", {}, {})[0] is False

        assert broker.unregister("limited") is True
        broker.register(
            ToolDef(name="limited", description="d", parameters={}, rate_limit_per_min=1),
            _ok_handler,
        )
        assert broker.invoke("limited", {}, {})[0] is True

    def test_rejection_is_audited(self) -> None:
        """被限流拒绝的调用要留审计，且原因是可读的。"""
        audit = _RecordingAudit()
        broker = _make_broker(1, audit_logger=audit)

        broker.invoke("limited", {}, {})
        broker.invoke("limited", {}, {})

        rejection = audit.calls[-1]
        assert rejection["result_ok"] is False
        assert str(rejection["error"]).startswith("限流：")

    def test_stats_report_process_scope(self) -> None:
        """统计里显式报出限流是进程内的，免得被当成全局配额。"""
        broker = _make_broker(60)
        broker.invoke("limited", {}, {})

        stats = broker.get_stats()
        assert stats["rate_limit_scope"] == "process"
        assert stats["tools"][0]["recent_calls_1min"] == 1


def _slow_handler(args: dict, context: dict) -> tuple[bool, str, dict]:
    """把 check-then-act 的窗口拉宽的 handler，仅用于并发用例。"""
    time.sleep(0.005)
    return True, "ok", {}
