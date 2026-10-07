"""tests.test_metrics —— Prometheus 导出端点。

三条不变量，各对应一类真实的坑：

1. **格式合法**：用一个小解析器**解析**输出，而不是断言字符串子串 ——
   格式错了（缺 TYPE、label 没转义、值不是数字）抓取端会整条丢弃，
   而字符串断言照样能过。
2. **数字会动**：造事件后对应的值**真的变了**，不是"有这一行"。
3. **label 基数有界**：输出里**不得出现 uuid / 会话 id 形状的值**。
   这是这类导出最容易埋的雷 —— 每个任务新增一条时间序列，只增不减，
   不会报错，只会在很久以后把监控拖垮。
"""

from __future__ import annotations

import re

import pytest

from harness.server.metrics import render_metrics

# ----------------------------------------------------------------------
# 一个极小的 Prometheus 文本解析器（只认本端点产出的形态）
# ----------------------------------------------------------------------
_SAMPLE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)"
    r"(?:\{(?P<labels>.*)\})?"
    r"\s+(?P<value>-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)$"
)
_LABEL = re.compile(r'(\w+)="((?:[^"\\]|\\.)*)"')
_UUIDISH = re.compile(r"[0-9a-fA-F]{8}-?[0-9a-fA-F]{4}|[0-9a-f]{16,}")


def _parse(text: str) -> tuple[dict, set[str]]:
    """解析成 ``{(指标名, 标签元组): 值}``；格式非法直接断言失败。"""
    samples: dict[tuple[str, tuple], float] = {}
    typed: set[str] = set()
    for line in text.splitlines():
        if not line.strip():
            continue
        if line.startswith("# TYPE "):
            _, _, name, _kind = line.split(None, 3)
            typed.add(name)
            continue
        if line.startswith("#"):
            continue
        m = _SAMPLE.match(line)
        assert m, f"不合法的样本行（抓取端会整条丢弃）：{line!r}"
        labels = tuple(sorted(_LABEL.findall(m.group("labels") or "")))
        samples[(m.group("name"), labels)] = float(m.group("value"))
    return samples, typed


def _value(samples: dict, name: str, **labels) -> float:
    return samples[(name, tuple(sorted(labels.items())))]


# ----------------------------------------------------------------------
# 脚手架
# ----------------------------------------------------------------------
@pytest.fixture
def service():
    import asyncio

    from harness.server.service import HarnessService
    from tests._smoke_server import ApprovalLLM
    from tests.test_console_api import _memory_saver

    async def build():
        svc = HarnessService(checkpointer=_memory_saver(), llm=ApprovalLLM(),
                             auto_assemble=False)
        await svc._ensure_ready()
        svc.assemble()
        return svc

    return asyncio.run(build())


# ----------------------------------------------------------------------
# 1. 格式
# ----------------------------------------------------------------------
def test_output_parses_and_every_metric_declares_its_type(service) -> None:
    text = render_metrics(service)
    samples, typed = _parse(text)

    assert samples, "端点不该什么都不导"
    undeclared = {name for name, _ in samples} - typed
    assert not undeclared, f"这些指标没有 # TYPE 声明：{sorted(undeclared)}"
    assert "governed_build_info" in typed


def test_endpoint_is_served_and_requires_the_same_auth_as_data(service) -> None:
    """挂在 FastAPI 上（不是只 import 得到函数），且**不在匿名白名单里**。"""
    from fastapi.testclient import TestClient

    from harness.server.app import create_app

    with TestClient(create_app(service)) as client:
        r = client.get("/metrics")
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/plain")
        _parse(r.text)

    from harness.server.auth import ANONYMOUS_PATHS

    assert "/metrics" not in ANONYMOUS_PATHS, (
        "指标暴露的是内部运行状况，不该匿名可读；抓取端用 Bearer 头取即可"
    )


# ----------------------------------------------------------------------
# 2. 数字会动
# ----------------------------------------------------------------------
def test_tool_call_counter_splits_by_outcome(service) -> None:
    from harness.models import ToolDef

    svc = service
    def _register(name: str, handler) -> None:
        svc.broker.register(
            ToolDef(name=name, description="x",
                    parameters={"type": "object", "properties": {}},
                    rate_limit_per_min=1000),
            handler,
        )

    def _boom(args, ctx):
        raise RuntimeError("下游炸了")

    _register("ok_tool", lambda a, c: (True, "done", {}))
    _register("boom_tool", _boom)

    svc.broker.invoke("ok_tool", {}, {"role": "admin"})
    svc.broker.invoke("boom_tool", {}, {"role": "admin"})

    samples, _ = _parse(render_metrics(svc))
    assert _value(samples, "governed_tool_calls_total", tool="ok_tool", ok="true") == 1
    assert _value(samples, "governed_tool_calls_total", tool="boom_tool", ok="false") == 1


def test_guard_and_approval_totals_move_with_real_events(service) -> None:
    """喂真实事件（走服务的聚合函数），断言导出值跟着动。"""
    svc = service
    svc._on_metric_event({"type": "GUARD_DECISION", "thread_id": "t1",
                          "data": {"layer": "pdp", "decision": "deny"}})
    svc._on_metric_event({"type": "GUARD_DECISION", "thread_id": "t2",
                          "data": {"layer": "pdp", "decision": "deny"}})
    svc._on_metric_event({"type": "APPROVAL_REQUIRED", "thread_id": "t1", "data": {}})
    svc._on_metric_event({"type": "APPROVAL_RESOLVED", "thread_id": "t1",
                          "data": {"approved": True}})

    samples, _ = _parse(render_metrics(svc))
    assert _value(samples, "governed_guard_decisions_total", layer="pdp", decision="deny") == 2
    assert _value(samples, "governed_approvals_required_total") == 1
    assert _value(samples, "governed_approvals_resolved_total", outcome="approved") == 1


def test_sandbox_gauge_reflects_runtime_not_configuration(service) -> None:
    """这个 gauge 的价值全在"运行时"三个字上：配置开着但服务没起，必须是 0。

    反过来也必须成立：真可用时是 1 —— 两边都钉死才叫"跟运行时走"。用替身注入
    两种运行时状态，**不能**依赖"这台机器上沙箱起没起"：曾经这条用例只断言 0，
    于是在沙箱真活着的环境里必然失败（同一个根源：把环境当成了断言前提）。
    """
    class _Down:
        def available(self):
            return False, "stub：沙箱服务未起"

    class _Up:
        def available(self):
            return True, "stub：沙箱可用"

    service.broker.sandbox = _Down()
    samples, _ = _parse(render_metrics(service))
    assert _value(samples, "governed_sandbox_available") == 0.0, "配置开着但服务不可用 → 必须是 0"

    service.broker.sandbox = _Up()
    samples, _ = _parse(render_metrics(service))
    assert _value(samples, "governed_sandbox_available") == 1.0, "服务真可用 → 必须是 1"


# ----------------------------------------------------------------------
# 3. label 基数（这条守着本模块最大的风险）
# ----------------------------------------------------------------------
def test_no_unbounded_label_values(service) -> None:
    """**不得**出现会话/任务 id 形状的 label 值。

    服务侧的聚合计数是按 thread_id 存的（控制台要按会话看），如果照搬进 label，
    每个任务都会在 Prometheus 里永久新增一条时间序列。导出前必须塌掉。
    """
    svc = service
    for tid in ("a3f1c2d4e5b64789a1b2c3d4e5f60718", "bbbbccccddddeeee1122334455667788"):
        svc._on_metric_event({"type": "GUARD_DECISION", "thread_id": tid,
                              "data": {"layer": "approval", "decision": "deny"}})

    text = render_metrics(svc)
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        labels = _LABEL.findall(line)
        for key, value in labels:
            assert not _UUIDISH.search(value), (
                f"label {key}={value!r} 是会话/任务 id 形状的 —— 基数无界：{line!r}"
            )


def test_error_text_never_becomes_a_label(service) -> None:
    """断路器的 last_error 是**自由文本**（含异常消息）—— 拿它做 label 既无界又泄漏。"""
    from harness.circuit_breaker import CircuitBreaker

    svc = service
    svc.broker.breaker = CircuitBreaker(failure_threshold=1, cooldown_seconds=30)
    svc.broker.breaker.record_failure("echo", "psycopg 连接失败：host=10.0.0.7 password=hunter2")

    text = render_metrics(svc)
    assert "hunter2" not in text, "错误原文不该出现在指标里"
    assert "10.0.0.7" not in text
