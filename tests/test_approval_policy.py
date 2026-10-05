"""tests.test_approval_policy —— 审批策略与风险分级。

设计见 docs/审批策略设计.md。核心不变量：**自动放行的理由必须是"有机制兜底"，
不能是"我判断它安全"** —— 判定函数是领域包写的代码，它会是错的。

本文件先只钉注册面（Task 1）：策略按工具名注册、可撤销、受限视图能读到。
决策函数（``decide_approval``）在 Task 2 补齐。
"""

from __future__ import annotations

import pytest

from harness.approval_policy import (
    DECISION_ASK, DECISION_AUTO, DECISION_DENY, RISK_CRITICAL, RISK_HIGH, RISK_LOW,
    RISK_MEDIUM, RISK_UNKNOWN, decide_approval,
)
from harness.events import EventBus, reset_event_bus
from harness.models import ToolDef
from harness.pdp import PDP
from harness.tool_broker import ToolBroker
from tests.test_approval_gate import _guarded_broker


def test_risk_policy_is_registered_and_cleared_per_tool() -> None:
    broker = _guarded_broker()
    assert broker.risk_policy_for("danger") is None, "默认没有策略"

    fn = lambda args: RISK_LOW
    broker.register_risk_policy("danger", fn)
    assert broker.risk_policy_for("danger") is fn

    assert broker.clear_risk_policy("danger") is True
    assert broker.risk_policy_for("danger") is None
    assert broker.clear_risk_policy("danger") is False, "重复清理由返回 False"


def test_scoped_broker_forwards_risk_policy_read() -> None:
    """子 Agent 也必须读得到父任务注册的策略（否则同一工具在子图里判法不同）。"""
    from harness.tool_broker import ScopedBroker

    broker = _guarded_broker()
    fn = lambda args: RISK_HIGH
    broker.register_risk_policy("danger", fn)
    scoped = ScopedBroker(broker, ["danger"])
    assert scoped.risk_policy_for("danger") is fn
    assert scoped.risk_policy_for("echo") is None, "白名单外不暴露"


# ======================================================================
# Task 2：决策函数（纯函数，四类分流）
# ======================================================================
def test_no_policy_defaults_to_ask() -> None:
    """零回归的根：不声明策略 = 与今天逐字节一致（每次都问）。"""
    d = decide_approval("danger", {}, risk_policy=None, fallback="sandbox")
    assert d.decision == DECISION_ASK
    assert d.risk == RISK_UNKNOWN
    assert d.policy == "default"


def test_low_risk_with_fallback_is_auto() -> None:
    d = decide_approval("danger", {}, risk_policy=lambda a: RISK_LOW, fallback="sandbox")
    assert d.decision == DECISION_AUTO
    assert d.fallback == "sandbox"


def test_low_risk_without_fallback_is_downgraded_to_ask() -> None:
    """命门：自动放行的唯一理由是"有机制兜底"，不是"我判断它安全"。"""
    d = decide_approval("danger", {}, risk_policy=lambda a: RISK_LOW, fallback=None)
    assert d.decision == DECISION_ASK
    assert "兜底" in d.reason


def test_high_risk_above_threshold_asks() -> None:
    d = decide_approval("danger", {}, risk_policy=lambda a: RISK_HIGH,
                        threshold=RISK_MEDIUM, fallback="sandbox")
    assert d.decision == DECISION_ASK


def test_unknown_risk_always_asks() -> None:
    for fallback in (None, "sandbox"):
        d = decide_approval("danger", {}, risk_policy=lambda a: RISK_UNKNOWN, fallback=fallback)
        assert d.decision == DECISION_ASK, "未知不放过"


def test_raising_policy_fails_closed() -> None:
    def boom(args):
        raise RuntimeError("策略作者写错了")

    d = decide_approval("danger", {}, risk_policy=boom, fallback="sandbox")
    assert d.decision == DECISION_ASK
    assert d.risk == RISK_UNKNOWN


@pytest.mark.parametrize("bad", ["", None, "MEDIUM ", "very-high", 3])
def test_illegal_risk_value_fails_closed(bad) -> None:
    """风险值**精确匹配**词表，不做 strip/lower 归一。

    "MEDIUM " 这种带杂讯的值按 unknown 处理 —— 归一化会把一个**拼错的 LOW** 悄悄
    变成自动放行，正是调研里 OpenAI #3863 / Pydantic #8060 那类 fail-open 事故。
    """
    d = decide_approval("danger", {}, risk_policy=lambda a: bad, fallback="sandbox")
    assert d.decision == DECISION_ASK


def test_illegal_threshold_asks_everything() -> None:
    """阈值配错时按最保守处理（全都问），而不是按最松处理。"""
    d = decide_approval("danger", {}, risk_policy=lambda a: RISK_LOW,
                        threshold="typo", fallback="sandbox")
    assert d.decision == DECISION_ASK


def test_critical_risk_hits_the_default_deny_line() -> None:
    """红线档默认就拒 —— 连人都别问（别在半夜为人不该被问到的事报警）。"""
    d = decide_approval("danger", {}, risk_policy=lambda a: RISK_CRITICAL, fallback="sandbox")
    assert d.decision == DECISION_DENY
    assert d.risk == RISK_CRITICAL
    assert "拒绝线" in d.reason


def test_deployment_can_lower_the_deny_line() -> None:
    """拒绝线是**部署方**配的：调低它，high 也直接拒；领域包管不着这条线。"""
    d = decide_approval("danger", {}, risk_policy=lambda a: RISK_HIGH,
                        fallback="sandbox", deny_threshold=RISK_HIGH)
    assert d.decision == DECISION_DENY


def test_deny_line_never_touches_unknown() -> None:
    """未知是"问人"，不是"拒" —— 判不出来就把用户的操作毙掉是另一种伤害。"""
    d = decide_approval("danger", {}, risk_policy=lambda a: RISK_UNKNOWN, fallback="sandbox")
    assert d.decision == DECISION_ASK


def test_illegal_deny_threshold_cannot_become_blanket_deny() -> None:
    """拒绝线写错 → 退回问人，而不是静默变成"无条件拒"（拒了就没法挽回）。"""
    d = decide_approval("danger", {}, risk_policy=lambda a: RISK_CRITICAL,
                        fallback="sandbox", deny_threshold="typo")
    assert d.decision == DECISION_ASK


def test_policy_receives_args() -> None:
    seen = {}

    def policy(args):
        seen.update(args)
        return RISK_LOW

    decide_approval("danger", {"code": "print(1)"}, risk_policy=policy, fallback="sandbox")
    assert seen == {"code": "print(1)"}


# ======================================================================
# Task 3：节点闸门接入三档
# ======================================================================
_POLICY_STATE = {"session_id": "s1", "agent_id": "a1", "trace_id": "t1",
                 "role": "analyst", "task_id": "task-1"}


def _tool_def_sandboxed() -> ToolDef:
    """一个需审批、且声明在沙箱里跑的工具 —— AUTO 的唯一合法来源。"""
    return ToolDef(
        name="sandboxed",
        description="沙箱内执行的低风险工具",
        parameters={},
        requires_approval=True,
        run_in_sandbox=True,
        sandbox_task="x",
        rate_limit_per_min=1000,
    )


class _StubSandbox:
    """沙箱替身：本组用例只关心"沙箱在不在"，不会真执行到它。"""

    def execute(self, tool_def, args, context, sandbox_config):
        return True, "sandbox ok", {}


def _sandboxed_broker(pdp=None) -> ToolBroker:
    broker = ToolBroker(
        pdp=pdp, sandbox_executor=_StubSandbox(), circuit_breaker=False, cache=False,
    )
    broker.register(_tool_def_sandboxed(), lambda a, c: (True, "ok", {}))
    return broker


def _nodes(broker: ToolBroker):
    from harness.nodes import ReActNodes

    return ReActNodes(llm=None, broker=broker)


def _never(payload):
    raise AssertionError("不应弹审批卡")


def test_auto_decision_skips_interrupt(monkeypatch) -> None:
    """低风险 + 兜底可用 → 不弹卡，凭证带 via=risk_auto 与归因字段。"""
    broker = _sandboxed_broker()
    broker.register_risk_policy("sandboxed", lambda a: RISK_LOW)
    nodes = _nodes(broker)
    monkeypatch.setattr("harness.nodes.interrupt", _never)

    approved, _, cred = nodes._request_tool_approval("sandboxed", {}, _POLICY_STATE, {})
    assert approved is True
    assert cred["via"] == "risk_auto"
    assert cred["risk"] == RISK_LOW and cred["fallback"] == "sandbox"


def test_auto_downgraded_when_sandbox_actually_unavailable(monkeypatch) -> None:
    """配置说在沙箱里跑、但沙箱实际没连上 → 不算有兜底 → 仍要问。"""
    broker = _guarded_broker()                     # sandbox_executor=False → sandbox is None
    broker.register(_tool_def_sandboxed(), lambda a, c: (True, "ok", {}))
    broker.register_risk_policy("sandboxed", lambda a: RISK_LOW)
    nodes = _nodes(broker)
    called = []
    monkeypatch.setattr("harness.nodes.interrupt",
                        lambda payload: called.append(payload) or {"approved": True})

    nodes._request_tool_approval("sandboxed", {}, _POLICY_STATE, {})
    assert len(called) == 1, "沙箱不可用时必须降级为按需审批"


class _DeadSandbox:
    """配置里开着、但**服务端是死的**沙箱：有探测口，探测结论是"不可用"。"""

    def __init__(self) -> None:
        self.probes = 0

    def execute(self, tool_def, args, context, sandbox_config):
        return False, "sandbox down", {}

    def available(self):
        self.probes += 1
        return False, "OpenSandbox 服务端不可达"


def test_fallback_consults_the_sandbox_not_the_config() -> None:
    """"非空"只能说明配置开着 —— 兜底结论必须来自**运行时**探测。"""
    broker = ToolBroker(sandbox_executor=_DeadSandbox(), circuit_breaker=False, cache=False)
    broker.register(_tool_def_sandboxed(), lambda a, c: (True, "ok", {}))
    assert broker.approval_fallback("sandboxed") is None


def test_configured_but_dead_sandbox_is_not_a_fallback(monkeypatch) -> None:
    """Review Focus 5 的真身：沙箱开着但服务死了 → 不算兜底 → 降级为按需审批。

    这条是**补**的：原先那条用例拿 ``sandbox_executor=False``（沙箱直接是 None）当替身，
    恰好绕开了真实场景 —— "非空但不可用"。接上真探测口之后才照得出来。
    """
    broker = ToolBroker(sandbox_executor=_DeadSandbox(), circuit_breaker=False, cache=False)
    broker.register(_tool_def_sandboxed(), lambda a, c: (True, "ok", {}))
    broker.register_risk_policy("sandboxed", lambda a: RISK_LOW)
    nodes = _nodes(broker)
    called = []
    monkeypatch.setattr("harness.nodes.interrupt",
                        lambda payload: called.append(payload) or {"approved": True})

    nodes._request_tool_approval("sandboxed", {}, _POLICY_STATE, {})
    assert len(called) == 1, "沙箱服务是死的 → 没有兜底 → 必须降级为按需审批"


def test_sandbox_probe_is_cached() -> None:
    """探测带超时，不能每次调用都打 —— 缓存窗口内复用上次结论。"""
    from harness.sandbox.executor import SandboxExecutor

    probe = _DeadSandbox()
    executor = SandboxExecutor(client=probe)
    for _ in range(5):
        assert executor.available() == (False, "OpenSandbox 服务端不可达")
    assert probe.probes == 1, "同一窗口内不该重复探测"

    executor.invalidate_availability()
    executor.available()
    assert probe.probes == 2, "显式失效后应重新探测"


def test_no_policy_never_probes_the_sandbox(monkeypatch) -> None:
    """没有风险策略时结论必然是 ASK —— 别为它花一次带超时的健康检查。"""
    dead = _DeadSandbox()
    broker = ToolBroker(sandbox_executor=dead, circuit_breaker=False, cache=False)
    broker.register(_tool_def_sandboxed(), lambda a, c: (True, "ok", {}))
    nodes = _nodes(broker)
    monkeypatch.setattr("harness.nodes.interrupt", lambda payload: {"approved": True})

    nodes._request_tool_approval("sandboxed", {}, _POLICY_STATE, {})
    assert dead.probes == 0


def test_auto_never_overrides_pdp(monkeypatch) -> None:
    """命门：AUTO 档 + 被 PDP 拒绝的工具 → 仍必须被拒（与豁免同一条）。"""
    pdp = PDP(default_policy="deny")
    broker = _sandboxed_broker(pdp=pdp)            # 兜底齐备，本该 AUTO
    broker.register_risk_policy("sandboxed", lambda a: RISK_LOW)
    nodes = _nodes(broker)
    monkeypatch.setattr("harness.nodes.interrupt", _never)

    approved, observation, _ = nodes._request_tool_approval("sandboxed", {}, _POLICY_STATE, {})
    assert approved is False, "风险判定不得越过 PDP"
    assert "未获授权" in observation


def test_deny_decision_blocks_without_interrupt(monkeypatch) -> None:
    """红线档：不问，直接拒；凭证也不签发。"""
    broker = _sandboxed_broker()
    broker.register_risk_policy("sandboxed", lambda a: RISK_CRITICAL)
    nodes = _nodes(broker)
    monkeypatch.setattr("harness.nodes.interrupt", _never)

    approved, observation, cred = nodes._request_tool_approval(
        "sandboxed", {}, _POLICY_STATE, {})
    assert approved is False and cred is None
    assert "拒绝" in observation


def test_no_policy_still_interrupts(monkeypatch) -> None:
    """零回归：没注册策略的工具行为不变，人工批准不带自动放行标记。"""
    broker = _guarded_broker()
    nodes = _nodes(broker)
    calls = []
    monkeypatch.setattr("harness.nodes.interrupt",
                        lambda payload: calls.append(payload) or {"approved": True, "comment": "ok"})

    approved, _, cred = nodes._request_tool_approval("danger", {}, _POLICY_STATE, {})
    assert approved is True and len(calls) == 1
    assert "via" not in cred, "人工批准不带自动放行标记"


def test_illegal_threshold_config_fails_at_startup() -> None:
    """两条线写错要**启动即报错**，不能静默回落到某种行为。"""
    from pydantic import ValidationError

    from harness.config import ServerSettings

    for bad in ("typo", "MEDIUM", ""):
        with pytest.raises(ValidationError):
            ServerSettings(approval_deny_threshold=bad)
    with pytest.raises(ValidationError):
        ServerSettings(approval_threshold="very-high")

    tuned = ServerSettings(approval_threshold="high", approval_deny_threshold="high")
    assert tuned.approval_deny_threshold == "high", "部署方可以调低拒绝线"


# ======================================================================
# Task 4：无人审批的两个问题
# ======================================================================
def _service_with_approval_tool(monkeypatch, **server_overrides):
    """一个装配后会挂上『需审批工具』的服务（用于启动一致性检查）。"""
    from harness.config import settings
    from harness.server.service import HarnessService
    from tests._smoke_server import ApprovalLLM
    from tests.test_console_api import _memory_saver

    for key, value in server_overrides.items():
        monkeypatch.setattr(settings.server, key, value)
    return HarnessService(checkpointer=_memory_saver(), llm=ApprovalLLM(), auto_assemble=False)


def test_missing_approval_channel_fails_at_assembly(monkeypatch) -> None:
    """有需审批工具却没回答"这个部署有没有人审" → 直接装配失败。"""
    svc = _service_with_approval_tool(monkeypatch, approval_channel=None)
    with pytest.raises(RuntimeError, match="SERVER_APPROVAL_CHANNEL"):
        svc.assemble()


def test_channel_none_auto_approves_approval_tools(monkeypatch) -> None:
    """无人值守模式（approval_channel='none'）：需审批工具自动批准，装配成功。

    P0-1 修复后，'none' 是合法的非交互部署模式——requires_approval 工具
    在运行时被自动批准（记审计事件，via='unattended_auto'），不再启动即报错。
    """
    svc = _service_with_approval_tool(monkeypatch, approval_channel="none")
    assembled = svc.assemble()
    assert assembled is not None
    # 需审批工具仍被注册（没有被静默剔除）
    assert any(t.requires_approval for t in svc.broker.list_tools())


def test_channel_http_assembles_normally(monkeypatch) -> None:
    svc = _service_with_approval_tool(monkeypatch, approval_channel="http")
    assert svc.assemble() is not None


def test_illegal_approval_channel_fails_fast() -> None:
    from pydantic import ValidationError

    from harness.config import ServerSettings

    with pytest.raises(ValidationError):
        ServerSettings(approval_channel="yes")


def test_illegal_unattended_mode_fails_at_startup() -> None:
    """配置写错必须启动即报错，不能静默回落到某种行为。"""
    from pydantic import ValidationError

    from harness.config import ServerSettings

    with pytest.raises(ValidationError):
        ServerSettings(approval_unattended="never")


def _sweep_scenario(expires_at: str):
    """装配一个 service，把待审批项固定为一条 expires_at 给定的记录。"""
    import asyncio
    from harness.server.service import HarnessService
    from tests._smoke_server import ApprovalLLM
    from tests.test_console_api import _memory_saver

    async def scenario():
        svc = HarnessService(checkpointer=_memory_saver(), llm=ApprovalLLM(),
                             auto_assemble=False)
        await svc._ensure_ready()
        svc.assemble()
        calls = []

        async def _list_tasks(limit=100):
            return [{"thread_id": "t1", "status": "awaiting_approval"}]

        async def _get_status(tid):
            return {"thread_id": tid, "status": "awaiting_approval",
                    "pending_approvals": [{"interrupt_id": "i1",
                                           "payload": {"expires_at": expires_at, "tool": "danger"}}]}

        async def _submit(tid, approved, comment="", **kw):
            calls.append((tid, approved, comment))
            return {"thread_id": tid, "status": "running", "goal": "g", "pending_approvals": []}

        svc.list_tasks, svc.get_status, svc.submit_approval = _list_tasks, _get_status, _submit
        return await svc.sweep_expired_approvals(), calls

    return asyncio.run(scenario())


def test_sweep_auto_rejects_expired_approval() -> None:
    """挂账 #16：没人处理的审批，超时后必须**由系统自己**驳回并推进。"""
    from datetime import datetime, timedelta

    past = (datetime.now() - timedelta(seconds=1)).isoformat()
    handled, calls = _sweep_scenario(past)

    assert handled == 1
    assert calls and calls[0][0] == "t1" and calls[0][1] is False, "必须自动驳回，不是批准"
    assert "超时" in calls[0][2]


def test_sweep_leaves_unexpired_alone() -> None:
    """没到期的不能动 —— 否则人工审批就被系统抢答了。"""
    from datetime import datetime, timedelta

    future = (datetime.now() + timedelta(hours=1)).isoformat()
    handled, calls = _sweep_scenario(future)

    assert handled == 0
    assert calls == []


@pytest.fixture
def bus():
    fresh = EventBus()
    reset_event_bus(fresh)
    yield fresh
    reset_event_bus(None)


def test_auto_emits_guard_event_with_attribution(bus, monkeypatch) -> None:
    """AUTO 是静默放宽，必须可见：事件带 risk / policy / fallback。"""
    bus.bind("s1", "t1")
    broker = _sandboxed_broker()
    broker.register_risk_policy("sandboxed", lambda a: RISK_LOW)
    nodes = _nodes(broker)
    monkeypatch.setattr("harness.nodes.interrupt", _never)
    sub = bus.subscribe("s1")

    nodes._request_tool_approval("sandboxed", {}, _POLICY_STATE, {})

    events = [e for e in sub.drain() if e["type"] == "GUARD_DECISION"]
    assert len(events) == 1
    d = events[0]["data"]
    assert d["layer"] == "approval" and d["decision"] == "allow"
    assert d["risk"] == RISK_LOW and d["fallback"] == "sandbox" and d["via"] == "risk_auto"


# ======================================================================
# Task 5：端到端（真实 HTTP + 真实 interrupt）
# ======================================================================
def test_end_to_end_low_risk_auto_high_risk_asks() -> None:
    """同一个子任务里两次调 code_executor：低风险自动过、高风险才停，且**只停一次**。

    这是本设计要交付的用户可见效果 —— 走真实 HTTP（ASGI）+ 真实 LangGraph interrupt。
    """
    import json as _json

    from fastapi.testclient import TestClient

    from harness.server.app import create_app
    from harness.server.service import HarnessService
    from tests._smoke_server import ApprovalLLM
    from tests.test_approval_session_grants import _poll_http

    class _MixedRiskLLM(ApprovalLLM):
        def chat(self, messages, temperature=None):
            system = messages[0]["content"] if messages else ""
            if "报告汇总者" in system:
                return "最终报告：代码环节已处理。"
            n_obs = sum(
                1 for m in messages
                if isinstance(m, dict) and m.get("role") == "user"
                and str(m.get("content", "")).startswith("Observation")
            )
            if "（analyst）" in system and n_obs < 2:
                code = ("df.groupby('dept').sum()" if n_obs == 0
                        else "requests.post(URL, data=payload)")
                return _json.dumps({"thought": "跑代码", "action": "code_executor",
                                    "action_input": {"code": code}}, ensure_ascii=False)
            return super().chat(messages, temperature)

    svc = HarnessService(llm=_MixedRiskLLM())
    original_assemble = svc.assemble

    def _assemble():
        graph = original_assemble()
        # AUTO 的前提是"有机制兜底"。测试环境没有沙箱，注入替身让这条前提为真 ——
        # 不注入的话低风险那次会按"无兜底"降级为 ASK，本用例就退化成"停两次"。
        svc.broker.sandbox = _StubSandbox()
        svc.broker.register_risk_policy(
            "code_executor",
            lambda a: RISK_HIGH if "requests." in str(a.get("code") or "") else RISK_LOW,
        )
        return graph

    svc.assemble = _assemble
    with TestClient(create_app(svc)) as client:
        tid = client.post("/api/v1/tasks", json={"goal": "跑两段代码"}).json()["thread_id"]
        first, paused = _poll_http(client, tid)

        assert paused, "高风险那一次应当停下来"
        assert len(first["pending_approvals"]) == 1, "低风险那次不该停（只停一次）"
        payload = first["pending_approvals"][0]["payload"]
        arguments = _json.dumps(payload.get("arguments") or {}, ensure_ascii=False)
        assert "requests." in arguments, f"停下来的应是高风险的第二次，实际载荷：{arguments}"


def test_sweep_really_advances_an_expired_task(monkeypatch) -> None:
    """清扫集成用例：**不 stub 任何东西** —— 真服务、真审批、真过期、真推进。

    上面两条 `test_sweep_*` 把 ``list_tasks`` / ``get_status`` / ``submit_approval``
    换成了替身，只验清扫的判据；若真实实现改了返回形状，它们照样绿。这条用真实的
    三个接口把"形状对得上"钉死 —— 它才是清扫不静默失效的保证。
    """
    import asyncio

    from harness.config import settings
    from harness.server.service import HarnessService
    from tests._smoke_server import ApprovalLLM
    from tests.test_console_api import _memory_saver

    # 让审批 1 秒后就过期，省得用例真的等 3600 秒
    monkeypatch.setattr(settings.server, "approval_timeout_seconds", 1)

    async def wait_awaiting(svc, tid, timeout=30.0):
        deadline = asyncio.get_running_loop().time() + timeout
        while asyncio.get_running_loop().time() < deadline:
            st = await svc.get_status(tid)
            if st and st.get("status") == "awaiting_approval":
                return st
            await asyncio.sleep(0.05)
        raise AssertionError("任务没有进入待审批状态")

    async def scenario():
        svc = HarnessService(checkpointer=_memory_saver(), llm=ApprovalLLM(),
                             auto_assemble=False)
        await svc._ensure_ready()
        svc.assemble()

        tid = await svc.create_task("运行代码")
        status = await wait_awaiting(svc, tid)

        payload = status["pending_approvals"][0]["payload"]
        assert payload["tool"] == "code_executor"
        assert payload.get("expires_at"), "审批卡片必须带过期时刻，否则清扫无从判定"

        while not svc._is_approval_expired(payload["expires_at"]):
            await asyncio.sleep(0.1)

        assert await svc.sweep_expired_approvals() == 1

        # 驳回后的 resume 走后台任务，给它一点时间把图推走 —— 但**必须**推走。
        deadline = asyncio.get_running_loop().time() + 15
        while asyncio.get_running_loop().time() < deadline:
            after = await svc.get_status(tid)
            if after["status"] != "awaiting_approval":
                break
            await asyncio.sleep(0.1)
        else:
            raise AssertionError("清扫报了处理 1 条，但任务仍停在 awaiting_approval")

    asyncio.run(scenario())


# ======================================================================
# 领域包第一次真的用上它（data_analysis 声明 code_executor 的风险）
# ======================================================================
def test_domain_package_classifies_code_risk() -> None:
    """领域包自己回答"多危险"：纯计算 low，出网/装包/起子进程 high。"""
    from packages.data_analysis.tools.code_executor import risk_of_code

    assert risk_of_code({"code": "df.groupby('dept').sum()"}) == RISK_LOW
    assert risk_of_code({"code": "import requests; requests.get(URL)"}) == RISK_HIGH
    assert risk_of_code({"code": "subprocess.run(['ls'])"}) == RISK_HIGH
    assert risk_of_code({}) == RISK_LOW


def test_mounted_package_actually_registers_the_policy() -> None:
    """策略要真的挂到运行链路上 —— 不是只在包代码里躺着（"已接入"三档口径）。"""
    import asyncio

    from harness.server.service import HarnessService
    from tests._smoke_server import ApprovalLLM
    from tests.test_console_api import _memory_saver

    async def scenario():
        svc = HarnessService(checkpointer=_memory_saver(), llm=ApprovalLLM(),
                             auto_assemble=False)
        await svc._ensure_ready()
        svc.assemble()
        return svc.broker

    broker = asyncio.run(scenario())
    assert broker.risk_policy_for("code_executor") is not None, "挂了包却没注册风险策略"
