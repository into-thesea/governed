"""tests.test_approval_session_grants —— 会话内审批豁免（「本任务内总是允许 / 总是拒绝」）。

设计见 ``docs/审批会话豁免设计.md``。核心不变量：**豁免只跳过人工审批那一步** ——
PDP（``authorize``）与沙箱永不被豁免；豁免按 ``(session_id, tool)`` 存于 Broker，
随任务终态清除，进程重启即丢（安全方向的失败模式）。

审计新增了 ``event="approval_grant"`` 这种记录类型，本文件同时钉住"它不得污染既有
工具调用指标"—— 指标聚合必须把没有 ``event`` 字段的历史行仍按工具调用计入。
"""

from __future__ import annotations

import json
import time

import pytest

from harness.audit import AuditLogger
from harness.events import build_event_bus, reset_event_bus
from harness.pdp import PDP
from harness.nodes import ReActNodes
from harness.server.service import HarnessService
from tests.test_approval_gate import _guarded_broker


def test_aggregate_audit_counts_legacy_rows_without_event_field() -> None:
    """升级前写的审计行没有 event 字段，必须仍按"工具调用"计入。"""
    legacy = {
        "tool_name": "sql_query", "session_id": "s1",
        "result_ok": True, "pdp_decision": "allow", "duration_ms": 12,
    }
    agg = HarnessService._aggregate_audit([legacy])
    assert agg["total"] == 1
    assert agg["succeeded"] == 1
    assert agg["by_tool"]["sql_query"]["calls"] == 1


def test_aggregate_audit_skips_grant_records() -> None:
    """豁免记录（event="approval_grant"）不是工具调用，不得计入调用数与成败比。"""
    rows = [
        {"event": "approval_grant", "tool_name": "code_executor", "session_id": "s1"},
        {"event": "tool_call", "tool_name": "code_executor", "session_id": "s1",
         "result_ok": True, "pdp_decision": "allow", "duration_ms": 5},
    ]
    agg = HarnessService._aggregate_audit(rows)
    assert agg["total"] == 1, "授权记录被误算成一次工具调用"
    assert agg["succeeded"] + agg["failed"] == agg["total"]
    assert agg["by_tool"]["code_executor"]["calls"] == 1


def test_record_approval_grant_writes_grant_event(tmp_path) -> None:
    """授予豁免落一条独立类型的审计记录，且不带参数原文。"""
    audit = AuditLogger(local_dir=str(tmp_path), enabled=True)
    rid = audit.record_approval_grant(
        tool_name="code_executor", session_id="s1", effect="allow",
        applied=False, granted_by="管理员A", granted_role="admin",
        grant_id="aprgrant_x", request_id="aprreq_y",
    )
    assert rid
    rows = audit.query(session_id="s1")
    assert len(rows) == 1
    rec = rows[0]
    assert rec["event"] == "approval_grant"
    assert rec["tool_name"] == "code_executor"
    assert rec["session_id"] == "s1"
    assert "args" not in rec and "args_hash" in rec, "沿用只存哈希的纪律"


# ======================================================================
# Task 2：Broker 的会话授权表
# ======================================================================
def _broker_with_audit(tmp_path):
    broker = _guarded_broker()
    broker.audit = AuditLogger(local_dir=str(tmp_path), enabled=True)
    return broker


def test_grant_then_apply_returns_effect_and_records_both(tmp_path) -> None:
    reset_event_bus()
    broker = _broker_with_audit(tmp_path)
    bus = build_event_bus()
    bus.bind("s1", "t1")
    sub = bus.subscribe("s1")

    broker.grant_session_approval(
        "s1", "danger", "allow", granted_by="管理员A", granted_role="admin",
        comment="本任务内免问", request_id="aprreq_1", trace_id="t1", agent_id="a1",
    )
    granted_events = [e["type"] for e in sub.drain()]
    assert "GUARD_DECISION" in granted_events, "授予必须可见（挂账 #13 定位）"

    grant = broker.apply_session_grant("s1", "danger", trace_id="t1", agent_id="a1")
    assert grant is not None and grant["effect"] == "allow"
    assert grant["grant_id"].startswith("aprgrant_")
    use_events = sub.drain()
    assert [e["type"] for e in use_events] == ["GUARD_DECISION"]
    assert use_events[0]["data"]["via"] == "session_grant"
    assert use_events[0]["data"]["applied"] is True

    # audit.query 是"最新在前"，故这里只断言集合：授予与使用各留一条痕
    rows = broker.audit.query(session_id="s1")
    assert sorted(r["applied"] for r in rows) == [False, True]


def test_apply_returns_none_without_grant(tmp_path) -> None:
    reset_event_bus()
    broker = _broker_with_audit(tmp_path)
    assert broker.apply_session_grant("s1", "danger") is None
    assert broker.audit.query(session_id="s1") == []


def test_grant_is_scoped_to_session_and_tool(tmp_path) -> None:
    reset_event_bus()
    broker = _broker_with_audit(tmp_path)
    broker.grant_session_approval("s1", "danger", "allow")
    assert broker.apply_session_grant("s2", "danger") is None, "不得跨会话"
    assert broker.apply_session_grant("s1", "echo") is None, "不得跨工具"
    assert broker.apply_session_grant("s1", "danger") is not None


def test_regrant_overwrites_instead_of_appending(tmp_path) -> None:
    """同一会话同一工具重复授予：覆盖旧条目，不叠成两条。"""
    reset_event_bus()
    broker = _broker_with_audit(tmp_path)
    broker.grant_session_approval("s1", "danger", "allow")
    broker.grant_session_approval("s1", "danger", "deny")
    grants = broker.list_session_grants("s1")
    assert len(grants) == 1
    assert grants[0]["effect"] == "deny"


def test_invalid_effect_raises(tmp_path) -> None:
    reset_event_bus()
    broker = _broker_with_audit(tmp_path)
    with pytest.raises(ValueError):
        broker.grant_session_approval("s1", "danger", "maybe")


def test_clear_session_grants(tmp_path) -> None:
    reset_event_bus()
    broker = _broker_with_audit(tmp_path)
    broker.grant_session_approval("s1", "danger", "allow")
    broker.grant_session_approval("s2", "danger", "allow")
    assert broker.clear_session_grants("s1") == 1
    assert broker.list_session_grants("s1") == []
    assert broker.apply_session_grant("s1", "danger") is None
    assert broker.list_session_grants("s2"), "别的会话不受影响"


# ======================================================================
# Task 3：安全边界 —— 豁免不越过 PDP；子 Agent 只读
# ======================================================================
def test_grant_cannot_override_pdp_denial(tmp_path) -> None:
    """命门：给一个被 PDP 拒绝的工具授予 allow 豁免，调用仍必须被拒。"""
    reset_event_bus()
    pdp = PDP(default_policy="deny")          # 默认拒绝：danger 不在白名单里
    broker = _guarded_broker(pdp=pdp)
    broker.audit = AuditLogger(local_dir=str(tmp_path), enabled=True)

    broker.grant_session_approval("s1", "danger", "allow")
    ctx = {"role": "analyst", "session_id": "s1"}

    allowed, reason = broker.authorize("danger", ctx)
    assert allowed is False, "豁免不得越过 PDP 授权"
    assert "不允许" in reason

    ok, text, _ = broker.invoke("danger", {}, {**ctx, "approval": {
        "id": "apr_x", "approved": True, "tool": "danger"}})
    assert ok is False, "持有凭证 + 有豁免，也不能越过 PDP"


def test_scoped_broker_reads_and_grants_within_scope_only(tmp_path) -> None:
    """受限视图：读得到父任务的豁免；授予**必须**透传（节点层就在子图里跑），
    但只能授予白名单内的工具。"""
    reset_event_bus()
    from harness.tool_broker import ScopedBroker

    broker = _broker_with_audit(tmp_path)
    broker.grant_session_approval("s1", "danger", "allow")
    scoped = ScopedBroker(broker, ["danger"])

    assert scoped.apply_session_grant("s1", "danger") is not None
    assert scoped.list_session_grants("s1")[0]["effect"] == "allow"

    # 白名单内可授予（子图的 ReActNodes 靠它把人工决定落成豁免）
    scoped.grant_session_approval("s1", "danger", "deny")
    assert broker.list_session_grants("s1")[0]["effect"] == "deny"

    # 白名单外不得授予：受限视图不能给越界工具开豁免
    with pytest.raises(PermissionError):
        scoped.grant_session_approval("s1", "echo", "allow")
    assert not hasattr(scoped, "clear_session_grants"), "清理只在任务收尾（service 层）"


# ======================================================================
# Task 4：节点审批闸门
# ======================================================================
def _nodes(broker) -> ReActNodes:
    """只装配审批闸门所需的依赖（LLM / 中间件都不参与本组用例）。"""
    return ReActNodes(llm=None, broker=broker)


_STATE = {"session_id": "s1", "agent_id": "a1", "trace_id": "t1",
          "role": "analyst", "task_id": "task-1"}


def _interrupt_must_not_fire(payload):
    raise AssertionError("不应弹审批卡")


def test_parse_approval_returns_remember() -> None:
    nodes = _nodes(None)
    assert nodes._parse_approval({"approved": True, "comment": "ok"}) == (True, "ok", None)
    assert nodes._parse_approval(
        {"approved": True, "remember": "allow"}) == (True, "", "allow")
    assert nodes._parse_approval(
        {"approved": False, "remember": "deny"}) == (False, "", "deny")
    # 非法值一律当 None（fail closed：不认识的值不产生豁免）
    assert nodes._parse_approval(
        {"approved": True, "remember": "whatever"}) == (True, "", None)
    # 与 approved 不同向：不猜，直接不授予
    assert nodes._parse_approval(
        {"approved": True, "remember": "deny"}) == (True, "", None)
    assert nodes._parse_approval(
        {"approved": False, "remember": "allow"}) == (False, "", None)


def test_apply_grant_allow_skips_interrupt(monkeypatch, tmp_path) -> None:
    """有 allow 豁免：不触发 interrupt，凭证带 via 标记。"""
    reset_event_bus()
    broker = _broker_with_audit(tmp_path)
    broker.grant_session_approval("s1", "danger", "allow")
    nodes = _nodes(broker)
    monkeypatch.setattr("harness.nodes.interrupt", _interrupt_must_not_fire)

    approved, observation, credential = nodes._request_tool_approval("danger", {}, _STATE, {})
    assert approved is True and observation == ""
    assert credential["via"] == "session_grant"
    assert credential["tool"] == "danger" and credential["approved"] is True


def test_apply_grant_deny_skips_interrupt(monkeypatch, tmp_path) -> None:
    """有 deny 豁免：不触发 interrupt，直接回驳回；换参数也一样（豁免是工具级）。"""
    reset_event_bus()
    broker = _broker_with_audit(tmp_path)
    broker.grant_session_approval("s1", "danger", "deny")
    nodes = _nodes(broker)
    monkeypatch.setattr("harness.nodes.interrupt", _interrupt_must_not_fire)

    approved, observation, credential = nodes._request_tool_approval(
        "danger", {"code": "完全不同的代码"}, _STATE, {})
    assert approved is False and credential is None
    assert "豁免" in observation


def test_no_grant_still_interrupts(monkeypatch, tmp_path) -> None:
    """回归：没有豁免时行为与今天完全一致 —— 仍然 interrupt。"""
    reset_event_bus()
    broker = _broker_with_audit(tmp_path)
    nodes = _nodes(broker)
    calls = []

    def _fake_interrupt(payload):
        calls.append(payload)
        return {"approved": True, "comment": "ok"}

    monkeypatch.setattr("harness.nodes.interrupt", _fake_interrupt)
    approved, _, credential = nodes._request_tool_approval("danger", {}, _STATE, {})
    assert approved is True and len(calls) == 1
    assert credential["approved"] is True
    assert "via" not in credential, "人工批准不带豁免标记"


def test_remember_grants_for_subsequent_requests(monkeypatch, tmp_path) -> None:
    """本次人工选择"总是允许"后：本次照旧走 interrupt，后续请求不再弹卡。"""
    reset_event_bus()
    broker = _broker_with_audit(tmp_path)
    nodes = _nodes(broker)
    calls = []

    def _fake_interrupt(payload):
        calls.append(payload)
        return {"approved": True, "comment": "可以", "remember": "allow",
                "approver": "管理员A", "approver_role": "admin"}

    monkeypatch.setattr("harness.nodes.interrupt", _fake_interrupt)

    approved, _, _ = nodes._request_tool_approval("danger", {}, _STATE, {})
    assert approved is True and len(calls) == 1, "触发豁免的那次仍走完整审批"
    grants = broker.list_session_grants("s1")
    assert grants[0]["effect"] == "allow"
    assert grants[0]["granted_by"] == "管理员A", "授予人必须留痕（否则审计记不住是谁放宽的）"
    assert grants[0]["granted_role"] == "admin"

    monkeypatch.setattr("harness.nodes.interrupt", _interrupt_must_not_fire)
    approved2, _, cred2 = nodes._request_tool_approval("danger", {}, _STATE, {})
    assert approved2 is True and cred2["via"] == "session_grant"


def test_remember_deny_grants_deny(monkeypatch, tmp_path) -> None:
    """驳回时选"总是拒绝"：本次驳回，后续直接拒绝。"""
    reset_event_bus()
    broker = _broker_with_audit(tmp_path)
    nodes = _nodes(broker)
    monkeypatch.setattr("harness.nodes.interrupt", lambda payload: {
        "approved": False, "comment": "不行", "remember": "deny"})

    approved, observation, credential = nodes._request_tool_approval("danger", {}, _STATE, {})
    assert approved is False and credential is None
    assert "拒绝" in observation, "沿用既有驳回文案（本轮不改它的措辞）"
    assert broker.list_session_grants("s1")[0]["effect"] == "deny"


# ======================================================================
# Task 5：Service 层
# ======================================================================
import asyncio  # noqa: E402 - 本组用例才开始需要

from tests._smoke_server import ApprovalLLM  # noqa: E402
from tests.test_console_api import _memory_saver  # noqa: E402


def _service_with_pending(monkeypatch, tmp_path):
    """一个装配好、但图被替身接管的 service：待审批项固定为一次 danger 工具审批。"""
    svc = HarnessService(checkpointer=_memory_saver(), llm=ApprovalLLM(), auto_assemble=False)
    captured: dict = {}

    class _Graph:
        # 注意：必须是**普通函数**。若写成 async def，调用它只是造出协程、函数体不执行，
        # 捕获就落空了 —— 这里的目的是验载荷，不是真驱动图。
        def ainvoke(self, payload, config):
            captured["resume"] = payload.resume

            async def _noop():
                return {}

            return _noop()

        async def aget_state(self, config):
            return type("Snap", (), {"tasks": [], "values": {}})()

    async def _get_status(thread_id):
        return {"status": "awaiting_approval", "pending_approvals": [{
            "interrupt_id": "i1",
            "payload": {"type": "tool_approval", "tool": "danger",
                        "approval_request_id": "aprreq_1", "task_id": "task-1"},
        }]}

    graph = _Graph()

    async def _scenario(call):
        # auto_assemble=False 时 _ensure_ready 只置 _ready、**不装配**，必须显式装配
        svc.assemble()
        svc.graph = graph
        svc.get_status = _get_status
        svc._trace_id = lambda tid: asyncio.sleep(0, result=None)
        await call()
        return captured

    return svc, _scenario


def test_resume_payload_carries_remember_and_approver(monkeypatch, tmp_path) -> None:
    """remember 与审批人身份都必须进 resume 载荷。

    没有审批人，授予记录的 granted_by 就是空串，审计答不出"谁放宽的"。
    """
    _svc, scenario = _service_with_pending(monkeypatch, tmp_path)

    async def call():
        await _svc.submit_approval(
            "t1", True, "可以", approver_name="管理员A", approver_role="admin",
            remember="allow")

    captured = asyncio.run(scenario(call))
    assert captured["resume"] == {
        "approved": True, "comment": "可以", "remember": "allow",
        "approver": "管理员A", "approver_role": "admin",
    }


def test_remember_mismatched_with_approved_is_dropped() -> None:
    """自相矛盾或非法的组合一律不授予（不猜）。"""
    assert HarnessService._normalize_remember(True, "deny") is None
    assert HarnessService._normalize_remember(False, "allow") is None
    assert HarnessService._normalize_remember(True, "allow") == "allow"
    assert HarnessService._normalize_remember(False, "deny") == "deny"
    assert HarnessService._normalize_remember(True, "nonsense") is None
    assert HarnessService._normalize_remember(True, None) is None
    assert HarnessService._normalize_remember(True, "  ALLOW  ") == "allow"


def test_settle_clears_session_grants() -> None:
    """任务到达终态后，该会话的豁免必须被清掉（否则是又一处只增不减的表）。"""

    async def scenario() -> None:
        svc = HarnessService(checkpointer=_memory_saver(), llm=ApprovalLLM(),
                             auto_assemble=False)
        await svc._ensure_ready()
        svc.assemble()
        svc.broker.grant_session_approval("t1", "danger", "allow")
        svc.broker.grant_session_approval("t2", "danger", "allow")

        class _Snap:
            tasks: list = []

        async def _aget_state(config):
            return _Snap()

        svc.graph = type("G", (), {"aget_state": staticmethod(_aget_state)})()
        await svc._settle_after_run("t1", trace_id=None)
        assert svc.broker.list_session_grants("t1") == []
        assert svc.broker.list_session_grants("t2"), "别的会话不受影响"

    asyncio.run(scenario())


def test_status_exposes_session_grants_and_control_plane_counts() -> None:
    """状态里能读出本任务生效的豁免；管控面报进程内总数。"""

    async def scenario() -> None:
        svc = HarnessService(checkpointer=_memory_saver(), llm=ApprovalLLM(),
                             auto_assemble=False)
        await svc._ensure_ready()
        svc.assemble()
        svc.broker.grant_session_approval("t1", "danger", "allow")

        st = await svc.get_status("t1")
        assert st is None or "session_grants" in st      # 任务不存在时返回 None 是既有语义

        cp = await svc.control_plane()
        assert cp["approval"]["session_grants_active"] == 1

    asyncio.run(scenario())


# ======================================================================
# Task 6：HTTP 层
# ======================================================================
def test_http_approval_accepts_remember_field() -> None:
    """端点必须能吃下 remember 字段（否则控制台的两个新按钮 422 / 静默失效）。"""
    from harness.server.schemas import ApprovalRequest

    req = ApprovalRequest.model_validate(
        {"approved": True, "comment": "免问", "remember": "allow"})
    assert req.remember == "allow"

    default = ApprovalRequest.model_validate({"approved": False})
    assert default.remember is None


def test_task_status_schema_carries_session_grants() -> None:
    """`session_grants` 必须能被 HTTP 响应带出去。

    `get_status` 返回的 dict 会经 `TaskStatusResponse` 序列化 —— schema 里没有这个字段，
    pydantic 会把它丢掉，控制台就永远看不到"哪些工具已豁免"。
    """
    from harness.server.schemas import TaskStatusResponse

    resp = TaskStatusResponse(
        thread_id="t1", status="running", goal="g", pending_approvals=[],
        session_grants=[{"tool": "danger", "effect": "allow"}],
    )
    assert resp.session_grants == [{"tool": "danger", "effect": "allow"}]
    assert resp.model_dump()["session_grants"], "HTTP 响应里必须带得出去"


def test_http_approval_endpoint_forwards_remember() -> None:
    """端点把 remember 交给 service（否则按钮是哑的）。"""
    from fastapi.testclient import TestClient

    from harness.server.app import create_app

    svc = HarnessService(checkpointer=_memory_saver(), llm=ApprovalLLM(), auto_assemble=False)
    seen: dict = {}

    async def _submit(thread_id, approved, comment="", **kwargs):
        seen["thread_id"] = thread_id
        seen["approved"] = approved
        seen.update(kwargs)
        return {"thread_id": thread_id, "status": "running", "goal": "g",
                "pending_approvals": []}

    svc.submit_approval = _submit
    with TestClient(create_app(svc)) as client:
        r = client.post("/api/v1/tasks/x/approval",
                        json={"approved": True, "comment": "免问", "remember": "allow"})
    assert r.status_code == 200, r.text
    assert seen.get("remember") == "allow"


# ======================================================================
# Task 7：端到端 —— 第二次不再弹卡
# ======================================================================
def test_end_to_end_second_request_skips_interrupt(monkeypatch, tmp_path) -> None:
    """同任务内同一高危工具提请两次：第一次弹卡并授予 allow，第二次不再弹卡。"""
    reset_event_bus()
    broker = _broker_with_audit(tmp_path)
    nodes = _nodes(broker)
    interrupts: list[dict] = []

    def _fake_interrupt(payload):
        interrupts.append(payload)
        return {"approved": True, "comment": "免问", "remember": "allow",
                "approver": "管理员A", "approver_role": "admin"}

    monkeypatch.setattr("harness.nodes.interrupt", _fake_interrupt)
    state = {"session_id": "thread-1", "agent_id": "a1", "trace_id": "t1",
             "role": "analyst", "task_id": "task-1"}

    # 第一次：弹卡 + 授予
    ok1, _, cred1 = nodes._request_tool_approval("danger", {"code": "print(1)"}, state, {})
    assert ok1 is True and len(interrupts) == 1
    assert "via" not in cred1, "本次是人工批准，不是豁免放行"

    # 第二次：同一工具、不同参数 —— 豁免是工具级，仍然不再问
    ok2, _, cred2 = nodes._request_tool_approval("danger", {"code": "print(2)"}, state, {})
    assert ok2 is True
    assert len(interrupts) == 1, "第二次不应该再弹卡"
    assert cred2["via"] == "session_grant"

    # 凭证能真正解锁 Broker 的凭证闸门
    ok3, text, _ = broker.invoke("danger", {"code": "print(2)"}, {
        "role": "analyst", "session_id": "thread-1", "approval": cred2})
    assert ok3 is True, text


# ======================================================================
# Task 8：图级别端到端（真实 HTTP + 真实 LangGraph interrupt）
# ======================================================================
class _TwiceApprovalLLM(ApprovalLLM):
    """像 ApprovalLLM，但在同一个子任务里**连问两次** code_executor。

    既有 ApprovalLLM 每任务只问一次，覆盖不到"第二次不再弹卡"。
    """

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
            # 高风险代码（`socket` 属 _RISKY_MARKERS）：两次调用都**必须**问人，
            # 与"沙箱在不在"无关。若用纯计算（low），沙箱真可用时第一次就被自动放行，
            # 根本不会停下来 —— 这两条用例的前提就没了。
            return json.dumps(
                {"thought": "再跑一次代码", "action": "code_executor",
                 "action_input": {"code": f"import socket; print(socket.gethostname(), {n_obs})"}},
                ensure_ascii=False,
            )

        last = ""
        for m in reversed(messages):
            if (isinstance(m, dict) and m.get("role") == "user"
                    and str(m.get("content", "")).startswith("Observation")):
                last = str(m["content"])
                break
        return json.dumps({"final_answer": last[:500]}, ensure_ascii=False)


def _poll_http(client, tid: str, timeout: float = 90.0):
    """轮询任务状态，返回 (最终快照, 是否又停在了审批)。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        st = client.get(f"/api/v1/tasks/{tid}").json()
        if st["status"] == "awaiting_approval":
            return st, True
        if st["status"] in ("finished", "failed"):
            return st, False
        time.sleep(0.2)
    raise AssertionError("轮询超时")


def test_http_end_to_end_second_call_not_asked(monkeypatch, tmp_path) -> None:
    """整条链路：同一任务里 code_executor 被请两次 —— 授予"总是允许"后第二次不再弹卡。

    这条走真实 HTTP（TestClient / ASGI）+ 真实 LangGraph interrupt，是本功能最强的
    自动化验证。`_TwiceApprovalLLM` 发的是高风险代码，两次调用在任何环境下都必须问人，
    所以"第二次是否弹卡"只由豁免决定 —— 与机器上沙箱起没起无关。
    """
    from fastapi.testclient import TestClient

    from harness.server.app import create_app

    # 不注入 checkpointer：让首次请求在应用循环内装配（与既有服务用例一致）
    svc = HarnessService(llm=_TwiceApprovalLLM())
    with TestClient(create_app(svc)) as client:
        tid = client.post("/api/v1/tasks", json={"goal": "运行代码"}).json()["thread_id"]

        first, paused = _poll_http(client, tid)
        assert paused, f"第一次调用应停在审批，实际 {first['status']}"
        payload = first["pending_approvals"][0]["payload"]
        assert payload["tool"] == "code_executor"

        r = client.post(f"/api/v1/tasks/{tid}/approval",
                        json={"approved": True, "comment": "本任务内免问", "remember": "allow"})
        assert r.status_code == 200, r.text

        # resume 是异步的：先等状态**离开**暂停态，否则会把"还没恢复"误读成"又弹了一次卡"
        deadline = time.time() + 30
        while time.time() < deadline:
            if client.get(f"/api/v1/tasks/{tid}").json()["status"] != "awaiting_approval":
                break
            time.sleep(0.1)

        final, paused_again = _poll_http(client, tid)
        assert not paused_again, "授予豁免后，第二次调用仍然弹了审批卡"
        assert final["status"] == "finished", final

    # 审计留痕：一次"授予" + 至少一次"命中使用"
    rows = svc.broker.audit.query(session_id=tid)
    granted = [r for r in rows if r.get("event") == "approval_grant" and not r.get("applied")]
    used = [r for r in rows if r.get("event") == "approval_grant" and r.get("applied")]
    assert granted and granted[0]["tool_name"] == "code_executor"
    assert granted[0]["grant_effect"] == "allow"
    assert used, "第二次调用是靠豁免放行的，必须留下一条命中的痕"


def test_http_end_to_end_without_effective_grant_asks_again(monkeypatch) -> None:
    """反向对照：豁免**没生效**时，第二次仍然弹卡。

    没有这条，"第二次不弹卡"的正向断言可能只是因为流程根本没走到第二次调用 ——
    它证明这条链路上确实存在第二次提请。
    """
    from fastapi.testclient import TestClient

    from harness.server.app import create_app
    from harness.tool_broker import ToolBroker

    # 让豁免永远读不命中：等价于"授予没生效"
    monkeypatch.setattr(ToolBroker, "apply_session_grant",
                        lambda self, session_id, tool_name, **kw: None)

    svc = HarnessService(llm=_TwiceApprovalLLM())
    with TestClient(create_app(svc)) as client:
        tid = client.post("/api/v1/tasks", json={"goal": "运行代码"}).json()["thread_id"]
        _first, paused = _poll_http(client, tid)
        assert paused, "对照组本身就应当先停一次审批"

        client.post(f"/api/v1/tasks/{tid}/approval",
                    json={"approved": True, "comment": "免问", "remember": "allow"})

        deadline = time.time() + 30
        while time.time() < deadline:
            if client.get(f"/api/v1/tasks/{tid}").json()["status"] != "awaiting_approval":
                break
            time.sleep(0.1)

        _final, paused_again = _poll_http(client, tid)
        assert paused_again, "豁免未生效时应当再次弹卡 —— 否则正向用例是空转的"


# ======================================================================
# 整支复核后的修复：Important 2 —— 非字符串 remember 不得挡在 422
# ======================================================================
def test_http_approval_normalizes_non_string_remember() -> None:
    """非字符串的 remember 必须被归一为 None，而不是让 pydantic 在入口抛 422。

    规格（Global Constraints）明确要求：非法值一律当 None **而不报错** ——
    报错会把整张审批卡死住，这恰恰是"手工客户端传了脏值"时最糟的结果。
    """
    from harness.server.schemas import ApprovalRequest

    for bad in (True, 123, 1.5, ["allow"], {"x": 1}):
        req = ApprovalRequest.model_validate({"approved": True, "remember": bad})
        assert req.remember is None, f"{bad!r} 应被归一为 None"

    # 合法值原样保留
    assert ApprovalRequest.model_validate(
        {"approved": True, "remember": "allow"}).remember == "allow"
    assert ApprovalRequest.model_validate({"approved": True}).remember is None


def test_http_approval_endpoint_tolerates_non_string_remember() -> None:
    """端点层：脏 remember 不能让审批返回 422（审批必须仍然提交成功）。"""
    from fastapi.testclient import TestClient

    from harness.server.app import create_app

    svc = HarnessService(checkpointer=_memory_saver(), llm=ApprovalLLM(), auto_assemble=False)
    seen: dict = {}

    async def _submit(thread_id, approved, comment="", **kwargs):
        seen.update(kwargs)
        return {"thread_id": thread_id, "status": "running", "goal": "g",
                "pending_approvals": []}

    svc.submit_approval = _submit
    with TestClient(create_app(svc)) as client:
        r = client.post("/api/v1/tasks/x/approval",
                        json={"approved": True, "comment": "c", "remember": True})
    assert r.status_code == 200, f"脏 remember 把审批挡在门外了：{r.status_code} {r.text}"
    assert seen.get("remember") is None
