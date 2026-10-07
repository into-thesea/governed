"""tests._smoke_server —— FastAPI 服务化端到端冒烟（无需 API Key，离线 Mock）。

覆盖：
1. 健康检查；
2. 主流程：创建任务 → 轮询 → finished → 最终报告；
3. 审批中断：高风险 code_executor 触发 interrupt → awaiting_approval →
   审批 payload 正确 → reject 恢复；
4. SSE 流式订阅（收到 open/status/done）；
5. 404（任务不存在）与 409（无待审批项却提交）。

运行（项目根）：
    .venv\\Scripts\\python.exe -m tests._smoke_server
"""

from __future__ import annotations

import json
import time

from fastapi.testclient import TestClient

from harness.server.app import create_app
from harness.server.service import HarnessService

# 冒烟测试显式注入脚本化 Mock LLM，确保离线 / 快速 / 确定性 —— 即使环境配了
# 真实 API Key 也不走网络。service 的"有 Key 用真实 LLM"是产品行为，另行验证。
import os as _os

from examples.data_analysis_demo import (
    RAW_FILE as _RAW, ScriptedAnalysisLLM, make_dirty_data,
)
from packages.data_analysis.tools.common import workspace_dir as _wsd

if not _os.path.exists(_os.path.join(_wsd({}), _RAW)):
    make_dirty_data()


def _offline_service() -> HarnessService:
    return HarnessService(llm=ScriptedAnalysisLLM(_RAW, "sales_cleaned"))


TERMINAL = ("finished", "failed")


def _poll(client: TestClient, thread_id: str, until: tuple[str, ...], timeout: float = 90) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = client.get(f"/api/v1/tasks/{thread_id}")
        state = r.json()
        if state["status"] in until:
            return state
        time.sleep(0.5)
    raise AssertionError(f"轮询超时，未进入 {until}")


# ----------------------------------------------------------------------
# 用于审批测试的最小脚本 LLM：规划单个 analyst 任务，analyst 调 code_executor
# ----------------------------------------------------------------------
class ApprovalLLM:
    def __init__(self) -> None:
        self.plan = {"tasks": [{
            "title": "运行代码", "description": "在沙箱运行一段代码",
            "assigned_to": "analyst", "depends_on": [],
            "acceptance_criteria": ["代码已运行"], "expected_artifacts": [],
        }]}

    def chat_json(self, messages):
        system = messages[0]["content"]
        if "质量门裁判" in system:
            return {"passed": True, "reason": "ok", "needs_human": False}
        return self.plan

    def chat(self, messages, temperature=None):
        system = messages[0]["content"] if messages else ""
        if "报告汇总者" in system:
            return "最终报告：代码环节已处理。"

        n_obs = sum(
            1 for m in messages
            if isinstance(m, dict) and m.get("role") == "user"
            and str(m.get("content", "")).startswith("Observation")
        )
        # 角色标记：analyst 的子 Agent 提示词里带"（analyst）"
        if "（analyst）" in system and n_obs == 0:
            # 故意用**高风险**代码（`socket` 是 _RISKY_MARKERS 之一）：风险高于阈值时
            # **任何环境**都要问人（见 harness/approval_policy.py「风险 > threshold → ask」，
            # 与兜底机制无关）。曾经这里用纯计算 `print('hi')`（low）——low 在**沙箱真可用**
            # 时会被自动放行，于是"停下来等审批"这一前提会随机器上沙箱起没起而翻转，
            # 让 test_server_auth / test_console_api / test_guard_events 等一批用例
            # 只在沙箱是死的降级环境下才绿。换成高风险后与兜底无关，两个环境都一致。
            # 代码本身无害（取主机名，不出网），万一被批准后真在沙箱里跑也无副作用。
            return json.dumps(
                {"thought": "运行代码", "action": "code_executor",
                 "action_input": {"code": "import socket; print(socket.gethostname())"}},
                ensure_ascii=False,
            )

        last = ""
        for m in reversed(messages):
            if (isinstance(m, dict) and m.get("role") == "user"
                    and str(m.get("content", "")).startswith("Observation")):
                last = str(m["content"])
                break
        return json.dumps({"final_answer": last[:500]}, ensure_ascii=False)


# ----------------------------------------------------------------------
# 1 & 2：健康检查 + 主流程
# ----------------------------------------------------------------------
def test_main_flow() -> None:
    service = _offline_service()
    app = create_app(service)
    with TestClient(app) as client:
        h = client.get("/health")
        assert h.status_code == 200 and h.json()["status"] == "ok"
        print("[1] 健康检查 ok")

        r = client.post("/api/v1/tasks", json={"goal": "对销售数据做端到端分析"})
        assert r.status_code == 200
        thread_id = r.json()["thread_id"]
        print("[2] 任务已创建：", thread_id)

        state = _poll(client, thread_id, TERMINAL)
        assert state["status"] == "finished", f"主流程未完成：{state['status']} {state.get('error')}"
        assert state["final_answer"], "缺少最终报告"
        print("[3] 主流程 finished，最终报告长度：", len(state["final_answer"]))

        # 5b：对已结束任务提交审批 → 409
        r = client.post(f"/api/v1/tasks/{thread_id}/approval",
                        json={"approved": True, "comment": ""})
        assert r.status_code == 409, f"expected 409, got {r.status_code}"
        print("[4] 无待审批项提交 → 409 ok")


# ----------------------------------------------------------------------
# 3：审批中断 → reject 恢复
# ----------------------------------------------------------------------
def test_approval_reject() -> None:
    service = HarnessService(llm=ApprovalLLM())
    app = create_app(service)
    with TestClient(app) as client:
        r = client.post("/api/v1/tasks", json={"goal": "运行一段代码"})
        thread_id = r.json()["thread_id"]

        state = _poll(client, thread_id, ("awaiting_approval",) + TERMINAL)
        assert state["status"] == "awaiting_approval", f"未进入审批：{state['status']}"
        assert state["pending_approvals"], "缺少待审批项"

        payload = state["pending_approvals"][0]["payload"]
        assert payload["tool"] == "code_executor", payload
        assert payload["run_in_sandbox"] is True
        print("[5] 触发工具审批 interrupt ok：", json.dumps(payload, ensure_ascii=False)[:160])

        # 列出待审批项
        r = client.get(f"/api/v1/tasks/{thread_id}/approvals")
        assert r.status_code == 200 and len(r.json()) == 1
        print("[6] 待审批项列表 ok")

        # reject 恢复
        r = client.post(f"/api/v1/tasks/{thread_id}/approval",
                        json={"approved": False, "comment": "不允许该操作"})
        assert r.status_code == 200
        state = _poll(client, thread_id, TERMINAL)
        assert state["status"] in TERMINAL
        print("[7] reject 恢复后任务收尾：", state["status"])


# ----------------------------------------------------------------------
# 4：SSE 流式订阅
# ----------------------------------------------------------------------
def test_sse() -> None:
    service = _offline_service()
    app = create_app(service)
    with TestClient(app) as client:
        r = client.post("/api/v1/tasks", json={"goal": "对销售数据做端到端分析"})
        thread_id = r.json()["thread_id"]

        lines: list[str] = []
        with client.stream("GET", f"/api/v1/tasks/{thread_id}/stream") as resp:
            assert resp.status_code == 200
            for line in resp.iter_lines():
                lines.append(line)
                # 在 done 事件处结束（done 在 final 之后，确保读全事件序列）
                if line.strip() == "event: done":
                    break
                if len(lines) > 400:
                    break

        blob = "\n".join(lines)
        assert "event: open" in blob, "缺少 open 事件"
        assert "event: status" in blob, "缺少 status 事件"
        assert "event: done" in blob, "缺少 done 事件"

        # 真实进展必须真的上到线：这三类事件此前在界面上完全不可见
        assert "event: RUN_STARTED" in blob, "缺少根 Span 事件（事件层没接进运行链路）"
        assert "event: TOOL_CALL_START" in blob, "缺少工具调用事件"
        assert "event: SUBAGENT_STARTED" in blob, "缺少子 Agent 事件"
        # 事件 payload 要能支撑界面分组（工具名 / 子 Agent 名）
        assert '"tool":' in blob and '"agent":' in blob, "事件缺少归因字段"
        print("[8] SSE 流式订阅 ok（事件行数", len(lines), "）")


# ----------------------------------------------------------------------
# 5：404
# ----------------------------------------------------------------------
def test_not_found() -> None:
    service = _offline_service()
    app = create_app(service)
    with TestClient(app) as client:
        assert client.get("/api/v1/tasks/nonexistent").status_code == 404
        assert client.get("/api/v1/tasks/nonexistent/approvals").status_code == 404
        print("[9] 不存在任务 → 404 ok")


def _main() -> None:
    test_main_flow()
    test_approval_reject()
    test_sse()
    test_not_found()
    print("\n=== 服务化端到端冒烟全部通过 ===")


if __name__ == "__main__":
    _main()
