"""tests.test_e2e_data_analysis_demo —— 端到端集成测试：脚本化 LLM 驱动完整数据分析链路。

本测试守护 Bug #2（analyst 子图不收尾）不复发：
  - 真实装配 Plan-and-Execute 图（含 checkpointer / 质量门 / 上下文管理）
  - 脚本化 LLM 回放正确决策，不依赖外部 API
  - 断言全部任务 completed、最终报告非空、子 Agent 结果 success=True

与单元测试的区别：不 mock 任何框架组件，只替换 LLM 决策层。
运行：.venv\\Scripts\\python.exe -m pytest tests/test_e2e_data_analysis_demo.py -v
"""

from __future__ import annotations

import pytest

from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from harness.agents.registry import AgentRegistry
from harness.audit import get_audit_logger
from harness.context import ContextManager
from harness.orchestrator import build_plan_execute_graph, make_plan_execute_state
from harness.planning import QualityGate, TaskPlanner, TaskStore
from harness.tool_broker import ToolBroker
from harness.vfs import VirtualFileSystem

from packages.data_analysis.agents import build_agents
from packages.data_analysis.offline import (
    CLEAN_STEM,
    RAW_FILE,
    ScriptedAnalysisLLM,
    make_dirty_data,
)
from packages.data_analysis.tools import register_builtin_tools

DEFAULT_GOAL = f"对销售数据 {RAW_FILE} 做端到端分析：体检 → 清洗 → EDA → 出图 → 报告"


def _build_graph():
    """与 examples/data_analysis_demo.py 完全一致的装配，供测试复用。"""
    broker = ToolBroker(audit_logger=get_audit_logger())
    register_builtin_tools(broker)

    registry = AgentRegistry()
    for _def in build_agents().values():
        registry.register(_def)

    store = TaskStore(backend="memory")
    # 直接构造脚本化 LLM，不依赖环境变量（避免 CI 中配置了真实 key 时走网络）
    llm = ScriptedAnalysisLLM(RAW_FILE, CLEAN_STEM)
    gate = QualityGate(llm=llm, use_critic=False)
    planner = TaskPlanner(
        llm,
        broker=broker,
        available_agents=registry.names(),
        agent_descriptions={d.name: d.description for d in registry.list_defs()},
    )

    from harness.config import settings

    context_manager = ContextManager(vfs=VirtualFileSystem())
    checkpointer = MemorySaver()
    graph = build_plan_execute_graph(
        llm, broker, planner=planner, store=store,
        registry=registry, gate=gate, max_replans=2,
        tool_mode=settings.runtime.agent_tool_mode,
        context_manager=context_manager,
        checkpointer=checkpointer,
    )
    return graph


def _run_with_auto_approve(graph, goal: str):
    """运行图，非交互模式下自动批准所有工具审批中断。"""
    run_config = {
        "recursion_limit": 200,
        "configurable": {"thread_id": "e2e-test-thread"},
    }
    state = graph.invoke(make_plan_execute_state(goal), config=run_config)

    _auto_approve = {
        "approved": True,
        "comment": "测试自动批准",
        "approver": "e2e-test",
        "approver_role": "system",
    }
    _cycle = 0
    while state.get("__interrupt__"):
        _cycle += 1
        state = graph.invoke(Command(resume=_auto_approve), config=run_config)
        if _cycle >= 50:
            pytest.fail("审批循环超过 50 次，图未正常收尾")
    return state


class TestEndToEndDataAnalysisDemo:
    """脚本化 LLM 端到端完整链路测试。"""

    def test_full_pipeline_all_tasks_completed(self):
        """完整流水线：5 个任务全部 completed，总状态 finished。"""
        make_dirty_data()
        graph = _build_graph()
        state = _run_with_auto_approve(graph, DEFAULT_GOAL)

        assert state["status"] == "finished", (
            f"总状态应为 finished，实际为 {state['status']}"
        )

        plan = state["plan"]
        task_statuses = [t.status.value for t in plan.tasks]
        assert all(s == "completed" for s in task_statuses), (
            f"存在未完成任务: {list(zip([t.title for t in plan.tasks], task_statuses))}"
        )
        assert len(plan.tasks) >= 5, f"计划任务数过少: {len(plan.tasks)}"

    def test_final_answer_non_empty(self):
        """最终报告（synthesize）非 None 且非空字符串。"""
        make_dirty_data()
        graph = _build_graph()
        state = _run_with_auto_approve(graph, DEFAULT_GOAL)

        final = state.get("final_answer")
        assert final is not None, "final_answer 为 None"
        assert isinstance(final, str), f"final_answer 类型应为 str，实际为 {type(final)}"
        assert len(final.strip()) > 100, (
            f"final_answer 过短（{len(final)} 字符），可能不是完整报告"
        )
        # 报告应包含各环节的关键词
        assert "体检" in final or "inspection" in final.lower() or "数据" in final
        assert "清洗" in final or "clean" in final.lower()
        assert "EDA" in final or "分析" in final

    def test_sub_agent_results_all_success(self):
        """所有子 Agent 执行结果 success=True，且有结论。"""
        make_dirty_data()
        graph = _build_graph()
        state = _run_with_auto_approve(graph, DEFAULT_GOAL)

        sub_results = state.get("sub_results", [])
        assert len(sub_results) >= 5, f"子 Agent 结果数过少: {len(sub_results)}"

        for r in sub_results:
            assert r.success is True, (
                f"子 Agent {r.sub_agent_name} 失败: {r.conclusion[:200]}"
            )
            assert r.steps_taken >= 1, (
                f"子 Agent {r.sub_agent_name} 步数为 0"
            )
            assert r.conclusion and len(r.conclusion.strip()) > 0, (
                f"子 Agent {r.sub_agent_name} 无结论"
            )

    def test_no_infinite_interrupt_loop(self):
        """回归测试：图能正常收尾，不会卡在审批中断无限循环。

        Bug #2 的核心症状：analyst 子图不产出 final_answer，
        导致顶层图永远等不到收尾。本测试断言 _run_with_auto_approve
        能在 50 次审批内返回（脚本化 LLM 不应触发任何审批）。
        """
        make_dirty_data()
        graph = _build_graph()
        # 如果图卡住，_run_with_auto_approve 会在 50 次后 pytest.fail
        state = _run_with_auto_approve(graph, DEFAULT_GOAL)
        assert state["status"] == "finished"

    def test_plan_version_stable(self):
        """脚本化 LLM 不应触发重规划：计划版本始终为 v1。

        重规划（replan）意味着质量门判定某个任务失败。
        脚本化 LLM 回放正确决策，所有任务应一次通过。
        """
        make_dirty_data()
        graph = _build_graph()
        state = _run_with_auto_approve(graph, DEFAULT_GOAL)

        plan = state["plan"]
        assert plan.version == 1, (
            f"计划版本应为 v1（无重规划），实际为 v{plan.version}。"
            f"重规划意味着质量门判定失败，脚本化 LLM 不应出现此情况。"
        )
