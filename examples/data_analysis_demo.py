"""examples.data_analysis_demo —— 端到端示例：Mock LLM 驱动完整数据分析链路。

与 mock_llm_demo.py（只跑单工具 ReAct）不同，本示例验证【除大模型决策外的整条工程链路】：
    Plan-and-Execute 顶层编排
      → data-explorer 探查清洗 → analyst 分析出图 → reporter 汇总成稿
    全程经过：ScopedBroker 子 Agent 白名单、PDP 权限、参数校验、限流、质量门，
    6 个真实数据分析工具被真实调用，CSV / Markdown / PNG 真实落盘到 VFS。

唯一"假"的是 ScriptedAnalysisLLM：它实现与真实 LLMClient 相同的 chat / chat_json，
按"当前子 Agent 角色标记 + 已收到的 Observation 条数"回放写死的正确决策。
未来把它替换成真实 LLMClient（填好 API Key），同一张图无需改动即可变成真实智能体。

运行（在项目根目录）：
    $env:PYTHONIOENCODING="utf-8"
    .venv\\Scripts\\python.exe -m examples.data_analysis_demo
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from harness.agents.registry import AgentRegistry
from harness.audit import get_audit_logger
from harness.context import ContextManager
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

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
from packages.data_analysis.tools.common import reports_dir, workspace_dir


def _snapshot_artifacts() -> set[str]:
    """记录 VFS 现有的全部文件，用于运行后 diff 出「本次新增产物」。

    不能用文件名前缀过滤：Mock 用固定名，真实 LLM 会自拟文件名
    （advanced_metrics.json、品类销售TopN.png …），前缀过滤会把它们全部漏掉。
    """
    files: set[str] = set()
    for base in (workspace_dir({}), reports_dir({})):
        for root, _, names in os.walk(base):
            files.update(os.path.join(root, n) for n in names)
    return files


def _human_size(n: int) -> str:
    """人类可读大小。不能用 n // 1024 —— 会把 613 字节的完整报告显示成「0 KB」，
    看起来像文件损坏。"""
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n / (1024 * 1024):.1f} MB"


def _list_artifacts(before: set[str]) -> None:
    print("\n" + "=" * 64)
    print("VFS 本次产物清单（与运行前快照 diff 出的新增文件）")
    print("=" * 64)
    for label, base in (("workspace 数据集", workspace_dir({})),
                        ("reports 报告/图表", reports_dir({}))):
        print(f"\n[{label}]  {base}")
        new = []
        for root, _, names in os.walk(base):
            for n in names:
                path = os.path.join(root, n)
                if path in before:
                    continue
                new.append((os.path.relpath(path, base), os.path.getsize(path)))
        for rel, size in sorted(new):
            print(f"  - {rel}  ({_human_size(size)})")
        if not new:
            print("  （无新增）")


def _build_llm():
    """真实 LLM 优先；未配置 API Key 时退回脚本化 Mock（离线可跑）。

    两者实现同一套 ``chat`` / ``chat_json`` 契约，因此下游的
    planner / nodes / gate / orchestrator 完全无感。
    """
    from harness.config import settings

    key = (settings.llm.api_key or "").strip()
    if key:
        from harness.llm_client import LLMClient

        print(f"LLM：真实模型 {settings.llm.model} @ {settings.llm.base_url}")
        return LLMClient(
            api_key=key,
            base_url=settings.llm.base_url,
            model=settings.llm.model,
            temperature=settings.llm.temperature,
            timeout=float(settings.llm.timeout_seconds),
        )

    print("LLM：未配置 DEEPSEEK_API_KEY，退回脚本化 Mock（离线模式）")
    return ScriptedAnalysisLLM(RAW_FILE, CLEAN_STEM)


def _print_context_budget(cm: ContextManager) -> None:
    """打印上下文预算（均来自配置，便于比对调参前后的差异）。"""
    b = cm.budget
    print(
        "上下文预算："
        f"沉淀阈值 {b.sink_threshold_chars} 字符 / "
        f"单条上限 {b.observation_char_limit} / "
        f"保留最近 {b.keep_recent_messages} 条 / "
        f"历史预算 {b.max_history_chars} 字符"
    )


def _report_context_activity(cm: ContextManager) -> None:
    """汇总上下文管理的实际动作 —— 这是调参要看的实测数据。

    关注两点：① 有多少工具结果真的超过了沉淀阈值（阈值是否过激）；
    ② 沉淀后的提示长度（是否达成了"大结果不进上下文"的目标）。
    """
    print("\n" + "=" * 64)
    print("上下文管理实际动作（实测）")
    print("=" * 64)
    refs_by_session = cm.snapshot().get("refs") or {}
    total = 0
    for session_id, refs in refs_by_session.items():
        print(f"\n[会话 {session_id[:8]}…] 沉淀 {len(refs)} 个大结果：")
        for ref in refs:
            print(f"  - {ref['tool']}  {ref['chars']} 字符 → {ref['path']}")
        total += len(refs)
    if total == 0:
        print(
            f"（本次无工具结果超过沉淀阈值 {cm.budget.sink_threshold_chars} 字符，未产生沉淀）"
        )
    else:
        print(f"\n合计沉淀 {total} 个大结果（阈值 {cm.budget.sink_threshold_chars} 字符）")

    # 运行统计计数器：沉淀/压缩/截断的真实计数（判断阈值与预算是否合理的依据）
    print()
    print(cm.stats.summary())


DEFAULT_GOAL = f"对销售数据 {RAW_FILE} 做端到端分析：体检 → 清洗 → EDA → 出图 → 报告"


def main(goal: str | None = None) -> None:
    # 真实组件：注册全部内置工具 + 默认 7 个子 Agent + 内存任务存储 + 硬校验质量门
    # 审计接入：每次工具调用落 data/audit/audit.jsonl，可用 sandbox_used 字段
    # 直接核对高风险工具是否真的走了沙箱。
    broker = ToolBroker(audit_logger=get_audit_logger())
    register_builtin_tools(broker)
    registry = AgentRegistry()
    # 框架不内置领域角色：从数据分析领域包挂载子 Agent 定义
    # （data-explorer / analyst / reporter），否则规划器的 available_agents 为空，
    # 真实 LLM 不知道可以分派给谁，脚本化 LLM 也会在 registry.get 时抛 KeyError。
    for _def in build_agents().values():
        registry.register(_def)
    store = TaskStore(backend="memory")
    llm = _build_llm()
    gate = QualityGate(llm=llm, use_critic=False)  # 关闭语义 Critic，只跑硬校验
    planner = TaskPlanner(
        llm,
        broker=broker,
        available_agents=registry.names(),
        agent_descriptions={d.name: d.description for d in registry.list_defs()},
    )

    # 工具调用范式取自全局配置（环境变量 AGENT_TOOL_MODE），非本示例私有的开关
    from harness.config import settings

    # 上下文管理在这里「装配一次」—— 框架提供参数，装配点负责构造实例并注入。
    # 阶段5 服务化后，装配点会移到服务层；本示例只是当前唯一的装配点。
    context_manager = ContextManager(vfs=VirtualFileSystem())
    _print_context_budget(context_manager)

    # checkpointer 是 interrupt()/Command(resume=...) 的前置条件
    checkpointer = MemorySaver()
    graph = build_plan_execute_graph(
        llm, broker, planner=planner, store=store,
        registry=registry, gate=gate, max_replans=2,
        tool_mode=settings.runtime.agent_tool_mode,
        context_manager=context_manager,
        checkpointer=checkpointer,
    )
    print(f"工具调用范式：agent_tool_mode={settings.runtime.agent_tool_mode}")

    make_dirty_data()
    artifacts_before = _snapshot_artifacts()

    print("\n" + "=" * 64)
    print("启动 Plan-and-Execute 端到端链路")
    print("=" * 64)
    run_config = {"recursion_limit": 200, "configurable": {"thread_id": "demo-thread"}}
    state = graph.invoke(
        make_plan_execute_state(goal or DEFAULT_GOAL),
        config=run_config,
    )
    # 非交互模式：工具审批中断自动批准（沙箱仍会拦截危险代码），循环恢复直到图收尾
    _auto_approve = {"approved": True, "comment": "非交互模式自动批准",
                     "approver": "auto", "approver_role": "system"}
    _cycle = 0
    while state.get("__interrupt__"):
        _cycle += 1
        _req = state["__interrupt__"][0].value
        _tool = _req.get("tool", "?") if isinstance(_req, dict) else "?"
        print(f"[审批 #{_cycle}] {_tool} 请求执行，非交互模式自动批准...")
        state = graph.invoke(Command(resume=_auto_approve), config=run_config)
        if _cycle >= 50:
            print("[警告] 审批循环超过 50 次，强制停止")
            break

    # ---- 结果展示 ----
    plan = state["plan"]
    print("\n" + "=" * 64)
    print("计划执行情况")
    print("=" * 64)
    for t in plan.tasks:
        print(f"  [{t.status.value:>11}] {t.assigned_to:<9} {t.title}")
    print(f"计划版本 v{plan.version}，进度 {plan.progress:.0%}，总状态：{state['status']}")

    print("\n" + "=" * 64)
    print("各子 Agent 执行结果（SubAgentResult）")
    print("=" * 64)
    for r in state["sub_results"]:
        art_keys = sorted(r.artifacts.keys())
        print(f"\n● {r.sub_agent_name:<9} success={r.success} "
              f"steps={r.steps_taken} 耗时={r.duration_ms}ms 产物键={art_keys}")
        print(f"  结论：{(r.conclusion or '')[:160]}")

    print("\n" + "=" * 64)
    print("最终报告（synthesize）")
    print("=" * 64)
    print(state["final_answer"])

    _list_artifacts(artifacts_before)
    _report_context_activity(context_manager)


if __name__ == "__main__":
    import sys

    # 可选传入自定义目标，用于驱动不同深度的分析链路（例如强制走沙箱）：
    #   python -m examples.data_analysis_demo "用 Python 代码计算各品类月度环比与异常值明细"
    main(sys.argv[1] if len(sys.argv) > 1 else None)
