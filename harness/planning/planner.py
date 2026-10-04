"""harness.planning.planner —— LLM 任务规划器 / 重规划器（Plan-and-Execute 的 Planner）。

输入用户目标，让 LLM 产出一个结构化 TaskPlan：
- 每个子任务带 title / description / assigned_to（专业子 Agent）/ depends_on（依赖）
  / acceptance_criteria（质量门验收标准）/ expected_artifacts（预期产物）。
- LLM 用"步骤序号（从 0 开始）"表达依赖，本模块负责把序号映射成稳定的 task_id，
  并校验负责人是否在已知子 Agent 名单内（未知则回退通用 executor）。

重规划 replan：执行偏离/失败时，保留已完成步骤（含结论与产物），让 LLM 只补
"剩余/调整后的步骤"，计划版本 +1。
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from harness.models import TaskPlan, TaskStatus, TaskStep
from harness.tool_broker import ToolBroker

logger = logging.getLogger(__name__)


class TaskPlanner:
    """把目标拆成 TaskPlan；执行受阻时重规划。

    **角色清单与描述由调用方给**（来自子 Agent 注册表，即来自领域包）—— 框架不内置
    任何领域角色，因此这里没有默认名录。原先的 ``DEFAULT_AGENT_ROLES`` 是「与
    agents 注册表保持一致」的**第二份手工名录**，已删除。

    Args:
        llm: 具备 ``chat_json(messages) -> dict`` 的对象（LLMClient 或 Mock）。
        broker: 可选 ToolBroker，用于把当前可用工具清单告诉规划器。
        available_agents: 可分派的子 Agent 名列表（装配点从注册表取）。
        agent_descriptions: 名字 → 一句话职责，用于渲染规划提示词。
        default_agent: LLM 给出未知负责人时的回退角色；缺省取 ``available_agents`` 第一个。
        middleware: 可选 MiddlewareManager；规划提示词里含用户原始目标，
            出站前必须过 ``before_llm``（PII 脱敏覆盖规划器，不只有推理节点）。
    """

    def __init__(
        self,
        llm: Any,
        broker: Optional[ToolBroker] = None,
        available_agents: Optional[list[str]] = None,
        agent_descriptions: Optional[dict[str, str]] = None,
        default_agent: Optional[str] = None,
        middleware: Optional[Any] = None,
    ) -> None:
        self.llm = llm
        self.broker = broker
        self.roles = list(available_agents or [])
        self.descriptions = dict(agent_descriptions or {})
        self.default_agent = default_agent or (self.roles[0] if self.roles else "")
        self.middleware = middleware
        if not self.roles:
            # 不是错误：框架可以不带任何领域包启动（那时也确实无人可分派）。
            # 但必须响亮 —— 否则表现为"规划器产出正常、执行时无处派发"。
            logger.warning(
                "规划器没有可用的子 Agent（available_agents 为空）："
                "请确认已挂载领域包，或装配点是否传了注册表"
            )

    def _outbound(self, messages: list[dict], operation: str) -> list[dict]:
        """LLM 出站前过一遍 before_llm 钩子（无中间件时原样返回）。"""
        if self.middleware is None:
            return messages
        from harness.middleware import MiddlewareContext

        return self.middleware.exec_before_llm(
            MiddlewareContext(operation=operation), messages
        )

    # ------------------------------------------------------------------
    # Prompt 构造
    # ------------------------------------------------------------------
    def _agent_hints(self) -> str:
        lines = []
        for name in self.roles:
            desc = self.descriptions.get(name, name)
            lines.append(f"- {name}: {desc}")
        return "\n".join(lines) if lines else "（当前没有任何可派发的子 Agent）"

    def _tool_hints(self) -> str:
        if self.broker is not None:
            desc = self.broker.list_tool_descriptions()
            return desc if desc.strip() else "（当前没有注册任何工具）"
        return "（未提供工具清单）"

    def _output_contract(self) -> str:
        return (
            "## 输出格式（只输出一个 JSON，不要 Markdown 代码块或多余文字）\n"
            '{"tasks": [\n'
            '  {"title": "简短标题",\n'
            '   "description": "这一步具体要做什么、输入是什么、产出什么",\n'
            '   "assigned_to": "上面某个子 Agent 名",\n'
            '   "depends_on": [前置步骤的序号，从 0 开始；没有依赖则给 []],\n'
            '   "acceptance_criteria": ["可判定的验收标准1", "验收标准2"],\n'
            '   "expected_artifacts": ["预期产物，如 /workspace/profile.json"]\n'
            "  }]}\n"
            "注意：depends_on 只能引用【本 JSON 中步骤的序号】（第一个步骤序号为 0）。\n"
        )

    def _plan_messages(self, goal: str, context: str) -> list[dict]:
        system = (
            "你是一名资深数据分析项目负责人，擅长把复杂的数据分析 / ETL 目标拆解为"
            "有序、可执行、可验收的子任务，并分派给合适的专业子 Agent。\n"
            "拆解原则：\n"
            "1. 遵循 数据体检 → 清洗 → 探索分析 → 图表/复杂计算 → 报告 的主干，按实际需要裁剪；\n"
            "2. 每个子任务应设计为对应子 Agent 能在 3-5 步工具调用内完成；复杂分析（如 EDA）必须拆成多个独立子任务（单变量画像 / 分组对比 / 时间趋势分开），不要把多类分析塞进一个任务；各子 Agent 步数上限：data-explorer 12 步、analyst 14 步、reporter 8 步，任务设计须留有余量；\n"
            "3. 明确先后依赖，能并行的不要强行串行；\n"
            "4. 为每个子任务写出可客观判定的验收标准与预期产物。\n\n"
            f"## 可分派的子 Agent\n{self._agent_hints()}\n\n"
            f"## 可调用的工具\n{self._tool_hints()}\n\n"
            f"{self._output_contract()}"
        )
        user = f"用户目标：\n{goal}\n"
        if context:
            user += f"\n补充背景：\n{context}\n"
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def _replan_messages(self, old_plan: TaskPlan, feedback: str) -> list[dict]:
        done_lines = []
        for i, t in enumerate(old_plan.tasks):
            if t.status in (TaskStatus.COMPLETED, TaskStatus.SKIPPED):
                done_lines.append(
                    f"[{i}] {t.title}（{t.status.value}）结论：{t.result or '(无)'}"
                )
        done_block = "\n".join(done_lines) or "（暂无已完成步骤）"

        system = (
            "你是数据分析项目负责人。原计划在执行中遇到问题，需要你重规划【尚未完成】的部分。\n"
            "要求：\n"
            "1. 只输出还需要执行（或需要调整）的新步骤，不要重复已经完成的步骤；\n"
            "2. 新步骤之间的依赖用【你本次输出列表内的序号（从 0 开始）】表示；\n"
            "3. 针对反馈中的失败原因调整方案，必要时换用不同子 Agent 或补一个前置修复步骤；\n"
            "4. **如果上次失败原因是步数超限 / 任务过大 / 子 Agent 反复调用同一工具不收尾，必须把原任务拆分为更细粒度的子任务**（每个 3-5 步可完成），不要原样重试；\n\n"
            f"## 可分派的子 Agent\n{self._agent_hints()}\n\n"
            f"{self._output_contract()}"
        )
        user = (
            f"原始目标：{old_plan.goal}\n\n"
            f"已完成步骤（系统会自动保留，不要重复）：\n{done_block}\n\n"
            f"重规划反馈 / 失败信息：\n{feedback}\n"
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    # ------------------------------------------------------------------
    # 解析 LLM 输出
    # ------------------------------------------------------------------
    def _parse_steps(self, data: dict[str, Any]) -> list[TaskStep]:
        raw_tasks = data.get("tasks")
        if not isinstance(raw_tasks, list) or not raw_tasks:
            raise ValueError("规划器返回中没有有效的 tasks 列表")

        # 第一轮：先实例化，拿到每个步骤的稳定 task_id 与 序号→id 映射
        steps: list[TaskStep] = []
        index_to_id: dict[int, str] = {}
        for i, item in enumerate(raw_tasks):
            if not isinstance(item, dict):
                raise ValueError(f"第 {i} 个任务不是对象")
            title = str(item.get("title", "")).strip() or f"步骤 {i + 1}"
            assigned = str(item.get("assigned_to", "")).strip()
            if assigned not in self.roles:
                logger.warning("规划器给出未知子 Agent %r，回退为 %s", assigned, self.default_agent)
                assigned = self.default_agent
            step = TaskStep(
                title=title,
                description=str(item.get("description", "")),
                assigned_to=assigned,
                acceptance_criteria=[str(x) for x in item.get("acceptance_criteria", [])],
                expected_artifacts=[str(x) for x in item.get("expected_artifacts", [])],
            )
            index_to_id[i] = step.task_id
            steps.append(step)

        # 第二轮：把"序号依赖"翻译成 task_id 依赖（非法序号忽略）
        for i, item in enumerate(raw_tasks):
            deps = item.get("depends_on", []) or []
            if not isinstance(deps, list):
                continue
            mapped = []
            for dep in deps:
                try:
                    dep_index = int(dep)
                except (TypeError, ValueError):
                    continue
                if dep_index == i:
                    continue  # 不允许自依赖
                target = index_to_id.get(dep_index)
                if target and target not in mapped:
                    mapped.append(target)
            steps[i].depends_on = mapped

        return steps

    # ------------------------------------------------------------------
    # 对外 API
    # ------------------------------------------------------------------
    def plan(self, goal: str, context: str = "") -> TaskPlan:
        """首次规划：目标 → TaskPlan（未持久化，交由 TaskStore.create_plan 校验保存）。"""
        data = self.llm.chat_json(
            self._outbound(self._plan_messages(goal, context), "plan")
        )
        steps = self._parse_steps(data)
        plan = TaskPlan(goal=goal, tasks=steps)
        plan.current_task_id = steps[0].task_id if steps else None
        plan.compute_progress()
        logger.info("Planner 产出计划：%d 个任务（goal=%.40s）", len(steps), goal)
        return plan

    def replan(self, old_plan: TaskPlan, feedback: str = "") -> TaskPlan:
        """重规划：保留已完成步骤，追加 LLM 给出的新步骤，版本 +1。"""
        data = self.llm.chat_json(
            self._outbound(self._replan_messages(old_plan, feedback), "replan")
        )
        new_steps = self._parse_steps(data)

        done = [
            t for t in old_plan.tasks
            if t.status in (TaskStatus.COMPLETED, TaskStatus.SKIPPED)
        ]

        # 让新步骤的第一步隐式依赖"最后一个已完成步骤"，衔接前后拓扑
        if done and new_steps:
            last_done_id = done[-1].task_id
            if last_done_id not in new_steps[0].depends_on:
                new_steps[0].depends_on.append(last_done_id)

        merged = done + new_steps
        plan = TaskPlan(
            plan_id=old_plan.plan_id,
            goal=old_plan.goal,
            tasks=merged,
            version=old_plan.version + 1,
            replan_count=old_plan.replan_count + 1,
        )
        plan.current_task_id = new_steps[0].task_id if new_steps else None
        plan.compute_progress()
        logger.info(
            "Replanner v%d：保留 %d 个已完成，新增 %d 个任务",
            plan.version, len(done), len(new_steps),
        )
        return plan


__all__ = ["TaskPlanner"]
