"""harness.planning.gate —— 子任务质量门（Quality Gate）。

每个子任务执行完、解锁下游前过一道 Gate，避免错误向下游传播。校验分两层：

1. 确定性硬校验（无需 LLM）：
   - SubAgentResult.success 是否成功；
   - 是否给出非空结论；
   - 可选 artifact_checker（由 Orchestrator 注入，通常结合 VFS 校验产物存在性、
     schema、行数、数据质量阈值）。
2. 语义验收（可选 Critic，LLM 裁判）：对照 TaskStep.acceptance_criteria 判断
   结论是否成立、能否支撑下游；裁判独立于执行子 Agent，避免"自己改卷"。

处置（decision）：
- PASS  通过，解锁下游；
- RETRY 可恢复问题，带反馈让同一子 Agent 重跑（受 max_retries 限制）；
- REPLAN 重试耗尽或计划本身走不通，回 Planner 重规划；
- HUMAN 高风险/结论矛盾，转人工审批（图 interrupt）；
- FAIL  不可恢复（HUMAN 无审批通道等），终止。
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Optional

from pydantic import BaseModel, Field

from harness.models import SubAgentResult, TaskStep

logger = logging.getLogger(__name__)


class GateDecision:
    PASS = "pass"  # nosec B105 - 质量门决策状态字符串，不是密码
    RETRY = "retry"
    REPLAN = "replan"
    HUMAN = "human"
    FAIL = "fail"


# 产物检查器签名：(task, result) -> (是否通过, 说明)
ArtifactChecker = Callable[[TaskStep, SubAgentResult], "tuple[bool, str]"]


class GateVerdict(BaseModel):
    """一次质量门判定结果。"""

    decision: str
    note: str = ""
    retry_feedback: str = ""          # 回灌给重跑子 Agent 的具体反馈
    checked_by_critic: bool = False


class QualityGate:
    """质量门评估器。

    Args:
        llm: 用于 Critic 语义裁判的 LLM（可与规划/执行同一个）。
        use_critic: 是否启用 LLM Critic；关闭时只做确定性硬校验。
        artifact_checker: 可选的产物确定性检查器（通常接 VFS / 数据质量规则）。
    """

    def __init__(
        self,
        llm: Any = None,
        use_critic: bool = False,
        artifact_checker: Optional[ArtifactChecker] = None,
        data_quality_checker: Optional[Any] = None,
        middleware: Optional[Any] = None,
    ) -> None:
        self.llm = llm
        self.use_critic = use_critic
        self.artifact_checker = artifact_checker
        self.data_quality_checker = data_quality_checker
        self.middleware = middleware

    # ------------------------------------------------------------------
    def evaluate(self, task: TaskStep, result: SubAgentResult) -> GateVerdict:
        hard_reasons: list[str] = []

        # 1) 确定性硬校验
        if not result.success:
            hard_reasons.append(f"子 Agent 执行失败：{result.error or '未知错误'}")
        if not (result.conclusion or "").strip():
            hard_reasons.append("子 Agent 未给出结论")
        if self.artifact_checker is not None:
            try:
                ok, msg = self.artifact_checker(task, result)
            except Exception as e:  # noqa: BLE001 - 检查器自身异常视为未通过并记录
                ok, msg = False, f"产物检查器异常：{type(e).__name__}: {e}"
            if not ok:
                hard_reasons.append(f"产物校验未通过：{msg}")

        # 2) 数据质量红线（确定性）：越线直接转人工，且不浪费一次 LLM 裁判
        if self.data_quality_checker is not None:
            try:
                pause_reason = self.data_quality_checker.check(task, result)
            except Exception as e:  # noqa: BLE001 - 检查器自身异常不能吞掉，升级为人工
                pause_reason = f"数据质量检查器异常：{type(e).__name__}: {e}"
            if pause_reason:
                logger.warning("Gate HUMAN (data quality) for %s: %s", task.title, pause_reason)
                return GateVerdict(decision=GateDecision.HUMAN, note=pause_reason)

        # 3) Critic 语义裁判（可选）
        critic_passed, critic_note, needs_human, checked = True, "", False, False
        if self.use_critic and self.llm is not None and task.acceptance_criteria:
            critic_passed, critic_note, needs_human, checked = self._critic_judge(task, result)

        if needs_human:
            return GateVerdict(
                decision=GateDecision.HUMAN,
                note=f"Critic 请求人工介入：{critic_note}",
                retry_feedback=critic_note,
                checked_by_critic=checked,
            )

        if not critic_passed:
            hard_reasons.append(f"验收标准未满足：{critic_note}")

        if hard_reasons:
            note = "；".join(r for r in hard_reasons if r)
            if task.retry_count < task.max_retries:
                logger.info("Gate RETRY for %s (retry %d/%d): %s",
                            task.title, task.retry_count, task.max_retries, note)
                return GateVerdict(
                    decision=GateDecision.RETRY, note=note,
                    retry_feedback=f"上一次执行未通过质量门：{note}。请针对性修正后重做。",
                    checked_by_critic=checked,
                )
            logger.info("Gate REPLAN for %s (retries exhausted): %s", task.title, note)
            return GateVerdict(
                decision=GateDecision.REPLAN, note=note, retry_feedback=note,
                checked_by_critic=checked,
            )

        note = "确定性校验通过"
        if checked:
            note += f"；Critic 验收通过：{critic_note or '符合验收标准'}"
        return GateVerdict(decision=GateDecision.PASS, note=note, checked_by_critic=checked)

    # ------------------------------------------------------------------
    def _critic_judge(
        self, task: TaskStep, result: SubAgentResult
    ) -> tuple[bool, str, bool, bool]:
        """返回 (是否通过, 理由, 是否需要人工, 是否实际完成裁判)。"""
        criteria = "\n".join(f"- {c}" for c in task.acceptance_criteria)
        artifacts = json.dumps(result.artifacts, ensure_ascii=False, default=str)[:1500]
        messages = [
            {"role": "system", "content": (
                "你是【质量门裁判 Critic】，独立于执行方，严格把关子任务交付质量。"
                "只输出一个 JSON：{\"passed\": true/false, \"reason\": \"简短理由\", "
                "\"needs_human\": true/false}。\n"
                "判定原则：逐条对照验收标准，结论必须由给出的证据支撑；证据不足、结论与"
                "产物矛盾则 passed=false；若涉及高风险操作、结论自相矛盾或无法据现有信息"
                "判定，置 needs_human=true。不要替执行方编造结果。\n"
                "若结论涉及统计分析，还须检查三类常见逻辑缺陷，并在 reason 中点名："
                "① 幸存者偏差（样本只覆盖了幸存/可见的部分）；② 辛普森悖论（分组趋势与"
                "整体趋势相反）；③ 数据泄露（目标泄漏：用到了分析时点不可得的信息，"
                "如结果字段参与了特征或口径）。发现任一项即 passed=false。"
            )},
            {"role": "user", "content": (
                f"子任务：{task.title}\n任务说明：{task.description}\n"
                f"验收标准：\n{criteria}\n\n"
                f"子 Agent 结论：\n{result.conclusion}\n\n产物：\n{artifacts}"
            )},
        ]
        if self.middleware is not None:
            from harness.middleware import MiddlewareContext

            messages = self.middleware.exec_before_llm(
                MiddlewareContext(operation="critic"), messages
            )
        try:
            data = self.llm.chat_json(messages)
            return (
                bool(data.get("passed", False)),
                str(data.get("reason", "")),
                bool(data.get("needs_human", False)),
                True,
            )
        except Exception as e:  # noqa: BLE001 - 裁判基础设施故障：保留确定性硬校验结论
            logger.warning("Critic judge failed, fall back to deterministic checks: %s", e)
            return True, f"(Critic 不可用，已跳过语义裁判: {e})", False, False


__all__ = ["QualityGate", "GateDecision", "GateVerdict", "ArtifactChecker"]
