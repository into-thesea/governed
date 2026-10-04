"""harness.nodes —— ReAct 执行内核的图节点（think / action / route）。

在最终的"Plan-and-Execute + ReAct 内核"架构里，本模块只负责【单个子任务】
内部的执行循环：think（LLM 决策）→ action（经 ToolBroker 执行）→ observation
喂回 → 再 think，直到该子任务产出结论。

LangGraph 节点铁律：每个节点接收 state（dict），返回"状态增量 dict"，
LangGraph 负责把增量合并回全局状态。

节点用类的方法实现（ReActNodes），便于把 LLM / Broker / 中间件等依赖
保存在 self 上，再把绑定方法交给 graph.add_node。
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Any, Optional

from langchain_core.messages import BaseMessage
from langgraph.types import interrupt

from harness.approval_policy import DECISION_AUTO, DECISION_DENY, decide_approval
from harness.events import GUARD_LAYER_APPROVAL, emit_guard_decision
from harness.llm_client import (
    LLMClient,
    _try_extract_json,
    bind_usage_context,
    reset_usage_context,
)
from harness.middleware import MiddlewareContext, MiddlewareManager
from harness.models import ThoughtStep
from harness.state import AgentState
from harness.tool_broker import ToolBroker
from harness.trace import span_for as _span

logger = logging.getLogger(__name__)

# LangChain 消息 type → OpenAI 角色名
_ROLE_MAP = {"human": "user", "ai": "assistant", "system": "system", "tool": "tool"}


def _approval_expires_at() -> Optional[str]:
    """审批卡片的过期时刻（ISO 字符串）；``SERVER_APPROVAL_TIMEOUT_SECONDS<=0`` 不过期。

    配置不可得时按"不过期"处理 —— 配置/观测问题不该把人工审批这条路堵死。
    """
    try:
        from harness.config import settings

        timeout_s = int(settings.server.approval_timeout_seconds)
    except Exception:  # noqa: BLE001 - 同上：配置异常不应阻断审批
        return None
    if timeout_s <= 0:
        return None
    return (datetime.now() + timedelta(seconds=timeout_s)).isoformat()


def _normalize_tool_calls(tool_calls: Any) -> list[dict]:
    """把两种 tool_calls 形态统一成 OpenAI 契约。

    LangChain 形态（LangGraph 的 add_messages 会把消息转成这种）:
        {"name": "add", "args": {"a": 1}, "id": "call_x", "type": "tool_call"}
        —— ``args`` 是 **dict**，且 ``type`` 是 ``"tool_call"``
    OpenAI 契约:
        {"id": "call_x", "type": "function",
         "function": {"name": "add", "arguments": "{\\"a\\": 1}"}}
        —— ``arguments`` 是 **JSON 字符串**

    原样回传 LangChain 形态会被服务端直接拒绝（实测 DeepSeek）：
        422 unknown variant `tool_call`, expected `function`
    """
    normalized: list[dict] = []
    for call in tool_calls or []:
        if not isinstance(call, dict):
            continue
        fn = call.get("function")
        if isinstance(fn, dict):
            # 已是 OpenAI 形态，只补齐 type
            normalized.append({
                "id": call.get("id", ""),
                "type": "function",
                "function": {
                    "name": fn.get("name", ""),
                    "arguments": fn.get("arguments") or "{}",
                },
            })
            continue
        args = call.get("args")
        normalized.append({
            "id": call.get("id", ""),
            "type": "function",
            "function": {
                "name": call.get("name", ""),
                "arguments": args if isinstance(args, str)
                else json.dumps(args if args is not None else {}, ensure_ascii=False),
            },
        })
    return normalized


class ReActNodes:
    """ReAct 执行内核的节点集合。

    用法：
        nodes = ReActNodes(llm=llm, broker=broker, middleware=manager)
        graph.add_node("think", nodes.think)
        graph.add_node("action", nodes.action)
        graph.add_conditional_edges("think", nodes.route, {...})
    """

    def __init__(
        self,
        llm: LLMClient,
        broker: ToolBroker,
        middleware: Optional[MiddlewareManager] = None,
        system_prefix: str = "",
        tool_mode: str = "react",
        context_manager: Optional[Any] = None,
        skill_registry: Optional[Any] = None,
        allowed_skills: Optional[list[str]] = None,
        datasources: Optional[Any] = None,
    ) -> None:
        if tool_mode not in ("react", "native"):
            raise ValueError(f"tool_mode 只能是 'react' 或 'native'，收到 {tool_mode!r}")
        self.llm = llm
        self.broker = broker
        self.middleware = middleware
        # 子 Agent 的专属角色/职责提示（专业子 Agent 委派时注入）；为空则通用
        self.system_prefix = system_prefix
        self.tool_mode = tool_mode
        # 上下文管理（harness.context.ContextManager）。None 表示不启用 —— 由上层
        # orchestrator 默认构造并透传；直接构造本类时可显式传入或留空。
        self.context_manager = context_manager
        # Skill 技能系统（harness.skills.SkillRegistry）。None 表示不注入技能指引。
        # allowed_skills 为该子 Agent 的 Skill 白名单（SubAgentDef.skills 非空时）；
        # None 表示不限定，按任务相关性匹配（skills 字段为空时的默认语义）。
        self.skill_registry = skill_registry
        self.allowed_skills = allowed_skills
        # 数据源管理器（harness.datasources.DataSourceManager）是运行时基础设施，
        # 经节点构造注入（不进 state、不被 checkpointer 序列化），sql_query 经
        # broker invoke 的 context 取用；None 时 sql_query 用进程默认单例。
        self.datasources = datasources

    # ------------------------------------------------------------------
    # 工具执行前审批闸门（Human-in-the-Loop）
    # ------------------------------------------------------------------
    def _request_tool_approval(
        self,
        tool_name: str,
        args: dict,
        state: AgentState,
        invoke_context: Optional[dict] = None,
    ) -> tuple[bool, str, Optional[dict]]:
        """高风险工具执行前的闸门：先授权预检，再人工审批（LangGraph interrupt）。

        ``ToolDef.requires_approval=True`` 时，在真正执行前 interrupt 暂停，
        把工具 / 参数 / 风险暴露给审批人；审批人经服务层 ``Command(resume=...)``
        下发决策。

        返回 ``(是否放行, 可直接回灌模型的 observation, 审批凭证)``：

        - 无需审批：``(True, "", None)``，零中断；
        - 授权预检拒绝（PDP / 子 Agent 工具边界）：``(False, <权限提示>, None)``，
          **不弹审批** —— 先判"准不准做"（authorize），再问"要不要做"（人工
          审批），避免对无权工具也打扰审批人、批准后仍被 Broker 拒（挂账 #12）；
        - 人工驳回：``(False, <驳回提示>, None)``；
        - 人工批准：``(True, "", <审批凭证>)``。凭证须随 invoke 上下文交给
          Broker，Broker 第 3.5 步校验匹配后才放行（挂账 #11：执行点 fail
          closed，MCP 等非交互通道拿不到凭证即无法绕过审批）。

        审批闸门在节点层而非 Broker 内部：interrupt 只能在 LangGraph 节点中
        调用，Broker 是与编排解耦的普通 Python；Broker 另以凭证闸门兜底。
        """
        tool_def = self.broker.get(tool_name)
        if tool_def is None or not tool_def.requires_approval:
            return True, "", None

        call_ctx = invoke_context or self._invoke_context(state)

        # 先授权（PDP / 子 Agent 白名单），再请求人工审批。
        allowed, authz_reason = self.broker.authorize(tool_name, call_ctx)
        if not allowed:
            observation = (
                f"工具调用未获授权，未进入人工审批：{authz_reason}。"
                "请改用你有权限的工具，不要再次请求该操作。"
            )
            logger.info("Tool %s blocked before approval: %s", tool_name, authz_reason)
            return False, observation, None

        # 会话豁免（"本任务内总是允许 / 总是拒绝"）：只跳过人工审批那一步。
        # 走到这里说明 authorize 已经放行 —— 豁免**不能**越过 PDP，这一层的顺序不能挪。
        grant = self.broker.apply_session_grant(
            state.get("session_id") or "",
            tool_name,
            trace_id=state.get("trace_id"),
            task_id=state.get("task_id"),
            agent_id=state.get("agent_id"),
        )
        if grant is not None:
            if grant["effect"] == "deny":
                logger.info("Tool %s denied by session grant", tool_name)
                return False, (
                    "本任务内已豁免：该工具已被审批人一次性拒绝，不要再次请求。"
                    "请调整方案或改做其他事。"
                ), None
            logger.info("Tool %s allowed by session grant", tool_name)
            return True, "", self._make_approval_credential(
                tool_name, state, via="session_grant", grant_id=grant["grant_id"]
            )

        # 风险分级：领域包只回答"多危险"，这里回答"要不要问人"。
        # 放在 authorize 之后 —— 自动放行**绝不能**越过 PDP（与豁免同一条）。
        # 没有风险策略时**不去探沙箱**：那条路径的结论必然是 ASK（零回归），而探测是
        # 一次带超时的健康检查，没必要为它花网络往返。也免得"无策略"这条最常见的路径
        # 依赖一个它根本用不到的运行时结论。
        risk_policy = self.broker.risk_policy_for(tool_name)
        fallback = (
            self.broker.approval_fallback(tool_name) if risk_policy is not None else None
        )
        decision = decide_approval(
            tool_name, args,
            risk_policy=risk_policy,
            threshold=self.broker.approval_threshold,
            deny_threshold=self.broker.approval_deny_threshold,
            fallback=fallback,
        )
        if decision.decision in (DECISION_AUTO, DECISION_DENY):
            # 自动档是"静默放宽"，比弹卡更需要可见（挂账 #13 的定位）
            logger.info("Tool %s decision=%s risk=%s: %s",
                        tool_name, decision.decision, decision.risk, decision.reason)
            emit_guard_decision(
                state.get("trace_id"), layer=GUARD_LAYER_APPROVAL,
                decision="allow" if decision.decision == DECISION_AUTO else "deny",
                reason=decision.reason, tool=tool_name,
                task_id=state.get("task_id"), agent_id=state.get("agent_id"),
                via="risk_auto", risk=decision.risk, policy=decision.policy,
                fallback=decision.fallback,
            )
            if decision.decision == DECISION_DENY:
                return False, f"该操作被审批策略拒绝：{decision.reason}", None
            return True, "", self._make_approval_credential(
                tool_name, state, via="risk_auto", risk=decision.risk,
                policy=decision.policy, fallback=decision.fallback,
            )

        # 无人值守部署（approval_channel="none"）：所有前置闸门（PDP / 会话豁免 /
        # 风险策略）均已放行，此处不再 interrupt 等人审，直接自动批准。
        # "none" 的语义：明确声明本部署没人审 → 高风险工具靠沙箱隔离兜底，
        # 审批闸门自动放行（仍留审计痕迹）。
        from harness.config import settings
        if settings.server.approval_channel == "none":
            logger.info("Tool %s auto-approved (approval_channel=none, unattended)",
                        tool_name)
            emit_guard_decision(
                state.get("trace_id"), layer=GUARD_LAYER_APPROVAL,
                decision="allow",
                reason="无人值守自动批准（approval_channel=none）",
                tool=tool_name,
                task_id=state.get("task_id"), agent_id=state.get("agent_id"),
                via="unattended_auto",
            )
            return True, "", self._make_approval_credential(
                tool_name, state, via="unattended_auto",
            )

        # 审批请求的关联 id 与过期时刻：request_id 用来把 APPROVAL_REQUIRED 与
        # 后续 APPROVAL_RESOLVED 配对（此刻还没有 LangGraph 的系统 interrupt id）；
        # expires_at 由服务层在 resume 时强制（超时只能驳回、不能批准）。
        request_id = f"aprreq_{uuid.uuid4().hex[:12]}"
        expires_at = _approval_expires_at()
        payload = {
            "type": "tool_approval",
            "tool": tool_name,
            "description": tool_def.description,
            "arguments": args,
            "run_in_sandbox": tool_def.run_in_sandbox,
            "approval_request_id": request_id,
            "expires_at": expires_at,
            "session_id": state.get("session_id"),
            "agent_id": state.get("agent_id"),
            "trace_id": state.get("trace_id"),
        }
        raw_decision = interrupt(payload)
        approved, comment, remember = self._parse_approval(raw_decision)
        logger.info("Tool %s approval: approved=%s comment=%s remember=%s",
                    tool_name, approved, comment, remember)

        # 审批人身份来自 **resume 值**（不是节点发出的 payload）—— 授予要把"谁放宽的"记下来，
        # 否则审计只留下一条没有责任人的豁免。
        approver = approver_role = ""
        if isinstance(raw_decision, dict):
            approver = str(raw_decision.get("approver") or "")
            approver_role = str(raw_decision.get("approver_role") or "")

        if remember:
            self.broker.grant_session_approval(
                state.get("session_id") or "", tool_name, remember,
                granted_by=approver, granted_role=approver_role,
                comment=comment, request_id=request_id,
                trace_id=state.get("trace_id"), task_id=state.get("task_id"),
                agent_id=state.get("agent_id"),
            )

        if not approved:
            observation = (
                f"工具调用被人工审批拒绝：{comment or '未说明原因'}。"
                "请调整方案，不要再次请求同样的操作。"
            )
            return False, observation, None

        return True, "", self._make_approval_credential(tool_name, state, comment=comment)

    def _make_approval_credential(
        self, tool_name: str, state: AgentState, comment: str = "", **extra: Any
    ) -> dict:
        """构造审批通过凭证。

        人工批准与会话豁免两条路径**必须复用本方法**：凭证字段一旦分叉，Broker 第 3.5 步的
        凭证闸门就会只认其中一条（本项目在"节点与 Broker 字段对不上"上踩过）。
        """
        return {
            "id": f"apr_{uuid.uuid4().hex[:12]}",
            "approved": True,
            "tool": tool_name,
            "comment": comment,
            "approved_at": datetime.now().isoformat(),
            "session_id": state.get("session_id"),
            "agent_id": state.get("agent_id"),
            "trace_id": state.get("trace_id"),
            **extra,
        }

    @staticmethod
    def _parse_approval(raw: Any) -> tuple[bool, str, Optional[str]]:
        """解析审批人下发的 resume 值 → ``(approved, comment, remember)``。

        ``remember`` 只接受 ``None | "allow" | "deny"``，且必须与 ``approved`` **同向**
        （allow↔True、deny↔False）；非法值或自相矛盾的组合一律归一为 ``None`` ——
        fail closed：不认识的输入不产生任何豁免，也不报错（报错会把审批卡死）。
        """
        if isinstance(raw, dict):
            comment = str(raw.get("comment") or raw.get("reason") or "")
            if "approved" in raw:
                approved = bool(raw["approved"])
            else:
                decision = str(raw.get("decision") or raw.get("action") or "").lower()
                approved = decision in ("approve", "approved", "allow", "pass", "true", "1")
            remember = str(raw.get("remember") or "").strip().lower() or None
        else:
            text = str(raw or "").strip().lower()
            approved = text in ("approve", "approved", "allow", "pass", "true", "1", "y", "yes")
            comment, remember = "", None

        if remember not in ("allow", "deny"):
            remember = None
        elif (remember == "allow") != bool(approved):
            remember = None                      # 不同向：不猜，直接不授予
        return approved, comment, remember

    # ------------------------------------------------------------------
    # 上下文管理的两个介入时机（未注入 ContextManager 时全部为透传，零开销）
    # ------------------------------------------------------------------
    def _settle_observation(self, tool_name: str, observation: str, state: AgentState) -> str:
        """时机一（action 之后）：工具大结果沉淀到 VFS，提示里只留摘要 + 文件卡片。"""
        if self.context_manager is None:
            return observation
        return self.context_manager.settle_observation(
            tool_name, observation, session_id=state.get("session_id") or "default"
        )

    def _compact_history(self, history: list[dict], state: AgentState) -> list[dict]:
        """时机二（think 之前）：按预算压缩历史，并把长期记忆前插。"""
        if self.context_manager is None:
            return history
        return self.context_manager.compact_history(
            history,
            long_term_context=state.get("long_term_context") or "",
            # 会话 id 必须传：被折叠的原文会落进 `VFS/<session>/`，不传就全挤在 default
            # 目录下 —— 多任务并行时彼此的历史会串到一起，而且没人会立刻发现。
            session_id=state.get("session_id") or "default",
        )

    def _skill_guidance(self, state: AgentState) -> str:
        """按当前任务匹配相关 Skill，渲染为渐进式披露文本；无匹配 / 未注入返回 ''。"""
        if self.skill_registry is None:
            return ""
        # 子图 goal 即 _compose_subtask 组合的任务标题 / 描述 / 上游结论
        goal = state.get("goal", "")
        return self.skill_registry.render_for_context(
            goal=goal, task=goal, allowed_skills=self.allowed_skills
        )

    def _bind_usage(self, state: AgentState):
        """绑定当前子任务的归因上下文（agent/session/trace），返回 reset token。"""
        return bind_usage_context(
            agent_id=state.get("agent_id", "") or "",
            session_id=state.get("session_id", "") or "",
            trace_id=state.get("trace_id", "") or "",
        )

    @staticmethod
    def _unbind_usage(token) -> None:
        reset_usage_context(token)

    # ------------------------------------------------------------------
    # 工作记忆资产索引（让 LLM 知道"已经获取了什么数据"）
    # ------------------------------------------------------------------
    @staticmethod
    def _format_working_memory(working_memory: dict) -> str:
        """把工作记忆中的工具产物格式化为「数据资产索引摘要」，注入 LLM 上下文。

        不注入完整 artifacts（那会回到上下文爆炸），只注入资产索引：
        来源工具、关键元数据（文件名 / 行列规模 / 列名 / 产出路径）。
        LLM 需要原始细节时，用文件工具按路径读取（渐进式披露）。

        背景：工具返回 (ok, text, artifacts)，artifacts 存入 working_memory 后
        若不注入上下文，LLM 只能看到精简 text，会因"不知道已有什么数据"而
        反复调用同一工具（实测单任务 71 次体检、撞步数上限）。
        """
        if not working_memory:
            return ""

        lines = [
            "【已获取的数据资产索引】（以下为摘要，非完整数据；"
            "需要原始细节时用文件工具按路径读取）"
        ]
        has_asset = False

        for key, artifacts in working_memory.items():
            if not key.startswith("result_") or not isinstance(artifacts, dict):
                continue
            tool_name = key[len("result_"):]
            has_asset = True

            parts: list[str] = []
            for art_key, art_val in artifacts.items():
                meta = ReActNodes._extract_asset_meta(art_key, art_val)
                if meta:
                    parts.append(meta)

            if parts:
                lines.append(f"- {tool_name}：" + " | ".join(parts))
            else:
                lines.append(f"- {tool_name}：（已获取结果，完整数据已落盘）")

        if not has_asset:
            return ""
        return "\n".join(lines)

    @staticmethod
    def _extract_asset_meta(art_key: str, art_val: Any) -> str:
        """从单个 artifact 值中提取关键元数据摘要（识别已知结构，未知结构兜底）。"""
        if not isinstance(art_val, dict):
            return f"{art_key}（非结构化结果）"

        meta: list[str] = []

        # ---- inspection 结构（data_inspector）----
        if isinstance(art_val.get("file"), dict):
            f = art_val["file"]
            if f.get("file_name"):
                meta.append(f"文件={f['file_name']}")
            if f.get("format"):
                meta.append(f"格式={f['format']}")

        if isinstance(art_val.get("quality"), dict):
            q = art_val["quality"]
            if "rows" in q and "cols" in q:
                meta.append(f"{q['rows']}行×{q['cols']}列")
            if q.get("overall_missing_rate") is not None:
                meta.append(f"缺失率{q['overall_missing_rate']:.1%}")

        if isinstance(art_val.get("schema"), list):
            col_names = [
                c.get("name", "?") for c in art_val["schema"]
                if isinstance(c, dict) and c.get("name")
            ]
            if col_names:
                shown = col_names[:8]
                suffix = "…" if len(col_names) > 8 else ""
                meta.append(f"列=[{','.join(shown)}]{suffix}")

        if isinstance(art_val.get("risks"), list):
            high = [
                r for r in art_val["risks"]
                if isinstance(r, dict) and r.get("level") == "high"
            ]
            if high:
                names = [str(r.get("column") or "整表") for r in high[:3]]
                meta.append(f"高风险{len(high)}项({','.join(names)})")

        # ---- eda overview 结构 ----
        if isinstance(art_val.get("overview"), dict):
            ov = art_val["overview"]
            if "rows" in ov and "cols" in ov:
                meta.append(f"{ov['rows']}行×{ov['cols']}列")
            if ov.get("numeric_cols"):
                meta.append(f"数值列{len(ov['numeric_cols'])}")
            if ov.get("categorical_cols"):
                meta.append(f"类别列{len(ov['categorical_cols'])}")
            if ov.get("target"):
                meta.append(f"目标={ov['target']}")

        # ---- cleaning 结构 ----
        if art_val.get("output"):
            meta.append(f"清洗输出={art_val['output']}")
        if art_val.get("report_file"):
            meta.append(f"报告={art_val['report_file']}")
        if isinstance(art_val.get("after"), dict) and "rows" in art_val["after"]:
            meta.append(f"清洗后{art_val['after']['rows']}行")

        # ---- chart 结构 ----
        if art_val.get("vfs_path"):
            meta.append(f"图表={art_val['vfs_path']}")
        elif art_val.get("abs_path"):
            meta.append(f"图表={art_val['abs_path']}")
        if art_val.get("chart_type"):
            meta.append(f"图类型={art_val['chart_type']}")

        # ---- 通用兜底：产出路径 ----
        for path_field in ("output_path", "report_path", "file_path"):
            if art_val.get(path_field) and not meta:
                meta.append(f"产出={art_val[path_field]}")

        if not meta:
            top_keys = list(art_val.keys())[:6]
            return f"{art_key}[{','.join(top_keys)}]" if top_keys else art_key
        return f"{art_key}: " + "；".join(meta)

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------
    @staticmethod
    def _now() -> str:
        return datetime.now().isoformat()

    def _build_system(self) -> str:
        """拼装 system 消息：子 Agent 角色（可选）+ 通用要求 + 工具列表 + JSON 契约。"""
        tool_descriptions = self.broker.list_tool_descriptions()
        prefix = f"{self.system_prefix.strip()}\n\n" if self.system_prefix else ""
        return (
            prefix
            + "你是一个严谨的数据分析 Agent，通过调用工具逐步完成【当前子任务】。\n\n"
            f"{tool_descriptions}\n\n"
            "## 输出要求\n"
            "你必须【只】输出一个 JSON 对象，不要输出任何解释文字或 Markdown 代码块。\n"
            "- 需要调用工具时输出：\n"
            '  {"thought": "简短说明你为什么调用该工具", '
            '"action": "工具名", "action_input": {"参数名": 参数值}}\n'
            "- 当前子任务已经能给出结论时输出：\n"
            '  {"final_answer": "本子任务的结论"}\n\n'
            "规则：\n"
            "1. 一次只调用一个工具，拿到 Observation 后再决定下一步；\n"
            "2. action_input 必须符合上面声明的参数；\n"
            "3. 工具报错时阅读错误并调整，不要重复同样的错误调用；\n"
            "4. 信息足够后用 final_answer 收尾，不要无谓调用工具。\n"
        )

    def _build_system_native(self) -> str:
        """原生 Function Calling 模式的 system 提示。

        与 ReAct 版的关键差别：**不内嵌工具清单、不要求 JSON 输出** ——
        工具经 API 的 ``tools`` 参数下发，调用结构由服务端保证。
        """
        prefix = f"{self.system_prefix.strip()}\n\n" if self.system_prefix else ""
        return (
            prefix
            + "你是一个严谨的数据分析 Agent，通过调用工具逐步完成【当前子任务】。\n\n"
            "## 工作方式\n"
            "1. 需要数据或计算时，调用可用工具；一次可以调用一个或多个；\n"
            "2. 阅读工具返回结果后再决定下一步，不要重复已经失败的同一种调用；\n"
            "3. 工具报错时先阅读错误信息（通常会告知可用的文件名/参数），据此调整；\n"
            "4. 信息足够时，直接用自然语言给出本子任务的结论，不再调用工具；\n"
            "5. 结论要具体、可溯源，引用关键数值与产物文件名。\n"
        )

    @staticmethod
    def _messages_to_dicts(messages: list) -> list[dict]:
        """把 state 中的消息统一成 OpenAI 风格 dict（LangChain 对象 → dict）。

        必须原样带过 ``tool_call_id`` / ``tool_calls``：原生 Function Calling 的
        多轮配对靠它们，丢掉后服务端会直接拒绝请求（assistant 的 tool_calls 与
        后续 role="tool" 消息必须成对且 id 一致）。
        """
        result: list[dict] = []
        for m in messages or []:
            if isinstance(m, dict):
                out: dict[str, Any] = {
                    "role": m.get("role", "user"),
                    "content": m.get("content", ""),
                }
                for key in ("tool_call_id", "name"):
                    if m.get(key) is not None:
                        out[key] = m[key]
                if m.get("tool_calls"):
                    out["tool_calls"] = _normalize_tool_calls(m["tool_calls"])
                result.append(out)
            elif isinstance(m, BaseMessage):
                out = {"role": _ROLE_MAP.get(m.type, m.type), "content": m.content}
                tool_call_id = getattr(m, "tool_call_id", None)
                if tool_call_id:
                    out["tool_call_id"] = tool_call_id
                tool_calls = getattr(m, "tool_calls", None)
                if tool_calls:
                    out["tool_calls"] = _normalize_tool_calls(tool_calls)
                result.append(out)
            else:
                result.append({"role": "user", "content": str(m)})
        return result

    # ------------------------------------------------------------------
    # think 节点：LLM 决策（调工具 or 给结论）
    # ------------------------------------------------------------------
    def think(self, state: AgentState) -> dict[str, Any]:
        """决策节点。按 ``tool_mode`` 分派到 ReAct 或原生 Function Calling。"""
        if self.tool_mode == "native":
            return self._think_native(state)
        return self._think_react(state)

    def _think_native(self, state: AgentState) -> dict[str, Any]:
        """原生 Function Calling 的决策：工具经 API 下发，读结构化 tool_calls。"""
        current_step = state.get("current_step", 0) + 1
        max_steps = state.get("max_steps", 12)

        history = self._messages_to_dicts(state.get("messages", []))
        if not history:
            history = [{"role": "user", "content": state.get("goal", "")}]
        history = self._compact_history(history, state)
        messages = [{"role": "system", "content": self._build_system_native()}]
        skill = self._skill_guidance(state)
        if skill:
            messages.append({"role": "system", "content": skill})
        wm_index = self._format_working_memory(state.get("working_memory", {}))
        if wm_index:
            messages.append({"role": "system", "content": wm_index})
        messages += history

        # 软停止：连续滥用同一需审批工具时注入引导提示（在 LLM 调用之前）
        _soft_warn = self._repetitive_tool_warning(state)
        if _soft_warn:
            messages.append({"role": "system", "content": _soft_warn})
            logger.info("Soft-stop warning injected at step %d", current_step)

        ctx = MiddlewareContext(
            operation="llm/think",
            trace_id=state.get("trace_id"),
            session_id=state.get("session_id"),
            agent_id=state.get("agent_id"),
            role=state.get("role", "default"),
        )

        steps = list(state.get("steps", []))
        update: dict[str, Any] = {"current_step": current_step, "updated_at": self._now()}

        # before_llm Hook（缓存命中可短路）
        short_circuit: Optional[str] = None
        if self.middleware is not None:
            messages = self.middleware.exec_before_llm(ctx, messages)
            if ctx.is_short_circuited:
                sc = ctx.short_circuit_result
                short_circuit = sc if isinstance(sc, str) else json.dumps(sc, ensure_ascii=False)

        # 短路结果在本模式下直接作为结论
        if short_circuit is not None:
            steps.append(
                ThoughtStep(step=current_step, thought="(中间件短路)", is_final=True,
                           final_answer=short_circuit, trace_span_id=ctx.trace_id)
            )
            update.update(
                messages=[{"role": "assistant", "content": short_circuit}], steps=steps,
                final_answer=short_circuit, status="finished",
                pending_tool_calls=[], last_action=None, last_action_input=None,
            )
            return update

        _usage_tok = self._bind_usage(state)
        try:
            with _span(state, "llm_call", "think_native"):
                result = self.llm.chat_with_tools(
                    messages, self.broker.list_tools_openai_format()
                )
        finally:
            self._unbind_usage(_usage_tok)
        new_messages = [result.raw_message]

        # 步数耗尽：无论模型想调工具还是没给结论，都强制收尾，避免死循环
        if current_step >= max_steps:
            guard = f"已达到最大步数 {max_steps}，停止继续调用工具。"
            answer = result.content.strip() or (
                f"已达到最大步数 {max_steps}，任务未完全完成。最后观察："
                f"{state.get('last_observation', '(无)')}"
            )
            if not result.content.strip():
                answer = (
                    f"已达到最大步数 {max_steps}，任务未完全完成。最后观察："
                    f"{state.get('last_observation', '(无)')}"
                )
            steps.append(
                ThoughtStep(step=current_step, thought=result.content or guard, is_final=True,
                           final_answer=answer, trace_span_id=ctx.trace_id)
            )
            update.update(
                messages=new_messages, steps=steps, final_answer=answer,
                status="finished", pending_tool_calls=[], last_action=None,
            )
            return update

        # 模型想调工具（可多个）
        if result.wants_tools:
            pending = [
                {"id": c.id, "name": c.name, "arguments": c.arguments}
                for c in result.tool_calls
            ]
            names = "、".join(c["name"] for c in pending)
            thought = result.content or f"调用工具：{names}"
            steps.append(
                ThoughtStep(step=current_step, thought=thought,
                           action=pending[0]["name"], action_input=pending[0]["arguments"],
                           trace_span_id=ctx.trace_id)
            )
            update.update(
                messages=new_messages, steps=steps, pending_tool_calls=pending,
                last_action=pending[0]["name"], last_action_input=pending[0]["arguments"],
                status="running",
            )
            return update

        # 没有工具调用 → 视为最终结论
        answer = result.content.strip()
        if not answer:
            new_messages.append({
                "role": "user",
                "content": "你既没有调用工具也没有给出结论。请给出本子任务的结论，或调用合适的工具。",
            })
            steps.append(ThoughtStep(step=current_step, thought="(空回复，要求给出结论)",
                                     observation="(已要求补充结论)"))
            update.update(messages=new_messages, steps=steps, pending_tool_calls=[],
                          last_action=None, status="running")
            return update

        steps.append(
            ThoughtStep(step=current_step, thought="", is_final=True,
                       final_answer=answer, trace_span_id=ctx.trace_id)
        )
        update.update(
            messages=new_messages, steps=steps, final_answer=answer, status="finished",
            pending_tool_calls=[], last_action=None, last_action_input=None,
        )
        return update


    def _repetitive_tool_warning(self, state: AgentState) -> Optional[str]:
        """软停止：检测同一 requires_approval 工具被连续调用 ≥3 次，返回引导提示。

        扫描最近的 assistant 消息（原始 JSON 决策），提取 action 字段。
        仅对 requires_approval=True 的工具告警——这类工具每次调用都触发
        interrupt/审批/沙箱，连续滥用代价最高（如 code_executor 死循环）。
        """
        raw_msgs = state.get("messages", [])
        if not raw_msgs:
            return None
        dicts = self._messages_to_dicts(raw_msgs)
        # 取最近的 assistant 消息，提取 action
        recent_actions: list[str] = []
        for m in reversed(dicts):
            if m.get("role") != "assistant":
                continue
            content = m.get("content", "")
            if not content:
                continue
            decision = _try_extract_json(content)
            if decision is None:
                continue
            action = decision.get("action")
            if action:
                recent_actions.append(str(action))
            if len(recent_actions) >= 5:
                break
        if len(recent_actions) < 3:
            return None
        # 最近 3 次是否同一工具
        if recent_actions[0] != recent_actions[1] or recent_actions[1] != recent_actions[2]:
            return None
        tool_name = recent_actions[0]
        tool_def = self.broker.get(tool_name) if self.broker else None
        if tool_def is None or not tool_def.requires_approval:
            return None
        return (
            f"【框架提示】你已连续调用 {tool_name} {len(recent_actions)} 次。"
            "请停下来反思：\n"
            f"1. 是否有专用工具可替代 {tool_name}？（如 eda / chart_generator / sql_query）\n"
            "2. 已有信息是否足够产出 final_answer？不要为了'再算点东西'而重复调用。\n"
            "3. 如果确实需要继续，请在 thought 中说明前几次结果的具体不足。"
        )

    def _think_react(self, state: AgentState) -> dict[str, Any]:
        """ReAct 决策：工具清单拼进 system 提示，解析模型输出的 JSON 决策。"""
        current_step = state.get("current_step", 0) + 1
        max_steps = state.get("max_steps", 12)

        # 组装消息：system（含最新工具列表）+ 资产索引 + 历史
        history = self._messages_to_dicts(state.get("messages", []))
        if not history:
            history = [{"role": "user", "content": state.get("goal", "")}]
        history = self._compact_history(history, state)
        messages = [{"role": "system", "content": self._build_system()}]
        skill = self._skill_guidance(state)
        if skill:
            messages.append({"role": "system", "content": skill})
        wm_index = self._format_working_memory(state.get("working_memory", {}))
        if wm_index:
            messages.append({"role": "system", "content": wm_index})
        messages += history

        # 软停止：连续滥用同一需审批工具时注入引导提示（在 LLM 调用之前）
        _soft_warn = self._repetitive_tool_warning(state)
        if _soft_warn:
            messages.append({"role": "system", "content": _soft_warn})
            logger.info("Soft-stop warning injected at step %d", current_step)

        ctx = MiddlewareContext(
            operation="llm/think",
            trace_id=state.get("trace_id"),
            session_id=state.get("session_id"),
            agent_id=state.get("agent_id"),
            role=state.get("role", "default"),
        )

        # before_llm Hook（缓存命中可短路）
        raw: Optional[str] = None
        if self.middleware is not None:
            messages = self.middleware.exec_before_llm(ctx, messages)
            if ctx.is_short_circuited:
                sc = ctx.short_circuit_result
                raw = sc if isinstance(sc, str) else json.dumps(sc, ensure_ascii=False)
                logger.info("think short-circuited by middleware")

        # 正常调用 LLM
        if raw is None:
            _usage_tok = self._bind_usage(state)
            try:
                with _span(state, "llm_call", "think"):
                    raw = self.llm.chat(messages)
            finally:
                self._unbind_usage(_usage_tok)
            if self.middleware is not None:
                raw = self.middleware.exec_after_llm(ctx, raw)

        steps = list(state.get("steps", []))
        new_messages = [{"role": "assistant", "content": raw}]
        update: dict[str, Any] = {"current_step": current_step, "updated_at": self._now()}

        # 解析模型 JSON
        decision = _try_extract_json(raw)

        # 情况 A：输出无法解析为 JSON → 回灌纠错，路由回 think 自纠
        if decision is None:
            new_messages.append(
                {"role": "user", "content": "你上一步的输出不是合法 JSON。请【只】输出一个 JSON 对象，不要其他文字。"}
            )
            steps.append(
                ThoughtStep(step=current_step, thought="(模型输出无法解析为 JSON)",
                           observation="(已要求重新输出 JSON)")
            )
            update.update(messages=new_messages, steps=steps, last_action=None,
                          last_action_input=None, status="running")
            return update

        thought = str(decision.get("thought", ""))
        final_answer = decision.get("final_answer")
        action = decision.get("action")
        action_input = decision.get("action_input") or {}
        if not isinstance(action_input, dict):
            action_input = {}

        # 情况 B：给出最终结论
        if final_answer and not action:
            steps.append(
                ThoughtStep(step=current_step, thought=thought, is_final=True,
                           final_answer=str(final_answer), trace_span_id=ctx.trace_id)
            )
            update.update(
                messages=new_messages, steps=steps, final_answer=str(final_answer),
                status="finished", last_action=None, last_action_input=None,
            )
            return update

        # 情况 C：JSON 合法但既无 action 也无 final_answer → 要求补字段
        if not action:
            new_messages.append(
                {"role": "user",
                 "content": "JSON 中缺少 action 或 final_answer 字段。需要工具请给 action，能回答请给 final_answer。"}
            )
            steps.append(ThoughtStep(step=current_step, thought=thought,
                                     observation="(字段缺失，要求重新输出)"))
            update.update(messages=new_messages, steps=steps, last_action=None, status="running")
            return update

        # 情况 D：达到最大步数仍要调工具 → 强制收尾，防止死循环
        if current_step >= max_steps:
            guard = f"已达到最大步数 {max_steps}，停止继续调用工具。"
            steps.append(
                ThoughtStep(step=current_step, thought=thought, action=action,
                           action_input=action_input, observation=guard, trace_span_id=ctx.trace_id)
            )
            update.update(
                messages=new_messages + [{"role": "user", "content": guard}],
                steps=steps, last_action=None,
                final_answer=f"已达到最大步数 {max_steps}，任务未完全完成。最后观察："
                             f"{state.get('last_observation', '(无)')}",
                status="failed",
            )
            return update

        # 情况 E：正常选择工具，交给 action 节点
        steps.append(
            ThoughtStep(step=current_step, thought=thought, action=action,
                       action_input=action_input, trace_span_id=ctx.trace_id)
        )
        update.update(
            messages=new_messages, steps=steps, last_action=action,
            last_action_input=action_input, status="running",
        )
        return update

    # ------------------------------------------------------------------
    # action 节点：经 ToolBroker 执行工具，回填 observation
    # ------------------------------------------------------------------
    def action(self, state: AgentState) -> dict[str, Any]:
        """执行节点。按 ``tool_mode`` 分派。"""
        if self.tool_mode == "native":
            return self._action_native(state)
        return self._action_react(state)

    def _invoke_context(self, state: AgentState) -> dict:
        """透传给 Broker 的上下文（权限/审计/追踪/记忆都从这里取）。"""
        return {
            "trace_id": state.get("trace_id"),
            "session_id": state.get("session_id"),
            "agent_id": state.get("agent_id"),
            "role": state.get("role", "default"),
            "working_memory": state.get("working_memory", {}),
            "data_source_manager": self.datasources,
            "skill_registry": self.skill_registry,
            "step": state.get("current_step"),
        }

    def _action_native(self, state: AgentState) -> dict[str, Any]:
        """执行本批 ``pending_tool_calls``，每个结果以 role="tool" 配对回传。

        模型可一次请求多个工具；这里**按序执行**而非真并行 —— 限流是按工具维度
        的滑动窗口、审计要保序、沙箱有并发上限，并行会破坏这三者的确定性。
        对外语义仍是"一次请求、全部执行、全部回传"。
        """
        pending = list(state.get("pending_tool_calls") or [])
        invoke_context = self._invoke_context(state)

        steps = list(state.get("steps", []))
        working_memory = dict(state.get("working_memory", {}))
        tool_messages: list[dict] = []
        observations: list[str] = []

        for call in pending:
            name = call.get("name") or ""
            args = call.get("arguments") or {}

            # 审批闸门：requires_approval 工具先授权、再 interrupt 等待人工决策
            approved, observation, approval = self._request_tool_approval(
                name, args, state, invoke_context
            )
            if not approved:
                observations.append(f"[{name}] {observation}")
                tool_messages.append(
                    self.llm.tool_result_message(call.get("id", ""), observation)
                )
                continue

            # 审批凭证注入独立副本：同批可能有多个工具，不能让 A 的凭证污染 B
            # （Broker 还会校验凭证里的工具名与本次调用一致）。
            call_ctx = dict(invoke_context)
            if approval:
                call_ctx["approval"] = approval

            # Broker 内部跑中间件、PDP、审批凭证校验、参数校验、限流、沙箱、审计
            with _span(state, "tool_call", name):
                ok, text, artifacts = self.broker.invoke(name, args, call_ctx)
            observation = text if ok else f"工具调用失败：{text}"
            observation = self._settle_observation(name, observation, state)
            observations.append(f"[{name}] {observation}")
            tool_messages.append(
                self.llm.tool_result_message(call.get("id", ""), observation)
            )
            if artifacts:
                working_memory[f"result_{name}"] = artifacts

        # 把 observation 回填到本轮 ThoughtStep（首个调用的那一步）
        if steps and observations:
            merged = "\n".join(observations)
            steps[-1] = steps[-1].model_copy(update={"observation": merged})

        return {
            "last_observation": "\n".join(observations),
            "messages": tool_messages,
            "steps": steps,
            "working_memory": working_memory,
            "pending_tool_calls": [],
            "updated_at": self._now(),
        }

    def _action_react(self, state: AgentState) -> dict[str, Any]:
        tool_name = state.get("last_action")
        tool_args = state.get("last_action_input") or {}

        # 透传上下文给 Broker（权限/审计/追踪/记忆都从这里取）
        invoke_context = {
            "trace_id": state.get("trace_id"),
            "session_id": state.get("session_id"),
            "agent_id": state.get("agent_id"),
            "role": state.get("role", "default"),
            "working_memory": state.get("working_memory", {}),
            "data_source_manager": self.datasources,
            "skill_registry": self.skill_registry,
            "step": state.get("current_step"),
        }

        # 审批闸门：requires_approval 工具先授权、再 interrupt 等待人工决策
        approved, observation, approval = self._request_tool_approval(
            tool_name, tool_args, state, invoke_context
        )

        # 把 observation 回填到本轮 ThoughtStep
        steps = list(state.get("steps", []))
        if not approved:
            if steps:
                steps[-1] = steps[-1].model_copy(update={"observation": observation})
            new_messages = [{"role": "user", "content": f"Observation:\n{observation}"}]
            return {
                "last_observation": observation,
                "messages": new_messages,
                "steps": steps,
                "updated_at": self._now(),
            }

        # 审批凭证注入独立副本后交给 Broker（Broker 第 3.5 步强制校验）
        call_ctx = dict(invoke_context)
        if approval:
            call_ctx["approval"] = approval

        # Broker 内部会跑工具中间件、PDP、审批凭证、参数校验、限流、沙箱、审计
        with _span(state, "tool_call", tool_name):
            ok, text, artifacts = self.broker.invoke(tool_name, tool_args, call_ctx)
        observation = text if ok else f"工具调用失败：{text}"
        observation = self._settle_observation(tool_name, observation, state)

        if steps:
            steps[-1] = steps[-1].model_copy(update={"observation": observation})

        new_messages = [{"role": "user", "content": f"Observation:\n{observation}"}]

        # 结构化产物沉淀进工作记忆，供后续工具/节点使用
        working_memory = dict(state.get("working_memory", {}))
        if artifacts:
            working_memory[f"result_{tool_name}"] = artifacts

        return {
            "last_observation": observation,
            "messages": new_messages,
            "steps": steps,
            "working_memory": working_memory,
            "updated_at": self._now(),
        }

    # ------------------------------------------------------------------
    # 路由：think 之后去哪
    # ------------------------------------------------------------------
    def route(self, state: AgentState) -> str:
        """条件边路由，返回 key，由 graph 的 path_map 映射到节点。"""
        if state.get("pending_tool_calls"):
            return "act"          # 原生模式：本批工具待执行
        if state.get("final_answer"):
            return "end"          # 有最终结论 → 结束
        if not state.get("last_action"):
            return "rethink"      # JSON 非法/缺字段 → 回 think 自纠
        return "act"             # 选了工具 → 去 action
