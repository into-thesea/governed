"""harness.tool_broker —— 工具注册与统一调度入口（Tool Broker）。

所有工具调用必须经过 Broker，不允许 Agent 直接调实现函数。

Broker 的完整调用链路：
    1. 中间件 before_tool Hook（PII检测/日志）
    2. 工具存在性检查
    3. PDP 权限检查（如果配置了 PDP）
    3.5 人工审批凭证（requires_approval 工具须带节点层审批产生的匹配凭证，否则 fail closed）
    4. 参数校验（JSON Schema 基础校验）
    5. 熔断准入（按工具名独立的三态熔断，下游持续故障时快速失败）
    6. 限流准入（滑动时间窗口，判定与记账在同一次加锁内完成）
    7. 工具结果缓存：命中则跳过第 8 步的执行，**其余各步照常**（见下）
    8. 调用实现函数（高风险工具走安全沙箱）
    9. 中间件 after_tool Hook（PII检测/日志/结果修改）
   10. 审计（记录最终结果，含 cache_hit 标记）
   11. 结果包装：统一返回 (ok, text, artifacts)

缓存为什么落在这个位置（而不是做成 before_tool 中间件）：命中要省的是"真正的
执行"，不能顺带省掉权限、参数校验、熔断、限流与 after_tool —— 那些是合规物，
且都不贵。缓存值写在 after_tool **之后**，存的是脱敏后的最终结果，否则缓存自己
就成了 PII 的泄漏面。详见 ``harness.cache`` 的模块说明。

工具实现函数的约定签名：
    handler(args: dict, context: dict) -> tuple[bool, str, dict]
        返回 (是否成功, 给 LLM 看的观察文本, 结构化附加信息)
"""

from __future__ import annotations

import copy
import json
import logging
import threading
import time
import uuid
from datetime import datetime
from typing import Any, Callable, Optional

from harness.approval_policy import RISK_CRITICAL, RISK_MEDIUM
from harness.audit import AuditLogger
from harness.circuit_breaker import CircuitBreaker, build_circuit_breaker
from harness.events import (
    GUARD_LAYER_APPROVAL,
    GUARD_LAYER_BREAKER,
    GUARD_LAYER_PDP,
    GUARD_LAYER_RATE_LIMIT,
    GUARD_LAYER_SANDBOX,
    emit_guard_decision,
)
from harness.middleware import MiddlewareContext, MiddlewareManager
from harness.models import ToolDef
from harness.rate_limiter import RateLimiter, build_rate_limiter

logger = logging.getLogger(__name__)

# 工具实现函数的类型别名
ToolHandler = Callable[[dict, dict], tuple[bool, str, dict]]

# 限流滑动窗口长度。字段名是 rate_limit_per_min，窗口因此固定为 60 秒 ——
# 这不是可调参数，改动它会让字段名与语义脱节。
_RATE_LIMIT_WINDOW_SECONDS = 60.0


def _default_sandbox() -> Optional[Any]:
    """按配置构造默认沙箱执行器。

    SandboxExecutor 构造不触碰网络（首次执行才连 OpenSandbox 服务端），
    因此这里默认构造是安全的。sandbox.enabled=False 时返回 None，此时
    标了 run_in_sandbox 的工具会 fail closed，不会退化为裸跑。
    """
    from harness.config import settings

    if not settings.sandbox.enabled:
        logger.info("sandbox.enabled=False，高风险工具将 fail closed")
        return None

    from harness.sandbox import SandboxExecutor

    return SandboxExecutor()


def _default_breaker() -> Optional[CircuitBreaker]:
    """按配置构造默认熔断器；关闭时返回 None（调用方据此跳过熔断这一环）。"""
    from harness.config import settings

    cb = settings.circuit
    return build_circuit_breaker(
        enabled=cb.enabled,
        failure_threshold=cb.failure_threshold,
        cooldown_seconds=cb.cooldown_seconds,
        half_open_trials=cb.half_open_trials,
    )


def _default_cache() -> Optional[Any]:
    """按配置构造默认工具结果缓存；关闭时返回 None。"""
    from harness.cache import ToolResultCache
    from harness.config import settings

    if not settings.cache.enabled:
        logger.info("cache.enabled=False，工具结果不做缓存")
        return None
    return ToolResultCache(max_entries=settings.cache.max_entries)


def render_tool_descriptions(tools: list[ToolDef]) -> str:
    """根据给定工具列表渲染给 LLM 看的工具描述文本（Broker 与其受限视图共用）。"""
    if not tools:
        return "（无可用工具）"

    lines = ["可用工具列表："]
    for i, tool_def in enumerate(tools, 1):
        lines.append(f"\n{i}. {tool_def.name}")
        lines.append(f"   描述：{tool_def.description}")
        if tool_def.parameters.get("properties"):
            lines.append("   参数：")
            for param_name, param_def in tool_def.parameters["properties"].items():
                required = param_name in tool_def.parameters.get("required", [])
                req_mark = "（必填）" if required else "（可选）"
                lines.append(f"     - {param_name}: {param_def.get('type', 'any')} {req_mark} - {param_def.get('description', '')}")
        if tool_def.requires_approval:
            lines.append("   注意：此工具需要人工审批")
    return "\n".join(lines)


class ToolBroker:
    """工具注册与统一调度入口。

    所有工具调用过 Broker，统一做存在性检查、权限校验、参数校验、
    限流、异常兜底、中间件 Hook、结果包装。

    使用方式：
        broker = ToolBroker()
        broker.register(tool_def, handler)
        ok, text, artifacts = broker.invoke("calculator", {"expression": "15*23"}, context)
    """

    def __init__(
        self,
        middleware_manager: Optional[MiddlewareManager] = None,
        pdp: Optional[Any] = None,
        sandbox_executor: Optional[Any] = None,
        audit_logger: Optional[AuditLogger] = None,
        circuit_breaker: Optional[Any] = None,
        cache: Optional[Any] = None,
        rate_limiter: Optional[Any] = None,
        approval_threshold: str = RISK_MEDIUM,
        approval_deny_threshold: str = RISK_CRITICAL,
    ):
        """初始化 Tool Broker。

        Args:
            middleware_manager: 可插拔中间件管理器（可选，没有则不执行 Hook）
            pdp: PDP 策略决策点实例（可选，没有则跳过权限检查）
            sandbox_executor: 安全沙箱执行器。缺省按 sandbox.enabled 自动构造
                （构造不触碰网络，首次执行才连接）；显式传 None 且沙箱启用时同样
                走自动构造；显式传 False 表示关闭沙箱，会归一为 None，标了
                run_in_sandbox 的工具因此走 fail-closed 分支而不是裸跑。
            audit_logger: 审计器实例（可选，没有则不记录审计日志）
            circuit_breaker: 按工具名的三态熔断器。缺省按 circuit.enabled 自动
                构造；显式传 False 表示关闭熔断。
            cache: 工具结果缓存（``harness.cache.ToolResultCache``）。缺省按
                cache.enabled 自动构造；显式传 False 表示关闭缓存。
            approval_threshold: 风险高过这条线就问人（默认 ``medium``）。
            approval_deny_threshold: 风险达到这条线**直接拒**，连问都不问（默认
                ``critical``）。这是**部署方**的旋钮 —— 领域包只能声明风险档，
                拒还是问由这条线拍板（见 ``docs/技术选型决策.md`` D-007）。
        """
        self._tools: dict[str, tuple[ToolDef, ToolHandler]] = {}
        # 限流：抽象为 RateLimiter ABC，单机默认 ProcessRateLimiter（进程内滑动窗口），
        # 多副本可切 RedisRateLimiter（zset+Lua 共享状态）。ToolBroker 只依赖 ABC。
        if rate_limiter is None:
            self._rate_limiter: RateLimiter = build_rate_limiter("process")
        elif rate_limiter is False:
            # 显式 False = 关闭限流（不推荐，仅用于测试/调试）
            from harness.rate_limiter import NoopRateLimiter
            self._rate_limiter = NoopRateLimiter()
        else:
            self._rate_limiter = rate_limiter
        # 累计调用次数的独立锁（与限流窗口分开：限流会涨落，计数只增不减）
        self._counts_lock = threading.Lock()
        # 会话级人工审批豁免：键 (session_id, tool) → 授予记录。
        # 与 PDP 的区别：PDP 是 (角色, 工具) 的**全局**策略表，写进去会影响所有任务且永久；
        # 这里只豁免"人工审批"这一步，且随任务收尾清除 —— 进程重启即丢是**有意**的失败方向
        # （重新问一次，而不是把放宽永久留下来）。
        # ponytail: 无 TTL。上界 = 该会话被授予过的工具数（工具总数是常数级）；任务若始终
        # 不到终态（进程被杀 / 被遗弃），条目留到进程结束。真要更严就加 TTL 清扫或按会话数封顶。
        self._session_grants: dict[tuple[str, str], dict[str, Any]] = {}
        self._grant_lock = threading.Lock()
        # 按工具的风险策略：领域包声明"多危险"，框架决定"要不要问人"。
        # 与工具定义分开放 —— `ToolDef` 是纯数据（要渲染给 LLM、可能被序列化），
        # 而策略是 (args) -> str 的**代码**（决策见 docs/技术选型决策.md D-006）。
        # 与豁免表同类：读多写少、随包回收，共用 _grant_lock。
        self._risk_policies: dict[str, Callable[[dict], str]] = {}
        # 累计调用次数，键 (工具名, 是否成功)。**累计**而不是窗口：`_call_log` 那个
        # 一分钟窗口是给限流用的，它会涨也会落，当监控指标用是错的。这一份只增不减，
        # 供 /metrics 导出（Prometheus 的 counter 语义，重启归零由 rate() 处理）。
        self._call_counts: dict[tuple[str, bool], int] = {}
        # 审批的两条线：都是**部署方配置**，不是领域包能改的。
        self.approval_threshold = approval_threshold
        self.approval_deny_threshold = approval_deny_threshold
        # 三个"可注入或关闭"的组件都是同一套开关语义：**实例 = 用这个 / None = 按配置
        # 自动构造 / False = 关闭**。传 True 是很自然的写法，但它不在这套语义里 ——
        # 早先会被原样存下去，直到 invoke 深处才炸成
        # `AttributeError: 'bool' object has no attribute 'fingerprint'`，离真正的原因
        # 十万八千里。当场拒绝并说清三种写法，比让人从调用栈里倒推便宜得多。
        for name, value in (
            ("sandbox_executor", sandbox_executor),
            ("circuit_breaker", circuit_breaker),
            ("cache", cache),
            ("rate_limiter", rate_limiter),
        ):
            if value is True:
                raise TypeError(
                    f"{name}=True 没有意义：不传或传 None = 按配置自动构造，"
                    f"传实例 = 用该实例，传 False = 关闭。"
                )

        self.middleware = middleware_manager
        self.pdp = pdp
        # 显式传 False = 关闭沙箱，归一成 None。不能把 False 直接存进来：invoke 与
        # get_stats 判的都是 `is None`，留着 False 会得到"既非有、也非无"的中间态 ——
        # 标了 run_in_sandbox 的工具会去调 False.execute()，把 fail-closed 提示换成
        # 一句 AttributeError，sandbox_enabled 也会误报 True。
        if sandbox_executor is False:
            self.sandbox = None
        elif sandbox_executor is None:
            self.sandbox = _default_sandbox()
        else:
            self.sandbox = sandbox_executor
        self.audit = audit_logger
        # 熔断器：开关语义与沙箱一致 —— 缺省按配置自动构造，显式 False 关闭
        # （同样归一为 None，让下游只需判 `is not None`）。
        if circuit_breaker is False:
            self.breaker: Optional[CircuitBreaker] = None
        elif circuit_breaker is None:
            self.breaker = _default_breaker()
        else:
            self.breaker = circuit_breaker

        # 工具结果缓存：开关语义同上，显式 False 归一为 None，下游只判 is not None。
        if cache is False:
            self.cache: Optional[Any] = None
        elif cache is None:
            self.cache = _default_cache()
        else:
            self.cache = cache

    # ------------------------------------------------------------------
    # 注册与注销
    # ------------------------------------------------------------------
    def register(self, tool_def: ToolDef, handler: ToolHandler) -> None:
        """注册工具：定义 + 实现函数。

        同名工具已存在时会被覆盖（支持运行时热更新）。
        """
        if not callable(handler):
            raise TypeError(f"handler must be callable, got {type(handler)}")
        self._tools[tool_def.name] = (tool_def, handler)
        logger.info("Registered tool: %s (role=%s, rate_limit=%d/min)", tool_def.name, tool_def.required_role, tool_def.rate_limit_per_min)

    def unregister(self, name: str) -> bool:
        """注销工具，返回是否成功。"""
        if name in self._tools:
            del self._tools[name]
            # 注销工具时清掉它的限流窗口记录（重新注册后配额干净）。
            self._rate_limiter.reset(name)
            logger.info("Unregistered tool: %s", name)
            return True
        return False

    def get(self, name: str) -> Optional[ToolDef]:
        """按工具名查定义，不存在返回 None。"""
        entry = self._tools.get(name)
        return entry[0] if entry else None

    def get_handler(self, name: str) -> Optional[ToolHandler]:
        """按工具名查实现函数，不存在返回 None。"""
        entry = self._tools.get(name)
        return entry[1] if entry else None

    def list_tools(self) -> list[ToolDef]:
        """列出所有已注册工具的定义，**按名排序**。

        顺序必须是确定的，且不能取决于注册顺序：工具定义块会被渲染进提示词，
        而服务端的提示词前缀缓存按字节比对 —— 顺序一变（例如多接了一个外部
        MCP server、或对端返回顺序不同），整块缓存静默失效，每一轮都按全价计费。
        MCP 规范同样要求列表结果确定性排序。
        """
        return sorted((entry[0] for entry in self._tools.values()), key=lambda t: t.name)

    def search(self, query: str) -> list[ToolDef]:
        """按关键词搜索工具（名称或描述匹配）。"""
        query_lower = query.lower()
        results = []
        for tool_def in self.list_tools():
            if query_lower in tool_def.name.lower() or query_lower in tool_def.description.lower():
                results.append(tool_def)
        return results

    # ------------------------------------------------------------------
    # 工具描述生成（给 LLM 看）
    # ------------------------------------------------------------------
    def list_tool_descriptions(self) -> str:
        """生成给 LLM 看的工具描述文本（ReAct Prompt 用）。"""
        return render_tool_descriptions(self.list_tools())

    def list_tools_openai_format(self) -> list[dict]:
        """生成 OpenAI Function Calling 格式的工具列表。"""
        tools = []
        for tool_def in self.list_tools():
            tools.append({
                "type": "function",
                "function": {
                    "name": tool_def.name,
                    "description": tool_def.description,
                    "parameters": tool_def.parameters,
                },
            })
        return tools

    def scoped(
        self,
        allowed_tools: list[str],
        force_role: Optional[str] = None,
    ) -> "ScopedBroker":
        """派生一个只暴露/允许指定工具的受限视图（供专业子 Agent 最小授权）。

        Args:
            allowed_tools: 允许的工具名列表；传 ["*"] 表示全部工具。
            force_role: 若提供，视图内所有调用强制以此角色过 PDP。
        """
        return ScopedBroker(self, allowed_tools, force_role=force_role)

    # ------------------------------------------------------------------
    # 会话级人工审批豁免（"本任务内总是允许 / 总是拒绝"）
    # ------------------------------------------------------------------
    def grant_session_approval(
        self,
        session_id: str,
        tool_name: str,
        effect: str,
        *,
        granted_by: str = "",
        granted_role: str = "",
        comment: str = "",
        request_id: str = "",
        trace_id: Optional[str] = None,
        task_id: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """授予"本任务内该工具不再询问审批人"。

        ``effect`` 为 ``"allow"``（后续直接放行）或 ``"deny"``（后续直接驳回）。
        **只影响人工审批那一步** —— ``invoke`` 的 PDP、参数校验、熔断、限流、沙箱与审计
        全部照旧（见设计 §3.1）。同一 (session, tool) 重复授予是**覆盖**，不是追加。

        ``trace_id`` / ``task_id`` / ``agent_id`` 只为留痕（审计与事件归因），不参与判定。
        """
        if effect not in ("allow", "deny"):
            raise ValueError(f"会话豁免的 effect 必须是 allow 或 deny，得到: {effect!r}")

        grant = {
            "grant_id": f"aprgrant_{uuid.uuid4().hex[:12]}",
            "session_id": session_id,
            "tool": tool_name,
            "effect": effect,
            "granted_by": granted_by,
            "granted_role": granted_role,
            "granted_at": datetime.now().isoformat(),
            "comment": comment,
            "request_id": request_id,
        }
        with self._grant_lock:
            self._session_grants[(session_id, tool_name)] = grant

        # 锁外留痕：授予是一次"放宽"，一旦发生必须可见、必须留痕（挂账 #13 的定位）
        self._record_grant(grant, applied=False, trace_id=trace_id, agent_id=agent_id)
        emit_guard_decision(
            trace_id, layer=GUARD_LAYER_APPROVAL, decision=effect,
            reason="已授予本任务内的审批豁免（该工具不再询问审批人）",
            tool=tool_name, task_id=task_id, agent_id=agent_id,
            via="session_grant", granted_by=granted_by, granted_role=granted_role,
            comment=comment, request_id=request_id, grant_id=grant["grant_id"],
        )
        return dict(grant)

    def apply_session_grant(
        self,
        session_id: str,
        tool_name: str,
        *,
        trace_id: Optional[str] = None,
        task_id: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        """读取该 (会话, 工具) 的豁免并**留痕**；没有则返回 None。

        唯一的读取入口，且自带留痕 —— 唯一的调用方（节点审批闸门）每次都需要这次留痕：
        靠豁免放行的调用是一次"静默放宽"，必须与拦截一样可见。不删除条目（"应用"不是
        "消费"：同一豁免在本任务内会生效多次）。
        """
        with self._grant_lock:
            grant = self._session_grants.get((session_id, tool_name))
            snapshot = dict(grant) if grant is not None else None
        if snapshot is None:
            return None

        self._record_grant(snapshot, applied=True, trace_id=trace_id, agent_id=agent_id)
        emit_guard_decision(
            trace_id, layer=GUARD_LAYER_APPROVAL, decision=snapshot["effect"],
            reason="命中本任务内的审批豁免，未再询问审批人",
            tool=tool_name, task_id=task_id, agent_id=agent_id,
            via="session_grant", applied=True, grant_id=snapshot["grant_id"],
        )
        return snapshot

    def list_session_grants(self, session_id: str) -> list[dict[str, Any]]:
        """该会话当前生效的全部豁免（控制台展示用）。"""
        with self._grant_lock:
            return [
                dict(g) for (sid, _), g in self._session_grants.items() if sid == session_id
            ]

    @property
    def session_grants_total(self) -> int:
        """当前进程内生效的豁免总条数（管控面展示用）。"""
        with self._grant_lock:
            return len(self._session_grants)

    def clear_session_grants(self, session_id: str) -> int:
        """清掉该会话的全部豁免，返回清掉的条数（任务收尾调用）。"""
        with self._grant_lock:
            keys = [k for k in self._session_grants if k[0] == session_id]
            for k in keys:
                del self._session_grants[k]
        return len(keys)

    def _record_grant(
        self,
        grant: dict[str, Any],
        *,
        applied: bool,
        trace_id: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> None:
        """落一条豁免审计；审计器缺失或抛错都不能影响主流程。"""
        if self.audit is None:
            return
        try:
            self.audit.record_approval_grant(
                tool_name=grant["tool"], session_id=grant["session_id"],
                effect=grant["effect"], applied=applied,
                granted_by=grant["granted_by"], granted_role=grant["granted_role"],
                comment=grant["comment"], request_id=grant["request_id"],
                grant_id=grant["grant_id"], agent_id=agent_id or "unknown",
                trace_id=trace_id,
            )
        except Exception:  # noqa: BLE001 - 观测面绝不拖垮主流程
            logger.exception("记录会话豁免审计失败")

    # ------------------------------------------------------------------
    # 风险策略（领域包声明"多危险"，框架决定"要不要问"）
    # ------------------------------------------------------------------
    def register_risk_policy(self, tool_name: str, policy: Callable[[dict], str]) -> None:
        """注册某工具的风险策略 ``(args) -> "low"|"medium"|"high"|"unknown"``。

        策略只回答"多危险"，**不回答"要不要问人"** —— 后者由阈值与兜底断言决定
        （见 ``harness.approval_policy``）。不注册 = 默认按需审批（与今天一致）。
        """
        with self._grant_lock:
            self._risk_policies[tool_name] = policy

    def risk_policy_for(self, tool_name: str) -> Optional[Callable[[dict], str]]:
        with self._grant_lock:
            return self._risk_policies.get(tool_name)

    def clear_risk_policy(self, tool_name: str) -> bool:
        with self._grant_lock:
            return self._risk_policies.pop(tool_name, None) is not None

    def call_counts(self) -> dict[tuple[str, bool], int]:
        """累计调用次数 ``{(工具名, 是否成功): 次数}``（进程内，只增不减）。

        与 ``get_stats()["tools"][*]["recent_calls_1min"]`` 的区别：那是**滚动窗口**，
        会涨也会落，当监控的 counter 用是错的。这一份是给 ``/metrics`` 的口径。
        """
        with self._counts_lock:
            return dict(self._call_counts)

    def approval_fallback(self, tool_name: str) -> Optional[str]:
        """这一次调用可依靠的确定性兜底机制名；没有则 None。

        必须看**运行时**状态，而且是**真的可用**，不是"配置里开了"：沙箱客户端
        构造时不碰网络，所以"``self.sandbox`` 非空"只说明配置开着 —— 服务端没起时
        照样是空的兜底。自动放行的理由只能是"有机制兜底"，不能是"我判断它安全"。
        """
        entry = self._tools.get(tool_name)
        tool_def = entry[0] if entry else None
        if tool_def is None or not tool_def.run_in_sandbox:
            return None
        return "sandbox" if self.sandbox_available() else None

    def sandbox_available(self) -> bool:
        """向沙箱要一个**运行时**的可用性结论（带缓存，见 ``SandboxExecutor.available``）。"""
        executor = self.sandbox
        if executor is None:
            return False
        probe = getattr(executor, "available", None)
        if probe is None:
            # 替身或第三方执行器没有探测口：按"存在即可用"处理，与旧行为一致。
            # 真实的 SandboxExecutor 有 available()，走下面的真探测。
            return True
        try:
            ok, reason = probe()
        except Exception:  # noqa: BLE001 - 探测失败按不可用处理（fail closed）
            logger.debug("沙箱可用性探测异常，按不可用处理", exc_info=True)
            return False
        if not ok:
            logger.info("沙箱当前不可用，自动放行降级为按需审批：%s", reason)
        return bool(ok)

    # ------------------------------------------------------------------
    # 授权预检（人工审批之前先判"准不准做"）
    # ------------------------------------------------------------------
    def authorize(self, tool_name: str, context: Optional[dict] = None) -> tuple[bool, str]:
        """授权预检：调用方是否允许调用此工具（存在性 + PDP），不做审批/限流/执行。

        节点层在发起 LangGraph interrupt 人工审批**之前**先调用本方法：先判
        "准不准做"（PDP），再问"要不要做"（人工审批），避免对一个根本无权
        调用的工具也弹出审批卡片（挂账 #12：审批曾发生在权限判定之前）。

        Returns:
            (是否允许, 理由)。允许时理由为空串；拒绝时理由说明原因。
        """
        context = context or {}
        role = context.get("role", "default")
        entry = self._tools.get(tool_name)
        if not entry:
            return False, (
                f"工具 '{tool_name}' 不存在。可用工具：{', '.join(self._tools.keys())}"
            )
        if self.pdp is not None:
            allowed, reason = self.pdp.check(role, tool_name, context)
            if not allowed:
                return False, f"角色 '{role}' 不允许调用工具 '{tool_name}'：{reason}"
        return True, ""

    # ------------------------------------------------------------------
    # 统一调用入口（核心方法）
    # ------------------------------------------------------------------
    def invoke(
        self,
        tool_name: str,
        args: dict,
        context: Optional[dict] = None,
    ) -> tuple[bool, str, dict]:
        """统一调用入口。

        完整链路：中间件 before → 存在性 → PDP权限 → 参数校验 → 限流 → 执行 → 中间件 after → 包装

        Args:
            tool_name: 工具名
            args: 工具参数
            context: 上下文（包含 agent_id/session_id/role/trace_id 等）

        Returns:
            (是否成功, 观察文本, 结构化附加信息)
        """
        context = context or {}
        trace_id = context.get("trace_id")
        session_id = context.get("session_id")
        agent_id = context.get("agent_id")
        role = context.get("role", "default")

        # 原始参数的快照：before_tool 的中间件（PII 脱敏）会**就地改写** args，
        # 而缓存指纹必须基于原始值 —— 否则两个不同手机号的查询会被脱敏成同一个
        # 键，命中出错误结果。见 harness.cache.ToolResultCache.fingerprint。
        original_args = copy.deepcopy(args)

        # 创建中间件上下文
        mw_ctx = MiddlewareContext(
            operation=f"tool_call/{tool_name}",
            trace_id=trace_id,
            session_id=session_id,
            agent_id=agent_id,
            role=role,
        )

        # ---- 1. 中间件 before_tool Hook ----
        if self.middleware:
            # 把工具定义放进上下文：中间件据此判断**工具自己的声明**（如 pii_skip、
            # cacheable），而不必由配置按名点名工具 —— 框架不该认识领域工具名。
            entry = self._tools.get(tool_name)
            mw_ctx.extra["tool_def"] = entry[0] if entry else None
            tool_name, args = self.middleware.exec_before_tool(mw_ctx, tool_name, args)
            if mw_ctx.is_short_circuited:
                logger.info("Tool %s short-circuited by middleware", tool_name)
                result = mw_ctx.short_circuit_result
                if isinstance(result, tuple) and len(result) == 3:
                    return result
                return False, f"中间件短路返回格式错误: {type(result)}", {}

        # ---- 2. 工具存在性检查 ----
        entry = self._tools.get(tool_name)
        if not entry:
            return False, f"工具 '{tool_name}' 不存在。可用工具：{', '.join(self._tools.keys())}", {}

        tool_def, handler = entry

        # ---- 3. PDP 权限检查 ----
        if self.pdp is not None:
            allowed, reason = self.pdp.check(role, tool_name, context)
            if not allowed:
                logger.info("PDP denied: role=%s tool=%s reason=%s", role, tool_name, reason)
                self._audit(
                    trace_id=trace_id, session_id=session_id, agent_id=agent_id, role=role,
                    tool_name=tool_name, args=args, pdp_decision="deny",
                    result_ok=False, error=reason,
                )
                emit_guard_decision(
                    trace_id, layer=GUARD_LAYER_PDP, reason=reason, tool=tool_name,
                    task_id=context.get("task_id"), agent_id=agent_id, role=role,
                )
                return False, f"权限不足：角色 '{role}' 不允许调用工具 '{tool_name}'。原因：{reason}", {}

        # ---- 3.5 人工审批凭证（HITL）----
        # requires_approval 的工具必须携带由**节点层 interrupt 人工审批**产生、
        # 且与本工具名匹配的批准凭证（context["approval"]）。审批 interrupt 只能
        # 在 LangGraph 节点里触发，Broker 是与编排解耦的普通 Python；MCP Server
        # 等非交互入口（mcp_adapter）直接调本方法、拿不到凭证 —— 这里 fail
        # closed，保证任何通道都无法绕过人工审批（挂账 #11）。
        approval_cred: Optional[dict] = (
            context.get("approval") if isinstance(context, dict) else None
        )
        if tool_def.requires_approval:
            cred_valid = (
                isinstance(approval_cred, dict)
                and approval_cred.get("approved") is True
                and approval_cred.get("tool") == tool_name
            )
            if not cred_valid:
                gate_reason = (
                    "该工具需人工审批，但当前调用通道未提供与本工具匹配的审批通过凭证"
                    "（fail closed；MCP/脚本等非交互入口不得绕过人工审批）"
                )
                logger.warning(
                    "Tool %s blocked by approval gate (role=%s agent=%s credential_present=%s)",
                    tool_name, role, agent_id, approval_cred is not None,
                )
                self._audit(
                    trace_id=trace_id, session_id=session_id, agent_id=agent_id, role=role,
                    tool_name=tool_name, args=args, pdp_decision="allow",
                    result_ok=False, error=f"审批闸门拦截：{gate_reason}",
                    approval_required=True,
                    approval_id=approval_cred.get("id") if isinstance(approval_cred, dict) else None,
                )
                emit_guard_decision(
                    trace_id, layer=GUARD_LAYER_APPROVAL, reason=gate_reason, tool=tool_name,
                    task_id=context.get("task_id"), agent_id=agent_id, role=role,
                    approval_id=approval_cred.get("id")
                    if isinstance(approval_cred, dict) else None,
                )
                return False, f"工具 '{tool_name}' 被人工审批闸门拦截：{gate_reason}", {}

        # ---- 4. 参数校验 ----
        args_ok, args_error = self._validate_args(tool_def, args)
        if not args_ok:
            self._audit(
                trace_id=trace_id, session_id=session_id, agent_id=agent_id, role=role,
                tool_name=tool_name, args=args, pdp_decision="allow",
                result_ok=False, error=f"参数校验失败：{args_error}",
            )
            return False, f"参数校验失败：{args_error}", {}

        # ---- 5. 熔断准入 ----
        # 放在限流之前：已经熔断的工具不该再消耗限流配额。熔断针对的是下游
        # 持续故障，限流针对的是调用过密，前者更该优先短路。
        if self.breaker is not None:
            allowed, breaker_error = self.breaker.allow(tool_name)
            if not allowed:
                self._audit(
                    trace_id=trace_id, session_id=session_id, agent_id=agent_id, role=role,
                    tool_name=tool_name, args=args, pdp_decision="allow",
                    result_ok=False, error=f"熔断：{breaker_error}",
                )
                emit_guard_decision(
                    trace_id, layer=GUARD_LAYER_BREAKER, reason=breaker_error, tool=tool_name,
                    task_id=context.get("task_id"), agent_id=agent_id,
                )
                return False, f"熔断：{breaker_error}", {}

        # ---- 6. 限流准入（判定 + 记账原子完成）----
        rate_ok, rate_error = self._rate_limiter.acquire(tool_name, tool_def.rate_limit_per_min)
        if not rate_ok:
            self._audit(
                trace_id=trace_id, session_id=session_id, agent_id=agent_id, role=role,
                tool_name=tool_name, args=args, pdp_decision="allow",
                result_ok=False, error=f"限流：{rate_error}",
            )
            emit_guard_decision(
                trace_id, layer=GUARD_LAYER_RATE_LIMIT, reason=rate_error, tool=tool_name,
                task_id=context.get("task_id"), agent_id=agent_id,
            )
            return False, f"限流：{rate_error}", {}

        # ---- 7. 工具结果缓存查找 ----
        # 位置有意放在限流之后：命中要省的是"真正的执行"，不该顺带省掉权限、
        # 参数校验、熔断与限流。指纹用**进入本方法时的原始参数**，不能用中间件
        # 改写后的 —— PII 脱敏会把两个不同号码的查询改成同一个键。
        cache_key: Optional[str] = None
        cached: Optional[tuple[bool, str, dict]] = None
        if self.cache is not None:
            cache_key = self.cache.fingerprint(tool_name, tool_def, original_args)
            if cache_key is not None:
                cached = self.cache.get(cache_key)

        # ---- 8. 调用实现函数（缓存命中则跳过）----
        start_time = time.time()
        raised: Optional[BaseException] = None
        if cached is not None:
            ok, text, artifacts = cached
            logger.info("Tool %s served from cache", tool_name)
        else:
            try:
                if tool_def.run_in_sandbox:
                    # 高风险工具必须走沙箱；沙箱缺失时 fail closed，绝不裸跑
                    if self.sandbox is None:
                        sandbox_reason = (
                            f"沙箱已禁用（sandbox.enabled=False），拒绝执行高风险工具 "
                            f"{tool_name!r}（fail closed）。"
                        )
                        emit_guard_decision(
                            trace_id, layer=GUARD_LAYER_SANDBOX, reason=sandbox_reason,
                            tool=tool_name, task_id=context.get("task_id"), agent_id=agent_id,
                        )
                        ok, text, artifacts = False, sandbox_reason, {}
                    else:
                        ok, text, artifacts = self.sandbox.execute(
                            tool_def, args, context, tool_def.sandbox_config
                        )
                else:
                    ok, text, artifacts = handler(args, context)

                # 确保返回值格式正确
                if not isinstance(ok, bool):
                    ok = bool(ok)
                if not isinstance(text, str):
                    text = str(text)
                if not isinstance(artifacts, dict):
                    artifacts = {"result": artifacts}

            except Exception as e:
                raised = e
                duration_ms = int((time.time() - start_time) * 1000)
                logger.error("Tool %s raised exception after %dms: %s", tool_name, duration_ms, e, exc_info=True)
                ok, text, artifacts = False, f"工具执行异常：{type(e).__name__}: {str(e)}", {}

        # 熔断记账：**只有抛异常才算下游故障**。业务上返回 ok=False（文件不存在、
        # SQL 被权限拒绝）说明下游是活的，不该熔断 —— 熔断器保护的是"依赖坏了"，
        # 不是"答案是坏的"。
        if self.breaker is not None:
            if raised is None:
                self.breaker.record_success(tool_name)
            else:
                self.breaker.record_failure(tool_name, f"{type(raised).__name__}: {raised}")

        duration_ms = int((time.time() - start_time) * 1000)
        mw_ctx.extra["duration_ms"] = duration_ms
        mw_ctx.extra["result_ok"] = ok
        mw_ctx.extra["cache_hit"] = cached is not None

        # ---- 9. 中间件 after_tool Hook ----
        if self.middleware:
            ok, text, artifacts = self.middleware.exec_after_tool(mw_ctx, tool_name, (ok, text, artifacts))

        # ---- 9.5 写入缓存 ----
        # 写在 after_tool **之后**：存的是脱敏后的最终结果，否则缓存自己就成了 PII
        # 的泄漏面。命中时不重复写入（值本就来自缓存）。失败结果不进缓存。
        if cache_key is not None and cached is None:
            self.cache.put(cache_key, tool_name, (ok, text, artifacts))

        # ---- 9.8 累计计数 ----
        # 与审计同一处取终值：`ok` 到这一步才定（中间件可能改写结果），口径必须与审计
        # 一致，否则"指标说 3 次失败、审计里 5 条"这种对不上会让人先怀疑监控再怀疑代码。
        with self._counts_lock:
            key = (tool_name, bool(ok))
            self._call_counts[key] = self._call_counts.get(key, 0) + 1

        # ---- 10. 审计：记录本次调用的最终结果（成功/执行异常） ----
        sandbox_used = bool(tool_def.run_in_sandbox and self.sandbox is not None)
        self._audit(
            trace_id=trace_id, session_id=session_id, agent_id=agent_id, role=role,
            tool_name=tool_name, args=args, pdp_decision="allow",
            result_ok=ok, duration_ms=duration_ms,
            error=None if ok else text,
            sandbox_used=sandbox_used,
            approval_required=tool_def.requires_approval,
            approval_id=(approval_cred or {}).get("id") if tool_def.requires_approval else None,
            cache_hit=cached is not None,
        )

        logger.info("Tool %s completed: ok=%s duration=%dms", tool_name, ok, duration_ms)
        return ok, text, artifacts

    def _audit(self, **kwargs: Any) -> None:
        """审计旁路：统一兜底，审计本身的任何异常都不得影响工具调用。"""
        if self.audit is None:
            return
        try:
            self.audit.record_tool_call(**kwargs)
        except Exception as e:  # noqa: BLE001 - 审计失败只记录，不抛出
            logger.debug("Audit record skipped due to error: %s", e)

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------
    def _validate_args(self, tool_def: ToolDef, args: dict) -> tuple[bool, str]:
        """简单参数校验：必填字段是否存在，类型是否匹配。"""
        required = tool_def.parameters.get("required", [])
        properties = tool_def.parameters.get("properties", {})

        # 检查必填字段
        for field in required:
            if field not in args:
                return False, f"缺少必填参数 '{field}'"

        # 简单类型检查
        type_map = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "object": dict,
            "array": list,
        }
        for field, value in args.items():
            if field in properties:
                expected_type = properties[field].get("type")
                if expected_type and expected_type in type_map:
                    if not isinstance(value, type_map[expected_type]):
                        return False, f"参数 '{field}' 类型错误：期望 {expected_type}，实际 {type(value).__name__}"

        return True, ""

    def _try_acquire(self, tool_name: str, max_per_min: int) -> tuple[bool, str]:
        """限流准入（兼容旧调用）：委托给 self._rate_limiter.acquire。

        新代码应直接调用 self._rate_limiter.acquire；保留此方法是为了兼容
        可能存在的外部/测试直接调用。
        """
        return self._rate_limiter.acquire(tool_name, max_per_min)

    # ------------------------------------------------------------------
    # 统计与调试
    # ------------------------------------------------------------------
    def get_stats(self) -> dict[str, Any]:
        """获取 Broker 统计信息。"""
        recent_by_tool = {
            td.name: self._rate_limiter.current_window_count(td.name)
            for td in self.list_tools()
        }
        return {
            "total_tools": len(self._tools),
            "tools": [
                {
                    "name": td.name,
                    "required_role": td.required_role,
                    "rate_limit_per_min": td.rate_limit_per_min,
                    "requires_approval": td.requires_approval,
                    "run_in_sandbox": td.run_in_sandbox,
                    "recent_calls_1min": recent_by_tool.get(td.name, 0),
                }
                for td in self.list_tools()
            ],
            "middleware_enabled": self.middleware is not None,
            # 接上 PDP ≠ 有强制力：默认策略是 allow 且没有任何规则时，它什么都不拦。
            # 把这两件事一起报出来，免得运维看到 True 以为鉴权在生效。
            "pdp_enabled": self.pdp is not None,
            "pdp_rules": len(getattr(self.pdp, "list_rules", dict)() or {}),
            "pdp_default_policy": getattr(self.pdp, "default_policy", None),
            "sandbox_enabled": self.sandbox is not None,
            "audit_enabled": self.audit is not None,
            # 只报非正常状态的 key（熔断中或有失败计数），免得快照被一堆 closed 淹没
            "circuit_breaker_enabled": self.breaker is not None,
            "circuit_breaker": self.breaker.snapshot() if self.breaker is not None else {},
            # 同理：限流窗口是进程内的，多副本时每个副本各算一份。报出来，免得
            # 看到 rate_limit_per_min 就以为全局总量被卡住了。
            "rate_limit_scope": self._rate_limiter.scope,
        }

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools


class ScopedBroker:
    """ToolBroker 的受限视图：共享底层已注册工具，但只暴露白名单内的工具。

    用于专业子 Agent 的最小权限隔离：
    - list_tool_descriptions / list_tools 只呈现白名单工具，模型看不到越权工具；
    - invoke 对越权工具直接拒绝（不进入 PDP/审计的正常放行链路）；
    - 可选 force_role，使视图内调用统一以子 Agent 角色通过 PDP。

    它与 ToolBroker 实现同一组被编排层依赖的方法（鸭子类型），可直接传给
    ReActNodes / executor 子图。``["*"]`` 表示放开全部工具。
    """

    def __init__(
        self,
        inner: ToolBroker,
        allowed_tools: list[str],
        force_role: Optional[str] = None,
    ) -> None:
        self._inner = inner
        self.force_role = force_role
        self._all = list(allowed_tools) == ["*"]
        self._allowed: set[str] = set(allowed_tools) if not self._all else set()

    def _is_allowed(self, name: str) -> bool:
        return self._all or name in self._allowed

    def _allowed_defs(self) -> list[ToolDef]:
        return [t for t in self._inner.list_tools() if self._is_allowed(t.name)]

    # ---- 与 ToolBroker 对齐的只读接口 ----
    def list_tools(self) -> list[ToolDef]:
        return self._allowed_defs()

    def list_tool_descriptions(self) -> str:
        return render_tool_descriptions(self._allowed_defs())

    def list_tools_openai_format(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in self._allowed_defs()
        ]

    def get(self, name: str) -> Optional[ToolDef]:
        return self._inner.get(name) if self._is_allowed(name) else None

    def search(self, query: str) -> list[ToolDef]:
        return [t for t in self._inner.search(query) if self._is_allowed(t.name)]

    def authorize(self, tool_name: str, context: Optional[dict] = None) -> tuple[bool, str]:
        """授权预检的受限视图：先拦越界工具，再透传底层 Broker 的 PDP 判定。

        与 ``invoke`` 的授权口径保持一致（force_role 同样在此生效），供节点层
        在人工审批之前做"先授权、后审批"的预检。
        """
        if not self._is_allowed(tool_name):
            allowed = "*" if self._all else ", ".join(sorted(self._allowed)) or "(无)"
            return False, f"工具 '{tool_name}' 对当前子 Agent 不可用（可用：{allowed}）"
        ctx = dict(context or {})
        if self.force_role:
            ctx["role"] = self.force_role
        return self._inner.authorize(tool_name, ctx)

    # ---- 会话豁免 ----
    # 子 Agent 的子图里跑的也是 ReActNodes，它的 self.broker 就是这个受限视图 ——
    # 所以授予**必须**经这里透传，否则节点层根本授予不了（异常还会被图运行器吞掉，
    # 表现为"每次重新弹卡"）。这不会让子 Agent 给自己开豁免：授予的入参只来自
    # `interrupt()` 返回的人工决定，模型的输出到不了这个方法。
    # 受限之处在于作用域：只能授予本视图白名单内的工具。
    def apply_session_grant(
        self, session_id: str, tool_name: str, **kwargs: Any
    ) -> Optional[dict[str, Any]]:
        if not self._is_allowed(tool_name):
            return None
        return self._inner.apply_session_grant(session_id, tool_name, **kwargs)

    def grant_session_approval(
        self, session_id: str, tool_name: str, effect: str, **kwargs: Any
    ) -> dict[str, Any]:
        if not self._is_allowed(tool_name):
            allowed = "*" if self._all else ", ".join(sorted(self._allowed)) or "(无)"
            raise PermissionError(
                f"工具 '{tool_name}' 不在当前子 Agent 的白名单内（可用：{allowed}），"
                "不能为它授予会话豁免"
            )
        return self._inner.grant_session_approval(session_id, tool_name, effect, **kwargs)

    def list_session_grants(self, session_id: str) -> list[dict[str, Any]]:
        return self._inner.list_session_grants(session_id)

    # ---- 风险策略读取 ----
    # 只读转发：策略由**包**注册在底层 Broker 上，子 Agent 只是恰好读不到就需要它
    # （子图里跑的也是 ReActNodes）。白名单外的工具不暴露，与 get() 同一口径。
    def risk_policy_for(self, tool_name: str):
        if not self._is_allowed(tool_name):
            return None
        return self._inner.risk_policy_for(tool_name)

    def approval_fallback(self, tool_name: str) -> Optional[str]:
        if not self._is_allowed(tool_name):
            return None
        return self._inner.approval_fallback(tool_name)

    @property
    def approval_threshold(self) -> str:
        return self._inner.approval_threshold

    @property
    def approval_deny_threshold(self) -> str:
        return self._inner.approval_deny_threshold

    def invoke(
        self,
        tool_name: str,
        args: dict,
        context: Optional[dict] = None,
    ) -> tuple[bool, str, dict]:
        """越权调用在视图层直接拦截；授权调用透传给底层 Broker 走完整管控链路。"""
        if not self._is_allowed(tool_name):
            allowed = "*" if self._all else ", ".join(sorted(self._allowed)) or "(无)"
            logger.warning("ScopedBroker denied out-of-scope tool: %s (allowed: %s)", tool_name, allowed)
            scope_reason = f"子 Agent 工具白名单越界（可用：{allowed}）"
            emit_guard_decision(
                (context or {}).get("trace_id"), layer=GUARD_LAYER_PDP,
                reason=scope_reason, tool=tool_name, task_id=(context or {}).get("task_id"),
                agent_id=(context or {}).get("agent_id"), scope="sub_agent",
            )
            return False, f"工具 '{tool_name}' 对当前子 Agent 不可用（可用：{allowed}）", {}
        ctx = dict(context or {})
        if self.force_role:
            ctx["role"] = self.force_role
        return self._inner.invoke(tool_name, args, ctx)

    def __len__(self) -> int:
        return len(self._allowed_defs())

    def __contains__(self, name: str) -> bool:
        return self._is_allowed(name) and name in self._inner


__all__ = ["ToolBroker", "ScopedBroker", "ToolHandler", "render_tool_descriptions"]
