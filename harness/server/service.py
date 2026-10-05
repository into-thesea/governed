"""harness.server.service —— 任务服务：装配组件、驱动图、状态查询与审批恢复。

职责：
- 作为服务化后的【装配点】（D16），构造 Broker / LLM / Registry / Planner /
  Gate / ContextManager / Checkpointer 并编译顶层 Plan-and-Execute 图；
- 以后台 asyncio Task 驱动图（跑到完成或 interrupt 暂停）；
- 提供状态快照（含待审批项）与审批恢复（Command resume）；
- 不直接处理 HTTP，由 app.py 的路由调用，便于单测与复用。
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import json
import logging
import threading
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Optional

from harness.agents.registry import AgentRegistry
from harness.audit import get_audit_logger
from harness.trace import cleanup_tracer, get_tracer
from harness.context import ContextManager
from harness.domain import FrameworkHandles, PackageManager, PackageState
from harness.events import (
    GUARD_LAYER_APPROVAL,
    build_event_bus,
    emit_approval_resolved,
    emit_guard_decision,
)
from harness.orchestrator import SubgraphCache, build_plan_execute_graph, make_plan_execute_state
from harness.planning import DataQualityChecker, QualityGate, TaskPlanner, TaskStore
from harness.skills import SkillRegistry
from harness.tool_broker import ToolBroker
from harness.vfs import VirtualFileSystem

logger = logging.getLogger(__name__)

VERSION = "1.0.0"


class HarnessService:
    """持有编译图与 checkpointer，管理任务的后台运行与人工审批。"""

    def __init__(
        self,
        *,
        checkpointer: Any = None,
        llm: Any = None,
        auto_assemble: bool = True,
    ) -> None:
        """初始化服务。

        Args:
            checkpointer: LangGraph checkpointer（审批 interrupt 必需）。
                默认按 ``CHECKPOINT_BACKEND`` 装配（默认 ``sqlite``，落盘，
                服务重启后仍能继续审批；``memory`` 为进程内实现，重启即丢）。
                注意：默认 saver 是异步实现、绑定事件循环，因此**装配推迟到第一个
                异步入口**（见 ``_ensure_ready``）。显式注入实例可立即装配。
            llm: 可注入的 LLM（真实 client / Mock）。None 时自动选择：
                配置了 API Key 用真实 LLM，否则用脚本化 Mock（离线可跑）。
            auto_assemble: 是否自动装配并编译图（测试可关闭后手动装配）。
        """
        self.checkpointer = checkpointer
        self._ready = False
        self._llm_override = llm
        self._bg_tasks: dict[str, asyncio.Task] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self.graph: Any = None
        self.llm: Any = None
        self.datasources: Any = None
        # 控制台只读端点要直接观测的组件：assemble() 里赋值（未装配前为 None）。
        self.broker: Any = None
        self.context_manager: Any = None
        self.middleware: Any = None
        self.gate: Any = None
        self.planner: Any = None
        self.registry: Any = None
        self.skill_registry: Any = None
        self.store: Any = None
        # 进程内按会话聚合的管控事件计数（全局事件观察者填充，不随 unbind 清空）。
        self._guard_metrics: dict[str, dict] = {}
        self._metrics_lock = threading.Lock()
        self._metrics_bus: Any = None
        self._auto_assemble = auto_assemble
        # 注入了 checkpointer 才能立即装配；否则等第一个异步入口（见 _ensure_ready）
        if auto_assemble and checkpointer is not None:
            self.assemble()

    async def _ensure_ready(self) -> None:
        """首个异步入口的幂等前置：解析 checkpointer 并装配图。

        异步 saver 绑定事件循环，只能在运行中的循环里构造，所以装配不能放在
        ``__init__``（见 ``harness/checkpoint.py`` 模块说明）。
        """
        if self._ready:
            return
        if self.checkpointer is None:
            from harness.checkpoint import build_checkpointer

            self.checkpointer = await build_checkpointer()
        if self._auto_assemble:
            self.assemble()
        else:
            self._ready = True

        # 无人处理的审批必须有个**非人触发**的判定点，否则任务永远停在 awaiting_approval
        # （进程活着、日志干净、不结束 —— 最难被发现的一类失败）。
        from harness.config import settings

        if settings.server.approval_unattended == "auto_reject":
            # 守护任务：进程退出随之结束，不需要单独取消
            asyncio.create_task(self._sweeper_loop())
        else:
            logger.warning(
                "SERVER_APPROVAL_UNATTENDED=block：无人处理审批时任务会一直等待。"
                "仅限有人值守的部署使用。"
            )

    # ------------------------------------------------------------------
    # 装配（D16：服务层是正式装配点）
    # ------------------------------------------------------------------
    def assemble(self) -> Any:
        """构造全部组件并编译图，返回编译后的图。"""
        from harness.config import settings
        from harness.middleware import MiddlewareManager, PIIDetectionMiddleware

        # 中间件链：PII 脱敏是**横切管控点**（D: 管控点横切，不侵入业务），
        # 必须同时挂到两条通路上 —— LLM 钩子走图的 middleware，工具钩子走 broker。
        middleware = MiddlewareManager()
        middleware.register(PIIDetectionMiddleware(settings.pii))

        # 工具级 PDP 此前从未接进装配（tool_broker 里 `if self.pdp is not None` 一直
        # 为假）—— 权限层等于空转。这里接上：未配置规则时默认放行，行为不变。
        from harness.pdp import PDP

        # 限流：按 RATE_LIMIT_BACKEND 选实现。process=进程内滑动窗口（单机默认），
        # redis=共享 Redis（多副本时限流配额全局一致）。未知 backend 显式报错不静默回退。
        from harness.rate_limiter import build_rate_limiter
        rl_backend = settings.rate_limit.backend
        if rl_backend == "redis":
            import redis as _redis
            rl_url = settings.rate_limit.redis_url
            if not rl_url:
                raise ValueError("RATE_LIMIT_BACKEND=redis 需要配置 RATE_LIMIT_REDIS_URL")
            _rl = build_rate_limiter("redis", redis_client=_redis.Redis.from_url(rl_url))
        else:
            _rl = build_rate_limiter(rl_backend)

        broker = ToolBroker(
            rate_limiter=_rl,
            middleware_manager=middleware,
            pdp=PDP.from_settings(settings.permission),
            audit_logger=get_audit_logger(),
            # 审批的两条线由**部署方**配（D-007）：领域包只能声明风险档。
            approval_threshold=settings.server.approval_threshold,
            approval_deny_threshold=settings.server.approval_deny_threshold,
        )
        # 框架侧对象先立起来（全部为空）：**工具、角色、技能都由领域包挂载进来**，
        # 框架不内置任何领域内容。
        registry = AgentRegistry()
        skill_registry = SkillRegistry()
        subgraph_cache = SubgraphCache()
        handles = FrameworkHandles(
            tools=broker,
            agents=registry,
            skills=skill_registry,
            result_cache=broker.cache,
            subgraph_cache=subgraph_cache,
        )
        # 发现（entry points）→ 挂载。依赖未就绪的包停在 PENDING，不算失败；
        # 一个包没挂上只会让对应能力缺失，不会把服务拖垮。
        self.packages = PackageManager(handles)
        self.packages.discover()
        self.packages.mount_all()
        for info in self.packages.list_packages():
            logger.info("领域包 %s：%s（%s）", info.name, info.state.value, info.contributes_summary)

        store = TaskStore(backend="memory")
        llm = self._llm_override or self._select_llm()
        gate = QualityGate(
            llm=llm,
            use_critic=settings.quality.critic_enabled,
            data_quality_checker=DataQualityChecker(settings.quality),
            middleware=middleware,
        )
        planner = TaskPlanner(
            llm,
            broker=broker,
            available_agents=registry.names(),
            # 角色职责描述取自注册表（即领域包声明的）—— 框架侧没有名录了，
            # 不传的话规划提示词会把角色渲染成裸名字。
            agent_descriptions={d.name: d.description for d in registry.list_defs()},
            middleware=middleware,
        )
        context_manager = ContextManager(vfs=VirtualFileSystem())

        # 多数据源（C5）：从 DATASOURCE_SOURCES 加载命名 MySQL/PostgreSQL 源
        from harness.datasources import DataSourceManager

        datasources = DataSourceManager()
        datasources.load_from_settings(settings.datasource)
        self.datasources = datasources

        # 长期记忆：跨会话经验沉淀。后端按 MEMORY_VECTOR_BACKEND 选（默认
        # pgvector），不可用时自动降级为本地实现 —— 降级是显式的，可从
        # long_term_memory.backend_name 看到实际生效的是谁。
        from harness.memory import LongTermMemory

        long_term_memory = LongTermMemory(agent_id="default")
        self.long_term_memory = long_term_memory
        if long_term_memory.probe():
            logger.info("长期记忆就绪：%s", long_term_memory.stats())
        else:
            # 记忆是增强项，不可用不阻断启动；但必须**响亮**，否则表现为
            # "接了线却永远没内容"，排查起来很费时。
            logger.warning(
                "长期记忆不可用，本次运行不会沉淀/检索任何经验。原因：%s。"
                "检查 MEMORY_VECTOR_BACKEND / MEMORY_PG_DSN 与 EMBEDDING_* 配置。",
                long_term_memory.degraded_reason or "未知（后端探活失败）",
            )

        self.llm = llm
        # 存下组件引用，供控制台只读端点（管控面 / 指标 / 产物）直接观测，不必再
        # 从编译图里反向捞。图仍用同一批对象，不存在两份实例。
        self.broker = broker
        self.middleware = middleware
        self.context_manager = context_manager
        self.gate = gate
        self.planner = planner
        self.registry = registry
        self.skill_registry = skill_registry
        self.store = store
        self.graph = build_plan_execute_graph(
            llm, broker,
            planner=planner, store=store, registry=registry, gate=gate,
            checkpointer=self.checkpointer,
            middleware=middleware,
            context_manager=context_manager,
            skill_registry=skill_registry,
            datasources=datasources,
            long_term_memory=long_term_memory,
            plan_review=settings.server.plan_approval,
        )
        # 注册进程级管控事件聚合观察者。按 bus 实例去重：测试 reset_event_bus 后总线
        # 换新，需在重新装配时挂到新 bus；同一 bus 不重复注册（否则计数翻倍）。
        bus = build_event_bus()
        if self._metrics_bus is not bus:
            bus.add_global_listener(self._on_metric_event)
            self._metrics_bus = bus

        # 领域包挂完了才知道有没有需审批的工具 —— 一致性检查必须在这之后。
        self._check_approval_channel()

        self._ready = True
        logger.info("HarnessService assembled (checkpointer=%s)", type(self.checkpointer).__name__)
        return self.graph

    def _check_approval_channel(self) -> None:
        """启动一致性检查：有需审批工具，就必须明确回答"这个部署有没有审批通道"。

        把"将来会不会有人来审"（**不可知**）换成"有没有接审批通道"（**可声明的配置事实**）。
        默认值一旦存在就等于没人回答过这个问题，所以 ``SERVER_APPROVAL_CHANNEL`` 不设默认。
        """
        from harness.config import settings

        if settings.server.plan_approval and settings.server.approval_channel != "http":
            raise RuntimeError(
                "SERVER_PLAN_APPROVAL=true（执行前计划审批）但 SERVER_APPROVAL_CHANNEL"
                f"={settings.server.approval_channel!r} —— 计划审批**需要有人**，"
                '没有人审的部署开它等于给自己挖一个永远等不到的坑。'
                '请设 SERVER_APPROVAL_CHANNEL="http"，或关掉计划审批。'
            )

        approval_tools = [t.name for t in self.broker.list_tools() if t.requires_approval]
        if not approval_tools and not settings.server.plan_approval:
            return
        channel = settings.server.approval_channel
        if channel == "http":
            return
        if channel == "none":
            # 无人值守：需审批工具由节点层自动批准（靠沙箱隔离兜底），不弹审批。
            # 这是合法部署形态（CI / 演示 / 内网开发），仅记 WARNING 提醒安全责任。
            logger.warning(
                "SERVER_APPROVAL_CHANNEL=none（无人值守）：需审批工具 %s "
                "将被自动批准，安全完全依赖沙箱隔离。生产环境请改为 http。",
                approval_tools,
            )
            return
        raise RuntimeError(
            f"存在需人工审批的工具 {approval_tools}，但未配置 SERVER_APPROVAL_CHANNEL。"
            '请明确回答这个部署有没有人审：设为 "http"（有人审）或 "none"（无人值守）。'
        )

    async def sweep_expired_approvals(self) -> int:
        """扫一遍：把已过期仍无人处理的待审批项**自动驳回**（返回处理条数）。

        一次调用只做一遍，不做调度 —— 便于测试直接驱动；调度在 :meth:`_sweeper_loop`。
        判据复用 ``_is_approval_expired``，与"晚到的审批人想批准会被 409"同源。
        """
        if self.broker is None:
            return 0
        handled = 0
        try:
            rows = await self.list_tasks()
        except Exception:  # noqa: BLE001 - 清扫失败不该影响主流程
            logger.exception("清扫过期审批：列举任务失败")
            return 0
        for row in rows:
            tid = row.get("thread_id")
            try:
                status = await self.get_status(tid)
                if not status or status.get("status") != "awaiting_approval":
                    continue
                pending = status.get("pending_approvals") or []
                if not pending:
                    continue
                expires_at = (pending[0].get("payload") or {}).get("expires_at")
                if not self._is_approval_expired(expires_at):
                    continue
                await self.submit_approval(
                    tid, False,
                    comment="超时无人处理，系统自动驳回（SERVER_APPROVAL_UNATTENDED=auto_reject）",
                )
                handled += 1
            except Exception:  # noqa: BLE001 - 单个任务出错不拖垮其余
                logger.exception("清扫过期审批失败：%s", tid)
        if handled:
            logger.info("自动驳回了 %d 个超时未处理的审批", handled)
        return handled

    async def _sweeper_loop(self) -> None:
        """按配置间隔反复清扫；仅在 ``auto_reject`` 模式下启动。"""
        from harness.config import settings

        interval = max(1, int(settings.server.approval_sweep_seconds))
        while True:
            await asyncio.sleep(interval)
            await self.sweep_expired_approvals()

    def _empty_guard_metrics(self) -> dict:
        return {
            "guard_by_layer": {},
            "guard_total": 0,
            "approval_required": 0,
            "approval_resolved": {"approved": 0, "rejected": 0, "expired": 0},
        }

    def _on_metric_event(self, event: dict) -> None:
        """全局事件观察者：把 GUARD / 审批事件按会话聚合成计数（控制台指标页）。

        只做常数时间的字典累加，不做 IO；可能从执行器工作线程回调，故全程持锁。
        与 per-thread history 的区别：history 在任务终态 unbind 时清空，这里保留，
        因此任务跑完后指标端点仍能给出本次进程内该会话被各管控点拦了多少次。
        """
        etype = event.get("type")
        tid = event.get("thread_id")
        data = event.get("data") or {}
        if not tid or etype not in ("GUARD_DECISION", "APPROVAL_REQUIRED", "APPROVAL_RESOLVED"):
            return
        with self._metrics_lock:
            m = self._guard_metrics.setdefault(tid, self._empty_guard_metrics())
            if etype == "GUARD_DECISION":
                layer = str(data.get("layer") or "unknown")
                decision = str(data.get("decision") or "deny")
                bucket = m["guard_by_layer"].setdefault(layer, {})
                bucket[decision] = bucket.get(decision, 0) + 1
                m["guard_total"] += 1
            elif etype == "APPROVAL_REQUIRED":
                m["approval_required"] += 1
            else:  # APPROVAL_RESOLVED
                resolved = m["approval_resolved"]
                if data.get("expired"):
                    resolved["expired"] += 1
                elif data.get("approved"):
                    resolved["approved"] += 1
                else:
                    resolved["rejected"] += 1

    def guard_totals(self) -> dict:
        """把**按会话**聚合的管控计数塌成**进程级**总量（给 ``/metrics`` 用）。

        控制台的指标页要按会话看，所以 ``_guard_metrics`` 的键是 thread_id；
        而监控系统绝不能吃这个键 —— 每个任务新增一条时间序列，**只增不减**，
        是 Prometheus 被单个应用拖垮的经典方式，且不会报错、只会在很久以后炸。
        导出前把它加总掉，label 只留**有界**的 layer / decision。
        """
        with self._metrics_lock:
            by_layer: dict[tuple[str, str], int] = {}
            required = approved = rejected = expired = 0
            for m in self._guard_metrics.values():
                for layer, bucket in m["guard_by_layer"].items():
                    for decision, n in bucket.items():
                        key = (layer, decision)
                        by_layer[key] = by_layer.get(key, 0) + n
                required += m["approval_required"]
                resolved = m["approval_resolved"]
                approved += resolved["approved"]
                rejected += resolved["rejected"]
                expired += resolved["expired"]
        return {
            "by_layer": by_layer,
            "approval_required": required,
            "approval_resolved": {"approved": approved, "rejected": rejected,
                                  "expired": expired},
        }

    def _select_llm(self) -> Any:
        """真实 LLM 优先；未配置 Key 时退回领域包贡献的脚本化 Mock。"""
        from harness.config import settings

        key = (settings.llm.api_key or "").strip()
        if key:
            from harness.llm_client import LLMClient

            logger.info("Using real LLM: %s @ %s", settings.llm.model, settings.llm.base_url)
            return LLMClient(
                api_key=key,
                base_url=settings.llm.base_url,
                model=settings.llm.model,
                temperature=settings.llm.temperature,
                timeout=float(settings.llm.timeout_seconds),
            )

        # 离线模式：用**已挂载领域包贡献出来的** LLM 工厂。
        # 框架不认识"演示脏数据""脚本化工作流"这类领域细节 —— 它只认贡献清单里的入口。
        logger.info("No API key, falling back to a package-contributed offline LLM")
        factory = self._offline_llm_factory()
        if factory is None:
            raise RuntimeError(
                "未配置 LLM_API_KEY，且已挂载的领域包都没有贡献离线 LLM"
                "（contributes['offline_llm']）。请配置 .env 里的 LLM_API_KEY，"
                "或挂载一个带离线能力的领域包。"
            )
        return factory()

    def _offline_llm_factory(self) -> Optional[Any]:
        """取已挂载领域包贡献的离线 LLM 工厂（contributions 里的 ``offline_llm``）。"""
        for info in self.packages.list_packages():
            if info.state is not PackageState.ACTIVE:
                continue
            spec = str(info.contributes.get("offline_llm") or "")
            if not spec:
                continue
            module_path, _, attr = spec.partition(":")
            try:
                import importlib

                factory = getattr(importlib.import_module(module_path), attr)
            except Exception as e:  # noqa: BLE001 - 贡献的入口坏了不该让服务起不来
                logger.error("领域包 %s 贡献的离线 LLM 无法加载（%s）：%s", info.name, spec, e)
                continue
            logger.info("离线 LLM 来自领域包 %s：%s", info.name, spec)
            return factory
        return None

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------
    @staticmethod
    def _config(thread_id: str) -> dict:
        return {"configurable": {"thread_id": thread_id}}

    def _lock(self, thread_id: str) -> asyncio.Lock:
        return self._locks.setdefault(thread_id, asyncio.Lock())

    def _spawn(self, thread_id: str, coro) -> asyncio.Task:
        """在后台驱动图，并记录任务（异常仅记录，不静默成功）。

        Tracer / 事件绑定的清理**不放在 done 回调里**：图遇到人工审批 interrupt 时
        本次 ``ainvoke`` 就结束了，但任务只是**暂停**、随后还要 resume —— 若在这里
        无条件 ``unbind``，会把中断前的事件历史一并清掉，晚到的控制台 / 审批卡片就
        看不到 ``APPROVAL_REQUIRED``，resume 阶段的事件也会丢。改由驱动协程按图的
        真实状态决定收尾（见 :meth:`_settle_after_run`）。
        """
        task = asyncio.create_task(coro)
        self._bg_tasks[thread_id] = task

        def _done(t: asyncio.Task) -> None:
            if t.cancelled():
                return
            exc = t.exception()
            if exc is not None:
                # 必须把异常对象本身传给 exc_info：只把 exc 当消息格式化，栈会丢光，
                # 只剩下 `name 'json' is not defined` 这种无从下手的单行（踩过）。
                logger.error(
                    "Background graph task %s failed", thread_id, exc_info=exc
                )

        task.add_done_callback(_done)
        return task

    async def _settle_after_run(self, thread_id: str, trace_id: str) -> None:
        """一次图驱动结束后的收尾：仍在等审批就保留绑定 / Tracer，否则释放。

        判据与 ``get_status`` 一致 —— 快照里任一 task 带 interrupts，即图暂停在
        人工审批上（可能跨越整个审批等待期），此时事件历史与绑定必须原样保留，
        供晚到的订阅者 replay 以及 resume 后继续路由。
        """
        try:
            snap = await self.graph.aget_state(self._config(thread_id))
            awaiting = any(t.interrupts for t in (snap.tasks or []))
        except Exception:  # noqa: BLE001 - 状态都查不到时按终态兜底，避免映射只增不减
            awaiting = False
        if awaiting:
            return
        if self.broker is not None:
            # 任务结束即作废该会话的审批豁免：豁免是"本任务内"的，留着就是跨任务放宽
            self.broker.clear_session_grants(thread_id)
        if trace_id:
            # 任务结束才释放该链路的 Tracer（不释放它持有的 Span 会在进程内常驻）
            cleanup_tracer(trace_id)
            # 事件绑定同理：终态后断开订阅、丢掉历史、删掉映射
            build_event_bus().unbind(thread_id)

    async def _drive_until_settled(self, thread_id: str, coro, trace_id: str):
        """跑一次图驱动（首次 / resume），并在结束后按状态收尾。"""
        try:
            return await coro
        finally:
            await self._settle_after_run(thread_id, trace_id)

    # ------------------------------------------------------------------
    # 任务生命周期
    # ------------------------------------------------------------------
    async def create_task(self, goal: str, context: str = "", role: str = "admin", origin_principal: str = "") -> str:
        """创建任务并后台驱动，返回 thread_id。"""
        await self._ensure_ready()
        thread_id = uuid.uuid4().hex
        initial = make_plan_execute_state(
            goal, context=context, session_id=thread_id, role=role, origin_principal=origin_principal
        )
        trace_id = str(initial.get("trace_id") or "")

        # 事件路由：埋点只带 trace_id，而订阅按 thread_id —— 绑定表把两者连起来。
        # 不绑的话事件层是空转的（发布了也没人收得到）。
        build_event_bus().bind(thread_id, trace_id)

        async def _drive():
            # 根 Span 覆盖整个请求；各节点在同一 Trace 上嵌套，构成一棵调用树
            with get_tracer(trace_id).span("request", operation="run_task"):
                return await self.graph.ainvoke(initial, self._config(thread_id))

        self._spawn(thread_id, self._drive_until_settled(thread_id, _drive(), trace_id))
        logger.info("Task created: %s (trace=%s)", thread_id, trace_id)
        return thread_id

    async def list_packages(self) -> list[dict]:
        """领域包清单与装载状态（装配后才有；未装配则先装配）。

        控制台「插件」页的数据源：框架挂了哪些领域、各自贡献了什么、现在什么状态。
        未挂载/导入失败的包**同样列出**（带上 ``state`` 与 ``error``）—— 界面要能如实
        显示"有但没起来"，而不是让它凭空消失。
        """
        await self._ensure_ready()
        return [
            {
                "name": info.name,
                "version": info.version,
                "description": info.description,
                "provider": info.provider,
                "state": info.state.value,
                "requires": list(info.requires),
                "contributes": dict(info.contributes),
                "contributes_summary": info.contributes_summary,
                "status_note": info.status_note,
                "error": info.error,
            }
            for info in self.packages.list_packages()
        ]

    async def get_status(self, thread_id: str) -> Optional[dict]:
        """返回任务状态快照；thread 不存在返回 None。"""
        await self._ensure_ready()
        snap = await self.graph.aget_state(self._config(thread_id))
        values = snap.values or {}
        # 未见过的 thread：values 为空（无 goal）、无待执行节点 / 任务 → 不存在
        if not values.get("goal") and not snap.next and not snap.tasks:
            return None
        return self._snap_to_dict(thread_id, snap)

    def _snap_to_dict(self, thread_id: str, snap: Any) -> dict:
        values = snap.values or {}
        pending: list[dict] = []
        for t in snap.tasks or []:
            for intr in (t.interrupts or []):
                pending.append({"interrupt_id": intr.id, "payload": intr.value})

        status = values.get("status", "unknown")
        if pending and status not in ("finished", "failed"):
            status = "awaiting_approval"

        plan = values.get("plan")
        progress = getattr(plan, "progress", None)
        token_usage = getattr(self.llm, "usage_total", None)

        return {
            "thread_id": thread_id,
            "status": status,
            "goal": values.get("goal", ""),
            "progress": progress,
            "final_answer": values.get("final_answer"),
            "error": values.get("error"),
            "pending_approvals": pending,
            "token_usage": dict(token_usage) if token_usage else None,
            # 计划与子任务状态机（运行详情 / 计划页的数据源）：含每个 TaskStep 的
            # 状态、分配、门结论、重试、依赖、验收标准、产物索引、计时。
            "plan": self._plan_to_dict(plan),
            # 本任务当前生效的审批豁免（控制台据此显示"哪些工具已经不再询问"）
            "session_grants": (
                self.broker.list_session_grants(thread_id) if self.broker is not None else []
            ),
        }

    @staticmethod
    def _plan_to_dict(plan: Any) -> Optional[dict]:
        """把 ``TaskPlan`` 序列化成 JSON 友好的 dict；无计划（尚未规划）返回 None。

        用 ``mode="json"`` 让 pydantic 一并转好嵌套的 datetime / Enum，再用
        ``default=str`` 兜底 artifacts 里可能出现的任意对象（产物内容不进状态，
        这里只有路径 / 类型等索引信息，正常都是标量）。
        """
        if plan is None:
            return None
        try:
            dumped = plan.model_dump(mode="json")
            return json.loads(json.dumps(dumped, ensure_ascii=False, default=str))
        except Exception:  # noqa: BLE001 - 观测/展示面不能因序列化失败拖垮状态查询
            logger.warning("计划序列化失败，退回精简字段", exc_info=True)
            return {
                "plan_id": getattr(plan, "plan_id", ""),
                "goal": getattr(plan, "goal", ""),
                "progress": getattr(plan, "progress", 0.0),
                "version": getattr(plan, "version", 1),
                "replan_count": getattr(plan, "replan_count", 0),
                "tasks": [],
            }

    async def list_tasks(self, limit: int = 100) -> list[dict]:
        """枚举 checkpointer 里的全部会话，返回任务列表页所需的摘要。

        跨后端（memory / sqlite）走官方 ``saver.alist`` 而不是直接读存储表：
        只取每个 thread **最新**（``metadata.step`` 最大）的快照，再复用
        :meth:`_snap_to_dict` 做权威序列化，不自己反序列化 msgpack。顶层图与子图
        共用一个 checkpointer，用 ``checkpoint_ns == ""`` 过滤掉子图检查点。

        枚举是只读观测面：任何异常都记日志后返回已收集到的部分，不向前端抛 500。
        """
        await self._ensure_ready()
        # alist 只用来枚举 distinct 顶层会话 id：它产出 CheckpointTuple（只有
        # config/metadata/checkpoint，没有反序列化好的 .values/.tasks，且 memory 后端
        # metadata 里不带 created_at）。权威快照（含 StateSnapshot.created_at）统一再走
        # 编译图的 aget_state，与 get_status 同一条读法。
        thread_ids: set[str] = set()
        try:
            # aclosing 确保异步生成器在中途 / 结束后正确关闭游标（sqlite 后端否则会
            # 在 "Cannot operate on a closed database" 上炸，探活时踩过）。
            # config 必须传 None 才是"列全部会话"：MemorySaver 里是
            # ``(...,) if config else self.storage``，传 {"configurable": {}} 反而是
            # 真值、会去取 config["configurable"]["thread_id"] 而 KeyError。
            async with contextlib.aclosing(self.checkpointer.alist(None)) as agen:
                async for item in agen:
                    conf = ((item.config or {}).get("configurable") or {})
                    tid = conf.get("thread_id")
                    if tid and not conf.get("checkpoint_ns"):
                        thread_ids.add(tid)  # 顶层图与子图共用 checkpointer，只收顶层
        except Exception:  # noqa: BLE001 - 列表是观测面，失败不应让控制台整页不可用
            logger.exception("枚举会话（checkpointer.alist）失败，返回已收集的 %d 条", len(thread_ids))

        items: list[dict] = []
        for tid in thread_ids:
            # aget_state 是编译图（而非裸 saver）的方法，与 get_status 走同一入口
            snap = await self.graph.aget_state(self._config(tid))
            if snap is None or not (snap.values or {}).get("goal"):
                continue  # 与 get_status 同一"会话是否存在"判据
            detail = self._snap_to_dict(tid, snap)
            plan = detail.get("plan") or {}
            tasks = plan.get("tasks") or []
            created = getattr(snap, "created_at", None)
            items.append({
                "thread_id": tid,
                "status": detail["status"],
                "goal": detail["goal"],
                "progress": detail["progress"],
                "awaiting_approval": bool(detail.get("pending_approvals")),
                "task_total": len(tasks),
                "task_completed": sum(1 for t in tasks if t.get("status") == "completed"),
                "has_final": bool(detail.get("final_answer")),
                "error": detail.get("error"),
                "updated_at": created.isoformat() if hasattr(created, "isoformat") else (str(created) if created else None),
            })

        items.sort(key=lambda x: (x.get("updated_at") or ""), reverse=True)
        return items[: max(1, int(limit))]

    async def control_plane(self) -> dict:
        """管控面只读快照（控制台「管控面」页）：各管控组件的**配置开关**与**运行时
        实际状态**并排报出。

        刻意区分两者：``configured_enabled`` 是配置/默认值，``connected``/``runtime``
        是装配后真正生效的状态（例如沙箱配置启用但 Docker 没连上、PDP 启用但零规则
        默认放行）——只报一个布尔会让运维误判"已经在拦了"。全部为只读，任何单项取数
        失败都降级为空值而不是让整页 500。
        """
        await self._ensure_ready()
        from harness.config import settings

        broker_stats: dict[str, Any] = {}
        if self.broker is not None:
            try:
                broker_stats = self.broker.get_stats()
            except Exception:  # noqa: BLE001 - 观测面容错
                logger.exception("读取 broker 管控统计失败")

        tools_view = broker_stats.get("tools") or []
        cache_runtime = None
        cache_obj = getattr(self.broker, "cache", None) if self.broker else None
        if cache_obj is not None:
            try:
                cache_runtime = cache_obj.stats()
            except Exception:  # noqa: BLE001
                logger.exception("读取缓存统计失败")

        memory_stats: dict[str, Any] = {}
        ltm = getattr(self, "long_term_memory", None)
        if ltm is not None:
            try:
                memory_stats = ltm.stats()
            except Exception:  # noqa: BLE001
                logger.exception("读取长期记忆统计失败")

        middleware_items: list[dict] = []
        if self.middleware is not None:
            try:
                middleware_items = self.middleware.list_all()
            except Exception:  # noqa: BLE001
                logger.exception("读取中间件清单失败")

        ds_names: list[str] = []
        if self.datasources is not None:
            try:
                ds_names = self.datasources.names()
            except Exception:  # noqa: BLE001
                logger.exception("读取数据源清单失败")

        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "checkpointer": {"backend": type(self.checkpointer).__name__},
            # 工具管控全景：每个工具的角色 / 限流 / 是否审批 / 是否沙箱，近 1 分钟调用，
            # 以及 PDP / 熔断 / 沙箱 / 审计 / 中间件开关与熔断快照。
            "tools": broker_stats,
            "cache": {
                "configured_enabled": settings.cache.enabled,
                "runtime": cache_runtime,
            },
            "middleware": {"items": middleware_items},
            "pii": {"configured_enabled": settings.pii.enabled},
            "permission": {
                "configured_enabled": settings.permission.enabled,
                "default_policy": getattr(settings.permission, "default_policy", None),
                "runtime": {
                    "pdp_enabled": broker_stats.get("pdp_enabled"),
                    "rules": broker_stats.get("pdp_rules"),
                    "effective_default_policy": broker_stats.get("pdp_default_policy"),
                },
            },
            "auth": {"configured_enabled": settings.auth.enabled},
            "sandbox": {
                "configured_enabled": settings.sandbox.enabled,
                "network_enabled": getattr(settings.sandbox, "network_enabled", None),
                "connected": broker_stats.get("sandbox_enabled"),
                "sandboxed_tools": [t["name"] for t in tools_view if t.get("run_in_sandbox")],
            },
            "circuit_breaker": {
                "configured_enabled": settings.circuit.enabled,
                "snapshot": broker_stats.get("circuit_breaker") or {},
            },
            "rate_limit": {"scope": broker_stats.get("rate_limit_scope", "process")},
            "approval": {
                "timeout_seconds": settings.server.approval_timeout_seconds,
                # 这个部署有没有人审、没人应怎么办、以及执行前计划审批开没开
                "channel": settings.server.approval_channel,
                "unattended": settings.server.approval_unattended,
                "plan_approval": settings.server.plan_approval,
                "requires_approval_tools": [
                    t["name"] for t in tools_view if t.get("requires_approval")
                ],
                # 进程内当前生效的会话豁免总条数（按会话清，故只能报总数）
                "session_grants_active": (
                    self.broker.session_grants_total if self.broker is not None else None
                ),
            },
            "memory": memory_stats,
            "datasources": {"names": ds_names},
            "packages": await self.list_packages(),
        }

    @staticmethod
    def _aggregate_audit(records: list[dict]) -> dict:
        """把审计记录聚合成指标页用的统计。

        只统计**工具调用**记录：``approval_grant`` 是会话豁免的授予 / 使用留痕，不是一次
        调用，计进去会让 ``succeeded + failed == total`` 破掉。缺 ``event`` 字段的行按
        工具调用处理 —— 升级前写下的审计行没有这个字段。
        """
        agg: dict[str, Any] = {
            "total": 0, "succeeded": 0, "failed": 0, "pdp_denied": 0,
            "cache_hits": 0, "sandbox_used": 0, "approval_required": 0,
            "duration_ms_total": 0, "by_tool": {},
        }
        for r in records:
            if r.get("event", "tool_call") != "tool_call":
                continue
            agg["total"] += 1
            if r.get("result_ok") is True:
                agg["succeeded"] += 1
            elif r.get("result_ok") is False:
                agg["failed"] += 1
            if r.get("pdp_decision") == "deny":
                agg["pdp_denied"] += 1
            if r.get("cache_hit"):
                agg["cache_hits"] += 1
            if r.get("sandbox_used"):
                agg["sandbox_used"] += 1
            if r.get("approval_required"):
                agg["approval_required"] += 1
            try:
                agg["duration_ms_total"] += int(r.get("duration_ms") or 0)
            except (TypeError, ValueError):
                pass
            tool = r.get("tool_name") or "unknown"
            bt = agg["by_tool"].setdefault(tool, {"calls": 0, "failed": 0, "denied": 0})
            bt["calls"] += 1
            if r.get("result_ok") is False:
                bt["failed"] += 1
            if r.get("pdp_decision") == "deny":
                bt["denied"] += 1
        agg["duration_ms_avg"] = (
            round(agg["duration_ms_total"] / agg["total"], 1) if agg["total"] else 0
        )
        return agg

    async def task_metrics(self, thread_id: str) -> Optional[dict]:
        """单会话运行指标（控制台「指标」页）；任务不存在返回 None。

        各指标的**可信时间跨度不同**，逐项标 scope，前端不能把它们当成同一口径：
        - audit：本地审计 JSONL 按 session 聚合，**持久、重启保留**（调用量 / 成败 /
          PDP 拒绝 / 缓存命中 / 沙箱 / 审批标记 / 时延）。
        - guard_events：全局事件观察者的**进程内**计数，覆盖 PDP 之外的全部管控层
          （限流 / 熔断 / 行列权限 / 沙箱 / 审批），重启清零，但任务终态后仍保留。
        - context.settled：该会话的大结果沉淀引用（进程内）；context.totals / cache /
          tokens 是组件单例的**进程累计**，并非按会话切分（计数器本身没有会话维度）。
        """
        await self._ensure_ready()
        status = await self.get_status(thread_id)
        if status is None:
            return None

        # ---- 1) 审计：持久、按会话 ----
        audit = getattr(self.broker, "audit", None) if self.broker else None
        records = audit.query(session_id=thread_id) if audit is not None else []
        agg = self._aggregate_audit(records)

        # ---- 2) GUARD / 审批：进程内事件聚合（不随终态清空）----
        with self._metrics_lock:
            guard = copy.deepcopy(self._guard_metrics.get(thread_id, self._empty_guard_metrics()))

        # ---- 3) 上下文沉淀（会话引用 + 进程累计计数器）----
        cm = self.context_manager
        settled_refs: list[dict] = []
        context_totals: Optional[dict] = None
        if cm is not None:
            settled_refs = [dict(r) for r in getattr(cm, "_refs", {}).get(thread_id, [])]
            context_totals = asdict(cm.stats)

        # ---- 4) 缓存 / token：组件单例的进程累计 ----
        cache_obj = getattr(self.broker, "cache", None) if self.broker else None
        cache_runtime = cache_obj.stats() if cache_obj is not None else None
        token_usage = getattr(self.llm, "usage_total", None)

        return {
            "thread_id": thread_id,
            "status": status["status"],
            "audit": {"scope": "persisted_local_jsonl", **agg},
            "guard_events": {"scope": "process_since_start", **guard},
            "context": {
                "settled_refs": settled_refs,
                "settled_count": len(settled_refs),
                "totals_scope": "process",
                "totals": context_totals,
            },
            "cache": {"scope": "process", "runtime": cache_runtime},
            "tokens": {"scope": "process", "usage": dict(token_usage) if token_usage else None},
        }

    async def task_artifacts(self, thread_id: str) -> Optional[dict]:
        """单会话产物索引（控制台「产物」页）；任务不存在返回 None。

        三类产物分开列，不臆造统一结构：
        - task_artifacts：计划里每个子任务 ``expected_artifacts`` 与工具实际回填的
          ``artifacts``（结构化索引，内容不进图状态，通常是路径 / 类型 / 行数等标量）；
        - settled_refs：因结果过大被上下文管理器**沉淀到 VFS** 的全文文件卡片；
        - vfs_root：该运行 VFS 根目录的直接子项（目录 / 文件 + 大小），供逐层浏览。
        """
        await self._ensure_ready()
        status = await self.get_status(thread_id)
        if status is None:
            return None

        tasks_view: list[dict] = []
        for t in (status.get("plan") or {}).get("tasks") or []:
            tasks_view.append({
                "task_id": t.get("task_id"),
                "title": t.get("title"),
                "status": t.get("status"),
                "assigned_to": t.get("assigned_to"),
                "expected_artifacts": t.get("expected_artifacts") or [],
                "artifacts": t.get("artifacts") or {},
            })

        cm = self.context_manager
        settled_refs: list[dict] = []
        vfs_root: list[dict] = []
        if cm is not None:
            settled_refs = [dict(r) for r in getattr(cm, "_refs", {}).get(thread_id, [])]
            vfs = getattr(cm, "vfs", None)
            if vfs is not None:
                try:
                    for f in vfs.list_directory("/"):
                        vfs_root.append({
                            "path": getattr(f, "path", ""),
                            "name": getattr(f, "name", ""),
                            "type": getattr(getattr(f, "type", None), "value", str(getattr(f, "type", ""))),
                            "size": getattr(f, "size", 0),
                        })
                except Exception:  # noqa: BLE001 - 观测面容错
                    logger.exception("列举 VFS 根目录失败")

        return {
            "thread_id": thread_id,
            "status": status["status"],
            "task_artifacts": tasks_view,
            "settled_refs": settled_refs,
            "vfs_root": vfs_root,
        }

    async def _origin_principal(self, thread_id: str) -> str:
        """任务发起者的身份指纹（存在图状态里，**不出现在任何 API 响应中**）。"""
        snap = await self.graph.aget_state(self._config(thread_id))
        return str((snap.values or {}).get("origin_principal") or "")

    async def _trace_id(self, thread_id: str) -> str:
        """从图状态取该任务的 trace_id（事件按 trace_id 路由）。"""
        snap = await self.graph.aget_state(self._config(thread_id))
        return str((snap.values or {}).get("trace_id") or "")

    @staticmethod
    def _is_approval_expired(expires_at: Any) -> bool:
        """审批是否已过 ``expires_at``；无过期时间或解析失败按"未过期"处理（不误伤）。"""
        if not expires_at:
            return False
        try:
            exp = datetime.fromisoformat(str(expires_at))
            now = datetime.now(exp.tzinfo) if exp.tzinfo else datetime.now()
            return now > exp
        except (ValueError, TypeError):
            return False

    @staticmethod
    def _normalize_remember(approved: bool, remember: Optional[str]) -> Optional[str]:
        """归一 ``remember``：只接受与 ``approved`` 同向的 allow / deny，其余一律 None。

        手写请求可能给出自相矛盾的组合（``approved=true`` + ``remember="deny"``）；不做猜测
        —— 猜错就是一次权限提升。非法值也不报错，只是不授予：报错会把整张审批卡卡死。
        """
        value = (remember or "").strip().lower() or None
        if value not in ("allow", "deny"):
            return None
        return value if (value == "allow") == bool(approved) else None

    async def submit_approval(
        self,
        thread_id: str,
        approved: bool,
        comment: str = "",
        approver_principal: str = "",
        approver_name: str = "",
        approver_role: str = "",
        remember: Optional[str] = None,
    ) -> dict:
        """提交审批决策并恢复图，返回提交后的状态快照。

        仅当任务处于 awaiting_approval（存在未处理 interrupt）时有效；
        同一 thread 的驱动经锁串行，避免并发恢复。

        Args:
            approver_principal: 提交审批者的身份指纹（令牌 hash）。与任务**发起者身份**
                相同则拒绝（职责分离：发起人不能自己批准自己触发的高危操作）。
            approver_name/approver_role: 审批人展示名与角色，写入 APPROVAL_RESOLVED 事件；
                同时随 resume 载荷交给节点 —— 节点据此记录"谁授予了会话豁免"。
            remember: ``"allow"`` / ``"deny"`` 表示本任务内不再询问该工具；``None`` 表示只对
                本次生效。必须与 ``approved`` 同向，否则按 ``None`` 处理（见
                :meth:`_normalize_remember`）。

        Raises:
            PermissionError: 发起人试图审批自己发起的任务。
            RuntimeError: 无待审批项，或审批已过期却试图"批准"。
        """
        from langgraph.types import Command

        await self._ensure_ready()
        async with self._lock(thread_id):
            current = await self.get_status(thread_id)
            if current is None:
                raise KeyError(f"任务 {thread_id} 不存在")
            pending = current["pending_approvals"]
            if not pending:
                raise RuntimeError(
                    f"任务当前无待审批项（状态 {current['status']}），无法提交审批"
                )
            if approver_principal:
                origin = await self._origin_principal(thread_id)
                if origin and origin == approver_principal:
                    raise PermissionError(
                        "发起者不能审批自己发起的任务（职责分离）"
                    )

            item = pending[0]
            interrupt_id = item.get("interrupt_id")
            payload = item.get("payload") or {}
            # 三类人工卡点（设计 §3.6）：高危工具审批 / 质量门 HUMAN 结论审查 / 执行前计划审批
            kind = {
                "gate_review": "gate",
                "plan_review": "plan",
            }.get(payload.get("type"), "tool")
            tool = payload.get("tool")
            request_id = payload.get("approval_request_id")
            task_id = payload.get("task_id")
            expires_at = payload.get("expires_at")
            trace_id = await self._trace_id(thread_id)
            # 归一"本任务内不再询问"：非法值 / 与 approved 不同向一律当作只对本次生效。
            # 放在过期判定之前 —— 过期的审批即便带了 remember 也不能放宽，由下面的分支拦。
            remember = self._normalize_remember(approved, remember)

            # 图在 interrupt 时首个后台任务即结束并在 _done 里 unbind；resume 是新的
            # 一次 ainvoke，必须先重新绑定事件路由，否则下面的 RESOLVED 及恢复阶段的
            # 工具/管控事件全都发不到控制台（bind 幂等）。
            if trace_id:
                build_event_bus().bind(thread_id, trace_id)

            # 过期强制：超时后不能再"批准"一个早已过时的现场，但始终允许"驳回"，
            # 让 Agent 重新提请。
            if approved and self._is_approval_expired(expires_at):
                reason = "审批已过期，不能再批准；请驳回后由 Agent 重新提请"
                emit_approval_resolved(
                    trace_id, approved=False, expired=True, kind=kind,
                    request_id=request_id, interrupt_id=interrupt_id, tool=tool,
                    task_id=task_id, approver=approver_name or None,
                    approver_role=approver_role or None, comment=reason,
                    expires_at=expires_at,
                )
                emit_guard_decision(
                    trace_id, layer=GUARD_LAYER_APPROVAL, reason=reason, tool=tool,
                    task_id=task_id, expired=True,
                )
                raise RuntimeError(reason)

            emit_approval_resolved(
                trace_id, approved=approved, kind=kind, request_id=request_id,
                interrupt_id=interrupt_id, tool=tool, task_id=task_id,
                approver=approver_name or None, approver_role=approver_role or None,
                comment=comment, expires_at=expires_at,
            )

            resume = {
                "approved": approved, "comment": comment, "remember": remember,
                # 审批人身份随载荷交给节点：授予会话豁免时要记下"谁放宽的"，
                # 否则审计里只留下一条没有责任人的放宽记录。
                "approver": approver_name or "", "approver_role": approver_role or "",
            }
            # 后台驱动恢复（恢复后可能再次 interrupt 或跑完）；收尾协程会在再次暂停
            # 时保留绑定、到达终态时释放 tracer / 事件路由。
            resume_coro = self.graph.ainvoke(Command(resume=resume), self._config(thread_id))
            self._spawn(
                thread_id,
                self._drive_until_settled(thread_id, resume_coro, trace_id),
            )

        # 稍让后台任务推进，再回快照（调用方也可随后轮询）
        await asyncio.sleep(0)
        return await self.get_status(thread_id)


__all__ = ["HarnessService", "VERSION"]
