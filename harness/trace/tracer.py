"""harness.trace.tracer —— 全链路追踪（Trace + Span 管理）。

对用户请求、Agent 规划、子 Agent 委派、记忆访问、工具调用、文件操作、
人工审批等关键步骤埋点，用 ``trace_id + span_id + parent_span_id`` 构建调用树。

设计约定：
- ``trace_id`` 全局唯一，一条用户请求对应一个 trace_id。
- ``span_id`` 每个操作唯一，``parent_span_id`` 表达调用关系；父节点取自当前
  线程的 Span 栈，所以**同一线程内嵌套 with 就能自动成树**。
- Span 出口由 :mod:`harness.trace.sink` 决定（本地 JSONL / Kafka / 丢弃），
  出口失败只记日志，不影响主流程。
- 采样按 **trace** 判定（不是按 span）：按 span 采会把调用树采残，反而更难用。
"""

from __future__ import annotations

import logging
import os
import random
import threading
import time
import uuid
from contextlib import contextmanager, nullcontext
from typing import Any, Optional

from ..config import settings
from ..models import SpanStatus, TraceSpan
from .sink import TraceSink, build_trace_sink

logger = logging.getLogger(__name__)

_sink: Optional[TraceSink] = None
_sink_lock = threading.Lock()


def get_trace_sink() -> TraceSink:
    """取全局 Span 出口单例（按配置构造一次）。"""
    global _sink
    if _sink is None:
        with _sink_lock:
            if _sink is None:
                t = settings.trace
                _sink = build_trace_sink(
                    sink=t.sink,
                    local_path=os.path.join(t.local_dir, "spans.jsonl"),
                    max_bytes=t.max_file_bytes,
                    backup_count=t.backup_count,
                )
                logger.info("Span 出口就绪：%s", _sink.name)
    return _sink


def reset_trace_sink(sink: Optional[TraceSink] = None) -> None:
    """替换全局出口（测试用；传 None 表示下次重新按配置构造）。"""
    global _sink
    with _sink_lock:
        _sink = sink


class Tracer:
    """全链路追踪器。

    管理 trace_id 和 Span 栈，自动构建调用树。

    使用方式::

        tracer = get_tracer(state["trace_id"])
        with tracer.span("agent_run", operation="run_agent"):
            with tracer.span("tool_call", operation="calculator"):
                result = calculator(...)
    """

    def __init__(self, trace_id: Optional[str] = None, service_name: Optional[str] = None):
        self.trace_id = trace_id or uuid.uuid4().hex
        self.service_name = service_name or settings.trace.service_name
        self._span_stack: list[str] = []
        self._spans: list[TraceSpan] = []
        # 同一 Tracer 可能被同一线程内的嵌套节点共用，栈与列表的复合操作要加锁
        self._lock = threading.RLock()
        # 采样在创建时定一次，整条链路保持一致 —— 按 span 采会把调用树采残
        self.sampled = random.random() < max(min(settings.trace.sample_rate, 1.0), 0.0)  # nosec B311 - trace 采样用伪随机，非安全/密码用途

    @property
    def recording(self) -> bool:
        """是否记录（总开关 + 采样）。"""
        return settings.trace.enabled and self.sampled

    def _emit(self, span: TraceSpan) -> None:
        if not self.recording:
            return
        try:
            get_trace_sink().emit(span.model_dump())
        except Exception:  # noqa: BLE001 - 埋点失败不得影响主流程
            logger.exception("Span 出口异常（忽略）")

    def _notify(self, kind: str, span: TraceSpan) -> None:
        """把 Span 的进出通知给事件总线（控制台看的就是这条流）。

        **不受 `TRACE_ENABLED` 与采样影响**：埋点可以停，实时视图不该跟着停 —— 采样若
        随机吃掉事件，控制台会随机空白，那是最难查的一类 bug。成本由总线把关：
        **未绑定的 trace（没有任务在跑）连事件对象都不构造**；绑定期间则一律记录
        （即使当下没人订阅，也要给晚连上的控制台留一段最近历史）。
        """
        try:
            from harness.events import build_event_bus

            build_event_bus().publish_span(kind, span)
        except Exception:  # noqa: BLE001 - 与埋点同一条纪律：观测失败不得影响主流程
            logger.exception("事件通知异常（忽略）")

    @contextmanager
    def span(self, name: str, operation: Optional[str] = None, tags: Optional[dict[str, Any]] = None):
        """创建一个 Span 上下文管理器。

        进入时创建 Span 并压栈，退出时结束、出栈并写出口。

        Args:
            name: Span 名称（如 tool_call、llm_call、file_read）
            operation: 操作名（如 calculator、think、/reports/analysis.md）
            tags: 自定义标签
        """
        span_id = uuid.uuid4().hex
        with self._lock:
            parent_span_id = self._span_stack[-1] if self._span_stack else None

        # `operation` 是**具体**那一件事（哪个工具 / 哪个子 Agent / plan_tasks），
        # 而 `name` 是这一段的**种类**（tool_call / delegate / plan / gate）。此前只留
        # operation，种类就丢了 —— 于是渲染时间线时分不清"一次工具调用"和"一个规划步"，
        # 只能靠猜 operation 的取值，而那正是领域耦合。把种类并进 tags 保下来。
        merged_tags = {"name": name, **(tags or {})}
        span = TraceSpan(
            trace_id=self.trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            service_name=self.service_name,
            operation=operation or name,
            tags=merged_tags,
        )

        with self._lock:
            self._span_stack.append(span_id)
            # 内存只留最近 max_spans_per_trace 条：get_trace_tree 是排查用的，
            # 长任务不该因为它把内存吃穿（出口仍然全量写出去）
            if len(self._spans) < settings.trace.max_spans_per_trace:
                self._spans.append(span)
            elif len(self._spans) == settings.trace.max_spans_per_trace:
                logger.warning("trace %s 的 Span 数已达内存上限 %d，后续只写出口不留内存",
                               self.trace_id, settings.trace.max_spans_per_trace)

        start_time = time.time()
        error_message: Optional[str] = None
        status = SpanStatus.OK
        # 进入时通知一次：这是"进行中"唯一的表达方式 —— 只在退出时写出口的话，
        # 长耗时的工具/子 Agent 在界面上永远只是"还没出现"。
        self._notify("start", span)

        try:
            yield span
        except Exception as e:
            status = SpanStatus.ERROR
            error_message = str(e)
            span.tags["error"] = str(e)
            raise
        finally:
            span.duration_ms = int((time.time() - start_time) * 1000)
            span.status = status
            span.error_message = error_message
            with self._lock:
                # 只弹掉自己：异常路径下栈可能已被更内层的 finally 弹过
                if self._span_stack and self._span_stack[-1] == span_id:
                    self._span_stack.pop()
                elif span_id in self._span_stack:
                    self._span_stack.remove(span_id)
            self._emit(span)
            self._notify("end", span)

    def add_tag(self, key: str, value: Any) -> None:
        """给当前活跃的 Span 添加标签；超过单 Span 标签上限则丢弃。"""
        with self._lock:
            if not self._span_stack:
                return
            current_span_id = self._span_stack[-1]
            for span in self._spans:
                if span.span_id == current_span_id:
                    if len(span.tags) >= settings.trace.max_tags_per_span:
                        return
                    span.tags[key] = value
                    return

    def get_spans(self) -> list[TraceSpan]:
        """获取内存中保留的 Span（受 max_spans_per_trace 限制）。"""
        with self._lock:
            return list(self._spans)

    def get_trace_tree(self) -> dict[str, Any]:
        """构建调用树（用于调试与展示）。"""
        with self._lock:
            spans = list(self._spans)
        span_map = {s.span_id: s for s in spans}
        children_map: dict[Optional[str], list[TraceSpan]] = {}
        for s in spans:
            children_map.setdefault(s.parent_span_id, []).append(s)

        def build_node(span: TraceSpan) -> dict[str, Any]:
            return {
                "span_id": span.span_id,
                "operation": span.operation,
                "duration_ms": span.duration_ms,
                "status": span.status.value,
                "tags": span.tags,
                "children": [build_node(c) for c in children_map.get(span.span_id, [])],
            }

        roots = children_map.get(None, [])
        return {
            "trace_id": self.trace_id,
            "service_name": self.service_name,
            "total_spans": len(spans),
            "tree": [build_node(r) for r in roots],
        }


# 全局 Tracer 存储（按 trace_id 索引）
_tracers: dict[str, Tracer] = {}
_tracers_lock = threading.Lock()

#: 进程内同时保留的 trace 数上限。正常路径上请求结束会调 cleanup_tracer 释放；
#: 这个上限是兜底 —— 漏调一次就永久泄漏一整条链路的 Span，代价太大。
_MAX_TRACERS = 256


def get_tracer(trace_id: Optional[str] = None) -> Tracer:
    """获取或创建一个 Tracer。

    同一 trace_id 返回同一实例，因此同一线程内跨模块嵌套 ``with span()``
    会自动串成调用树。
    """
    if trace_id:
        with _tracers_lock:
            existing = _tracers.get(trace_id)
            if existing is not None:
                return existing
            tracer = Tracer(trace_id=trace_id)
            if len(_tracers) >= _MAX_TRACERS:
                # 兜底淘汰：丢掉最早插入的一个（dict 保序）
                oldest = next(iter(_tracers))
                _tracers.pop(oldest, None)
                logger.warning("Tracer 数量达上限 %d，淘汰最早的 %s", _MAX_TRACERS, oldest)
            _tracers[trace_id] = tracer
            return tracer

    tracer = Tracer()
    with _tracers_lock:
        _tracers[tracer.trace_id] = tracer
    return tracer


def span_for(state: Any, name: str, operation: Optional[str] = None):
    """按状态里的 ``trace_id`` 取 Span 上下文。

    埋点未启用、或状态里没有 trace_id 时返回**空上下文** —— 调用点不必到处写
    条件判断，埋点的有无也不改变任何行为。
    """
    trace_id = state.get("trace_id") if isinstance(state, dict) else None
    if not trace_id or not settings.trace.enabled:
        return nullcontext()
    return get_tracer(str(trace_id)).span(name, operation=operation)


def cleanup_tracer(trace_id: str) -> None:
    """清理一个 Tracer（请求结束后调用）。

    不调的话这条 trace 的 Span 会在进程内永久驻留 —— 这是接埋点时最容易漏的
    一步，所以 :data:`_MAX_TRACERS` 还留了个兜底上限。
    """
    with _tracers_lock:
        tracer = _tracers.pop(trace_id, None)
    if tracer is not None:
        logger.debug("Tracer %s 已释放（%d 个 Span）", trace_id, len(tracer.get_spans()))


__all__ = [
    "Tracer",
    "get_tracer",
    "cleanup_tracer",
    "get_trace_sink",
    "reset_trace_sink",
    "span_for",
]
