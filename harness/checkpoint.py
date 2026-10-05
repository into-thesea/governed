"""harness.checkpoint —— checkpointer 装配（D16：配置决定实现，装配点构造）。

审批中断（LangGraph ``interrupt``）的状态由 checkpointer 持有：进程内
``MemorySaver`` 一重启即丢，**待审批任务随之蒸发** —— "工具级审批"就只在单进程
生命周期内成立。所以默认后端是 SQLite，让"重启后仍能继续审批"成为默认行为。

为什么是 **async** 构造函数：服务用 ``ainvoke`` / ``aget_state`` 驱动图，必须用
异步实现（同步 ``SqliteSaver`` 的异步方法直接 ``NotImplementedError``），而
``AsyncSqliteSaver`` 在 ``__init__`` 里就 ``asyncio.get_running_loop()`` ——
**异步 saver 绑定事件循环，只能在运行中的循环里构造**。同理，它的建表是首次使用
时自动完成的（库文档明确要求不要手工调用 ``setup()``）。
"""

from __future__ import annotations

import inspect
import os
from typing import Any

from harness.config import settings as default_settings

# 落盘时要反序列化的**状态类型**（见 orchestrator.PlanExecuteState 的字段类型，
# 全部定义在 harness.models）。
#
# 为什么不放行全部：容忍任意类型反序列化等于把检查点库变成反序列化攻击面，
# 而这里存的是图状态（含计划、子结果、工具参数）。所以显式登记 —— 但登记必须是
# **完整**的：漏掉的类型会被库**阻断**（不是忽略），状态退化成 dict 并在下游炸出
# `'dict' object has no attribute 'model_copy'`。失败响亮，可定位，故取此路。
# 因此这里按模块穷举，而不是手写几个名字：模型增删自动跟随。
STATE_TYPE_MODULE = "harness.models"


def state_types() -> list[tuple[str, str]]:
    """返回 ``harness.models`` 中全部模型/枚举类型（``(模块, 类名)``）。"""
    import enum
    import inspect

    from pydantic import BaseModel

    import harness.models as models

    return [
        (models.__name__, name)
        for name, obj in vars(models).items()
        if inspect.isclass(obj)
        and obj.__module__ == models.__name__
        and issubclass(obj, (BaseModel, enum.Enum))
    ]


async def build_checkpointer(config: Any = None) -> Any:
    """按配置构造 checkpointer（必须在事件循环内调用）。

    Args:
        config: ``CheckpointSettings``（配置段本身）；``None`` 时取全局配置。

    Raises:
        ValueError: 未知的 ``CHECKPOINT_BACKEND`` —— 显式报错，因为静默回退到
            内存实现正好会掩盖本模块要解决的问题（不报错的错最贵）。
    """
    cfg = config or default_settings.checkpoint
    backend = (cfg.backend or "").strip().lower()

    if backend == "memory":
        from langgraph.checkpoint.memory import MemorySaver
        from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

        # 与 sqlite 分支登记同一份状态类型：不登记的话每个类型打一条
        # "unregistered type" 警告，且库声明未来版本会阻断
        return MemorySaver(serde=JsonPlusSerializer(allowed_msgpack_modules=state_types()))

    if backend == "sqlite":
        import aiosqlite
        from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        path = cfg.sqlite_path
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)
        # 连接惰性建立（真正连上在首次 await），建表由 saver 首次使用时自动完成
        return AsyncSqliteSaver(
            aiosqlite.connect(path),
            serde=JsonPlusSerializer(allowed_msgpack_modules=state_types()),
        )

    if backend == "postgres":
        import psycopg
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

        dsn = getattr(cfg, "postgres_dsn", "") or ""
        if not dsn:
            raise ValueError(
                "CHECKPOINT_BACKEND=postgres 需要配置 CHECKPOINT_POSTGRES_DSN"
                "（如 postgresql://user:pass@host:5432/db）"
            )
        # 连接惰性建立（与 sqlite 同模式：传 coroutine，首次 await 时真正连上）。
        # Postgres 表不会自动创建，需显式 setup()——在首次使用前调用，幂等。
        saver = AsyncPostgresSaver(
            psycopg.AsyncConnection.connect(dsn),
            serde=JsonPlusSerializer(allowed_msgpack_modules=state_types()),
        )
        await saver.setup()
        return saver

    raise ValueError(
        f"未知的 checkpointer backend: {backend!r}"
        "（CHECKPOINT_BACKEND 支持 memory | sqlite | postgres）"
    )


async def close_checkpointer(saver: Any) -> None:
    """释放 checkpointer 自持的连接（进程退出、或测试模拟"重启"时调用）。

    只装配未使用的 saver 其连接从未启动，关闭是空操作 —— 这种情形静默跳过。
    """
    conn = getattr(saver, "conn", None)
    if conn is None:
        return  # MemorySaver 等无连接实现
    try:
        result = conn.close()
    except ValueError:
        return  # aiosqlite：连接从未启动
    if inspect.isawaitable(result):
        await result


__all__ = ["build_checkpointer", "close_checkpointer", "state_types"]
