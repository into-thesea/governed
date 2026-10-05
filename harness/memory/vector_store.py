"""harness.memory.vector_store —— 向量后端的统一接口与三份实现。

长期记忆需要一个「存向量、按相似度查」的后端。后端选择取决于部署形态：
单机开发用进程内实现即可，生产已有 PostgreSQL 时用 pgvector 最省运维，
已有 Milvus 集群时也能接。三者对本模块以上完全等价，切换只改一个配置项。

职责边界：
- **本模块只负责存取向量**，不碰 Embedding 计算 —— 那是 :class:`LongTermMemory`
  的事。这样后端实现不依赖任何模型 SDK，可离线测。
- 相似度统一约定为**余弦相似度**，取值 [-1, 1]，**越大越相关**。各后端自己
  把底层的距离/相似度换算到这个口径，上层不必关心。

失败语义：后端不可用时 ``is_available()`` 为 False，由上层决定降级到哪个后端；
本模块的构造与调用都**不抛异常打断任务**。
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class VectorRecord:
    """一条待写入的向量记录。"""

    id: str
    content: str
    vector: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class VectorHit:
    """一条检索结果。``score`` 是余弦相似度，越大越相关。"""

    id: str
    content: str
    metadata: dict[str, Any]
    score: float


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """余弦相似度；任一向量为零向量时返回 0（避免除零）。"""
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


# ===========================================================================
# 接口
# ===========================================================================
class VectorStore(ABC):
    """向量后端接口。实现必须线程安全，且不向上抛后端异常。"""

    #: 后端名，用于日志与统计上报
    backend_name: str = "unknown"

    @abstractmethod
    def add(self, records: list[VectorRecord]) -> int:
        """写入一批记录，返回成功写入条数。同 id 覆盖。"""

    @abstractmethod
    def search(
        self,
        vector: list[float],
        top_k: int,
        *,
        agent_id: Optional[str] = None,
        min_score: float = -1.0,
    ) -> list[VectorHit]:
        """按余弦相似度检索，返回至多 ``top_k`` 条，按相似度降序。"""

    @abstractmethod
    def delete(self, ids: list[str]) -> int:
        """按 id 删除，返回删除条数。"""

    @abstractmethod
    def clear(self, *, agent_id: Optional[str] = None) -> None:
        """清空；给了 ``agent_id`` 只清该主体的。"""

    @abstractmethod
    def count(self, *, agent_id: Optional[str] = None) -> int:
        """条数统计。"""

    @abstractmethod
    def prune(self, *, agent_id: str, keep: int) -> int:
        """每个主体只保留最新的 ``keep`` 条，返回淘汰条数。

        长期记忆必须有容量上界 —— 没有淘汰策略的向量库在跑得久的部署里会
        无限增长，最后把存储和检索都拖垮。
        """

    def is_available(self) -> bool:
        """后端当前是否可用。不可用时上层应降级。"""
        return True

    def close(self) -> None:
        """释放连接等资源；幂等。"""


# ===========================================================================
# 进程内实现（默认；也是其它后端不可用时的降级目标）
# ===========================================================================
class LocalVectorStore(VectorStore):
    """进程内向量存储，数据落本地 JSON。

    真实计算余弦相似度，不是关键词兜底 —— 长期记忆的检索质量在单机形态下
    同样成立，只是规模受限。

    ponytail: 每次检索 O(n) 全量扫描，且全部记录常驻内存。容量由
    ``MEMORY_LONG_TERM_MAX_ITEMS`` 封顶（默认几百条量级）时完全够用；
    要到万级以上再换 pgvector / Milvus。
    """

    backend_name = "local"

    def __init__(self, path: str) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._records: dict[str, VectorRecord] = {}
        self._load()

    # ---- 持久化 ----
    def _load(self) -> None:
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            for item in raw:
                rec = VectorRecord(
                    id=item["id"],
                    content=item["content"],
                    vector=list(item["vector"]),
                    metadata=item.get("metadata") or {},
                    timestamp=float(item.get("timestamp") or 0.0),
                )
                self._records[rec.id] = rec
            logger.info("LocalVectorStore 载入 %d 条：%s", len(self._records), self.path)
        except Exception:
            logger.exception("LocalVectorStore 载入失败，按空库启动：%s", self.path)
            self._records = {}

    def _flush(self) -> None:
        """原子落盘：先写临时文件再 replace，避免中途崩溃留下半个文件。"""
        try:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            tmp = f"{self.path}.tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(
                    [
                        {"id": r.id, "content": r.content, "vector": r.vector,
                         "metadata": r.metadata, "timestamp": r.timestamp}
                        for r in self._records.values()
                    ],
                    f,
                    ensure_ascii=False,
                )
            os.replace(tmp, self.path)
        except Exception:
            logger.exception("LocalVectorStore 落盘失败：%s", self.path)

    # ---- 接口 ----
    def add(self, records: list[VectorRecord]) -> int:
        with self._lock:
            for rec in records:
                self._records[rec.id] = rec
            self._flush()
        return len(records)

    def search(
        self,
        vector: list[float],
        top_k: int,
        *,
        agent_id: Optional[str] = None,
        min_score: float = -1.0,
    ) -> list[VectorHit]:
        with self._lock:
            candidates = [
                r for r in self._records.values()
                if agent_id is None or r.metadata.get("agent_id") == agent_id
            ]
        scored = [
            VectorHit(
                id=r.id, content=r.content, metadata=r.metadata,
                score=cosine_similarity(vector, r.vector),
            )
            for r in candidates
        ]
        hits = [h for h in scored if h.score >= min_score]
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:top_k]

    def delete(self, ids: list[str]) -> int:
        with self._lock:
            removed = sum(1 for i in ids if self._records.pop(i, None) is not None)
            if removed:
                self._flush()
        return removed

    def clear(self, *, agent_id: Optional[str] = None) -> None:
        with self._lock:
            if agent_id is None:
                self._records.clear()
            else:
                for rid in [
                    r.id for r in self._records.values()
                    if r.metadata.get("agent_id") == agent_id
                ]:
                    self._records.pop(rid, None)
            self._flush()

    def count(self, *, agent_id: Optional[str] = None) -> int:
        with self._lock:
            if agent_id is None:
                return len(self._records)
            return sum(
                1 for r in self._records.values()
                if r.metadata.get("agent_id") == agent_id
            )

    def prune(self, *, agent_id: str, keep: int) -> int:
        with self._lock:
            mine = sorted(
                (r for r in self._records.values() if r.metadata.get("agent_id") == agent_id),
                key=lambda r: r.timestamp,
            )
            victims = mine if keep <= 0 else mine[: max(len(mine) - keep, 0)]
            for r in victims:
                self._records.pop(r.id, None)
            if victims:
                self._flush()
            return len(victims)


# ===========================================================================
# PostgreSQL + pgvector
# ===========================================================================
class PgVectorStore(VectorStore):
    """PostgreSQL + pgvector 后端。

    建表与扩展由本类在首次连接时保证（``CREATE EXTENSION IF NOT EXISTS vector``），
    所以只要库里装了 pgvector，无需人工预置。

    向量以 pgvector 的字面量形式（``'[0.1,0.2,...]'``）经参数化 SQL 传入/传出，
    因此**不需要额外的 Python 包**，psycopg 就够了。
    """

    backend_name = "pgvector"

    def __init__(self, dsn: str, table: str, dim: int, *, index: str = "hnsw") -> None:
        # 表名是**标识符**，没法参数化，只能拼进 SQL 文本（本类有 8 处这么拼）。
        # 既然要拼，就必须先保证它真的是标识符 —— 配置里一个手滑的分号或空格就是
        # 注入面。校验放在这里，后面每一处拼接都因此变得安全。
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", table or ""):
            raise ValueError(
                f"向量表名必须是合法 SQL 标识符（字母/下划线开头，只含字母数字下划线）：{table!r}"
            )
        self.dsn = dsn
        self.table = table
        self.dim = dim
        self.index = index
        self._conn: Any = None
        self._lock = threading.Lock()
        self._error: Optional[str] = None
        self._connect()

    # ---- 连接与建表 ----
    def _connect(self) -> None:
        if not self.dsn:
            self._error = "未配置 MEMORY_PG_DSN"
            return
        try:
            import psycopg

            self._conn = psycopg.connect(self.dsn, autocommit=True)
            self._ensure_schema()
            self._error = None
            logger.info("PgVectorStore 就绪：table=%s dim=%d index=%s", self.table, self.dim, self.index)
        except Exception as e:  # noqa: BLE001 - 任何失败都视为不可用
            self._conn = None
            self._error = f"{type(e).__name__}: {e}"
            logger.warning("PgVectorStore 不可用，将由上层降级：%s", self._error)

    def _ensure_schema(self) -> None:
        with self._lock, self._conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.table} (
                    id          TEXT PRIMARY KEY,
                    agent_id    TEXT NOT NULL,
                    content     TEXT NOT NULL,
                    embedding   vector({self.dim}) NOT NULL,
                    metadata    JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    created_at  DOUBLE PRECISION NOT NULL
                )
                """
            )
            # 表可能早就存在。核对它的向量维度与当前配置是否一致：换过 Embedding
            # 模型的话，库里的旧向量与新查询不在同一空间，继续用只会得到毫无意义
            # 的相似度 —— 与其静默给错结果，不如当场判定不可用。
            cur.execute(
                "SELECT format_type(atttypid, atttypmod) FROM pg_attribute "
                "WHERE attrelid = %s::regclass AND attname = 'embedding'",
                (self.table,),
            )
            row = cur.fetchone()
            if row and row[0] and "vector" in row[0]:
                existing = int(row[0].split("(")[-1].rstrip(")"))
                if existing != self.dim:
                    raise ValueError(
                        f"表 {self.table} 的向量维度是 {existing}，当前配置是 {self.dim}；"
                        "换过 Embedding 模型时需先重建该表（旧向量与新模型不可比）"
                    )
            if self.index == "hnsw":
                cur.execute(
                    f"CREATE INDEX IF NOT EXISTS {self.table}_emb_hnsw "
                    f"ON {self.table} USING hnsw (embedding vector_cosine_ops)"
                )
            elif self.index == "ivfflat":
                cur.execute(
                    f"CREATE INDEX IF NOT EXISTS {self.table}_emb_ivf "
                    f"ON {self.table} USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)"
                )
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS {self.table}_agent_idx ON {self.table} (agent_id)"
            )

    def is_available(self) -> bool:
        if self._conn is None:
            return False
        try:
            with self._lock, self._conn.cursor() as cur:
                cur.execute("SELECT 1")
            return True
        except Exception as e:  # noqa: BLE001
            self._error = f"{type(e).__name__}: {e}"
            logger.warning("PgVectorStore 探活失败：%s", self._error)
            return False

    @property
    def last_error(self) -> Optional[str]:
        return self._error

    @staticmethod
    def _to_pg_vector(vector: list[float]) -> str:
        """pgvector 字面量。经参数化传入，不拼接进 SQL 文本。"""
        return "[" + ",".join(repr(float(x)) for x in vector) + "]"

    # ---- 接口 ----
    def add(self, records: list[VectorRecord]) -> int:
        if not records:
            return 0
        with self._lock, self._conn.cursor() as cur:
            cur.executemany(
                f"INSERT INTO {self.table} (id, agent_id, content, embedding, metadata, created_at) "  # nosec B608 - 表名在 __init__ 已校验为合法 SQL 标识符（re.fullmatch）
                f"VALUES (%s, %s, %s, %s::vector, %s::jsonb, %s) "
                f"ON CONFLICT (id) DO UPDATE SET content = EXCLUDED.content, embedding = EXCLUDED.embedding, metadata = EXCLUDED.metadata",
                [
                    (
                        r.id,
                        str(r.metadata.get("agent_id") or "default"),
                        r.content,
                        self._to_pg_vector(r.vector),
                        json.dumps(r.metadata, ensure_ascii=False),
                        r.timestamp,
                    )
                    for r in records
                ],
            )
        return len(records)

    def search(
        self,
        vector: list[float],
        top_k: int,
        *,
        agent_id: Optional[str] = None,
        min_score: float = -1.0,
    ) -> list[VectorHit]:
        sql = (
            f"SELECT id, content, metadata, 1 - (embedding <=> %s::vector) AS score "  # nosec B608 - 表名在 __init__ 已校验为合法 SQL 标识符（re.fullmatch）
            f"FROM {self.table} "
        )
        params: list[Any] = [self._to_pg_vector(vector)]
        if agent_id is not None:
            sql += "WHERE agent_id = %s "
            params.append(agent_id)
        sql += "ORDER BY embedding <=> %s::vector LIMIT %s"
        params.extend([self._to_pg_vector(vector), top_k])

        with self._lock, self._conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
        return [
            VectorHit(id=r[0], content=r[1], metadata=r[2] or {}, score=float(r[3]))
            for r in rows
            if float(r[3]) >= min_score
        ]

    def delete(self, ids: list[str]) -> int:
        if not ids:
            return 0
        with self._lock, self._conn.cursor() as cur:
            cur.execute(f"DELETE FROM {self.table} WHERE id = ANY(%s)", (ids,))  # nosec B608 - 表名在 __init__ 已校验为合法 SQL 标识符（re.fullmatch）
            return cur.rowcount or 0

    def clear(self, *, agent_id: Optional[str] = None) -> None:
        with self._lock, self._conn.cursor() as cur:
            if agent_id is None:
                cur.execute(f"DELETE FROM {self.table}")  # nosec B608 - 表名在 __init__ 已校验为合法 SQL 标识符（re.fullmatch）
            else:
                cur.execute(f"DELETE FROM {self.table} WHERE agent_id = %s", (agent_id,))  # nosec B608 - 表名在 __init__ 已校验为合法 SQL 标识符（re.fullmatch）

    def count(self, *, agent_id: Optional[str] = None) -> int:
        with self._lock, self._conn.cursor() as cur:
            if agent_id is None:
                cur.execute(f"SELECT COUNT(*) FROM {self.table}")  # nosec B608 - 表名在 __init__ 已校验为合法 SQL 标识符（re.fullmatch）
            else:
                cur.execute(f"SELECT COUNT(*) FROM {self.table} WHERE agent_id = %s", (agent_id,))  # nosec B608 - 表名在 __init__ 已校验为合法 SQL 标识符（re.fullmatch）
            return int(cur.fetchone()[0])

    def prune(self, *, agent_id: str, keep: int) -> int:
        with self._lock, self._conn.cursor() as cur:
            cur.execute(
                f"DELETE FROM {self.table} "  # nosec B608 - 表名在 __init__ 已校验为合法 SQL 标识符（re.fullmatch）
                f"WHERE agent_id = %s AND id NOT IN ("
                f"SELECT id FROM {self.table} WHERE agent_id = %s ORDER BY created_at DESC LIMIT %s)",
                (agent_id, agent_id, max(keep, 0)),
            )
            return cur.rowcount or 0

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:  # noqa: BLE001
                pass
            self._conn = None


# ===========================================================================
# Milvus（可选后端）
# ===========================================================================
class MilvusStore(VectorStore):
    """Milvus 后端。

    Milvus 面向 10M+ 向量、多节点、有专门运维团队的场景；本项目的记忆量级
    （10K~1M）用 pgvector 或本地实现即可，故作为**可选**后端保留，供已有
    Milvus 集群的部署直接接上。

    ``pymilvus`` 不是必装依赖，缺失时本后端判定不可用、由上层降级。
    """

    backend_name = "milvus"

    def __init__(
        self,
        host: str,
        port: str,
        collection: str,
        dim: int,
        *,
        index_type: str = "HNSW",
        metric_type: str = "COSINE",
    ) -> None:
        self.host = host
        self.port = port
        self.collection_name = collection
        self.dim = dim
        self.index_type = index_type
        self.metric_type = metric_type
        self._client: Any = None
        self._error: Optional[str] = None
        self._connect()

    def _connect(self) -> None:
        try:
            from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility

            connections.connect(alias="default", host=self.host, port=self.port)
            self._client = {
                "Collection": Collection, "CollectionSchema": CollectionSchema,
                "DataType": DataType, "FieldSchema": FieldSchema,
                "connections": connections, "utility": utility,
            }
            self._ensure_collection()
            self._error = None
            logger.info("MilvusStore 就绪：%s:%s collection=%s", self.host, self.port, self.collection_name)
        except Exception as e:  # noqa: BLE001
            self._client = None
            self._error = f"{type(e).__name__}: {e}"
            logger.warning("MilvusStore 不可用，将由上层降级：%s", self._error)

    def _ensure_collection(self) -> None:
        m = self._client
        if not m["utility"].has_collection(self.collection_name):
            fields = [
                m["FieldSchema"](name="id", dtype=m["DataType"].VARCHAR, is_primary=True, max_length=64),
                m["FieldSchema"](name="agent_id", dtype=m["DataType"].VARCHAR, max_length=128),
                m["FieldSchema"](name="content", dtype=m["DataType"].VARCHAR, max_length=65535),
                m["FieldSchema"](name="embedding", dtype=m["DataType"].FLOAT_VECTOR, dim=self.dim),
                m["FieldSchema"](name="metadata", dtype=m["DataType"].VARCHAR, max_length=65535),
                m["FieldSchema"](name="created_at", dtype=m["DataType"].DOUBLE),
            ]
            schema = m["CollectionSchema"](fields=fields, description="Long-term memory")
            collection = m["Collection"](name=self.collection_name, schema=schema)
            collection.create_index(
                field_name="embedding",
                index_params={
                    "index_type": self.index_type,
                    "metric_type": self.metric_type,
                    "params": {"M": 16, "efConstruction": 200},
                },
            )
            collection.load()
        else:
            m["Collection"](self.collection_name).load()

    def _collection(self) -> Any:
        return self._client["Collection"](self.collection_name)

    def is_available(self) -> bool:
        if self._client is None:
            return False
        try:
            self._collection().num_entities
            return True
        except Exception as e:  # noqa: BLE001
            self._error = f"{type(e).__name__}: {e}"
            return False

    @property
    def last_error(self) -> Optional[str]:
        return self._error

    def add(self, records: list[VectorRecord]) -> int:
        if not records:
            return 0
        collection = self._collection()
        collection.insert([
            {
                "id": r.id,
                "agent_id": str(r.metadata.get("agent_id") or "default"),
                "content": r.content,
                "embedding": [float(x) for x in r.vector],
                "metadata": json.dumps(r.metadata, ensure_ascii=False),
                "created_at": r.timestamp,
            }
            for r in records
        ])
        collection.flush()
        return len(records)

    def search(
        self,
        vector: list[float],
        top_k: int,
        *,
        agent_id: Optional[str] = None,
        min_score: float = -1.0,
    ) -> list[VectorHit]:
        expr = f'agent_id == "{agent_id}"' if agent_id else None
        results = self._collection().search(
            data=[[float(x) for x in vector]],
            anns_field="embedding",
            param={"metric_type": self.metric_type, "params": {"ef": 128}},
            limit=top_k,
            expr=expr,
            output_fields=["content", "metadata"],
        )
        hits: list[VectorHit] = []
        for group in results:
            for hit in group:
                # COSINE 口径下距离即相似度，越大越相关，与 VectorStore 约定一致
                score = float(hit.distance)
                if score < min_score:
                    continue
                try:
                    meta = json.loads(hit.entity.get("metadata") or "{}")
                except Exception:  # noqa: BLE001
                    meta = {}
                hits.append(
                    VectorHit(id=str(hit.id), content=hit.entity.get("content", ""),
                              metadata=meta, score=score)
                )
        return hits

    def delete(self, ids: list[str]) -> int:
        if not ids:
            return 0
        collection = self._collection()
        quoted = ", ".join(f'"{i}"' for i in ids)
        collection.delete(expr=f"id in [{quoted}]")
        collection.flush()
        return len(ids)

    def clear(self, *, agent_id: Optional[str] = None) -> None:
        utility = self._client["utility"]
        if agent_id is None:
            if utility.has_collection(self.collection_name):
                utility.drop_collection(self.collection_name)
                self._ensure_collection()
            return
        self._collection().delete(expr=f'agent_id == "{agent_id}"')

    def count(self, *, agent_id: Optional[str] = None) -> int:
        collection = self._collection()
        if agent_id is None:
            return int(collection.num_entities)
        return len(
            collection.query(expr=f'agent_id == "{agent_id}"', output_fields=["id"], limit=16384)
        )

    def prune(self, *, agent_id: str, keep: int) -> int:
        rows = self._collection().query(
            expr=f'agent_id == "{agent_id}"',
            output_fields=["id", "created_at"],
            limit=16384,
        )
        if len(rows) <= keep:
            return 0
        rows.sort(key=lambda r: float(r.get("created_at") or 0.0))
        victims = [str(r["id"]) for r in (rows if keep <= 0 else rows[: len(rows) - keep])]
        return self.delete(victims) if victims else 0

    def close(self) -> None:
        if self._client is not None:
            try:
                self._client["connections"].disconnect("default")
            except Exception:  # noqa: BLE001
                pass
            self._client = None


# ===========================================================================
# 工厂
# ===========================================================================
def build_vector_store(*, backend: str, dim: int, table: str, local_path: str,
                       pg_dsn: str = "", pg_index: str = "hnsw",
                       milvus_host: str = "127.0.0.1", milvus_port: str = "19530",
                       milvus_index: str = "HNSW", milvus_metric: str = "COSINE") -> VectorStore:
    """按配置构造向量后端。

    首选后端不可用时**降级到本地实现**并记一条 warning —— 长期记忆是增强项，
    它挂掉不该让任务失败。降级是显式的：调用方可用 ``backend_name`` 看到
    当前实际生效的是谁。
    """
    name = (backend or "").strip().lower()

    if name == "pgvector":
        store = PgVectorStore(pg_dsn, table, dim, index=pg_index)
        if store.is_available():
            return store
        logger.warning("pgvector 后端不可用，降级到本地实现：%s", store.last_error)
        store.close()
    elif name == "milvus":
        store = MilvusStore(milvus_host, milvus_port, table, dim,
                            index_type=milvus_index, metric_type=milvus_metric)
        if store.is_available():
            return store
        logger.warning("Milvus 后端不可用，降级到本地实现：%s", store.last_error)
        store.close()
    elif name not in ("local", ""):
        logger.warning("未知的向量后端 %r，回退本地实现", backend)

    return LocalVectorStore(local_path)


__all__ = [
    "VectorRecord",
    "VectorHit",
    "VectorStore",
    "LocalVectorStore",
    "PgVectorStore",
    "MilvusStore",
    "build_vector_store",
    "cosine_similarity",
]
