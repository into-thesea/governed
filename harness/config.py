"""harness.config —— 全局配置管理（pydantic-settings）。

从环境变量（.env 文件）加载所有配置项，按功能域分组。
所有模块通过 ``from harness.config import settings`` 获取配置单例。

设计约定：
- 配置项名与 .env.example 中的环境变量名一一对应（大写）。
- 提供合理的默认值，确保最小配置也能运行（本地回退模式）。
- 敏感信息（API Key）不设默认值，必须从环境变量传入。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# 把 .env 注入 os.environ 后再实例化任何 Settings。
#
# 为什么需要这一步：pydantic-settings 的 env_file **不会传播到嵌套模型** ——
# 顶层 Settings 虽然声明了 env_file=".env"，但其嵌套字段（llm / sandbox / vfs …）
# 各自是独立的 BaseSettings，只从 os.environ 读取。结果是 .env 整体失效，
# 框架一直静默运行在代码默认值上（例如 llm.api_key 恒为空、llm.model 恒为默认）。
# 显式 load_dotenv 让所有嵌套模型都能读到，且不覆盖已存在的真实环境变量。
load_dotenv(Path(__file__).resolve().parent.parent / ".env")


class LLMSettings(BaseSettings):
    """LLM 调用配置。"""

    model_config = SettingsConfigDict(env_prefix="DEEPSEEK_", extra="ignore")

    api_key: str = ""
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-chat"
    temperature: float = 0.0
    timeout_seconds: int = 60
    max_retries: int = 2


class EmbeddingSettings(BaseSettings):
    """Embedding 模型配置（用于长期记忆向量检索）。"""

    model_config = SettingsConfigDict(env_prefix="EMBEDDING_", extra="ignore")

    # 提供方：local（本地 sentence-transformers 模型，无需外部服务与凭据）| openai（OpenAI 兼容端点）
    provider: str = "local"
    # 模型。provider=local 时接受 HF 模型名、本地模型目录，或 HF 缓存目录
    # （models--<org>--<name> 形式会自动解开到 snapshots/<rev>）；
    # provider=openai 时是端点上的模型名。
    model: str = "models--BAAI--bge-base-zh-v1.5"
    # 向量维度。**仅供参考**：实际维度以提供方返回的为准（local 从模型配置读，
    # openai 从首次响应读），两者不符会在探活时被拦下。
    dim: int = 768
    # provider=local：推理设备（cpu / cuda）与 HF 缓存根目录（含 models--* 的目录）
    device: str = "cpu"
    cache_dir: str = ""
    # 检索查询的前缀。BGE 中文系列按官方用法需要指令前缀；留空则按模型名自动判定
    query_prefix: str = ""
    api_key: str = ""
    base_url: str = "https://api.openai.com/v1"
    batch_size: int = 32


class MilvusSettings(BaseSettings):
    """Milvus 配置（长期向量记忆，可选后端）。"""

    model_config = SettingsConfigDict(env_prefix="MILVUS_", extra="ignore")

    # 用 IPv4 字面量，别写 "localhost"：Docker 默认只把端口发布在 IPv4 上，而
    # "localhost" 在 Windows 上优先解析到 ::1，连接会先在 IPv6 上死等约 20 秒
    # 才回落到 IPv4。实测 Milvus / MinIO 都会中招。
    host: str = "127.0.0.1"
    port: int = 19530
    collection_prefix: str = "governed_"
    index_type: str = "HNSW"
    metric_type: str = "COSINE"
    top_k: int = 5
    consistency_level: str = "Session"


class KafkaSettings(BaseSettings):
    """Kafka 配置（审计日志、链路追踪、事件总线）。"""

    model_config = SettingsConfigDict(env_prefix="KAFKA_", extra="ignore")

    bootstrap_servers: str = "localhost:9092"
    audit_topic: str = "governed_audit"
    trace_topic: str = "governed_trace"
    event_topic: str = "governed_events"
    consumer_group: str = "governed_consumer"
    enable_audit_produce: bool = True
    enable_trace_produce: bool = True
    # spool（投递缓冲）的文件数上限。Kafka 长期不可达时 spool 会只进不出，
    # 没有上限就会写满磁盘。超出时丢最旧的并告警 —— spool 是**投递缓冲**不是
    # 存储，被丢的消息只是没上送，审计的本地留档（audit.jsonl）不受影响。
    # 0 表示不限制（不建议）。
    spool_max_files: int = 5000
    producer_acks: str = "1"
    retries: int = 3
    linger_ms: int = 5
    batch_size: int = 16384


class MinIOSettings(BaseSettings):
    """MinIO 配置（VFS 虚拟文件系统后端）。"""

    model_config = SettingsConfigDict(env_prefix="MINIO_", extra="ignore")

    # 默认关闭：VFS 默认落本地磁盘，需要对象存储时显式打开并指向自己的实例。
    # 理由见 storage.get_storage_backend —— 撞上"别人家的 MinIO"不会报错，
    # 只会让 VFS 数据悄悄落到别人的对象存储里（9000 这类端口在开发机上常被
    # 别的项目占着，比如 Milvus 自带的 MinIO）。
    enabled: bool = False
    # 同 MilvusSettings.host：写 IPv4 字面量。MinIO 每个 VFS 构造都要连一次，
    # 用 "localhost" 会让每个任务白等 20 秒。
    endpoint: str = "127.0.0.1:9000"
    access_key: str = "minioadmin"
    secret_key: str = "minioadmin"
    bucket: str = "governed"
    secure: bool = False
    region: Optional[str] = None


class ServerSettings(BaseSettings):
    """FastAPI 服务层配置。"""

    model_config = SettingsConfigDict(env_prefix="SERVER_", extra="ignore")

    host: str = "0.0.0.0"  # nosec B104 - 服务端绑定所有接口是预期默认（部署方可配 SERVER_HOST）
    port: int = 8000
    workers: int = 1
    reload: bool = False
    cors_origins: str = "*"
    api_prefix: str = "/api/v1"
    request_timeout_seconds: int = 300
    stream_heartbeat_seconds: int = 15
    approval_timeout_seconds: int = 3600
    """人工审批等待超时（秒）。

    节点发起审批时在卡片与 ``APPROVAL_REQUIRED`` 事件上写
    ``expires_at = now + 本值``；超时后审批人**不能再"批准"**（服务层按 409
    拒绝，避免对早已过时的现场放行），但始终可以**驳回**让 Agent 重新提请。
    设为 0 或负数表示不过期。
    """

    approval_threshold: str = "medium"
    """风险高过这条线就问人（``low`` / ``medium`` / ``high`` / ``critical``）。

    与 ``approval_deny_threshold`` 一样是**部署方的旋钮**：领域包只声明风险档，
    "问还是放"由这里拍板（见 ``docs/技术选型决策.md`` D-007）。
    """

    approval_deny_threshold: str = "critical"
    """风险达到这条线**直接拒**，连问都不问。默认只有红线档触发。

    调低它会让更低的档位也直接拒（例如设成 ``high``）。**写错会在启动时报错**，
    不会静默回落 —— 这条线关系到"什么不问就毙掉"，必须响亮。
    """

    @field_validator("approval_threshold", "approval_deny_threshold")
    @classmethod
    def _check_risk_threshold(cls, value: str) -> str:
        from harness.approval_policy import RISK_CRITICAL, RISK_HIGH, RISK_LOW, RISK_MEDIUM

        allowed = (RISK_LOW, RISK_MEDIUM, RISK_HIGH, RISK_CRITICAL)
        if value not in allowed:
            raise ValueError(
                f"风险阈值取值非法：{value!r}（只支持 {' | '.join(allowed)}）"
            )
        return value

    # ------------------------------------------------------------------
    # 无人值守：两个问题、两个配置
    # ------------------------------------------------------------------
    approval_channel: Optional[str] = None
    """这个部署**有没有**审批通道：``"http"``（有人审）| ``"none"``（明确无人值守）。

    **不设默认值** —— 默认值一旦存在，就等于没人回答过"这个部署有没有人审"。

    - ``"http"``：有人审。需审批工具走 LangGraph interrupt，由审批通道（HTTP API）
      下发批准/驳回。
    - ``"none"``：明确无人值守。需审批工具由节点层**自动批准**（不 interrupt），
      安全完全依赖沙箱隔离。适用于 CI / 演示 / 内网开发。装配时记 WARNING。
    - ``None``（未配置）：存在需审批工具时**装配即失败**——必须明确回答有没有人审。
    """

    approval_unattended: str = "auto_reject"
    """有通道但一直没人应怎么办：``auto_reject``（超时自动驳回并推进图）| ``block``（一直等）。

    默认不是 ``block``：没人处理时任务会永远停在 ``awaiting_approval``（进程活着、
    日志干净、不结束），这是最难被发现的一类失败。选 ``block`` 会打 WARNING。
    """

    approval_sweep_seconds: int = 30
    """``auto_reject`` 的清扫间隔（秒）。"""

    plan_approval: bool = False
    """是否在**执行前**把计划摆给审批人过一眼（默认关，零回归）。

    打开后图会在 `plan` 与 `dispatch` 之间停下，等人批准了才动手。它与
    ``approval_channel`` 是绑定关系：**没有人审的部署不该开它**（开了就是给自己挖
    一个永远等不到的坑），装配期会直接失败。
    """

    @field_validator("approval_channel")
    @classmethod
    def _check_approval_channel(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        v = str(value).strip().lower()
        if v not in ("http", "none"):
            raise ValueError(
                f"SERVER_APPROVAL_CHANNEL 取值非法：{value!r}（只支持 http | none，或不配置）"
            )
        return v

    @field_validator("approval_unattended")
    @classmethod
    def _check_approval_unattended(cls, value: str) -> str:
        allowed = ("auto_reject", "block")
        v = str(value).strip().lower()
        if v not in allowed:
            raise ValueError(
                f"SERVER_APPROVAL_UNATTENDED 取值非法：{value!r}（只支持 {' | '.join(allowed)}）"
            )
        return v


class SandboxSettings(BaseSettings):
    """安全沙箱配置（OpenSandbox 控制面 + Docker 容器隔离）。

    沙箱由独立的 OpenSandbox 服务端承载（infra/opensandbox-server/），
    本框架只持有客户端连接信息，不直接操作 Docker。
    """

    model_config = SettingsConfigDict(env_prefix="SANDBOX_", extra="ignore")

    enabled: bool = True

    # --- OpenSandbox 控制面（对应 infra/opensandbox-server/sandbox.toml）---
    server_url: str = "http://127.0.0.1:8080"
    # 无内置默认凭据：必须在 .env 显式配置 SANDBOX_API_KEY，与服务端
    # OPENSANDBOX_SERVER_API_KEY 一致；留空时沙箱调用一律 fail closed
    api_key: str = ""

    # --- 沙箱容器（自建镜像见 infra/Dockerfile.sandbox）---
    image: str = "governed-sandbox:latest"
    workdir: str = "/home/sandbox"
    cpu_limit: float = 1.0
    memory_limit: str = "512m"
    ready_timeout_seconds: int = 180
    # 同时运行的沙箱数上限。每个沙箱要起一个容器、按 cpu_limit / memory_limit
    # 占资源，不限并发时一批并行子任务能瞬间把宿主压垮。
    #   0  = 按 CPU 核数 / 单沙箱核数推导（默认，跟着机器走，不拍数字）
    #   >0 = 显式指定
    #   <0 = 不限制（不建议）
    max_concurrency: int = 0

    # --- 单次执行（信任边界，勿放宽）---
    timeout_seconds: int = 30
    max_timeout_seconds: int = 300
    artifact_max_bytes: int = 32 * 1024 * 1024

    # --- 网络：False 时下发 NetworkPolicy(default_action="deny")，容器无出网 ---
    network_enabled: bool = False


class TraceSettings(BaseSettings):
    """链路追踪配置。"""

    model_config = SettingsConfigDict(env_prefix="TRACE_", extra="ignore")

    enabled: bool = True
    # 采样率：按 **trace** 判定（不是按 span），否则同一条链路的调用树会残缺
    sample_rate: float = 1.0
    service_name: str = "governed"
    environment: str = "development"
    # 单条 Span 的标签数上限，超出丢弃（防止某处循环 add_tag 把内存撑爆）
    max_tags_per_span: int = 50

    # Span 出口：local（落 JSONL，单机默认）| kafka（上送，供链路可视化消费）| none
    sink: str = "local"
    # local 出口：落盘目录、单文件上限、保留份数（按大小轮转，留最近几份）
    local_dir: str = "data/trace"
    max_file_bytes: int = 33554432
    backup_count: int = 3
    # 单条 trace 在内存里保留的 Span 上限：超出后仍写出口，只是不再留内存供
    # get_trace_tree 查询 —— 否则长任务的内存占用无上限
    max_spans_per_trace: int = 2000


class EventSettings(BaseSettings):
    """事件落盘配置（供控制台历史回放）。

    一个任务一个 append-only JSONL（``<dir>/<thread_id>.jsonl``）。决策见
    ``docs/技术选型决策.md`` D-008。
    """

    model_config = SettingsConfigDict(env_prefix="EVENT_", extra="ignore")

    enabled: bool = True
    """是否落盘。关掉则控制台只能看正在跑的任务（跑完即无历史可回放）。"""

    dir: str = "data/events"
    """落盘目录。"""

    retention_days: int = 30
    """保留天数，超龄的任务文件在启动时清理。``<=0`` 表示不清理。"""

    max_file_bytes: int = 33554432
    """单个任务文件的封顶（32MB）。**封顶不是轮转**：超过后不再写，
    以便"回放 = 读一个文件"这个前提始终成立。"""


class VFSSettings(BaseSettings):
    """虚拟文件系统配置。"""

    model_config = SettingsConfigDict(env_prefix="VFS_", extra="ignore")

    local_root: str = "data/vfs"
    max_file_size_mb: int = 50
    max_versions_per_file: int = 20
    default_directories: list[str] = Field(
        default_factory=lambda: ["/workspace", "/reports", "/logs", "/policies", "/memories"]
    )


class MemorySettings(BaseSettings):
    """记忆管理配置。"""

    model_config = SettingsConfigDict(env_prefix="MEMORY_", extra="ignore")

    long_term_top_k: int = 5
    # 检索相似度阈值。**这个值取决于 Embedding 模型**，换模型要重新标定：
    # BGE 中文系列在"相关/无关"两档上整体比 OpenAI 系偏低（实测 bge-base-zh-v1.5：
    # 相关对 0.44~0.53，无关对 0.24~0.34），沿用 0.5 会把大部分相关记忆滤掉。
    # 阈值取在两档之间，宁松勿紧 —— 多注入一条无关记忆只是噪声，漏掉相关记忆
    # 则等于这层能力不存在。
    long_term_similarity_threshold: float = 0.38
    local_dir: str = "data/memory"

    # ---- 长期向量记忆 ----
    # 后端：pgvector（默认）/ milvus / local。首选不可用时自动降级为 local，
    # 降级是显式的（见 LongTermMemory.backend_name），不会静默换个后端。
    vector_backend: str = "pgvector"
    # pgvector 连接串。注意它需要**写权限**，与只读的数据源账号不同；
    # 留空即判定后端不可用。生产用环境变量注入，别写进版本库。
    pg_dsn: str = ""
    # pgvector 索引类型：hnsw / ivfflat / none
    pg_index: str = "hnsw"
    # 单条记忆的内容上限：超出截断，避免把整篇报告灌进向量库
    long_term_max_content_chars: int = 2000
    # 每个主体（agent_id）保留的条数上限，超出按时间淘汰最旧的
    long_term_max_items: int = 500
    # 查询短于这个长度就不检索：避免"分析""报告"这类泛词命中一堆无关记忆
    long_term_min_query_chars: int = 8


class RuntimeSettings(BaseSettings):
    """运行时通用配置。"""

    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    log_level: str = "INFO"
    log_format: str = "json"
    checkpoint_dir: str = "data/checkpoints"
    audit_dir: str = "data/audit"
    max_workers: int = 10
    # 单个调度批次里**并发执行**的子任务数上限。
    #   0  = 按就绪集大小，不额外限制（默认）
    #   >0 = 显式指定
    # 不额外限制的理由：并发度天然受"就绪集大小"约束（计划里同时无依赖的任务数），
    # 而真正稀缺的资源（沙箱）已由 SANDBOX_MAX_CONCURRENCY 单独限流，其余开销是
    # LLM 网络等待。若实测出现线程/限流压力，再显式收窄。
    max_parallel_subtasks: int = 0
    default_max_steps: int = 20
    default_timeout_seconds: int = 300

    # 子任务执行体的工具调用范式（见 harness.nodes.ReActNodes）：
    #   "react"  —— 工具清单拼进提示、解析模型 JSON 输出；过程可见、模型无关，
    #               用于教学/调试/兼容弱模型
    #   "native" —— OpenAI 原生 Function Calling，工具结构由 API 保证，
    #               支持一次多个 tool_calls；用于生产
    agent_tool_mode: str = "react"


class ContextSettings(BaseSettings):
    """上下文管理配置（ContextManager 的预算与沉淀策略）。

    这组值决定「多大的工具结果沉淀到 VFS」与「历史压到多长」，直接改变
    长任务的 token 消耗与信息保留度，故全部配置化，便于按模型上下文窗口调整。

    注意：调大 `sink_threshold_chars` 会让更多原文留在提示里（省 VFS 但费 token）；
    调小 `max_history_chars` 会让更早的中间过程被折叠（省 token 但可能丢脉络）。
    两者是此消彼长的取舍，改前建议先看实测数据。
    """

    model_config = SettingsConfigDict(env_prefix="CONTEXT_", extra="ignore")

    sink_threshold_chars: int = 500
    """工具结果超过该字符数即沉淀到 VFS（全文落盘，提示里只留摘要 + 文件卡片）。"""

    observation_head_chars: int = 200
    """沉淀后，在提示中保留的结果头部摘要字符数。"""

    observation_char_limit: int = 1200
    """无 VFS（或沉淀失败）时，单条 observation 允许进入提示的最大字符数，超出硬截断。"""

    keep_recent_messages: int = 8
    """压缩历史时，始终保留最近多少条消息的原文。"""

    max_history_chars: int = 6000
    """历史（不含 system 与工具说明）的字符软上限，超出则折叠更早的消息。"""

    summary_head_chars: int = 120
    """确定性降级摘要中，每条旧消息最多保留多少字符。"""


class PermissionSettings(BaseSettings):
    """权限配置：工具级 PDP + 行列级数据权限。

    **默认全部不介入**（``enabled=False``、PDP 默认放行）—— 未配置时的行为与从前
    完全一致；一旦配置规则，行/列级校验**fail closed**（改不了的 SQL 形状直接拒绝，
    不勉强改写）。

    ``rules`` 是行列级规则数组（JSON）：
        [{"role": "analyst", "data_source": "pg_dev", "table": "sales",
          "allow_columns": ["order_id", "dept_id", "amount"],
          "row_filter": "dept_id = 7"}]
    ``role`` 用 ``"*"`` 表示任意角色；``data_source`` 省略表示不限数据源。

    ``pdp_rules`` 是工具级规则数组（JSON）：
        [{"role": "analyst", "tool": "sql_query", "effect": "allow"}]
    """

    model_config = SettingsConfigDict(env_prefix="PERMISSION_", extra="ignore")

    enabled: bool = False
    """是否启用行列级数据权限（数据层强制注入 WHERE / 列白名单）。"""

    rules: str = ""
    """行列级规则（JSON 数组）。启用却为空会**显式报错**（空规则等于放行一切）。"""

    unmatched_role: str = "deny"
    """角色在规则表里一条都没命中时的态度：``deny``（默认）| ``allow``。

    默认 ``deny``：规则表一旦启用就是白名单。默认放行会让"只给 analyst 写了规则"
    静默变成"其他角色（含以 ``senior_analyst`` 运行的子 Agent）不受限"。
    """

    pdp_rules: str = ""
    """工具级 PDP 规则（JSON 数组）；为空时工具级检查不做拦截。"""

    pdp_default_policy: str = "allow"
    """PDP 未命中任何规则时的默认策略：``allow`` | ``deny``。

    默认 ``allow`` 是为了不改变既有行为（此前 PDP 根本没接进装配）；
    要"默认拒绝"就把这里改成 ``deny`` 并显式列出允许项。
    """


class AuthSettings(BaseSettings):
    """服务端鉴权配置（对内网/对外提供服务的前提）。

    ``tokens`` 是**令牌 → 身份**的映射（JSON 对象），角色只能来自这里：

        {"<长随机令牌>": {"role": "analyst", "name": "张三"},
         "<另一个令牌>":  {"role": "admin",   "name": "李四"}}

    **不设默认值**：凭据的默认值一旦进仓库就等于没有鉴权。``enabled=true`` 而
    ``tokens`` 为空时服务端**直接启动失败**——加鉴权最怕的是"以为开了、实际没开"。
    """

    model_config = SettingsConfigDict(env_prefix="AUTH_", extra="ignore")

    enabled: bool = True
    """是否启用鉴权。关闭仅供本机开发，服务端会打 WARNING。"""

    tokens: str = ""
    """令牌表（JSON 对象）。无令牌时启用鉴权会启动失败。"""

    approver_roles: str = "admin"
    """可执行人工审批的角色（逗号分隔）。"""


class PIISettings(BaseSettings):
    """PII 脱敏配置（C4：自研中文规则层，不引 Presidio）。

    规则命中即替换为 ``[MASKED_手机号]`` 形式的占位符。校验位/号段/Luhn 校验
    默认开启：不校验的话，普通 18 位编号会被误当身份证、16 位订单号会被误当卡号。
    """

    model_config = SettingsConfigDict(env_prefix="PII_", extra="ignore")

    enabled: bool = True
    """是否启用 PII 脱敏。"""

    validate_checksum: bool = True
    """是否校验身份证校验位 / 银行卡 Luhn；关闭则只按正则形态命中。"""

    mask_artifacts: bool = True
    """是否脱敏工具返回的结构化产物（真实行数据在这里，不脱敏等于没脱敏）。"""

    skip_tools: str = ""
    """入参不脱敏的工具名（逗号分隔）。**默认空**。

    "哪些工具的入参不该脱敏"是**工具自己**的事，由 ``ToolDef.pii_skip`` 声明 ——
    框架配置按名点名领域工具，等于让框架认识领域名词（默认值写错就是全错）。
    本项只作部署期的兜底覆盖，默认不点名任何工具。
    """


class QualitySettings(BaseSettings):
    """质量门配置：确定性数据质量红线 + Critic 语义裁判。

    数据质量红线是**确定性**的（缺失率/重复率超阈值 → 暂停分析待人工确认）；
    Critic 是对照验收标准的 LLM 语义裁判（审查分析逻辑缺陷，见 gate.py）。
    """

    model_config = SettingsConfigDict(env_prefix="QUALITY_", extra="ignore")

    data_check_enabled: bool = True
    """是否启用确定性数据质量校验（缺失率/重复率红线）。"""

    missing_rate_max: float = 0.3
    """单列缺失率上限（0..1）；**超过**（不含等于）即暂停分析。"""

    duplicate_rate_max: float = 0.3
    """整表重复率上限（0..1）；**超过**即暂停分析。"""

    critic_enabled: bool = True
    """是否启用质量门 Critic（每个子任务一次 LLM 裁判）。关闭可省调用。"""


class CacheSettings(BaseSettings):
    """工具结果缓存配置（见 ``harness.cache``）。

    缓存的对象是**工具结果**，不是 LLM 响应 —— 后者的成本目标已由服务端的提示词
    前缀缓存覆盖。键由「工具名 + 参数 + 输入文件身份」构成，因此**没有 TTL**：
    文件没变就命中，变了立即失效。
    """

    model_config = SettingsConfigDict(env_prefix="CACHE_", extra="ignore")

    enabled: bool = True
    """是否启用工具结果缓存。"""

    max_entries: int = 128
    """条目上限（有界 LRU，超出淘汰最久未用）。"""


class CheckpointSettings(BaseSettings):
    """图状态检查点配置（审批 interrupt 的持久化）。

    ``backend="sqlite"``（默认）把中断状态落盘：服务重启后仍能继续审批。
    ``backend="memory"`` 是进程内实现，重启即丢 —— 仅用于测试与短命令流程。
    ``backend="postgres"`` 用共享 PG 存检查点：多副本部署时审批中断状态跨副本一致，
    配合 Redis 限流一起构成"共享会话/窗口状态"层（见 deploy/README）。
    """

    model_config = SettingsConfigDict(env_prefix="CHECKPOINT_", extra="ignore")

    backend: str = "sqlite"
    """``sqlite`` | ``memory`` | ``postgres``。未知取值会显式报错，不静默回退。"""

    sqlite_path: str = "data/checkpoints.sqlite"
    """sqlite 后端的状态文件（项目相对路径，父目录自动创建）。"""

    postgres_dsn: str = ""
    """postgres 后端的连接串，如 ``postgresql://user:pass@host:5432/db``。
    ``backend="postgres"`` 时必填，为空会显式报错。"""


class RateLimitSettings(BaseSettings):
    """工具调用限流配置（滑动窗口）。

    ``backend="process"``（默认）是进程内滑动窗口：单机够用，多副本时各副本各算各的，
    实际放行量 = 副本数 × 配额。
    ``backend="redis"`` 用共享 Redis（zset + Lua 原子滑动窗口）：多副本时限流配额全局一致。
    """

    model_config = SettingsConfigDict(env_prefix="RATE_LIMIT_", extra="ignore")

    backend: str = "process"
    """``process`` | ``redis``。未知取值会显式报错，不静默回退。"""

    redis_url: str = ""
    """redis 后端的连接串，如 ``redis://host:6379/0``。
    ``backend="redis"`` 时必填，为空会显式报错。"""


class DataSourceSettings(BaseSettings):
    """命名数据源配置（C5：SQLAlchemy 统一连接层）。

    通过环境变量 ``DATASOURCE_SOURCES`` 配置多个命名数据源，值为 JSON 对象：
        {"mysql_prod": "mysql+pymysql://ro:pass@host:3306/db",
         "pg_dwh": "postgresql+psycopg://ro:pass@host:5432/dwh"}

    注意驱动名：PG 用 **psycopg（v3）**，即 ``postgresql+psycopg://``。
    写成 ``psycopg2`` 在本项目里连不上 —— 依赖里装的是 psycopg 3，没有 psycopg2
    （早先此处示例写错，照抄即报 ModuleNotFoundError）。

    也支持简单的 ``name=url,name=url`` 形式（密码中的特殊字符请做 URL 编码）。
    SQLite 文件无需在此配置 —— sql_query 的 db_path 会自动注册只读源。

    本地开发可 ``docker compose -f infra/docker-compose.yml up -d postgres mysql``
    起库（端口 55432 / 53306，只读账号 harness_ro）。
    """

    model_config = SettingsConfigDict(env_prefix="DATASOURCE_", extra="ignore")

    sources: str = ""
    default: str = Field("", description="默认数据源名（未指定 data_source/db_path 时使用）")


class CircuitBreakerSettings(BaseSettings):
    """工具熔断配置（按工具名独立维护三态熔断器）。

    熔断与限流是两件事：**限流挡的是"调用太密"，熔断挡的是"调了也没用"**。
    后者保护的是下游持续性故障 —— 每次调用都要等一次超时才失败，而 LLM 看到失败
    还会重试，把一次等待放大成好几倍。
    """

    model_config = SettingsConfigDict(env_prefix="CIRCUIT_", extra="ignore")

    enabled: bool = True
    # 连续失败多少次后熔断。这是**策略阈值**而不是可推导量：太小会因偶发抖动误熔断，
    # 太大则响应太慢。5 是连续失败型熔断器的常规取值。
    failure_threshold: int = 5
    # 熔断后隔离多久再放试探。默认取一个**完整的限流窗口**（60 秒）—— 隔离一个窗口，
    # 给下游同等的时间恢复。
    cooldown_seconds: float = 60.0
    # 半开状态下允许几个试探同时在飞（默认 1：刚恢复不该立刻被打满）
    half_open_trials: int = 1


class Settings(BaseSettings):
    """全局配置聚合。

    所有子配置作为嵌套属性访问：
        settings.milvus.host
        settings.kafka.bootstrap_servers
        settings.llm.api_key
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    llm: LLMSettings = Field(default_factory=LLMSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    milvus: MilvusSettings = Field(default_factory=MilvusSettings)
    kafka: KafkaSettings = Field(default_factory=KafkaSettings)
    minio: MinIOSettings = Field(default_factory=MinIOSettings)
    server: ServerSettings = Field(default_factory=ServerSettings)
    sandbox: SandboxSettings = Field(default_factory=SandboxSettings)
    circuit: CircuitBreakerSettings = Field(default_factory=CircuitBreakerSettings)
    trace: TraceSettings = Field(default_factory=TraceSettings)
    event: EventSettings = Field(default_factory=EventSettings)
    vfs: VFSSettings = Field(default_factory=VFSSettings)
    memory: MemorySettings = Field(default_factory=MemorySettings)
    context: ContextSettings = Field(default_factory=ContextSettings)
    checkpoint: CheckpointSettings = Field(default_factory=CheckpointSettings)
    rate_limit: RateLimitSettings = Field(default_factory=RateLimitSettings)
    quality: QualitySettings = Field(default_factory=QualitySettings)
    cache: CacheSettings = Field(default_factory=CacheSettings)
    pii: PIISettings = Field(default_factory=PIISettings)
    auth: AuthSettings = Field(default_factory=AuthSettings)
    permission: PermissionSettings = Field(default_factory=PermissionSettings)
    datasource: DataSourceSettings = Field(default_factory=DataSourceSettings)
    runtime: RuntimeSettings = Field(default_factory=RuntimeSettings)


# 全局配置单例
settings = Settings()


__all__ = ["settings", "Settings"]
