# ============================================================
# Governed —— LangGraph 之上的 Agent 管控运行时
# 多阶段构建：builder 装编译依赖，runtime 只留运行时
# ============================================================
FROM python:3.14-slim AS builder

WORKDIR /app

# 编译依赖：psycopg[binary] 是预编译 wheel 不需要，但部分间接依赖可能需要
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# 先装依赖层（代码不变时缓存命中）
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ============================================================
FROM python:3.14-slim AS runtime

WORKDIR /app

# 运行时系统依赖：curl 供 healthcheck，libpq 供 psycopg
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

# 从 builder 拷贝已安装的包
COPY --from=builder /install /usr/local

# 拷贝代码
COPY harness/ ./harness/
COPY packages/ ./packages/
COPY pyproject.toml ./

# 安装 governed 本地包（依赖已在 builder 层装好，这里只注册包与 entry points）
RUN pip install --no-cache-dir --no-deps .

# 数据目录（checkpoint / trace / events / vfs / memory）
RUN mkdir -p /app/data

EXPOSE 8000

# 健康检查：FastAPI /health 端点
HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=5 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "harness.server.run:app", "--host", "0.0.0.0", "--port", "8000"]
