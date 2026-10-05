# ============================================================
# Governed —— LangGraph 之上的 Agent 管控运行时
# 单阶段构建。
#
# 为什么不用多阶段（builder/runtime 分离）：
#   Docker Desktop BuildKit 在 Windows + overlay2 上，跨阶段
#   COPY --from=builder /install /usr/local 复制大量小文件时，
#   会把目标文件截断为 0 字节（曾导致 11210 个 .py 为空，
#   uvicorn/fastapi/pandas 全部失效但镜像仍构建成功）。
#   本项目零 apt（所有 C 扩展包均有 manylinux wheel），
#   不需要 builder 阶段装编译工具，单阶段既简单又可靠。
#
# 设计取舍：
#   - 不装 build-essential / libpq-dev：psycopg[binary] 是预编译 wheel（自带
#     libpq），其余有 C 扩展的包（numpy/pandas/matplotlib/pydantic-core）在
#     Python 3.14 上均有 manylinux wheel，无需源码编译。
#   - 不装 curl：healthcheck 用 Python 自带 urllib 完成。
#   - 允许构建时传入国内 pip 镜像：
#       docker build --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple .
#     （compose 下：docker compose build --build-arg PIP_INDEX_URL=... governed）
# ============================================================
FROM python:3.14-slim

WORKDIR /app

ARG PIP_INDEX_URL=
ENV PIP_INDEX_URL=${PIP_INDEX_URL} \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# 先装依赖层（requirements 不变时缓存命中）
COPY requirements.txt .
# 安装后立即删除 __pycache__：减少约一半文件数，降低 BuildKit 处理 layer 时的
# 内存峰值（Docker VM 内存有限时避免 OOM），同时减小镜像体积。
RUN pip install --no-cache-dir -r requirements.txt \
 && find /usr/local/lib/python3.14/site-packages -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true

# 拷贝代码与本地包
COPY harness/ ./harness/
COPY packages/ ./packages/
COPY pyproject.toml ./

# 注册 governed 本地包与 entry points（依赖已装好，--no-deps）
RUN pip install --no-cache-dir --no-deps .

# 数据目录（checkpoint / trace / events / vfs / memory）
RUN mkdir -p /app/data

EXPOSE 8000

# 健康检查：用 Python urllib 探测 /health，无需额外装 curl
HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=5 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health',timeout=4).status==200 else 1)"

# python -m uvicorn 比直接 uvicorn 更可靠（不依赖入口脚本的 shebang/PATH）
CMD ["python", "-m", "uvicorn", "harness.server.run:app", "--host", "0.0.0.0", "--port", "8000"]
