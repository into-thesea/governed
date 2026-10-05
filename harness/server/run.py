"""harness.server.run —— uvicorn 启动入口。

运行（项目根目录）：
    .venv\\Scripts\\python.exe -m harness.server.run
或：
    .venv\\Scripts\\uvicorn harness.server.run:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

from harness.server.app import create_app

# 模块级 app，供 uvicorn / gunicorn 引用：harness.server.run:app
app = create_app()


if __name__ == "__main__":
    import uvicorn

    from harness.config import settings
    uvicorn.run("harness.server.run:app", host=settings.server.host, port=settings.server.port, reload=False)  # nosec B104 - 服务端绑定地址来自配置（默认 0.0.0.0 是预期默认）
