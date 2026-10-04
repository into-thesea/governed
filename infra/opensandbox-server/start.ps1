# ============================================================================
# 启动 Governed 的沙箱层基础设施（OpenSandbox 服务端）
#
# 服务端以独立 venv 运行在本目录，不与项目 .venv 混装：
#   项目 .venv        只装轻量客户端 SDK（opensandbox）
#   本目录 .venv      装控制面（opensandbox-server，依赖 docker/fastapi/grpcio/...）
#
# 首次准备（若 .venv 不存在）：
#   python -m venv .venv
#   .venv\Scripts\python.exe -m pip install opensandbox-server==0.2.3
#   Copy-Item sandbox.toml.example sandbox.toml
#
# api_key 不写进配置文件/版本库，启动前通过环境变量注入：
#   $env:OPENSANDBOX_SERVER_API_KEY = "一条长随机串"
# 客户端 .env 的 SANDBOX_API_KEY 必须与之完全一致。
#
# 用法：
#   infra\opensandbox-server\start.ps1
# ============================================================================
$ErrorActionPreference = "Stop"
$here      = Split-Path -Parent $MyInvocation.MyCommand.Path
$py        = Join-Path $here ".venv\Scripts\python.exe"
$conf      = Join-Path $here "sandbox.toml"
$example   = Join-Path $here "sandbox.toml.example"
$stateDir  = Join-Path $here "state"

# 注意：用 python.exe 而非 .venv\Scripts\opensandbox-server.exe —— 该 console
# 入口脚本在部分 Windows 环境下会静默退出（exit 1、无任何输出）。直接经
# python -c 调用 CLI 入口更可靠，且不改变任何运行语义。
if (-not (Test-Path $py)) {
    Write-Host "未找到服务端 Python：$py" -ForegroundColor Red
    Write-Host "请先在本目录准备独立 venv：" -ForegroundColor Yellow
    Write-Host "  python -m venv .venv"
    Write-Host "  .venv\Scripts\python.exe -m pip install opensandbox-server==0.2.3"
    exit 1
}

if (-not (Test-Path $conf)) {
    Write-Host "未找到配置：$conf" -ForegroundColor Red
    if (Test-Path $example) {
        Write-Host "请先从模板复制一份（该文件不进版本库）：" -ForegroundColor Yellow
        Write-Host "  Copy-Item `"$example`" `"$conf`""
    }
    exit 1
}

# api_key 必须由环境变量注入；未设置即拒绝启动（避免无认证/弱凭据的服务端）
if (-not $env:OPENSANDBOX_SERVER_API_KEY) {
    Write-Host "未设置环境变量 OPENSANDBOX_SERVER_API_KEY。" -ForegroundColor Red
    Write-Host "请先设置一条长随机串（客户端 .env 的 SANDBOX_API_KEY 须与之相同）：" -ForegroundColor Yellow
    Write-Host '  $env:OPENSANDBOX_SERVER_API_KEY = "your-long-random-key"'
    exit 1
}

# 切到脚本目录：让 sandbox.toml 中的相对路径（state/opensandbox.db）基于本目录解析
Set-Location $here
if (-not (Test-Path $stateDir)) {
    New-Item -ItemType Directory -Path $stateDir | Out-Null
}

Write-Host "启动 OpenSandbox 服务端（Ctrl+C 停止）..." -ForegroundColor Cyan
$cliCode = "from opensandbox_server.cli import main; main()"
& $py -c $cliCode --config $conf
