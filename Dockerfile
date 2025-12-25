# 使用 Astral 官方 uv 镜像，支持多架构（amd64/arm64/armv7）
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# 设置工作目录
WORKDIR /app

# 先复制依赖文件（如果有 pyproject.toml / requirements.txt）
# 如果你没有 pyproject.toml，可以直接跳过这步，用 pip 安装固定包
# 这里我们直接用 uv pip install 固定依赖（最简单可靠）
RUN uv pip install --system --no-cache gemini_webapi fastapi uvicorn python-multipart

# 复制你的 main.py
COPY main.py .

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
