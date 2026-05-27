FROM python:3.10-slim

# 强制 matplotlib 使用无界面后端（服务器必备）
ENV MPLBACKEND=Agg

WORKDIR /app

# 系统依赖（你原来的版本，最稳定）
RUN apt-get update && apt-get install -y \
    gcc g++ \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# 安装 Python 依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制全部代码（包括你的 main.py）
COPY . .

# 启动程序
CMD ["python", "main.py"]