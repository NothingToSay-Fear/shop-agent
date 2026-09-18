FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
# 嵌入模型在默认配置下使用 CPU 推理；预装 CPU 轮子以避免 pip 拉取数 GB 的 CUDA 运行库。
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.7.1+cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY evaluation ./evaluation
COPY alembic.ini .
COPY migrations ./migrations

EXPOSE 8000

CMD ["sh", "-c", "alembic upgrade head && python -m app.scripts.seed_demo_data && uvicorn app.main:app --host 0.0.0.0 --port 8000"]
