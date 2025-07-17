FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libssl-dev \
    libffi-dev \
    python3-dev \
    git \
    poppler-utils \
    libgl1-mesa-glx \
    libsm6 \
    libxext6 \
    libxrender-dev \
        libheif1 \
        libheif-dev \
        libglib2.0-0 \
        && rm -rf /var/lib/apt/lists/*
    # <-- 新增 安裝libglib基本共享函式庫，解決libgthread找不到問題

COPY requirements.txt .

# 升級 pip、setuptools、wheel 並安裝Python依賴
RUN python -m ensurepip --upgrade && \
    pip install --upgrade pip setuptools wheel && \
    pip cache purge && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p data/chroma_db backend/uploaded_files

EXPOSE 5000

CMD ["python", "backend/app_Docling.py"]
