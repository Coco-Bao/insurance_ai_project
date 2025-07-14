# 選擇基礎映像，使用官方 Python 3.9 slim 版本，體積較小且兼容性好
FROM python:3.10-slim

# 設定環境變數，確保 Python 輸出不緩衝，方便日誌即時查看
ENV PYTHONUNBUFFERED=1

# 安裝系統依賴，滿足 pdf2image、opencv、pillow_heif 等函式庫的運行需求
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    build-essential \
    poppler-utils \
    libgl1-mesa-glx \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libheif1 \
    libheif-dev && rm -rf /var/lib/apt/lists/*

# 以下是註解說明，放在獨立行避免解析錯誤
# poppler-utils：pdf2image 依賴的 PDF 轉換工具
# libgl1-mesa-glx：opencv 依賴的圖形庫
# libheif1：pillow_heif 運行時庫
# libheif-dev：pillow_heif 編譯依賴

# 設定工作目錄
WORKDIR /app

# 安装系统依赖（PaddlePaddle 推荐）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y poppler-utils

RUN pip install --upgrade pip

# 先安装 paddlepaddle，确保版本兼容
RUN pip install paddlepaddle==2.6.2 -i https://pypi.tuna.tsinghua.edu.cn/simple

# 统一安装其他依赖，包括 paddleocr 和 paddlex
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --upgrade paddlepaddle
RUN pip install --upgrade paddlex

# 複製專案所有程式碼到容器中
COPY . /app

# 建立運行時必要目錄，確保程式運行時目錄存在
RUN mkdir -p data/chroma_db backend/uploaded_files

# 開放 Flask 預設埠號
EXPOSE 5000

# 說明：OpenAI 相關敏感環境變數建議於 docker 運行時透過 -e 參數傳入
# 例如：docker run -e OPENAI_API_KEY=your_key -e OPENAI_API_BASE=your_base ...

# 容器啟動時執行 Flask 應用，假設入口檔案為 app.py，請依實際情況修改
CMD ["python", "backend/app.py"]
