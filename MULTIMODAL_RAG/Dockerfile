FROM python:3.11-slim

WORKDIR /app

# System deps needed by unstructured[pdf], camelot-py, pymupdf, pillow/pi-heif
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    poppler-utils \
    tesseract-ocr \
    libgl1 \
    ghostscript \
    libheif-dev \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

# Ensure runtime data dirs exist (uploads, images)
RUN mkdir -p data/uploads data/images

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
