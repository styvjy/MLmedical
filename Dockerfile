# Production Dockerfile for Pneumonia Classifier
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY src/ ./src/
COPY web/ ./web/
COPY models/ ./models/
COPY download_dataset.py .

ENV PORT=5000
ENV PYTHONUNBUFFERED=1

EXPOSE 5000

CMD ["python", "web/app.py"]
