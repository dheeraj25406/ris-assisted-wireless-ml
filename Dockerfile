# Lightweight Python 3.9 image
FROM python:3.9-slim

# Prevent Python from writing .pyc files & buffer stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency specifications
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy complete project code & assets
COPY . .

# Expose default service port
EXPOSE 8501

# Healthcheck to verify app is serving
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8501/api/health || exit 1

# Launch FastAPI modern web application
CMD ["python3", "-m", "uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8501"]
