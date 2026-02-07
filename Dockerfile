# Predictive Router Oracle - Dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p reports/output router models

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Expose ports
# 8000 - API
# 8080 - Dashboard
EXPOSE 8000 8080

# Default command (API server)
CMD ["python", "-m", "uvicorn", "api.prediction_api:app", "--host", "0.0.0.0", "--port", "8000"]