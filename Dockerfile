# Dockerfile for Customer Churn Prediction API
# Multi-stage build for optimized production image

# ============================================================================
# Stage 1: Builder - Install dependencies
# ============================================================================
FROM python:3.9-slim as builder

LABEL maintainer="ML Engineering Team <ml-team@example.com>"
LABEL description="Customer Churn Prediction API - Builder Stage"

# Set working directory
WORKDIR /build

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ============================================================================
# Stage 2: Runtime - Create final image
# ============================================================================
FROM python:3.9-slim

LABEL maintainer="ML Engineering Team <ml-team@example.com>"
LABEL description="Customer Churn Prediction API - Production"
LABEL version="1.0.0"

# Create non-root user for security
RUN useradd -m -u 1000 apiuser && \
    mkdir -p /app && \
    chown -R apiuser:apiuser /app

# Set working directory
WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application files
COPY --chown=apiuser:apiuser api/ ./api/
COPY --chown=apiuser:apiuser model_pipeline.py .
COPY --chown=apiuser:apiuser config_loader.py .
COPY --chown=apiuser:apiuser config/ ./config/
COPY --chown=apiuser:apiuser models/ ./models/

# Switch to non-root user
USER apiuser

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Run the application
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
