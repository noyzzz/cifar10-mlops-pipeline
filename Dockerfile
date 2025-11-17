FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies (curl for healthcheck, libgomp1 for PyTorch)
RUN apt-get update && apt-get install -y \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (API only - minimal set)
# Use CPU-only versions to reduce image size
# Pin numpy to 1.x for compatibility with PyTorch 2.0.0
RUN pip install --no-cache-dir \
    "numpy<2" \
    torch==2.0.0+cpu \
    torchvision==0.15.0+cpu \
    fastapi==0.100.0 \
    uvicorn[standard]==0.23.0 \
    python-multipart==0.0.6 \
    pillow==10.0.0 \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Copy only necessary source files for API serving
COPY src/__init__.py ./src/
COPY src/ml/__init__.py ./src/ml/
COPY src/ml/model.py ./src/ml/
COPY src/ml/infer.py ./src/ml/
COPY src/api/ ./src/api/

# Copy model artifacts (trained models)
COPY artifacts/ ./artifacts/

# Create logs directory
RUN mkdir -p logs

# Expose API port
EXPOSE 8000

# Health check - automatically verify container is healthy
HEALTHCHECK --interval=30s --timeout=3s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the API server
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
