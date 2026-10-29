# Stage 1: Build & Dependency Stage
FROM python:3.11-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Minimal Runtime Environment
FROM python:3.11-slim AS runner

WORKDIR /app

# Force Python to locate local modules ('src' and 'ui') from /app
ENV PYTHONPATH=/app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy source code and default structure
COPY src/ ./src/
COPY ui/ ./ui/
COPY app.py .
COPY pytest.ini .
COPY tests/ ./tests/
COPY data/ ./data/

# Expose Streamlit default port
EXPOSE 8501

# Healthcheck to verify the web server is responding
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["python", "-m", "streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]