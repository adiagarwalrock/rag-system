# Stage 1: Build
FROM ghcr.io/astral-sh/uv:python3.12-slim AS builder

# Set working directory
WORKDIR /app

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Copy only dependency files first to leverage Docker cache
COPY pyproject.toml uv.lock ./

# Install dependencies into a virtual environment
# We use --no-install-project to avoid re-installing the project itself when source changes
RUN uv sync --frozen --no-install-project --no-dev

# Stage 2: Runtime
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install Node/npm for LiteParse CLI tooling.
RUN apt-get update \
    && apt-get install -y --no-install-recommends nodejs npm \
    && npm install -g @llamaindex/liteparse \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PARSED_ARTIFACTS_DIR=/app/artifacts/parsed \
    RAW_DATA_DIR=/app/data/raw \
    PATH="/app/.venv/bin:$PATH"

# Copy the virtual environment from the builder stage
COPY --from=builder /app/.venv /app/.venv

# Copy the application code
COPY app /app/app
COPY ui /app/ui
COPY streamlit_app.py api.py init_db.py ./

# Ensure local storage roots exist in container runtime.
RUN mkdir -p /app/data/raw /app/artifacts/parsed

# Expose Streamlit default port
EXPOSE 8501

# Run the Streamlit application
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
