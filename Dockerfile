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

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH"

# Copy the virtual environment from the builder stage
COPY --from=builder /app/.venv /app/.venv

# Copy the application code
COPY app /app/app
COPY ui /app/ui
COPY app.py api.py init_db.py ./

# Expose Streamlit default port
EXPOSE 8501

# Run the Streamlit application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
