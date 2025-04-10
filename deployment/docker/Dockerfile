# Multi-stage Dockerfile for AI Platform Trainer
# Supports both development and inference environments

#------------------------------------------------------------------------------
# Base Stage - Used as the foundation for all other stages
#------------------------------------------------------------------------------
FROM python:3.9-slim AS base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml README.md ./
COPY src ./src/

#------------------------------------------------------------------------------
# CPU Development Stage - For local development without GPU
#------------------------------------------------------------------------------
FROM base AS cpu-dev

# Install development dependencies
RUN pip install -e ".[dev]"

# Default command runs environment check
CMD ["python", "-m", "src.ai_platform_trainer.utils.environment"]

#------------------------------------------------------------------------------
# GPU Development Stage - For development with GPU support
#------------------------------------------------------------------------------
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04 AS gpu-dev

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install Python and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    python3.9 \
    python3.9-dev \
    python3-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3.9 /usr/bin/python

# Copy project files
COPY pyproject.toml README.md ./
COPY src ./src/

# Install development dependencies
RUN pip install -e ".[dev]"

# Default command runs environment check
CMD ["python", "-m", "src.ai_platform_trainer.utils.environment"]

#------------------------------------------------------------------------------
# Production Stage - Minimal image for deployment
#------------------------------------------------------------------------------
FROM base AS production

# Install only production dependencies
RUN pip install -e "."

# Create a non-root user
RUN useradd -m appuser
USER appuser

# Set entry point
ENTRYPOINT ["python", "-m", "src.ai_platform_trainer.main"]
