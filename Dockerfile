# Use the official Python image as the base
FROM python:3.11-slim

# Update and install system dependencies
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# Create a non-root user for security best practices
RUN useradd -ms /bin/bash appuser

# Set the working directory
WORKDIR /home/appuser/app

# Copy and install dependencies first (better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Ensure the files belong to our non-root user
RUN chown -R appuser:appuser /home/appuser/app
USER appuser

# Define the default command
CMD ["python", "-m", "ai_platform_trainer.main"]