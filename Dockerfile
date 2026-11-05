# Base image: Official lightweight Python runtime
FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app

# Install system dependencies (needed for compiling some scientific libraries)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency specifications and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and scripts into the container
COPY . .

# Default command to run the simulation
CMD ["python", "main.py"]

