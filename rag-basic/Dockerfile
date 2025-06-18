FROM ubuntu:22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links for python and pip
RUN ln -sf /usr/bin/python3 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Create virtual environment
RUN python3 -m venv /opt/venv

# Set PATH to include virtual environment
ENV PATH="/opt/venv/bin:$PATH"

# Activate virtual environment and upgrade pip
RUN /opt/venv/bin/pip install --upgrade pip

# Install Python dependencies in virtual environment
RUN /opt/venv/bin/pip install --no-cache-dir -r requirements.txt
RUN /opt/venv/bin/pip install 'protobuf<=3.20.1' --force-reinstall
RUN /opt/venv/bin/pip install pypdf

# Debug: Check what's installed and where
RUN echo "=== Virtual environment contents ===" && \
    ls -la /opt/venv/bin/ && \
    echo "=== Checking streamlit installation ===" && \
    /opt/venv/bin/pip list | grep streamlit && \
    echo "=== Python path ===" && \
    /opt/venv/bin/python -c "import sys; print(sys.path)" && \
    echo "=== Trying to find streamlit ===" && \
    find /opt/venv -name "*streamlit*" -type f

# Copy application code
COPY . .

# Expose port for Streamlit
EXPOSE 8501

# Set PATH to include virtual environment
ENV PATH="/opt/venv/bin:$PATH"
RUN pip show protobuf

# Command to run the application
CMD ["/bin/bash", "-c", "echo 'Starting application...' && ls -la /opt/venv/bin/ && /opt/venv/bin/python -m streamlit run app.py --server.address 0.0.0.0 --server.port 8501"]