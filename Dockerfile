# Use multi-stage build for efficiency
FROM nvidia/cuda:12.1-devel-ubuntu22.04 as base

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-dev \
    python3.12-distutils \
    python3-pip \
    git \
    wget \
    curl \
    build-essential \
    cmake \
    ninja-build \
    libhdf5-dev \
    libboost-all-dev \
    libeigen3-dev \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -s /usr/bin/python3.12 /usr/bin/python

# Upgrade pip
RUN python -m pip install --upgrade pip

# Set working directory
WORKDIR /app

# Development stage
FROM base as development

# Copy requirements first for better caching
COPY requirements_modern.txt ./
COPY pyproject.toml ./

# Install Python dependencies
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
RUN pip install -r requirements_modern.txt

# Copy source code
COPY src/ ./src/
COPY tests/ ./tests/
COPY README.md ./
COPY LICENSE ./

# Install package in development mode
RUN pip install -e ".[dev,vis,medical]"

# Create non-root user
RUN useradd -m -u 1000 synful && chown -R synful:synful /app
USER synful

# Set default command
CMD ["python", "-c", "from synful import *; print('Synful development environment ready!')"]

# Production stage
FROM base as production

# Copy only necessary files
COPY requirements_modern.txt ./
COPY pyproject.toml ./

# Install production dependencies only
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
RUN pip install -r requirements_modern.txt

# Copy source code
COPY src/ ./src/
COPY README.md ./
COPY LICENSE ./

# Install package
RUN pip install .

# Create non-root user
RUN useradd -m -u 1000 synful && chown -R synful:synful /app
USER synful

# Expose port for Jupyter/visualization
EXPOSE 8888

# Set default command
CMD ["synful", "info"]

# Training stage (optimized for training workloads)
FROM base as training

# Install additional training dependencies
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

COPY requirements_modern.txt ./
RUN pip install -r requirements_modern.txt

# Add wandb and additional training tools
RUN pip install wandb tensorboard neptune-client comet-ml

# Copy source
COPY src/ ./src/
COPY pyproject.toml ./
COPY README.md ./
COPY LICENSE ./

RUN pip install -e ".[dev,vis]"

# Create directories for data and outputs
RUN mkdir -p /data /outputs /models && chown -R 1000:1000 /data /outputs /models

# Create non-root user
RUN useradd -m -u 1000 synful && chown -R synful:synful /app
USER synful

# Set volumes
VOLUME ["/data", "/outputs", "/models"]

CMD ["synful", "train", "--help"]

# Inference stage (optimized for inference)
FROM base as inference

# Minimal dependencies for inference
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

COPY requirements_modern.txt ./
RUN pip install -r requirements_modern.txt

# Copy source
COPY src/ ./src/
COPY pyproject.toml ./
COPY README.md ./
COPY LICENSE ./

RUN pip install .

# Create directories
RUN mkdir -p /data /outputs /models && chown -R 1000:1000 /data /outputs /models

# Create non-root user
RUN useradd -m -u 1000 synful && chown -R synful:synful /app
USER synful

# Set volumes
VOLUME ["/data", "/outputs", "/models"]

CMD ["synful", "predict", "--help"]

# Jupyter stage (for interactive development)
FROM development as jupyter

# Install Jupyter and extensions
USER root
RUN pip install jupyterlab jupyter-widgets ipywidgets
RUN jupyter labextension install @jupyter-widgets/jupyterlab-manager

# Install additional visualization tools
RUN pip install napari[all] neuroglancer

# Create jupyter config
USER synful
RUN mkdir -p /home/synful/.jupyter
RUN echo "c.ServerApp.ip = '0.0.0.0'" > /home/synful/.jupyter/jupyter_lab_config.py
RUN echo "c.ServerApp.port = 8888" >> /home/synful/.jupyter/jupyter_lab_config.py
RUN echo "c.ServerApp.open_browser = False" >> /home/synful/.jupyter/jupyter_lab_config.py
RUN echo "c.ServerApp.allow_root = False" >> /home/synful/.jupyter/jupyter_lab_config.py

# Create example notebooks directory
RUN mkdir -p /home/synful/notebooks

EXPOSE 8888

CMD ["jupyter", "lab", "--no-browser", "--ip=0.0.0.0", "--port=8888"]