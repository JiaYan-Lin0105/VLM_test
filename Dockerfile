# Use an official PyTorch image with CUDA support
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel

# Set environment variable to prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    cmake \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
# Note: explicitly ensuring pip is up to date
RUN pip install --upgrade pip

# Install llama-cpp-python with CUDA support
# We set CMAKE_ARGS to enable CUDA support during compilation
# RUN CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python

# Install other requirements
RUN pip install -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . /app

# Command to run the RAG script by default, or open a shell
CMD ["python", "RAG/RAG.py"]
