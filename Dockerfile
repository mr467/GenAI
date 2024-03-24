# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Metadata indicating an image maintainer
LABEL authors="rieder"

# Set the working directory in the container to /app
WORKDIR /app

# Install system dependencies required for compiling certain Python packages,
# including gcc, python3-dev, HDF5 libraries, and pkg-config. libhdf5-dev is required for h5py.
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    libhdf5-dev \
    pkg-config && \
    rm -rf /var/lib/apt/lists/*

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variables
ENV OPENAI_API_KEY=your_api_key_here
ENV TOKENIZERS_PARALLELISM=false
ENV TRANSFORMERS_FORCE_CPU=True

# Run chatbot.py when the container launches
CMD ["python", "./chatbot.py"]
