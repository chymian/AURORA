# Use the official Python image from the Docker Hub
FROM python:3.12

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Update the package list and install g++, PortAudio, and other dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    g++ \
    portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install the required packages
RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Command to run on container start
CMD ["python", "main.py"]
