# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the current directory contents into the container at /usr/src/app
COPY . .

# Install packages
RUN pip install --no-cache-dir uvicorn fastapi python-multipart transformers accelerate tensorflow[and-cuda] huggingface_hub python-dotenv

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Define environment variable
ENV NAME Architecht

# Run app.py when the container launches
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--reload"]
