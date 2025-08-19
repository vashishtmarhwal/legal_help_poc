# Stage 1: Use an official Python runtime as a parent image
# We use the 'slim' version for a smaller image size.
FROM python:3.10-slim

# Stage 2: Set environment variables
# Prevents Python from writing pyc files to disc and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Stage 3: Set the working directory in the container
WORKDIR /app

# Stage 4: Install dependencies
# Copy the requirements file first to leverage Docker's layer caching.
# This way, dependencies are only re-installed if requirements.txt changes.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Stage 5: Copy the application source code
# Copy the 'src' directory from our project into the container's working directory.
COPY ./src /app/src

# Stage 6: Expose the port the app runs on
# The container will listen on port 8000 for incoming connections.
EXPOSE 8000

# Stage 7: Define the command to run the application
# This tells the container how to start the FastAPI server with Uvicorn.
# We use src.main:app because our main.py file is inside the 'src' directory.
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]