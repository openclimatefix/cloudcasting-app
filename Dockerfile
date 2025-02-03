# Use an official Python runtime as a parent image
FROM python:3.11-slim as base

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install
COPY pyproject.toml ./
RUN pip install .

# Copy the application files
COPY src /app/src

# Set the entrypoint command to run the app
CMD ["python", "src/cloudcasting_app/app.py"]