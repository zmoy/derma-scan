# Use an official, lightweight Python image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install dependencies
# --no-cache-dir keeps the Docker image smaller
COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the api code and the models directory into the container
COPY api/ ./api/
COPY models/ ./models/

# Change working directory to the api folder for execution
WORKDIR /app/api

# Expose the port FastAPI will run on
EXPOSE 8000

# Command to run the application (bound to 0.0.0.0 to be accessible from outside the container)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]