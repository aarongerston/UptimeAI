# Use the official Python image
FROM python:3.12-slim

# Set the working directory
WORKDIR /app

# Install only necessary dependencies
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Copy the application files
COPY aux.py .
COPY backend.py .
COPY scaler.pkl .
COPY backend_reqs.txt .

# Install dependencies
RUN pip install --no-cache-dir -r backend_reqs.txt

# Run the FastAPI app with Uvicorn
EXPOSE 8080
CMD ["uvicorn", "backend:app", "--host", "0.0.0.0", "--port", "8080"]
