# Use a minimal Python base image
FROM python:3.10-slim

# Set working directory in container
WORKDIR /app

# Copy dependency list and install packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your source code and model
COPY src/ src/
COPY models/ models/

# Run the prediction script
CMD ["python", "src/predict.py"]
