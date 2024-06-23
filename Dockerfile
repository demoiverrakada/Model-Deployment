# Use the official PyTorch image from the Docker Hub
FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-runtime

# Set the working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port the app runs on
EXPOSE 5000

# Command to run the application
CMD ["python", "app.py"]
