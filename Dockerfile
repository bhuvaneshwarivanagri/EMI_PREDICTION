# EMIPredict AI — Hugging Face Docker Deployment
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all app files
COPY app_hf.py .
COPY pages/ ./pages/
COPY .streamlit/ ./.streamlit/
COPY predict_utils.py .
COPY download_models.py .

# Create necessary directories
RUN mkdir -p models data artifacts logs

# Expose Streamlit port
EXPOSE 7860

# Hugging Face Spaces uses port 7860
ENV STREAMLIT_SERVER_PORT=7860
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Run the app
CMD ["streamlit", "run", "app_hf.py", \
     "--server.port=7860", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
