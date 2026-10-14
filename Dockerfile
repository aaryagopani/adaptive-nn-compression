FROM python:3.11-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY openenv.yaml .
COPY models.py .
COPY client.py .
COPY data/ data/
COPY server/ server/
COPY assets/ assets/

# Expose the server port
EXPOSE 7860

# Run the FastAPI server
CMD ["python", "server/app.py"]
