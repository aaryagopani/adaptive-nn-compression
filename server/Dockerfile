FROM python:3.11-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements from project root
COPY ../requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy server and project files
COPY ../openenv.yaml ./openenv.yaml
COPY ../models.py ./models.py
COPY ../client.py ./client.py
COPY . ./server/
COPY ../data/ ./data/
COPY ../assets/ ./assets/

EXPOSE 7860

CMD ["python", "server/app.py"]
