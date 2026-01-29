FROM python:3.11-slim

WORKDIR /app

# system deps (optional, aber oft hilfreich)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# LiveKit Agent process
CMD ["python", "agent.py", "start"]
