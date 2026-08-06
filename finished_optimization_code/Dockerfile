FROM python:3.11-slim

WORKDIR /app

# System deps for casadi (IPOPT solver)
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Data volume for logs, SQLite, calibration
VOLUME /app/data

ENV PYTHONUNBUFFERED=1

CMD ["python", "optimizer_daemon.py", "--verbose"]
