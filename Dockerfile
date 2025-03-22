FROM python:3.11-slim-buster

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the data and source code
COPY atp_tennis.csv .
COPY main.py .
COPY api.py .

# Set environment variables
ENV PORT=8004
ENV PYTHONUNBUFFERED=1

# Expose the API port
EXPOSE 8004

# Run the API
CMD ["python", "api.py"] 