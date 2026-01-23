FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install minimal system deps required for geospatial wheels and runtime.
# Keep layers small and remove apt lists to reduce image size.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
       gdal-bin libgdal-dev \
       libproj-dev proj-bin \
       pkg-config \
       curl \
    && rm -rf /var/lib/apt/lists/*

# Create unprivileged user for running the app
RUN useradd --create-home --shell /bin/bash appuser

# Copy only requirements first to leverage Docker cache
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy application code
COPY . /app
RUN chown -R appuser:appuser /app

USER appuser

EXPOSE 8501

ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=8501

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
