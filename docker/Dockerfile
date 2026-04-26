FROM python:3.11-slim

# Install basic tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps (lightweight by default)
COPY requirements.txt /app/requirements.txt
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_REQUIRE_HASHES=0
ARG WITH_BERTOPIC=0
RUN python -m pip install --upgrade pip setuptools wheel && \
    pip config unset global.require-hashes || true && \
    pip install --no-cache-dir -r requirements.txt && \
    if [ "$WITH_BERTOPIC" = "1" ]; then \
        pip install --no-cache-dir -i https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cpu \
          torch==2.4.1+cpu sentence-transformers==3.1.1 "bertopic[visualization]==0.16.4"; \
    fi

# Copy app code
COPY . /app
ENV PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501

# Run Streamlit via Python (robust across platforms)
CMD ["python", "-m", "streamlit", "run", "src/automated_slr.py"]
