# SwagBot LangGraph Implementation with Datadog LLM Observability
ARG PYTHON_BASE_IMAGE=python:3.11-slim
FROM ${PYTHON_BASE_IMAGE}

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Upgrade pip first for better dependency resolution
RUN pip install --no-cache-dir --upgrade pip==24.0

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy essential shared files
COPY swagbot_langgraph_config.py .
COPY swagbot_utils.py .

# Copy LangGraph implementation files
COPY swagbot_langgraph_workflow.py .
COPY swagbot_app.py .

# Copy static resources
COPY resources/ resources/
COPY templates/ templates/
COPY static/ static/
COPY scripts/ scripts/

# Set environment variables
ENV FLASK_APP=swagbot_app.py
ENV FLASK_ENV=development
ENV PYTHONPATH=/app

# Expose port
EXPOSE 3000

# Run the application (instrumentation handled by docker-compose)
CMD ["python", "swagbot_app.py"]

