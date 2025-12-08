# Build frontend
FROM node:20-alpine AS frontend-builder

WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

# Final image with nginx + Python backend
FROM python:3.11-slim

WORKDIR /app

# Install nginx and supervisor
RUN apt-get update && apt-get install -y --no-install-recommends \
    nginx \
    supervisor \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for Python dependency management
RUN pip install uv

# Copy backend code
COPY ../lemonade_bench /app/backend
WORKDIR /app/backend

# Install Python dependencies
RUN uv pip install --system -e .

# Copy frontend build
COPY --from=frontend-builder /app/dist /usr/share/nginx/html

# Copy nginx config
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copy supervisor config
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8080/api/health || exit 1

# Start supervisor (manages both nginx and uvicorn)
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]

