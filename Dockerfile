# Stage 1: build the frontend (no secrets involved).
FROM node:22-alpine AS frontend
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY index.html vite.config.js eslint.config.js ./
COPY public ./public
COPY src ./src
RUN npm run build

# Stage 2: Python runtime serving the API and the built frontend.
FROM python:3.11-slim AS runtime
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY smc_agents ./smc_agents
COPY api ./api
COPY data/processed ./data/processed
COPY --from=frontend /app/dist ./dist

# GROQ_API_KEY is provided at RUNTIME only - never baked into the image.
ENV PORT=8000
EXPOSE 8000
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT}"]
