FROM node:22 AS build

WORKDIR /app

ARG VITE_GROQ_API_KEY
ARG VITE_GROQ_MODEL
ARG VITE_GROQ_MODELS
ENV VITE_GROQ_API_KEY=$VITE_GROQ_API_KEY
ENV VITE_GROQ_MODEL=$VITE_GROQ_MODEL
ENV VITE_GROQ_MODELS=$VITE_GROQ_MODELS

COPY package*.json ./
RUN npm ci

COPY . .
RUN npm run build

FROM node:22-alpine AS runtime

WORKDIR /app

COPY --from=build /app/package*.json ./
COPY --from=build /app/node_modules ./node_modules
COPY --from=build /app/dist ./dist

ENV NODE_ENV=production
EXPOSE 4173

CMD ["npm", "run", "preview", "--", "--host", "0.0.0.0", "--port", "4173"]
