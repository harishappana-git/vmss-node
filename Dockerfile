# syntax=docker/dockerfile:1

FROM node:20-alpine AS deps
WORKDIR /app
COPY package*.json ./
RUN npm ci --omit=dev

FROM node:20-alpine AS runtime
WORKDIR /app

ARG GIT_SHA=unknown
ENV NODE_ENV=production
ENV PORT=3000
ENV GIT_SHA=${GIT_SHA}

LABEL org.opencontainers.image.source="https://github.com/example/vmss-node" \
      org.opencontainers.image.title="Todo API" \
      org.opencontainers.image.revision="${GIT_SHA}"

COPY --from=deps /app/node_modules ./node_modules
COPY src ./src

USER node
EXPOSE 3000
CMD ["node", "src/index.js"]
