FROM node:24-bookworm-slim AS deps
WORKDIR /app
COPY package.json package-lock.json* ./
COPY apps/web/package.json apps/web/package.json
COPY packages/avatar_protocol/package.json packages/avatar_protocol/package.json
RUN npm install

FROM deps AS build
COPY apps/web apps/web
COPY packages/avatar_protocol packages/avatar_protocol
RUN npm run web:build

FROM nginx:1.29-alpine
COPY --from=build /app/apps/web/dist /usr/share/nginx/html
COPY apps/web/nginx.conf.template /etc/nginx/templates/default.conf.template
EXPOSE 80
