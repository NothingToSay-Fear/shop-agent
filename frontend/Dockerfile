FROM node:20-alpine AS build

WORKDIR /app
# 使用锁文件，确保镜像内安装的依赖版本可复现。
COPY package.json package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY . .
# Vite 在构建期注入公开变量，而不是在 Nginx 启动时注入。
ARG VITE_API_URL=http://localhost:8000
ENV VITE_API_URL=$VITE_API_URL
RUN npm run build

FROM nginx:1.27-alpine
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
