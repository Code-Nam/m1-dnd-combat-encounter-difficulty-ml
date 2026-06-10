# --- dev : serveur Vite avec hot reload (utilisé par docker-compose) ---------
FROM node:22-alpine AS dev

WORKDIR /app

COPY package*.json ./
RUN npm ci

COPY . .

EXPOSE 5173

# --host 0.0.0.0 : nécessaire pour accéder au serveur depuis l'extérieur du conteneur
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]

# --- prod : build statique servi par nginx -----------------------------------
FROM dev AS build
RUN npm run build

FROM nginx:alpine AS prod
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
