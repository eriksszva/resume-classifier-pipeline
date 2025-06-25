cd docker/
docker compose -f docker-compose-monitoring.yaml up -d --build
docker compose -f docker-compose-model.yaml up -d --build