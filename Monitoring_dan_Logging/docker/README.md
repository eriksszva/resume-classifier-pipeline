# Docker Deployment Guide

This project uses multiple Docker Compose configurations to separate infrastructure components like **monitoring tools** and the **machine learning model service**, allowing for modular development and deployment.

## Directory Structure

```bash
docker/
├── docker-compose-monitoring.yaml   # Stack for observability tools 
├── docker-compose-model.yaml        # Stack for ML model API service
````

##  How to Run

1. **Navigate to the `docker/` directory:**

   ```bash
   cd docker/
   ```

2. **Start the monitoring stack:**

   ```bash
   docker compose -f docker-compose-monitoring.yaml up -d --build
   ```

3. **Start the model API service:**

   ```bash
   docker compose -f docker-compose-model.yaml up -d --build
   ```

## 📦 Services Overview

| Compose File                     | Description                                          |
| -------------------------------- | ---------------------------------------------------- |
| `docker-compose-monitoring.yaml` | Observability tools like **Grafana**, **Prometheus** |
| `docker-compose-model.yaml`      | Machine learning **inference API** (e.g., Logistic Regression FastAPI)   |

## 📤 Exposed Ports (Default)

| Service    | Port | Description        |
| ---------- | ---- | ------------------ |
| Grafana    | 3000 | Dashboard UI       |
| Prometheus | 9090 | Metrics UI         |
| Model API  | 8000 | Inference endpoint |

> 🔧 Make sure to adjust ports in the `.yaml` files if changed the defaults.

## How to Stop the Services

To stop all services and remove the containers:

```bash
docker compose -f docker-compose-monitoring.yaml down
docker compose -f docker-compose-model.yaml down
```

## Related Docs

* [Grafana Documentation](https://grafana.com/docs/)
* [Prometheus Documentation](https://prometheus.io/docs/)
* [FastAPI Documentation](https://fastapi.tiangolo.com/)
