# Sentence Embedding API (FastAPI + Sentence Transformers)

This service provides a lightweight HTTP API for generating sentence embeddings using `sentence-transformers`. It is designed to run **outside the main Docker container** to keep your deployment lightweight and flexible.

## Location

```bash
Monitoring_dan_Logging/resume-api/embedder/
````

---

## How to Run (Local)

Make sure you have the dependencies installed:

```bash
pip install -r requirements.txt
```

Then start the API using `uvicorn`:

```bash
cd Monitoring_dan_Logging/resume-api/embedder
uvicorn app.main:app --host 0.0.0.0 --port 5000
```

The API will be accessible at:

```
http://localhost:5000
```

## 📦 Features

* Uses `sentence-transformers` (e.g., `all-MiniLM-L6-v2`) to generate embeddings.
* Accepts raw text input via JSON.
* Returns 384-dimensional sentence vectors.
* Can be queried by any service via HTTP (e.g., classifier, retriever).

## Why Run This Outside Docker?

Running `sentence-transformers` inside a Docker container would require large dependencies, including:

* `torch` (200–500MB)
* `transformers` (\~200MB)
* Pretrained model weights (\~100–300MB)

This significantly increases image size, build time, and cold start latency—especially problematic in autoscaled environments. Therefore, this component runs separately as a local or dedicated inference service.

## 🧪 Example Request

```http
POST /embed
Content-Type: application/json

{
  "sentences": ["Data science is cool.", "Transformers are powerful models."]
}
```

### Response

```json
{
  "vectors": [
    [0.123, -0.234, ..., 0.015],
    [0.987, -0.654, ..., 0.032]
  ]
}
```

## Related

* [Sentence-Transformers Documentation](https://www.sbert.net/)
* [FastAPI Documentation](https://fastapi.tiangolo.com/)
* [Uvicorn ASGI Server](https://www.uvicorn.org/)
