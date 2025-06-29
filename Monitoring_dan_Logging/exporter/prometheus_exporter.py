from flask import Flask, request, jsonify, Response
import requests
import time
import psutil
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

app = Flask(__name__)

# ---- Metrics Definitions ----
# Request count and latency (for /predict)
REQUEST_COUNT = Counter(
    'http_requests_total',
    'Total HTTP Requests',
    ['method', 'endpoint']
)
REQUEST_LATENCY = Histogram(
    'http_request_duration_seconds',
    'HTTP Request Latency',
    ['endpoint']
)
THROUGHPUT = Counter(
    'http_requests_throughput_total',
    'Total number of requests (for throughput estimation)',
    ['endpoint']
)

# Inference-specific latency (to separate model latency from network)
INFERENCE_LATENCY = Histogram(
    'model_inference_duration_seconds',
    'ML model inference latency in seconds'
)

# System metrics
CPU_USAGE = Gauge('system_cpu_usage', 'CPU Usage Percentage')
RAM_USAGE = Gauge('system_ram_usage', 'RAM Usage Percentage')

# ---- Routes ----
@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()
    endpoint = request.path

    # Count request
    REQUEST_COUNT.labels(method=request.method, endpoint=endpoint).inc()
    THROUGHPUT.labels(endpoint=endpoint).inc()

    # Get input data
    data = request.get_json()

    # Send to model service
    api_url = "http://classifier:5000/predict"
    try:
        infer_start = time.time()
        response = requests.post(api_url, json=data)
        infer_duration = time.time() - infer_start
        INFERENCE_LATENCY.observe(infer_duration)

        # Log latency
        duration = time.time() - start_time
        REQUEST_LATENCY.labels(endpoint=endpoint).observe(duration)

        return jsonify(response.json()), response.status_code

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/metrics', methods=['GET'])
def metrics():
    # Update system metrics each time /metrics is scraped
    CPU_USAGE.set(psutil.cpu_percent(interval=1))
    RAM_USAGE.set(psutil.virtual_memory().percent)
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)