from flask import Flask, request, jsonify, Response
import requests
import time
import psutil  # for monitoring sistem
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
 
app = Flask(__name__)
 
# metric for API model
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP Requests')  # Total accepted requests 
REQUEST_LATENCY = Histogram('http_request_duration_seconds', 'HTTP Request Latency')  # response time of API
THROUGHPUT = Counter('http_requests_throughput', 'Total number of requests per second')  # throughput
 
# metric for sistem
CPU_USAGE = Gauge('system_cpu_usage', 'CPU Usage Percentage')  # CPU usage
RAM_USAGE = Gauge('system_ram_usage', 'RAM Usage Percentage')  # RAM usage
 
# endpoint for Prometheus
@app.route('/metrics', methods=['GET'])
def metrics():
    # update system metric everytime /metrics is accessed
    CPU_USAGE.set(psutil.cpu_percent(interval=1))  # take CPU data usage (percentage)
    RAM_USAGE.set(psutil.virtual_memory().percent)  # take RAM data usage (percentage)
    
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)
 
# endpoint for access API model and log request
@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()
    REQUEST_COUNT.inc()  # add total request
    THROUGHPUT.inc()  # add throughput (request per second)
 
    # send request to API model
    api_url = "http://127.0.0.1:5005/invocations"
    data = request.get_json()
 
    try:
        response = requests.post(api_url, json=data)
        duration = time.time() - start_time
        REQUEST_LATENCY.observe(duration)  # log response time (latency)
        
        return jsonify(response.json())
 
    except Exception as e:
        return jsonify({"error": str(e)}), 500
 
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000)