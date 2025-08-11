from flask import Flask, request, jsonify, Response
from logger import log_request, get_metrics_json, PROM

app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    return "<h1>Admin Portal</h1><p>Login <a href='/login'>here</a></p>"

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "GET":
        return '''
            <form method="POST">
                <input name="username" placeholder="Username">
                <input name="password" type="password" placeholder="Password">
                <button type="submit">Login</button>
            </form>
        '''
    else:
        log_request(request)
        return "Invalid credentials!"

@app.route("/api/data", methods=["GET", "POST"])
def api_data():
    log_request(request)
    return {"status": "ok", "data": []}

# Human-/dashboard-friendly JSON metrics
@app.route("/metrics.json", methods=["GET"])
def metrics_json():
    return jsonify(get_metrics_json())

# Prometheus metrics endpoint (if prometheus_client available)
@app.route("/metrics", methods=["GET"])
def metrics_prom():
    if not PROM:
        return Response("prometheus_client not installed\n", status=501, mimetype="text/plain")
    content = PROM["generate_latest"](PROM["REG"])
    return Response(content, mimetype=PROM["CONTENT_TYPE"])

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)
