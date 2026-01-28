from flask import Flask, request, jsonify, Response
from logger import log_request, get_metrics_json, PROM

app = Flask(__name__)

# Don't include /login here because you define a real /login below
COMMON_BAIT = [
    "/wp-login.php", "/phpmyadmin", "/server-status", "/admin",
    "/.git/config", "/.env", "/backup.zip", "/db.sql", "/api",
    "/vendor/phpunit/phpunit/src/Util/PHP/eval-stdin.php"
]

@app.after_request
def spoof_headers(response):
    response.headers["Server"] = "Apache/2.4.41 (Ubuntu)"
    return response

@app.route("/robots.txt")
def robots():
    data = (
        "User-agent: *\n"
        "Disallow: /admin\n"
        "Disallow: /phpmyadmin\n"
        "Disallow: /.git\n"
        "Disallow: /.env\n"
        "Disallow: /backup.zip\n"
    )
    return Response(data, mimetype="text/plain")

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

# bait routes -> 200 OK + log (register AFTER real routes)
for path in COMMON_BAIT:
    def _mk_handler(p=path):
        def handler():
            log_request(request)
            return "OK", 200
        handler.__name__ = ("h_" + p.strip("/").replace("/", "_").replace(".", "_")) or "root"
        return handler
    app.add_url_rule(path, view_func=_mk_handler(), methods=["GET", "POST"])

# catch-all 404 logger
@app.errorhandler(404)
def all_404(e):
    log_request(request)
    return "Not found", 404

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
