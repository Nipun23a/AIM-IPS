from fastapi import FastAPI, Request
from main import pipeline

app = FastAPI(title="AIM-IPS Demo API")

@app.post("/test")
async def test_request(request: Request):
    body = await request.body()

    raw_request = {
        "ip": request.client.host,
        "method": request.method,
        "path": request.url.path,
        "headers": dict(request.headers),
        "body": body.decode("utf-8", errors="ignore")
    }

    decision, info = pipeline.process(raw_request)

    return {
        "decision": str(decision),
        "info": info
    }
