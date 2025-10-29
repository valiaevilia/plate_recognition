import os
import json
from typing import Any, Dict

import httpx
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # можно сузить до своих IP/доменов
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPSTREAM_URL = os.environ.get(
    "UPSTREAM_URL",
    "https://integrate.api.nvidia.com/v1/chat/completions"
)

NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "").strip()
if not NVIDIA_API_KEY:
    print("[WARN] NVIDIA_API_KEY is empty. Proxy will 401.")
PROXY_CLIENT_TOKEN = os.environ.get("PROXY_CLIENT_TOKEN", "").strip()

@app.get("/healthz")
async def healthz():
    return {"ok": True}

@app.post("/v1/chat/completions")
async def relay_chat(req: Request):
    # опциональная лёгкая защита:
    # если задан PROXY_CLIENT_TOKEN, то клиент должен передать
    #   Authorization: Bearer <тот_токен>
    if PROXY_CLIENT_TOKEN:
        auth = req.headers.get("authorization", "")
        if not auth.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Missing bearer token")
        given = auth.split(" ", 1)[1].strip()
        if given != PROXY_CLIENT_TOKEN:
            raise HTTPException(status_code=403, detail="Bad token")

    if not NVIDIA_API_KEY:
        raise HTTPException(status_code=401, detail="Server missing NVIDIA_API_KEY")

    try:
        body_bytes = await req.body()
        try:
            body_json: Dict[str, Any] = json.loads(body_bytes.decode("utf-8"))
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid JSON body")

        # минимальная валидация
        if "messages" not in body_json:
            raise HTTPException(status_code=400, detail="messages required")

        headers = {
            "Authorization": f"Bearer {NVIDIA_API_KEY}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

        # пробрасываем запрос к NVIDIA
        async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, read=30.0, write=30.0, connect=10.0)) as client:
            upstream_resp = await client.post(
                UPSTREAM_URL,
                headers=headers,
                json=body_json,
            )

        # если NVIDIA вернула ошибку — пробрасываем код и текст
        if upstream_resp.status_code >= 400:
            return Response(
                content=upstream_resp.text,
                status_code=upstream_resp.status_code,
                media_type="application/json",
            )

        # всё ок — вернули как есть
        return Response(
            content=upstream_resp.text,
            status_code=200,
            media_type="application/json",
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"proxy error: {e}")
