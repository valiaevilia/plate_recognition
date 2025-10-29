#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, io, sys, json, base64, argparse, csv, time
from typing import Dict, Any, Optional, Tuple
import requests
import cv2
from PIL import Image

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

PROXY_URL = os.getenv("PROXY_URL")
PROXY_CLIENT_TOKEN = os.getenv("PROXY_CLIENT_TOKEN", "")
INVOKE_URL = os.getenv("NVIDIA_INVOKE_URL", "https://integrate.api.nvidia.com/v1/chat/completions")
API_KEY = os.getenv("NVIDIA_API_KEY") or os.getenv("NGC_API_KEY")
DEFAULT_MODEL_NAME = os.getenv("NIM_MODEL", "meta/llama-3.2-90b-vision-instruct")

IMG_MAX_DIM = int(os.getenv("IMG_MAX_DIM", "1280"))
IMG_JPEG_QUALITY = int(os.getenv("IMG_JPEG_QUALITY", "90"))
HTTP_CONNECT_TIMEOUT = float(os.getenv("NVIDIA_CONNECT_TIMEOUT", "10"))
HTTP_READ_TIMEOUT = float(os.getenv("NVIDIA_READ_TIMEOUT", "25"))
ROI_MARGIN = float(os.getenv("ROI_MARGIN", "0.20"))
MAX_RETRIES = int(os.getenv("NVIDIA_RETRIES", "3"))
BACKOFF_BASE = float(os.getenv("NVIDIA_BACKOFF_BASE", "0.7"))

DEFAULT_SAMPLE_FPS = float(os.getenv("SAMPLE_FPS", "5"))
DEFAULT_DEDUPE_WINDOW_S = float(os.getenv("DEDUPE_WINDOW", "2.0"))

CYR_TO_LAT = str.maketrans({
    "А":"A","В":"B","Е":"E","К":"K","М":"M","Н":"H","О":"O","Р":"P","С":"C","Т":"T","У":"Y","Х":"X",
    "а":"A","в":"B","е":"E","к":"K","м":"M","н":"H","о":"O","р":"P","с":"C","т":"T","у":"Y","х":"X",
})
RU_LAT_SET = "ABEKMHOPCTYX"
PLATE_REGEX_FULL = re.compile(rf"^[{RU_LAT_SET}]\d{{3}}[{RU_LAT_SET}]{{2}}\d{{2,3}}$")
PLATE_REGEX_PART = re.compile(rf"^[{RU_LAT_SET}#][0-9#]{{3}}[{RU_LAT_SET}#]{{2}}[0-9#]{{2,3}}$")
FIND_REGEX_PART = re.compile(r"[A-Za-zА-Яа-я#][0-9#]{3}[A-Za-zА-Яа-я#]{2}[0-9#]{2,3}")

def normalize_plate(t: str) -> str:
    t = (t or "").strip().upper().translate(CYR_TO_LAT)
    return re.sub(r"[^A-Z0-9#]", "", t)

def mask_non_ru_letters(token: str) -> str:
    if not token: return token
    s = list(token)
    for idx in (0,4,5):
        if idx < len(s) and s[idx].isalpha() and s[idx] not in RU_LAT_SET and s[idx] != "#":
            s[idx] = "#"
    return "".join(s)

def is_valid_full(t: str) -> bool: return bool(PLATE_REGEX_FULL.match(t))
def is_valid_partial(t: str) -> bool: return bool(PLATE_REGEX_PART.match(t)) and any(ch != "#" for ch in t)

def format_mmss_msec(msec: float) -> str:
    total = int(round(msec)); mm = total//60000; ss = (total%60000)//1000; ms = total%1000
    return f"{mm:02d}:{ss:02d}.{ms:03d}"

def bgr_to_jpeg(frame_bgr, max_dim: int, quality: int) -> bytes:
    h, w = frame_bgr.shape[:2]
    scale = min(1.0, max_dim / float(max(h, w)))
    if scale < 1.0:
        frame_bgr = cv2.resize(frame_bgr, (max(1,int(w*scale)), max(1,int(h*scale))), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    buf = io.BytesIO(); Image.fromarray(rgb).save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()

def jpeg_to_data_url(jpg: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(jpg).decode("ascii")

_TRANSIENT = {408, 429, 500, 502, 503, 504}

def call_api(payload: Dict[str, Any]) -> Optional[str]:
    target_url = PROXY_URL or INVOKE_URL
    headers = {"Accept":"application/json","Content-Type":"application/json"}
    if PROXY_URL:
        if PROXY_CLIENT_TOKEN:
            headers["Authorization"] = f"Bearer {PROXY_CLIENT_TOKEN}"
    else:
        if not (API_KEY or "").strip():
            return None
        headers["Authorization"] = f"Bearer {API_KEY}"
    timeouts = (HTTP_CONNECT_TIMEOUT, HTTP_READ_TIMEOUT)
    for attempt in range(1, MAX_RETRIES+1):
        try:
            with requests.post(target_url, headers=headers, json=payload, timeout=timeouts) as resp:
                if resp.status_code in _TRANSIENT:
                    raise requests.HTTPError(f"Transient {resp.status_code}: {resp.text[:200]}")
                if resp.status_code >= 400:
                    return None
                obj = resp.json()
                ch = (obj.get("choices") or [{}])[0]
                msg = ch.get("message", {})
                content = msg.get("content", "")
                if isinstance(content, list):
                    return " ".join([c.get("text","") for c in content if c.get("type")=="text"]).strip()
                return str(content)
        except (requests.exceptions.ReadTimeout,
                requests.exceptions.ConnectTimeout,
                requests.exceptions.ConnectionError,
                requests.exceptions.ChunkedEncodingError,
                requests.HTTPError):
            if attempt == MAX_RETRIES:
                return None
            time.sleep(BACKOFF_BASE*(2**(attempt-1)))
        except Exception:
            return None
    return None

def ask_bbox(frame_bgr) -> Optional[Tuple[float,float,float,float]]:
    jpg = bgr_to_jpeg(frame_bgr, IMG_MAX_DIM, IMG_JPEG_QUALITY)
    prompt = (
        "You are a precise detector. Decide if a vehicle with a visible RUSSIAN license plate exists.\n"
        "Return ONLY JSON:\n"
        "{ \"has_vehicle\": true|false, \"bbox\": [x0,y0,x1,y1] | null }\n"
        "bbox is normalized [0,1] tightly around the PLATE. If no plate -> has_vehicle=false, bbox=null."
    )
    payload = {
        "model": DEFAULT_MODEL_NAME,
        "temperature": 0.0,
        "max_tokens": 128,
        "messages": [{
            "role":"user","content":[
                {"type":"text","text":prompt},
                {"type":"image_url","image_url":{"url": jpeg_to_data_url(jpg)}}
            ]}]
    }
    ans = call_api(payload)
    if not ans: return None
    m = re.search(r"\{.*\}", ans, flags=re.DOTALL)
    if not m: return None
    try:
        obj = json.loads(m.group(0))
        if not bool(obj.get("has_vehicle", False)): return None
        bbox = obj.get("bbox")
        if isinstance(bbox, list) and len(bbox)==4:
            x0,y0,x1,y1 = [float(max(0.0,min(1.0,float(v)))) for v in bbox]
            if x1>x0 and y1>y0: return (x0,y0,x1,y1)
    except Exception:
        return None
    return None

def ask_ocr(frame_bgr) -> Optional[str]:
    jpg = bgr_to_jpeg(frame_bgr, IMG_MAX_DIM, IMG_JPEG_QUALITY)
    prompt = (
        "You are a STRICT OCR for RUSSIAN license plates.\n"
        "Return JSON ONLY:\n"
        "{ \"plate\": { \"text\": \"<8 or 9 chars, LDDDLLDD[D], use # for unreadable>\" } }\n"
        "LATIN uppercase RU subset only: A,B,E,K,M,H,O,P,C,T,Y,X. Always 8 or 9 chars with # if unsure."
    )
    payload = {
        "model": DEFAULT_MODEL_NAME,
        "temperature": 0.0,
        "max_tokens": 128,
        "messages": [{
            "role":"user","content":[
                {"type":"text","text":prompt},
                {"type":"image_url","image_url":{"url": jpeg_to_data_url(jpg)}}
            ]}]
    }
    ans = call_api(payload)
    if not ans: return None
    token = None
    m = re.search(r"\{.*\}", ans, flags=re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            plate = obj.get("plate", {})
            if isinstance(plate, dict):
                token = plate.get("text")
        except Exception:
            token = None
    if not token:
        m2 = FIND_REGEX_PART.search(ans)
        token = m2.group(0) if m2 else None
    if not token:
        return None
    token = mask_non_ru_letters(normalize_plate(token))
    if not (is_valid_full(token) or is_valid_partial(token)):
        return None
    return token

def enhance_roi_for_ocr(roi_bgr):
    g = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    g = clahe.apply(g)
    blur = cv2.GaussianBlur(g, (0,0), 1.0)
    sharp = cv2.addWeighted(g, 1.5, blur, -0.5, 0)
    return cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR)

def ask_ocr_multi(roi_bgr):
    t = ask_ocr(roi_bgr)
    if t: return t
    t = ask_ocr(enhance_roi_for_ocr(roi_bgr))
    if t: return t
    os.environ["IMG_JPEG_QUALITY"] = "95"
    return ask_ocr(roi_bgr)

def crop_by_norm_bbox(frame_bgr, bbox, margin: float):
    h, w = frame_bgr.shape[:2]
    x0,y0,x1,y1 = bbox
    cx0 = max(0, int((x0 - margin) * w))
    cy0 = max(0, int((y0 - margin) * h))
    cx1 = min(w, int((x1 + margin) * w))
    cy1 = min(h, int((y1 + margin) * h))
    if cx1<=cx0 or cy1<=cy0: return None
    return frame_bgr[cy0:cy1, cx0:cx1].copy()

def process_video(video_path: str, out_csv_path: str,
                  sample_fps: float, dedupe_window_s: float,
                  start_sec: float, max_frames: int, no_full_frame_ocr: bool) -> None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): raise RuntimeError(f"Cannot open video: {video_path}")
    if start_sec>0: cap.set(cv2.CAP_PROP_POS_MSEC, int(start_sec*1000))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_step = max(1, int(round(fps / sample_fps))) if sample_fps>0 else 1
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    os.makedirs(os.path.dirname(out_csv_path) or ".", exist_ok=True)
    with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f); wr.writerow(["time","plate_num"])
        last_emit: Dict[str,int] = {}
        idx = -1; processed = 0
        print(f"Opened {video_path}; FPS≈{fps:.2f}, step={frame_step}, total={total_frames}")
        print(f"Using {'PROXY' if PROXY_URL else 'DIRECT'} endpoint.")
        while True:
            ok, frame = cap.read()
            if not ok: break
            idx += 1
            if idx % frame_step != 0: continue
            pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            if pos_ms <= 0: pos_ms = (idx / fps) * 1000.0
            ts_dbg = format_mmss_msec(pos_ms)
            print(f"→ frame {idx} @ {ts_dbg}", flush=True)
            bbox = ask_bbox(frame)
            plate_text = None
            if bbox:
                print(f"  bbox={bbox}", flush=True)
                roi = crop_by_norm_bbox(frame, bbox, ROI_MARGIN)
                plate_text = ask_ocr_multi(roi if roi is not None else frame)
            else:
                if not no_full_frame_ocr:
                    print("  no bbox, OCR full frame", flush=True)
                    plate_text = ask_ocr_multi(frame)
                else:
                    print("  no bbox -> skip", flush=True)
            processed += 1
            if max_frames and processed>=max_frames: break
            if not plate_text:
                print("  no plate", flush=True); continue
            plate = mask_non_ru_letters(normalize_plate(plate_text))
            if not (is_valid_full(plate) or is_valid_partial(plate)):
                print(f"  normalized invalid: {plate}", flush=True); continue
            now_ms = int(round(pos_ms))
            last = last_emit.get(plate, -10**9)
            if (now_ms - last) < int(dedupe_window_s*1000):
                print("  deduped", flush=True); continue
            last_emit[plate] = now_ms
            wr.writerow([format_mmss_msec(now_ms), plate]); f.flush()
            print(f"[{format_mmss_msec(now_ms)}] {plate}", flush=True)
    cap.release()
    print(f"Done. CSV saved to: {out_csv_path}", flush=True)

def main():
    ap = argparse.ArgumentParser(description="RU plate OCR via NVIDIA Integrate (proxy-friendly)")
    ap.add_argument("--video", type=str, required=True)
    ap.add_argument("--out", type=str, default="plates.csv")
    ap.add_argument("--sample-fps", type=float, default=DEFAULT_SAMPLE_FPS)
    ap.add_argument("--dedupe-window", type=float, default=DEFAULT_DEDUPE_WINDOW_S)
    ap.add_argument("--start-sec", type=float, default=0.0)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--no-full-frame-ocr", action="store_true")
    args = ap.parse_args()
    out_csv = args.out
    if out_csv.endswith(os.sep) or os.path.isdir(out_csv):
        base = os.path.splitext(os.path.basename(args.video))[0] or "output"
        out_csv = os.path.join(out_csv, f"{base}_plates.csv")
    process_video(args.video, out_csv, args.sample_fps, args.dedupe_window,
                  args.start_sec, args.max_frames, args.no_full_frame_ocr)

if __name__ == "__main__":
    main()
