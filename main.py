import os
import uuid
import time
import math
from io import BytesIO
from typing import Any, Dict, List, Optional

import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from PIL import Image, ImageDraw, ImageFont

app = FastAPI(title="Sidewalk Annotation API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUT_DIR = os.path.join(os.getcwd(), "static")
os.makedirs(OUTPUT_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=OUTPUT_DIR), name="static")

# Optional: set API_KEY in Render environment variables.
API_KEY = os.getenv("API_KEY", "").strip()


class AnnotationItem(BaseModel):
    problem_id: str = ""
    severity: str = "P3"
    problem_type: str = "问题"
    label: str = ""
    locatable: bool = True
    bbox: Optional[List[float]] = None
    arrow_start: Optional[List[float]] = None
    arrow_end: Optional[List[float]] = None
    legend_text: str = ""


class DrawRequest(BaseModel):
    img: Any = Field(..., description="Image URL, or Coze image object containing url/file_url")
    anno_lst: List[AnnotationItem] = []
    prob_lst: List[Dict[str, Any]] = []
    title: str = "人行道现状问题标注图"


@app.get("/")
def root():
    return {
        "service": "Sidewalk Annotation API",
        "status": "ok",
        "endpoints": ["GET /health", "POST /draw"],
    }


@app.get("/health")
def health():
    return {"draw_st": "ok"}


@app.post("/draw")
async def draw_annotation(payload: DrawRequest, request: Request):
    check_auth(request)

    try:
        image = load_image(payload.img)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"无法读取图片：{e}")

    # Limit very large images for speed and memory.
    image = normalize_image_size(image, max_side=1800)
    base_w, base_h = image.size

    # Add a white legend panel on the right.
    panel_w = max(360, int(base_w * 0.30))
    canvas = Image.new("RGB", (base_w + panel_w, base_h), "white")
    canvas.paste(image, (0, 0))
    draw = ImageDraw.Draw(canvas)

    font_big, font_mid, font_small = load_fonts(base_w)

    legend_items = []
    visible_count = 0

    for idx, item in enumerate(payload.anno_lst):
        if not item.locatable or not item.bbox or len(item.bbox) != 4:
            continue

        bbox = clamp_bbox(item.bbox)
        if bbox is None:
            continue

        severity = normalize_severity(item.severity)
        color = get_color(severity)
        x1 = int(bbox[0] * base_w)
        y1 = int(bbox[1] * base_h)
        x2 = int(bbox[2] * base_w)
        y2 = int(bbox[3] * base_h)

        label = item.label.strip() or f"{severity}-{visible_count + 1:02d}"
        legend_text = item.legend_text.strip() or item.problem_type or "问题"

        line_w = max(4, int(base_w * 0.006))
        # Rectangle
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_w)

        # Label badge; use short label only to avoid Chinese font issues on image.
        badge_text = label
        tx, ty = x1, max(0, y1 - 38)
        tw = min(260, max(130, 14 * len(badge_text) + 24))
        th = 34
        draw.rounded_rectangle([tx, ty, tx + tw, ty + th], radius=8, fill=color)
        draw.text((tx + 8, ty + 7), badge_text, fill="white", font=font_small)

        # Arrow line if provided.
        if item.arrow_start and item.arrow_end and len(item.arrow_start) == 2 and len(item.arrow_end) == 2:
            sx = int(clamp_float(item.arrow_start[0]) * base_w)
            sy = int(clamp_float(item.arrow_start[1]) * base_h)
            ex = int(clamp_float(item.arrow_end[0]) * base_w)
            ey = int(clamp_float(item.arrow_end[1]) * base_h)
            draw_arrow(draw, (sx, sy), (ex, ey), color=color, width=line_w)

        visible_count += 1
        legend_items.append({
            "label": label,
            "severity": severity,
            "problem_type": item.problem_type,
            "legend_text": legend_text,
        })

    draw_legend_panel(
        draw=draw,
        x0=base_w,
        y0=0,
        panel_w=panel_w,
        panel_h=base_h,
        title=payload.title,
        legend_items=legend_items,
        fonts=(font_big, font_mid, font_small),
    )

    filename = f"anno_{int(time.time())}_{uuid.uuid4().hex[:8]}.jpg"
    out_path = os.path.join(OUTPUT_DIR, filename)
    canvas.save(out_path, quality=94)

    base_url = str(request.base_url).rstrip("/")
    return {
        "anno_url": f"{base_url}/static/{filename}",
        "legend": legend_items,
        "draw_st": "success",
        "count": visible_count,
    }


def check_auth(request: Request):
    if not API_KEY:
        return
    provided = request.headers.get("x-api-key", "").strip()
    if provided != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


def load_image(img_input: Any) -> Image.Image:
    url = None
    if isinstance(img_input, dict):
        url = img_input.get("url") or img_input.get("file_url") or img_input.get("image_url")
    elif isinstance(img_input, str):
        url = img_input

    if not url:
        raise ValueError("img中没有可用的图片URL")

    if url.startswith("http://") or url.startswith("https://"):
        headers = {"User-Agent": "sidewalk-annotation-api/1.0"}
        resp = requests.get(url, timeout=30, headers=headers)
        resp.raise_for_status()
        return Image.open(BytesIO(resp.content)).convert("RGB")

    # Local path is mostly for testing.
    if os.path.exists(url):
        return Image.open(url).convert("RGB")

    raise ValueError("图片地址不是http(s) URL，也不是本地文件路径")


def normalize_image_size(image: Image.Image, max_side: int = 1800) -> Image.Image:
    w, h = image.size
    scale = min(1.0, max_side / max(w, h))
    if scale < 1.0:
        return image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return image


def load_fonts(base_w: int):
    # The app uses short ASCII labels on image. Chinese details are returned in legend JSON.
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    path = None
    for p in candidates:
        if os.path.exists(p):
            path = p
            break
    try:
        if path:
            return (
                ImageFont.truetype(path, max(22, int(base_w * 0.026))),
                ImageFont.truetype(path, max(18, int(base_w * 0.020))),
                ImageFont.truetype(path, max(15, int(base_w * 0.017))),
            )
    except Exception:
        pass
    return ImageFont.load_default(), ImageFont.load_default(), ImageFont.load_default()


def normalize_severity(sev: str) -> str:
    sev = (sev or "P3").upper().strip()
    return sev if sev in ["P1", "P2", "P3"] else "P3"


def get_color(severity: str):
    if severity == "P1":
        return (220, 40, 40)  # red
    if severity == "P2":
        return (245, 140, 30)  # orange
    return (40, 120, 220)  # blue


def clamp_float(v: Any) -> float:
    try:
        return max(0.0, min(1.0, float(v)))
    except Exception:
        return 0.0


def clamp_bbox(bbox: List[float]):
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    x1, y1, x2, y2 = [clamp_float(v) for v in bbox]
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def draw_arrow(draw: ImageDraw.ImageDraw, start, end, color, width: int = 4):
    draw.line([start, end], fill=color, width=width)
    sx, sy = start
    ex, ey = end
    angle = math.atan2(ey - sy, ex - sx)
    head_len = max(14, width * 4)
    head_angle = math.pi / 7
    p1 = (ex - head_len * math.cos(angle - head_angle), ey - head_len * math.sin(angle - head_angle))
    p2 = (ex - head_len * math.cos(angle + head_angle), ey - head_len * math.sin(angle + head_angle))
    draw.polygon([end, p1, p2], fill=color)


def draw_legend_panel(draw, x0, y0, panel_w, panel_h, title, legend_items, fonts):
    font_big, font_mid, font_small = fonts
    pad = 24
    draw.rectangle([x0, y0, x0 + panel_w, panel_h], fill=(255, 255, 255))
    draw.line([x0, y0, x0, panel_h], fill=(210, 210, 210), width=2)

    # English title to avoid missing Chinese glyphs in server font.
    draw.text((x0 + pad, y0 + pad), "Problem Legend", fill=(30, 30, 30), font=font_big)
    y = y0 + pad + 48
    draw.text((x0 + pad, y), "P1 red / P2 orange / P3 blue", fill=(90, 90, 90), font=font_small)
    y += 42

    if not legend_items:
        draw.text((x0 + pad, y), "No locatable items", fill=(90, 90, 90), font=font_mid)
        return

    for item in legend_items[:18]:
        sev = item.get("severity", "P3")
        color = get_color(sev)
        label = item.get("label", "")
        problem_type = item.get("problem_type", "")
        # Keep image text short and mostly ASCII-safe.
        text = f"{label}  {sev}"
        draw.rounded_rectangle([x0 + pad, y, x0 + pad + 18, y + 18], radius=4, fill=color)
        draw.text((x0 + pad + 28, y - 2), text, fill=(40, 40, 40), font=font_mid)
        y += 32
        if problem_type:
            # May not render Chinese in every deployment; does not affect API JSON legend.
            draw.text((x0 + pad + 28, y - 4), str(problem_type)[:24], fill=(100, 100, 100), font=font_small)
            y += 28
        y += 8
        if y > panel_h - 60:
            draw.text((x0 + pad, y), "...", fill=(100, 100, 100), font=font_mid)
            break
