import os
import uuid
import time
import math
import json
import base64
from io import BytesIO
from typing import Any, Dict, List, Optional

import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from PIL import Image, ImageDraw, ImageFont
from openai import OpenAI


app = FastAPI(title="Sidewalk Annotation API", version="3.0.0")

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

# Optional: protect Render API with x-api-key.
API_KEY = os.getenv("API_KEY", "").strip()

# OpenAI-compatible proxy config.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "").strip()

# Recommended model from your proxy platform.
VISION_MODEL = os.getenv("VISION_MODEL", "gemini-3-flash-preview").strip()


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


class GeminiElementRequest(BaseModel):
    img: Any


class GeminiCoordRequest(BaseModel):
    img: Any
    prob_lst: List[Dict[str, Any]] = []


@app.get("/")
def root():
    return {
        "service": "Sidewalk Annotation API",
        "status": "ok",
        "version": "3.0.0",
        "mode": "openai-compatible",
        "model": VISION_MODEL,
        "endpoints": [
            "GET /health",
            "POST /gemini_element",
            "POST /gemini_coord",
            "POST /draw",
        ],
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "draw_st": "ok",
        "mode": "openai-compatible",
        "model": VISION_MODEL,
        "has_openai_key": bool(OPENAI_API_KEY),
        "has_base_url": bool(OPENAI_BASE_URL),
    }


@app.post("/gemini_element")
async def gemini_element(payload: GeminiElementRequest, request: Request):
    check_auth(request)

    try:
        image = load_image(payload.img)
        image = normalize_image_size(image, max_side=1600)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"无法读取图片：{e}")

    prompt = """
你是一名人行道现状元素识别Agent。

你的任务是根据用户上传的人行道照片，识别照片中可见的空间元素，并输出“元素状态记录表”。

你只能做三件事：
1. 识别照片中存在的人行道相关元素；
2. 判断元素状态；
3. 描述元素在图片中的位置。

禁止直接输出问题判断。
禁止提出更新建议。
禁止推测照片中不可见的信息。

一、识别对象

1. 边界与建筑立面：
实体围墙、围栏式围墙、商铺界面、透明橱窗、建筑入口、建筑台阶、无障碍坡道、建筑雨棚、店招系统、外挂机设备、沿街绿化。

2. 退界缓冲区：
店铺外摆、临时摊点、排队等候区、座椅、花箱、快递堆放、外卖停放、骑楼空间。

3. 核心通行区：
铺装、铺装平整度、防滑状况、盲道、盲道连续性、盲道占用、井盖、雨水口、障碍物、横坡、积水、净通行宽度。

4. 设施与绿化带：
行道树、树池、树池盖板、绿化隔离带、灌木绿化、座椅、垃圾桶、路灯、电杆、配电箱、消防栓、公交站、监控杆、标志杆、共享单车停放区、非机动车停车位、充电设施。

5. 路缘阻隔带：
路缘石、人非隔离栏、机非隔离带、路侧停车位、机动车违停、缘石坡道、人行横道衔接、路口转角空间、安全缓冲区。

二、状态判断字段

每个元素输出以下字段：
element_id：元素编号
space_zone：所属空间分区
element_name：元素名称
exist：是 / 否 / 不确定
status：良好 / 一般 / 较差 / 严重 / 不确定
location：左侧 / 右侧 / 中部 / 前景 / 中景 / 远景 / 路口 / 连续分布 / 不确定
occupation_status：是否占用通行空间，是 / 否 / 不确定
barrier_status：是否形成障碍，是 / 否 / 不确定
accessibility_status：是否影响无障碍，是 / 否 / 不确定
safety_status：是否存在安全隐患，是 / 否 / 不确定
visible_confidence：高 / 中 / 低
evidence：简要说明判断依据

如果照片中无法判断，请输出“不确定”，不得猜测。

三、宽度估算

如照片中可见地砖、行人、自行车、汽车、树池等参考物，请估算：
total_width：人行道总宽度
clear_walking_width：核心净通行宽度
width_confidence：高 / 中 / 低
reference_objects：用于估算的参考物

如果无法估算，请输出“不确定”。

四、输出格式

严格输出JSON，不要输出解释、分析过程或Markdown。

{
  "element_table": [
    {
      "element_id": "C01",
      "space_zone": "核心通行区",
      "element_name": "铺装",
      "exist": "是",
      "status": "较差",
      "location": "中部连续分布",
      "occupation_status": "否",
      "barrier_status": "否",
      "accessibility_status": "不确定",
      "safety_status": "是",
      "visible_confidence": "高",
      "evidence": "铺装表面可见破损或不平整"
    }
  ],
  "width_estimate": {
    "total_width": "不确定",
    "clear_walking_width": "不确定",
    "width_confidence": "低",
    "reference_objects": []
  },
  "image_observation_note": "仅记录照片中可见内容，不做问题判断"
}
"""

    try:
        result = call_vision_model(prompt=prompt, image=image)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini元素识别失败：{e}")

    return {
        "gemini_st": "success",
        "elem_raw": result,
    }


@app.post("/gemini_coord")
async def gemini_coord(payload: GeminiCoordRequest, request: Request):
    check_auth(request)

    try:
        image = load_image(payload.img)
        image = normalize_image_size(image, max_side=1600)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"无法读取图片：{e}")

    prompt = f"""
你是一名人行道问题图像定位Agent。

你将接收：
1. 原始人行道照片；
2. 已经由规则引擎生成的问题清单 prob_lst。

你的任务是为每个可定位问题生成图像标注坐标。

你不能新增问题。
你不能删除问题。
你不能修改问题严重度。
你不能重新判断问题。
你只能根据问题清单和原图，为问题寻找可见位置。

坐标要求：
1. bbox 使用归一化坐标，格式为 [x1, y1, x2, y2]；
2. 所有数值必须在 0 到 1 之间；
3. x1 < x2，y1 < y2；
4. bbox 应框选问题对象或问题区域；
5. 如果无法定位，locatable=false，bbox=null。

标注规则：
P1：优先定位，必须尽量给出bbox；
P2：可以定位则给bbox；
P3：如果位置不明确，可以设为不可定位。

问题清单 prob_lst：
{json.dumps(payload.prob_lst, ensure_ascii=False)}

严格输出JSON，不要输出解释、分析过程或Markdown。

{
  "anno_lst": [
    {
      "problem_id": "P-001",
      "severity": "P1",
      "problem_type": "通行问题",
      "label": "P1-01",
      "locatable": true,
      "bbox": [0.10, 0.20, 0.40, 0.50],
      "arrow_start": null,
      "arrow_end": null,
      "legend_text": "核心通行空间被占用"
    }
  ]
}
"""

    try:
        result = call_vision_model(prompt=prompt, image=image)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini坐标定位失败：{e}")

    return {
        "gemini_st": "success",
        "anno_raw": result,
    }


@app.post("/draw")
async def draw_annotation(payload: DrawRequest, request: Request):
    check_auth(request)

    try:
        image = load_image(payload.img)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"无法读取图片：{e}")

    image = normalize_image_size(image, max_side=1800)
    base_w, base_h = image.size

    panel_w = max(360, int(base_w * 0.30))
    canvas = Image.new("RGB", (base_w + panel_w, base_h), "white")
    canvas.paste(image, (0, 0))
    draw = ImageDraw.Draw(canvas)

    font_big, font_mid, font_small = load_fonts(base_w)

    legend_items = []
    visible_count = 0

    for item in payload.anno_lst:
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
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_w)

        badge_text = label
        tx, ty = x1, max(0, y1 - 38)
        tw = min(260, max(130, 14 * len(badge_text) + 24))
        th = 34
        draw.rounded_rectangle([tx, ty, tx + tw, ty + th], radius=8, fill=color)
        draw.text((tx + 8, ty + 7), badge_text, fill="white", font=font_small)

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


def call_vision_model(prompt: str, image: Image.Image) -> str:
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is not set")

    if not OPENAI_BASE_URL:
        raise ValueError("OPENAI_BASE_URL is not set")

    image_b64 = image_to_base64(image)

    client = OpenAI(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
    )

    response = client.chat.completions.create(
        model=VISION_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_b64}"
                        },
                    },
                ],
            }
        ],
        temperature=0.1,
        stream=False,
    )

    content = response.choices[0].message.content

    if not content:
        raise ValueError("模型返回内容为空")

    return content


def image_to_base64(image: Image.Image) -> str:
    buf = BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=92)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def check_auth(request: Request):
    if not API_KEY:
        return

    provided = request.headers.get("x-api-key", "").strip()

    if provided != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


def load_image(img_input: Any) -> Image.Image:
    url = None

    if isinstance(img_input, dict):
        url = (
            img_input.get("url")
            or img_input.get("file_url")
            or img_input.get("image_url")
            or img_input.get("uri")
        )
    elif isinstance(img_input, str):
        url = img_input

    if not url:
        raise ValueError("img中没有可用的图片URL")

    if url.startswith("http://") or url.startswith("https://"):
        headers = {"User-Agent": "sidewalk-annotation-api/3.0"}
        resp = requests.get(url, timeout=30, headers=headers)
        resp.raise_for_status()
        return Image.open(BytesIO(resp.content)).convert("RGB")

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
        return (220, 40, 40)
    if severity == "P2":
        return (245, 140, 30)
    return (40, 120, 220)


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

    p1 = (
        ex - head_len * math.cos(angle - head_angle),
        ey - head_len * math.sin(angle - head_angle),
    )
    p2 = (
        ex - head_len * math.cos(angle + head_angle),
        ey - head_len * math.sin(angle + head_angle),
    )

    draw.polygon([end, p1, p2], fill=color)


def draw_legend_panel(draw, x0, y0, panel_w, panel_h, title, legend_items, fonts):
    font_big, font_mid, font_small = fonts
    pad = 24

    draw.rectangle([x0, y0, x0 + panel_w, panel_h], fill=(255, 255, 255))
    draw.line([x0, y0, x0, panel_h], fill=(210, 210, 210), width=2)

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

        text = f"{label}  {sev}"

        draw.rounded_rectangle(
            [x0 + pad, y, x0 + pad + 18, y + 18],
            radius=4,
            fill=color,
        )
        draw.text((x0 + pad + 28, y - 2), text, fill=(40, 40, 40), font=font_mid)
        y += 32

        if problem_type:
            draw.text(
                (x0 + pad + 28, y - 4),
                str(problem_type)[:24],
                fill=(100, 100, 100),
                font=font_small,
            )
            y += 28

        y += 8

        if y > panel_h - 60:
            draw.text((x0 + pad, y), "...", fill=(100, 100, 100), font=font_mid)
            break
