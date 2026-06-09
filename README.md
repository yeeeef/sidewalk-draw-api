# Sidewalk Annotation API

A small FastAPI service for drawing annotation boxes on sidewalk photos.

## Endpoints

- `GET /health` checks whether the service is running.
- `POST /draw` draws bounding boxes and labels on the image.

## Request example

```json
{
  "img": "https://example.com/photo.jpg",
  "anno_lst": [
    {
      "problem_id": "P-001",
      "severity": "P1",
      "problem_type": "通行问题",
      "label": "P1-01",
      "locatable": true,
      "bbox": [0.25, 0.35, 0.58, 0.70],
      "arrow_start": [0.15, 0.25],
      "arrow_end": [0.40, 0.55],
      "legend_text": "核心通行空间被占用"
    }
  ],
  "prob_lst": []
}
```

## Response example

```json
{
  "anno_url": "https://your-service.onrender.com/static/anno_xxx.jpg",
  "legend": [],
  "draw_st": "success",
  "count": 1
}
```
