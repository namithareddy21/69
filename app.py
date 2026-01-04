from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import onnxruntime as ort
import os
from collections import Counter

app = Flask(__name__)

# ---------------------------
# Load YOLOv8 ONNX model
# ---------------------------
MODEL_PATH = "yolov8n.onnx"
session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# COCO class names
CLASSES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella",
    "handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite",
    "baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle",
    "wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich",
    "orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch",
    "potted plant","bed","dining table","toilet","tv","laptop","mouse","remote",
    "keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","book",
    "clock","vase","scissors","teddy bear","hair drier","toothbrush"
]

def nms(boxes, scores, score_thresh=0.25, iou_thresh=0.45):
    # cv2.dnn.NMSBoxes expects boxes as [x,y,w,h]
    if not boxes:
        return []
    idx = cv2.dnn.NMSBoxes(boxes, scores, score_thresh, iou_thresh)
    if idx is None or len(idx) == 0:
        return []
    return idx.flatten().tolist()

def clamp(val, lo, hi):
    return max(lo, min(hi, val))

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/detect", methods=["POST"])
def detect():
    if "image" not in request.files:
        return jsonify({"objects": [], "class_counts": {}})

    file = request.files["image"]
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"objects": [], "class_counts": {}})

    h, w = img.shape[:2]

    # Preprocess
    img_resized = cv2.resize(img, (640, 640))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_norm = img_rgb.astype(np.float32) / 255.0
    img_input = np.transpose(img_norm, (2, 0, 1))[None, :, :, :]

    # Inference
    out = session.run([output_name], {input_name: img_input})[0]
    preds = np.squeeze(out)

    # Robust shape handling:
    # common: (84, 8400) -> transpose to (8400, 84)
    # or:     (8400, 84) already
    if preds.ndim == 2 and preds.shape[0] < preds.shape[1]:
        preds = preds.T

    boxes = []
    scores = []
    class_ids = []

    for pred in preds:
        # YOLOv8 ONNX usually: [cx,cy,w,h, 80 class scores] => len 84
        # Some exports include objectness: [cx,cy,w,h,obj, 80 class scores] => len 85
        if pred.shape[0] == 84:
            cx, cy, bw, bh = pred[:4]
            cls_scores = pred[4:]
        elif pred.shape[0] == 85:
            cx, cy, bw, bh = pred[:4]
            obj = float(pred[4])
            cls_scores = pred[5:] * obj
        else:
            continue

        class_id = int(np.argmax(cls_scores))
        conf = float(cls_scores[class_id])

        if conf < 0.25:
            continue

        # Scale boxes back to original image size
        x = int((cx - bw / 2) * w / 640)
        y = int((cy - bh / 2) * h / 640)
        bw_px = int(bw * w / 640)
        bh_px = int(bh * h / 640)

        # Clamp to image bounds
        x = clamp(x, 0, w - 1)
        y = clamp(y, 0, h - 1)
        bw_px = clamp(bw_px, 1, w - x)
        bh_px = clamp(bh_px, 1, h - y)

        boxes.append([x, y, bw_px, bh_px])
        scores.append(conf)
        class_ids.append(class_id)

    keep = nms(boxes, scores, score_thresh=0.25, iou_thresh=0.45)

    objects = []
    labels = []

    for i in keep:
        x, y, bw_px, bh_px = boxes[i]
        label = CLASSES[class_ids[i]] if 0 <= class_ids[i] < len(CLASSES) else str(class_ids[i])

        objects.append({
            "label": label,
            "confidence": round(scores[i], 2),
            "box": [x, y, x + bw_px, y + bh_px]  # x1,y1,x2,y2
        })
        labels.append(label)

    class_counts = dict(Counter(labels))
    return jsonify({"objects": objects, "class_counts": class_counts})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
