from flask import Flask, render_template, request, jsonify
import os
from collections import Counter

import cv2
import numpy as np
import onnxruntime as ort

app = Flask(__name__, template_folder="templates")

MODEL_PATH = "yolov8n.onnx"

CLASSES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
    "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
    "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
    "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake",
    "chair","couch","potted plant","bed","dining table","toilet","tv","laptop",
    "mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink",
    "refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"
]

def load_session():
    if not os.path.exists(MODEL_PATH):
        print("MODEL MISSING:", MODEL_PATH)
        return None, None, None
    sess = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name
    return sess, in_name, out_name

session, input_name, output_name = load_session()

def nms_xywh(boxes_xywh, scores, score_thresh=0.4, iou_thresh=0.5):
    if not boxes_xywh:
        return np.array([], dtype=np.int32)
    idx = cv2.dnn.NMSBoxes(boxes_xywh, scores, score_thresh, iou_thresh)
    if idx is None or len(idx) == 0:
        return np.array([], dtype=np.int32)
    return idx.flatten()

def clamp(v, lo, hi):
    return max(lo, min(int(v), hi))

@app.get("/ping")
def ping():
    return "pong"

@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": session is not None,
        "model_path": MODEL_PATH
    })

@app.get("/")
def index():
    if session is None:
        return "Upload yolov8n.onnx in repo root and redeploy.", 500
    return render_template("index.html")

@app.post("/detect")
def detect():
    if session is None:
        return jsonify(error="Model not loaded"), 500

    if "image" not in request.files:
        return jsonify(count_per_class={}, objects=[], description=[])

    file = request.files["image"]
    img_bytes = file.read()
    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

    if img is None:
        return jsonify(error="Invalid image"), 400

    orig_h, orig_w = img.shape[:2]

    # preprocess -> 640x640 RGB, CHW, float32
    img_resized = cv2.resize(img, (640, 640))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_norm = img_rgb.astype(np.float32) / 255.0
    img_input = np.transpose(img_norm, (2, 0, 1))[None, :, :, :]

    # inference
    out = session.run([output_name], {input_name: img_input})[0]

    # expected shapes:
    # (1, 84, 8400) or (1, 8400, 84) depending on export
    if out.ndim == 3 and out.shape[0] == 1:
        out = out[0]

    # make it (N, 84)
    if out.shape[0] == 84:
        preds = out.T
    else:
        preds = out

    boxes_xywh = []
    scores = []
    class_ids = []

    for pred in preds:
        class_scores = pred[4:]
        cls = int(np.argmax(class_scores))
        conf = float(class_scores[cls])

        if conf <= 0.4:
            continue

        cx, cy, bw, bh = pred[:4]

        # scale from 640 space to original image space
        x1 = (cx - bw / 2.0) * orig_w / 640.0
        y1 = (cy - bh / 2.0) * orig_h / 640.0
        x2 = (cx + bw / 2.0) * orig_w / 640.0
        y2 = (cy + bh / 2.0) * orig_h / 640.0

        x1 = clamp(x1, 0, orig_w - 1)
        y1 = clamp(y1, 0, orig_h - 1)
        x2 = clamp(x2, 0, orig_w - 1)
        y2 = clamp(y2, 0, orig_h - 1)

        if x2 <= x1 or y2 <= y1:
            continue

        w_box = x2 - x1
        h_box = y2 - y1

        boxes_xywh.append([x1, y1, w_box, h_box])  # for cv2 NMSBoxes
        scores.append(conf)
        class_ids.append(cls)

    keep = nms_xywh(boxes_xywh, scores, 0.4, 0.5)

    objects = []
    labels = []
    for i in keep:
        x, y, w_box, h_box = boxes_xywh[int(i)]
        label = CLASSES[class_ids[int(i)]]

        obj = {
            "label": label,
            "confidence": round(float(scores[int(i)]), 2),
            "box": [int(x), int(y), int(x + w_box), int(y + h_box)],
            "width_px": int(w_box),
            "height_px": int(h_box)
        }
        objects.append(obj)
        labels.append(label)

    count_per_class = dict(Counter([o["label"] for o in objects]))

    return jsonify(
        count_per_class=count_per_class,
        objects=objects,
        description=list(set(labels))
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, threaded=True)
