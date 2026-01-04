from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import onnxruntime as ort
import os
from collections import Counter

# Make template folder explicit (important on some deploys)
app = Flask(__name__, template_folder="templates")

MODEL_PATH = "yolov8n.onnx"

# COCO-80 classes
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
        print(f"❌ Model not found: {MODEL_PATH} (must be in repo root)")
        return None, None, None
    try:
        sess = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
        in_name = sess.get_inputs()[0].name
        out_name = sess.get_outputs()[0].name
        print("✅ ONNX model loaded")
        return sess, in_name, out_name
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None, None, None

session, input_name, output_name = load_session()

def nms_xywh(boxes_xywh, scores, score_thresh=0.4, iou_thresh=0.5):
    if len(boxes_xywh) == 0:
        return np.array([], dtype=np.int32)
    idx = cv2.dnn.NMSBoxes(boxes_xywh, scores, score_thresh, iou_thresh)
    if idx is None or len(idx) == 0:
        return np.array([], dtype=np.int32)
    return idx.flatten()

def clamp_box(x1, y1, x2, y2, w, h):
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(0, min(int(x2), w - 1))
    y2 = max(0, min(int(y2), h - 1))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2

@app.route("/ping")
def ping():
    return "pong"

@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": session is not None,
        "model_path": MODEL_PATH
    })

@app.route("/")
def index():
    # If you see "Not Found" in browser, it usually means the wrong start command/app is running,
    # but if the app is running and model is missing, show a clear message here.
    if session is None:
        return "Model not loaded. Put yolov8n.onnx in repo root and redeploy.", 500
    return render_template("index.html")

@app.route("/detect", methods=["POST"])
def detect():
    if session is None:
        return jsonify(error="Model not loaded"), 500

    if "image" not in request.files:
        return jsonify(count_per_class={}, objects=[], description=[])

    file = request.files["image"]
    data = file.read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)

    if img is None:
        return jsonify(error="Invalid image"), 400

    orig_h, orig_w = img.shape[:2]

    # Preprocess
    img_resized = cv2.resize(img, (640, 640))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_norm = img_rgb.astype(np.float32) / 255.0
    img_input = np.transpose(img_norm, (2, 0, 1))[None, ...]

    # Inference
    outputs = session.run([output_name], {input_name: img_input})
    out = outputs[0]

    # Handle common YOLOv8 ONNX output shapes safely
    # Typical: (1, 84, 8400) -> want (8400, 84)
    if out.ndim == 3 and out.shape[0] == 1:
        out = out[0]  # (84, 8400)
    if out.shape[0] < out.shape[1]:
        preds = out.T  # (8400, 84)
    else:
        preds = out  # already (N, 84)

    boxes_xywh, confidences, classids = [], [], []

    for pred in preds:
        # pred: [cx, cy, w, h, ...80 class scores...]
        class_scores = pred[4:]
        classid = int(np.argmax(class_scores))
        conf = float(class_scores[classid])

        if conf > 0.4:
            cx, cy, bw, bh = pred[:4]

            # Scale back to original image size (because model input is 640x640)
            x1 = (cx - bw / 2) * orig_w / 640
            y1 = (cy - bh / 2) * orig_h / 640
            x2 = (cx + bw / 2) * orig_w / 640
            y2 = (cy + bh / 2) * orig_h / 640

            x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, orig_w, orig_h)
            w_box = max(1, x2 - x1)
            h_box = max(1, y2 - y1)

            boxes_xywh.append([x1, y1, w_box, h_box])  # for NMS
            confidences.append(conf)
            classids.append(classid)

    idxs = nms_xywh(boxes_xywh, confidences, score_thresh=0.4, iou_thresh=0.5)

    results = []
    labels = []
    for i in idxs:
        x, y, w_box, h_box = boxes_xywh[i]
        label = CLASSES[classids[i]]
        results.append({
            "label": label,
            "confidence": round(float(confidences[i]), 2),
            "box": [int(x), int(y), int(x + w_box), int(y + h_box)],
            "width_px": int(w_box),
            "height_px": int(h_box)
        }).
        labels.append(label)

    count_per_class = dict(Counter([r["label"] for r in results]))

    return jsonify(
        count_per_class=count_per_class,
        objects=results,
        description=list(set(labels))
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, threaded=True)
