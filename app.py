from flask import Flask, render_template, request, jsonify
import base64
import cv2
import numpy as np
from ultralytics import YOLO

app = Flask(__name__)

# Load YOLOv8 LVIS model (1200+ classes)
model = YOLO("yolov8l-lvis.pt")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/detect", methods=["POST"])
def detect():
    data = request.json["image"]

    # Decode base64 image
    img_data = base64.b64decode(data.split(",")[1])
    np_arr = np.frombuffer(img_data, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    results = model(frame, conf=0.4)

    detections = []
    counts = {}

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            label = model.names.get(cls, "Unknown")
            counts[label] = counts.get(label, 0) + 1

            detections.append({
                "label": label,
                "confidence": round(conf, 2),
                "box": [x1, y1, x2, y2],
                "width": x2 - x1,
                "height": y2 - y1
            })

    return jsonify({
        "detections": detections,
        "counts": counts
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
