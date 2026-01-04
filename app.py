from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import onnxruntime as ort
import os
import math
from collections import OrderedDict

app = Flask(__name__)

# ---------------------------
# Load YOLOv8 ONNX model
# ---------------------------
MODEL_PATH = "yolov8n.onnx"

session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# COCO classes
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

# ---------------------------
# Centroid Tracker
# ---------------------------
class CentroidTracker:
    def __init__(self, max_disappeared=5, max_distance=50):
        self.nextID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid, label, box):
        self.objects[self.nextID] = (centroid, label, box)
        self.disappeared[self.nextID] = 0
        self.nextID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, detections):
        if len(detections) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.max_disappeared:
                    self.deregister(objectID)
            return self.objects

        input_centroids = []
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            input_centroids.append((cx, cy))

        if len(self.objects) == 0:
            for i, det in enumerate(detections):
                self.register(input_centroids[i], det["label"], det["box"])
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = [self.objects[objID][0] for objID in objectIDs]

            for i, centroid in enumerate(input_centroids):
                distances = [math.dist(centroid, oc) for oc in objectCentroids]
                minDist = min(distances)
                idx = distances.index(minDist)

                if minDist < self.max_distance:
                    objectID = objectIDs[idx]
                    self.objects[objectID] = (
                        centroid,
                        detections[i]["label"],
                        detections[i]["box"]
                    )
                    self.disappeared[objectID] = 0
                else:
                    self.register(centroid, detections[i]["label"], detections[i]["box"])

        return self.objects


tracker = CentroidTracker()

# ---------------------------
# Routes
# ---------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/detect", methods=["POST"])
def detect():
    if "image" not in request.files:
        return jsonify({"objects": [], "counts": {}})

    img = cv2.imdecode(
        np.frombuffer(request.files["image"].read(), np.uint8),
        cv2.IMREAD_COLOR
    )

    h, w = img.shape[:2]

    # Preprocess
    img_resized = cv2.resize(img, (640, 640))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_norm = img_rgb.astype(np.float32) / 255.0
    img_input = np.transpose(img_norm, (2, 0, 1))[None, :, :, :]

    # Inference
    outputs = session.run([output_name], {input_name: img_input})[0]
    preds = np.squeeze(outputs).T

    detections = []

    for pred in preds:
        scores = pred[4:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        if confidence > 0.5:
            cx, cy, bw, bh = pred[:4]

            x1 = int((cx - bw / 2) * w / 640)
            y1 = int((cy - bh / 2) * h / 640)
            x2 = int((cx + bw / 2) * w / 640)
            y2 = int((cy + bh / 2) * h / 640)

            detections.append({
                "label": CLASSES[class_id],
                "box": [x1, y1, x2, y2]
            })

    tracked = tracker.update(detections)

    results = []
    counts = {}

    for objectID, (_, label, box) in tracked.items():
        counts[label] = counts.get(label, 0) + 1
        results.append({
            "id": objectID,
            "label": label,
            "box": box
        })

    return jsonify({
        "objects": results,
        "counts": counts
    })

# ---------------------------
# Run
# ---------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)


