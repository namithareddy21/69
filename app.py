from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import onnxruntime as ort
import os
import math
from collections import OrderedDict, defaultdict

app = Flask(__name__)

# ---------------------------
# Load YOLOv8 ONNX
# ---------------------------
MODEL_PATH = "yolov8n.onnx"
session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

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
# Centroid Tracker (PER CLASS)
# ---------------------------
class CentroidTracker:
    def __init__(self, max_disappeared=8, max_distance=80):
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
            for oid in list(self.disappeared.keys()):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.deregister(oid)
            return self.objects

        input_centroids = []
        for d in detections:
            x1,y1,x2,y2 = d["box"]
            input_centroids.append(((x1+x2)//2, (y1+y2)//2))

        if len(self.objects) == 0:
            for i,d in enumerate(detections):
                self.register(input_centroids[i], d["label"], d["box"])
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = [self.objects[i][0] for i in objectIDs]

            used = set()

            for i,centroid in enumerate(input_centroids):
                distances = [math.dist(centroid, oc) for oc in objectCentroids]
                idx = np.argmin(distances)

                if distances[idx] < self.max_distance and idx not in used:
                    oid = objectIDs[idx]
                    self.objects[oid] = (centroid, detections[i]["label"], detections[i]["box"])
                    self.disappeared[oid] = 0
                    used.add(idx)
                else:
                    self.register(centroid, detections[i]["label"], detections[i]["box"])

        return self.objects


# ONE TRACKER PER CLASS (IMPORTANT)
trackers = defaultdict(CentroidTracker)

# ---------------------------
# Routes
# ---------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/detect", methods=["POST"])
def detect():
    img = cv2.imdecode(
        np.frombuffer(request.files["image"].read(), np.uint8),
        cv2.IMREAD_COLOR
    )
    h,w = img.shape[:2]

    img_r = cv2.resize(img,(640,640))
    img_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2RGB)
    img_r = img_r.astype(np.float32)/255.0
    img_r = np.transpose(img_r,(2,0,1))[None]

    preds = session.run([output_name], {input_name: img_r})[0]
    preds = np.squeeze(preds).T

    boxes, scores, labels = [], [], []

    for p in preds:
        cls_scores = p[4:]
        cls = np.argmax(cls_scores)
        conf = cls_scores[cls]

        if conf > 0.5:
            cx,cy,bw,bh = p[:4]
            x1 = int((cx-bw/2)*w/640)
            y1 = int((cy-bh/2)*h/640)
            bw = int(bw*w/640)
            bh = int(bh*h/640)

            boxes.append([x1,y1,bw,bh])
            scores.append(float(conf))
            labels.append(cls)

    # 🔥 NMS (THIS FIXES COUNT EXPLOSION)
    idxs = cv2.dnn.NMSBoxes(boxes, scores, 0.5, 0.4)

    detections_by_class = defaultdict(list)

    if len(idxs) > 0:
        for i in idxs.flatten():
            x,y,bw,bh = boxes[i]
            label = CLASSES[labels[i]]
            detections_by_class[label].append({
                "label": label,
                "box": [x,y,x+bw,y+bh]
            })

    results = []
    counts = {}

    for label,dets in detections_by_class.items():
        tracked = trackers[label].update(dets)
        counts[label] = len(tracked)

        for oid,(_,_,box) in tracked.items():
            results.append({
                "id": oid,
                "label": label,
                "box": box
            })

    return jsonify({
        "objects": results,
        "counts": counts
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
