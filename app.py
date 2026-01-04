from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import onnxruntime as ort
from collections import Counter
import os

app = Flask(__name__)

# Update with your model path
MODEL_PATH = 'yolov8n.onnx'  # or your model file
session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

def nms(boxes, scores, score_thresh=0.4, iou_thresh=0.5):
    indices = cv2.dnn.NMSBoxes(boxes, scores, score_thresh, iou_thresh)
    return indices

@app.route('/')
def index():
    return render_template('index.html')  # Assumes index.html in templates/

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify(class_counts={}, objects=[], description=[])
    
    file = request.files['image']
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    
    # Preprocess
    img_resized = cv2.resize(img, (640, 640))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_norm = img_rgb.astype(np.float32) / 255.0
    img_input = np.transpose(img_norm, (2, 0, 1))[None, ...]
    
    # Inference
    outputs = session.run([output_name], {input_name: img_input})[0]
    preds = np.squeeze(outputs).T
    
    boxes, confidences, classids = [], [], []
    for pred in preds:
        class_scores = pred[4:]
        classid = np.argmax(class_scores)
        confidence = class_scores[classid]
        if confidence > 0.4:
            cx, cy, bw, bh = pred[:4]
            x = int((cx - bw / 2) * w / 640)
            y = int((cy - bh / 2) * h / 640)
            bw_px = int(bw * w / 640)
            bh_px = int(bh * h / 640)
            boxes.append([x, y, bw_px, bh_px])
            confidences.append(float(confidence))
            classids.append(classid)
    
    indices = nms(boxes, confidences)
    results, labels = [], []
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, bw_px, bh_px = boxes[i]
            label = CLASSES[classids[i]]
            results.append({
                'label': label,
                'confidence': round(confidences[i], 2),
                'box': [x, y, x + bw_px, y + bh_px],
                'width_px': bw_px,
                'height_px': bh_px
            })
            labels.append(label)
    
    class_counts = dict(Counter(labels))
    unique_labels = list(set(labels))
    
    return jsonify(
        class_counts=class_counts,
        objects=results,
        description=unique_labels
    )

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, threaded=True)
