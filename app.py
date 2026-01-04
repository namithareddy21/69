from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import onnxruntime as ort
from collections import Counter
import os

app = Flask(__name__)

MODEL_PATH = 'yolov8n.onnx'
session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
           'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
           'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
           'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
           'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
           'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
           'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
           'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
           'hair drier', 'toothbrush']

def nms(boxes, scores, score_thresh=0.25, iou_thresh=0.45):
    if len(boxes) == 0:
        return []
    indices = cv2.dnn.NMSBoxes(boxes, scores, score_thresh, iou_thresh)
    return indices.flatten().tolist() if indices is not None else []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify(class_counts={}, objects=[], description=[])
    
    file = request.files['image']
    nparr = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    orig_h, orig_w = img.shape[:2]
    
    # Letterbox resize + pad
    r = min(640 / orig_w, 640 / orig_h)
    new_w, new_h = int(orig_w * r), int(orig_h * r)
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((640, 640, 3), 114, dtype=np.uint8)
    canvas[(640 - new_h) // 2:(640 - new_h) // 2 + new_h, 
           (640 - new_w) // 2:(640 - new_w) // 2 + new_w] = img_resized
    
    img_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    img_norm = img_rgb.astype(np.float32) / 255.0
    img_input = np.transpose(img_norm, (2, 0, 1))[np.newaxis, ...]
    
    # Inference
    outputs = session.run([output_name], {input_name: img_input})
    predictions = np.squeeze(outputs[0]).T  # (8400, 84)
    
    boxes, confidences, class_ids = [], [], []
    for i in range(len(predictions)):
        pred = predictions[i]
        scores = pred[4:84]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        
        if confidence > 0.25:
            cx, cy, w, h = pred[0:4]
            # Scale back with padding
            x1 = ((cx - w / 2) * 640) / orig_w
            y1 = ((cy - h / 2) * 640) / orig_h
            x2 = ((cx + w / 2) * 640) / orig_w
            y2 = ((cy + h / 2) * 640) / orig_h
            boxes.append([float(x1), float(y1), float(x2), float(y2)])
            confidences.append(float(confidence))
            class_ids.append(int(class_id))
    
    indices = nms(boxes, confidences)
    
    results, labels = [], []
    for i in indices:
        x1, y1, x2, y2 = boxes[i]
        w, h = x2 - x1, y2 - y1
        label = CLASSES[class_ids[i]]
        conf = confidences[i]
        results.append({
            'label': label,
            'confidence': round(conf, 2),
            'box': [int(x1), int(y1), int(x2), int(y2)],
            'width_px': int(w),
            'height_px': int(h),
            'trackId': i  # For frontend tracking
        })
        labels.append(label)
    
    class_counts = dict(Counter(labels))
    
    return jsonify(
        class_counts=class_counts,
        objects=results,
        description=list(set(labels))
    )

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, threaded=True, debug=False)
