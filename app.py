from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import onnxruntime as ort
import os
from collections import Counter

app = Flask(__name__)

MODEL_PATH = 'yolov8n.onnx'

def load_session():
    if not os.path.exists(MODEL_PATH):
        print("Model file yolov8n.onnx not found")
        return None
    return ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])

session = load_session()
if session is not None:
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
else:
    input_name = None
    output_name = None

CLASSES = [
    'person','bicycle','car','motorcycle','airplane','bus','train','truck','boat',
    'traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat',
    'dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack',
    'umbrella','handbag','tie','suitcase','frisbee','skis','snowboard','sports ball',
    'kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket',
    'bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple',
    'sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake',
    'chair','couch','potted plant','bed','dining table','toilet','tv','laptop',
    'mouse','remote','keyboard','cell phone','microwave','oven','toaster','sink',
    'refrigerator','book','clock','vase','scissors','teddy bear','hair drier','toothbrush'
]

def nms(boxes, scores, score_thresh=0.4, iou_thresh=0.5):
    if len(boxes) == 0:
        return []
    return cv2.dnn.NMSBoxes(boxes, scores, score_thresh, iou_thresh)

@app.route('/')
def index():
    if session is None:
        return "Upload yolov8n.onnx to server root", 500
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    if session is None:
        return jsonify(error="Model not loaded"), 500

    if 'image' not in request.files:
        return jsonify(count_per_class={}, objects=[], description=[])

    file = request.files['image']
    nparr = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    orig_h, orig_w = img.shape[:2]

    img_resized = cv2.resize(img, (640, 640))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_norm = img_rgb.astype(np.float32) / 255.0
    img_input = np.transpose(img_norm, (2, 0, 1))[None, ...]

    outputs = session.run([output_name], {input_name: img_input})
    preds = np.squeeze(outputs[0].T)

    boxes, confidences, classids = [], [], []
    for pred in preds:
        class_scores = pred[4:]
        classid = np.argmax(class_scores)
        confidence = class_scores[classid]
        if confidence > 0.4:
            cx, cy, bw, bh = pred[:4]
            x1 = int((cx - bw / 2) * orig_w / 640)
            y1 = int((cy - bh / 2) * orig_h / 640)
            x2 = int((cx + bw / 2) * orig_w / 640)
            y2 = int((cy + bh / 2) * orig_h / 640)
            boxes.append([x1, y1, x2 - x1, y2 - y1])
            confidences.append(float(confidence))
            classids.append(classid)

    indices = nms(boxes, confidences)
    results, labels = [], []
    if len(indices) > 0:
        for i in indices.flatten():
            x1, y1, w, h = boxes[i]
            label = CLASSES[classids[i]]
            results.append({
                'label': label,
                'confidence': round(confidences[i], 2),
                'box': [x1, y1, x1 + w, y1 + h],
                'width_px': w,
                'height_px': h
            })
            labels.append(label)

    count_per_class = dict(Counter(r['label'] for r in results))
    return jsonify(count_per_class=count_per_class, objects=results, description=list(set(labels)))

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, threaded=True)
