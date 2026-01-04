from flask import Flask, request, jsonify, render_template_string
import cv2
import numpy as np
import onnxruntime as ort
from collections import Counter
import os

app = Flask(__name__)

MODEL_PATH = 'yolov8n.onnx'
ort_session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
input_name = ort_session.get_inputs()[0].name

CLASSES = ['person','bicycle','car','motorcycle','airplane','bus','train','truck','boat','traffic light',
           'fire hydrant','stop sign','parking meter','bench','bird','cat','dog','horse','sheep','cow',
           'elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee',
           'skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard',
           'tennis racket','bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple',
           'sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair','couch',
           'potted plant','bed','dining table','toilet','tv','laptop','mouse','remote','keyboard','cell phone',
           'microwave','oven','toaster','sink','refrigerator','book','clock','vase','scissors','teddy bear',
           'hair drier','toothbrush']

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({'class_counts': {}, 'objects': [], 'description': []})
    
    file = request.files['image']
    nparr = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    h, w = frame.shape[:2]
    
    # Fast preprocessing (416x416)
    img_resized = cv2.resize(frame, (416, 416))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_norm = img_rgb.astype(np.float32) / 255.0
    img_input = np.transpose(img_norm, (2, 0, 1))[np.newaxis, ...]
    
    # Inference
    outputs = ort_session.run(None, {input_name: img_input})
    predictions = np.squeeze(outputs[0]).T  # (8400, 85)
    
    boxes, confs, classids = [], [], []
    for i in range(len(predictions)):
        pred = predictions[i]
        scores = pred[4:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        
        if confidence > 0.3:  # Fast threshold
            cx, cy, bw, bh = pred[:4]
            x1 = int((cx - bw/2) * w / 416)
            y1 = int((cy - bh/2) * h / 416)
            x2 = int((cx + bw/2) * w / 416)
            y2 = int((cy + bh/2) * h / 416)
            boxes.append([x1, y1, x2-x1, y2-y1])
            confs.append(float(confidence))
            classids.append(class_id)
    
    # Fast NMS
    indices = cv2.dnn.NMSBoxes(boxes, confs, 0.3, 0.4)
    objects = []
    labels = []
    
    if len(indices) > 0:
        for i in indices.flatten():
            x1, y1, w, h = boxes[i]
            obj = {
                'label': CLASSES[min(classids[i], len(CLASSES)-1)],
                'confidence': round(confs[i], 2),
                'box': [x1, y1, x1+w, y1+h],
                'width_px': w,
                'height_px': h
            }
            objects.append(obj)
            labels.append(obj['label'])
    
    return jsonify({
        'class_counts': dict(Counter(labels)),
        'objects': objects,
        'description': list(set(labels))
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, threaded=True)
