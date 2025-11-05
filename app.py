from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import cv2
import base64
import numpy as np
from ultralytics import YOLO
import torch

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Load YOLOv5 model
model = YOLO('yolov5su.pt')

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('video_frame')
def handle_video_frame(data):
    # Decode base64 image
    img_data = base64.b64decode(data.split(',')[1])
    nparr = np.frombuffer(img_data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Run object detection
    results = model(frame)

    # Draw bounding boxes
    annotated_frame = results[0].plot()

    # Encode back to base64
    _, buffer = cv2.imencode('.jpg', annotated_frame)
    img_str = base64.b64encode(buffer).decode('utf-8')
    emit('detection_result', 'data:image/jpeg;base64,' + img_str)

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
