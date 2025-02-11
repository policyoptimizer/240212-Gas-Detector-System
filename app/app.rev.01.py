from flask import Flask, request, jsonify
import cv2
import numpy as np
from ultralytics import YOLO

app = Flask(__name__)

# YOLOv8 모델 로드
model = YOLO(
    r'D:\#.Secure Work Folder\BIG\Toy\24~28Y\240212 유해 화학물질 가스 누출 실시간 감지 시스템\train-yolov8-object-detection-on-custom-dataset_rev.01\train\weights\best.pt')


@app.route('/predict', methods=['POST'])
def predict():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400

    file = request.files['video']
    file_bytes = np.frombuffer(file.read(), np.uint8)
    video = cv2.VideoCapture(file_bytes)

    if not video.isOpened():
        return jsonify({'error': 'Invalid video file'}), 400

    results_list = []
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        # 예측 수행
        results = model(frame)
        result_data = results.pandas().xyxy[0].to_dict(orient="records")
        results_list.append(result_data)

    video.release()

    return jsonify(results_list)


if __name__ == '__main__':
    app.run(debug=True)

