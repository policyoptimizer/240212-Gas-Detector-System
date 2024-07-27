# video.py 같이 실행해야 함
# 작동됨!!!
# 장문의 response text 로 출력되지만

from flask import Flask, request, jsonify
import cv2
import numpy as np
import os
import tempfile
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

    # 임시 파일에 저장
    temp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    temp.write(file.read())
    temp.close()

    video = cv2.VideoCapture(temp.name)

    if not video.isOpened():
        os.remove(temp.name)
        return jsonify({'error': 'Invalid video file'}), 400

    results_list = []
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        # 예측 수행
        results = model(frame)
        # 결과를 수동으로 처리
        for result in results:
            boxes = result.boxes  # 각 결과의 바운딩 박스 가져오기
            for box in boxes:
                result_data = {
                    "xmin": box.xyxy[0][0].item(),
                    "ymin": box.xyxy[0][1].item(),
                    "xmax": box.xyxy[0][2].item(),
                    "ymax": box.xyxy[0][3].item(),
                    "confidence": box.conf[0].item(),
                    "class": box.cls[0].item()
                }
                results_list.append(result_data)

    video.release()
    os.remove(temp.name)

    return jsonify(results_list)


if __name__ == '__main__':
    app.run(debug=True)

