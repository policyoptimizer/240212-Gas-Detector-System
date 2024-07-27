# video.py 같이 실행해야 함
# 영상이 빠른 속도로 저장됨
# output 경로에 파일로 저장됨

import torch
from flask import Flask, request, jsonify
import cv2
import numpy as np
import os
import tempfile
from ultralytics import YOLO

app = Flask(__name__)

# GPU 설정 (가능한 경우)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# YOLOv8 모델 로드 및 GPU로 이동
model = YOLO(
    r'D:\#.Secure Work Folder\BIG\Toy\24~28Y\240212 유해 화학물질 가스 누출 실시간 감지 시스템\train-yolov8-object-detection-on-custom-dataset_rev.01\train\weights\best.pt')
model.to(device)


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

    # 결과 동영상 저장 설정
    output_path = r'D:\#.Secure Work Folder\BIG\Toy\24~28Y\240212 유해 화학물질 가스 누출 실시간 감지 시스템\output\output_video.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    results_list = []
    frame_count = 0
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        # 프레임 간격 설정 (예: 5프레임마다 처리)
        frame_count += 1
        if frame_count % 5 != 0:
            continue

        # 예측 수행
        results = model(frame)
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # 바운딩 박스 그리기
                xmin, ymin, xmax, ymax = map(int, box.xyxy[0])
                confidence = box.conf[0].item()
                cls = int(box.cls[0].item())
                label = f'{cls} {confidence:.2f}'
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

                result_data = {
                    "xmin": xmin,
                    "ymin": ymin,
                    "xmax": xmax,
                    "ymax": ymax,
                    "confidence": confidence,
                    "class": cls
                }
                results_list.append(result_data)

        # 프레임을 결과 동영상에 저장
        out.write(frame)

    video.release()
    out.release()
    os.remove(temp.name)

    return jsonify(results_list)


if __name__ == '__main__':
    app.run(debug=True)

