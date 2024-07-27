# 여러 웹페이지로 구성함

# video.py 실행 안해도 됨
# 대신 templates/index.html 있어야 함
# 실시간으로 detect 되는 것이 보임

# 모델 가중치 파일 경로
# r'D:\#.Secure Work Folder\BIG\Toy\24~28Y\240212 유해 화학물질 가스 누출 실시간 감지 시스템\train-yolov8-object-detection-on-custom-dataset_rev.01\train\weights\best.pt'
# r"D:\회사_더샵\BIG\Toy\24~28Y\240212 유해 화학물질 가스 누출 실시간 감지 시스템\train-yolov8-object-detection-on-custom-dataset_rev.01\train\weights\best.pt"

# video_path = r"D:\회사_더샵\BIG\Toy\24~28Y\240212 유해 화학물질 가스 누출 실시간 감지 시스템\test video\FireDay\fatalcrash_cut.mp4"
# video_path = r"D:\회사_더샵\BIG\Toy\24~28Y\240212 유해 화학물질 가스 누출 실시간 감지 시스템\test video\FireNight\distancefire7_cut.mp4"

import os
import cv2
import time
from ultralytics import YOLO
from flask import Flask, render_template, Response, jsonify

app = Flask(__name__)

# 모델 가중치 파일 경로
model_path = r"D:\회사_더샵\BIG\Toy\24~28Y\240212 유해 화학물질 가스 누출 실시간 감지 시스템\train-yolov8-object-detection-on-custom-dataset_rev.01\train\weights\best.pt"

# YOLOv8 모델 로드
model = YOLO(model_path)

# 연기 감지 상태를 저장할 변수
smoke_detected = False
detected_time = None
detection_logs = []

def generate_frames(video_path, save_path):
    global smoke_detected, detected_time
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(save_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 모델을 사용하여 객체 감지 수행
        results = model(frame)

        if isinstance(results, list):
            results = results[0]

        # 결과를 이미지에 그리기
        annotated_frame = results.plot()  # 'plot' 메소드 사용

        # 연기 감지 여부 확인
        smoke_detected = False
        if len(results.boxes) > 0:  # 객체가 감지되었을 경우
            for box in results.boxes:
                label = model.names[int(box.cls)]
                if label == "smoke":  # 감지된 객체가 연기일 경우
                    # 알람 표시 (이미지에 텍스트 추가)
                    cv2.putText(annotated_frame, 'Smoke Detected!', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)
                    smoke_detected = True
                    detected_time = time.strftime('%Y-%m-%d %H:%M:%S')
                    detection_logs.append(f"Smoke detected at {detected_time}")

        out.write(annotated_frame)

        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
    out.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    video_path = r"D:\회사_더샵\BIG\Toy\24~28Y\240212 유해 화학물질 가스 누출 실시간 감지 시스템\test video\FireDay\fatalcrash_cut.mp4"
    save_path = r"D:\회사_더샵\BIG\Toy\24~28Y\240212 유해 화학물질 가스 누출 실시간 감지 시스템\saved_videos\output.avi"
    return Response(generate_frames(video_path, save_path),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/smoke_status')
def smoke_status():
    global smoke_detected, detected_time
    return jsonify({"smoke_detected": smoke_detected, "detected_time": detected_time})

@app.route('/review')
def review():
    saved_video_path = r"D:\회사_더샵\BIG\Toy\24~28Y\240212 유해 화학물질 가스 누출 실시간 감지 시스템\saved_videos\output.avi"
    return render_template('review.html', video_path=saved_video_path)

@app.route('/report')
def report():
    return render_template('report.html', detection_logs=detection_logs)

if __name__ == '__main__':
    app.run(debug=True)
