import os
import cv2
import time
from ultralytics import YOLO
from flask import Flask, render_template, Response, jsonify, send_from_directory, request, url_for
import json
import threading
from pygame import mixer

app = Flask(__name__)

# 모델 가중치 파일 경로
model_path = r"D:\회사_더샵\BIG\Toy\24~28Y\240212 유해 화학물질 가스 누출 실시간 감지 시스템\train-yolov8-object-detection-on-custom-dataset_rev.01\train\weights\best.pt"

# YOLOv8 모델 로드
model = YOLO(model_path)

# 감지 상태를 저장할 변수
detection_logs = []

alarm_duration = 3
alarm_volume = 50
alarm_active = False

mixer.init()


def play_alarm(duration, volume):
    global alarm_active
    if not alarm_active:
        alarm_active = True
        mixer.music.set_volume(volume / 100.0)
        mixer.music.load(r"D:\회사_더샵\BIG\Toy\24~28Y\240212 유해 화학물질 가스 누출 실시간 감지 시스템\flask\sound\alarm_sound.mp3")
        mixer.music.play()
        time.sleep(duration)
        mixer.music.stop()
        alarm_active = False


def generate_frames(video_path, save_path):
    global detection_logs, alarm_duration, alarm_volume
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(save_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        elapsed_time = time.time() - start_time
        minutes, seconds = divmod(int(elapsed_time), 60)

        # 모델을 사용하여 객체 감지 수행
        results = model(frame)

        if isinstance(results, list):
            results = results[0]

        # 결과를 이미지에 그리기
        annotated_frame = results.plot()  # 'plot' 메소드 사용

        # 객체 감지 여부 확인
        for box in results.boxes:
            label = model.names[int(box.cls)]
            score = box.conf[0]
            detection_time = f"{minutes}min {seconds}s"
            cv2.putText(annotated_frame, f'{label} {score:.2f} {detection_time}', (10, 30 + 30 * int(box.cls)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
            detected_time = time.strftime('%Y-%m-%d %H:%M:%S')
            detection_logs.append((label, score, detected_time, minutes, seconds))

            # 알람 실행
            threading.Thread(target=play_alarm, args=(alarm_duration, alarm_volume)).start()

        out.write(annotated_frame)

        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
    out.release()

    # 감지 로그를 요약하여 파일에 저장
    summarized_logs = summarize_logs(detection_logs)
    with open('detection_report.json', 'w') as report_file:
        json.dump(summarized_logs, report_file, indent=4)


def summarize_logs(logs):
    summary = {}
    current_detection = None
    for log in logs:
        label, score, detected_time, minutes, seconds = log
        key = f"{label} (Score: {score:.2f})"
        if key not in summary:
            summary[key] = []
        if current_detection and current_detection["label"] == label and current_detection["score"] == score:
            current_detection["end"] = detected_time
            current_detection[
                "duration"] = f"{(minutes - current_detection['start_minutes']) * 60 + (seconds - current_detection['start_seconds'])} seconds"
        else:
            current_detection = {
                "label": label,
                "score": score,
                "start": detected_time,
                "end": detected_time,
                "duration": "1 second",
                "start_minutes": minutes,
                "start_seconds": seconds
            }
            summary[key].append(current_detection)
    return summary


@app.route('/')
def index():
    return render_template('index.html', alarm_duration=alarm_duration, alarm_volume=alarm_volume)


@app.route('/video_feed')
def video_feed():
    video_path = r"D:\회사_더샵\BIG\Toy\24~28Y\240212 유해 화학물질 가스 누출 실시간 감지 시스템\test video\FireDay\fatalcrash_cut.mp4"
    save_path = r"D:\회사_더샵\BIG\Toy\24~28Y\240212 유해 화학물질 가스 누출 실시간 감지 시스템\saved_videos\output.avi"
    return Response(generate_frames(video_path, save_path),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/smoke_status')
def smoke_status():
    return jsonify({"detection_logs": detection_logs})


@app.route('/review')
def review():
    return render_template('review.html')


@app.route('/saved_video')
def saved_video():
    return send_from_directory(directory=r"D:\회사_더샵\BIG\Toy\24~28Y\240212 유해 화학물질 가스 누출 실시간 감지 시스템\saved_videos",
                               path="output.avi")


@app.route('/report')
def report():
    summarized_logs = summarize_logs(detection_logs)
    return render_template('report.html', detection_logs=summarized_logs)


@app.route('/download_report')
def download_report():
    return send_from_directory(directory=os.getcwd(), path="detection_report.json", as_attachment=True)


@app.route('/update_alarm_settings', methods=['POST'])
def update_alarm_settings():
    global alarm_duration, alarm_volume
    alarm_duration = int(request.form['duration'])
    alarm_volume = int(request.form['volume'])
    return jsonify({"status": "alarm settings updated"})


if __name__ == '__main__':
    app.run(debug=True)
