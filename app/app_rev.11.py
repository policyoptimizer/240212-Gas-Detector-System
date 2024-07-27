# Smoke 0.80, Fire 0.70 등 화면 상에서 object 가 detect 됩니다. 거기에 바로 시간을 표기할 수는 없을까요? 예를 들면 Smoke 0.80 26s 이런 식으로요
#
# 보고서는 웹페이지에서 결과물이 없어요
# 전혀 안 보여요
#
# 보고서 다운로드 버튼은 좋은데
# 로그 기록보다는 뭔가 좀 보고서 형식으로
# 요약된 결과가 다운로드 되었으면 좋겠어요
#
# 다시 보기 페이지는 여전히 영상 재생이 안되요
# 아래 경로에 output.avi 영상이 있는데요
# D:\회사_더샵\BIG\Toy\24~28Y\240212 유해 화학물질 가스 누출 실시간 감지 시스템\saved_videos\
#
# smoke 나 fire detect 됐을 때
# 알람을 울릴 수 있나요? 필요하시면 알람 소리를 구글에서 검색해서 파일로 다운 받도록 하겠습니다.
#
# 그리고 옆에 작은 버튼을 만들어서
# 사용자가 알람 지속시간(30s, 20s, 10s, 5s, 3s, 1s 등), 알람 소리 크기 등을 설정할 수 있었으면 좋겠어요

# 동영상 저장: 동영상이 PC에 저장되도록 설정합니다.
# 시간 정보 추가 저장: 연기 감지 시 몇 초째 감지되는지 화면에 표시합니다.
# 보고서 생성: 감지된 시간 정보를 보고서로 작성하여 PC에 저장합니다.

# 모델 가중치 파일 경로
# r'D:\#.Secure Work Folder\BIG\Toy\24~28Y\240212 유해 화학물질 가스 누출 실시간 감지 시스템\train-yolov8-object-detection-on-custom-dataset_rev.01\train\weights\best.pt'
# r"D:\회사_더샵\BIG\Toy\24~28Y\240212 유해 화학물질 가스 누출 실시간 감지 시스템\train-yolov8-object-detection-on-custom-dataset_rev.01\train\weights\best.pt"

# 테스트 영상 경로
# video_path = r"D:\회사_더샵\BIG\Toy\24~28Y\240212 유해 화학물질 가스 누출 실시간 감지 시스템\test video\FireDay\fatalcrash_cut.mp4"
# video_path = r"D:\회사_더샵\BIG\Toy\24~28Y\240212 유해 화학물질 가스 누출 실시간 감지 시스템\test video\FireNight\distancefire7_cut.mp4"

# 영상 저장 경로
# save_path = r"D:\회사_더샵\BIG\Toy\24~28Y\240212 유해 화학물질 가스 누출 실시간 감지 시스템\saved_videos\output.avi"
# saved_video_path = r"D:\회사_더샵\BIG\Toy\24~28Y\240212 유해 화학물질 가스 누출 실시간 감지 시스템\saved_videos\output.avi"

import os
import cv2
import time
from ultralytics import YOLO
from flask import Flask, render_template, Response, jsonify, send_from_directory, request
import json
import threading

app = Flask(__name__)

# 모델 가중치 파일 경로
model_path = r"D:\회사_더샵\BIG\Toy\24~28Y\240212 유해 화학물질 가스 누출 실시간 감지 시스템\train-yolov8-object-detection-on-custom-dataset_rev.01\train\weights\best.pt"

# YOLOv8 모델 로드
model = YOLO(model_path)

# 감지 상태를 저장할 변수
detection_logs = []

alarm_duration = 3
alarm_volume = 50


def play_alarm(duration, volume):
    from pygame import mixer
    mixer.init()
    mixer.music.set_volume(volume / 100.0)
    mixer.music.load(r"D:\회사_더샵\BIG\Toy\24~28Y\240212 유해 화학물질 가스 누출 실시간 감지 시스템\flask\sound\alarm_sound.mp3")
    mixer.music.play()
    time.sleep(duration)
    mixer.music.stop()


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
            detection_logs.append(
                f"{label} {score:.2f} detected at {detected_time} for {minutes} minutes and {seconds} seconds")

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
    for log in logs:
        parts = log.split(" ")
        label, score = parts[0], parts[1]
        detected_time, duration = " ".join(parts[4:6]), parts[-3] + " " + parts[-2]
        if label not in summary:
            summary[label] = []
        summary[label].append({"score": score, "time": detected_time, "duration": duration})
    return summary


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
    return jsonify({"detection_logs": detection_logs})


@app.route('/review')
def review():
    return render_template('review.html')


@app.route('/saved_video')
def saved_video():
    return send_from_directory(directory=r"D:\회사_더샵\BIG\Toy\24~28Y\240212 유해 화학물질 가스 누출 실시간 감지 시스템\saved_videos",
                               filename="output.avi")


@app.route('/report')
def report():
    return render_template('report.html', detection_logs=summarize_logs(detection_logs))


@app.route('/download_report')
def download_report():
    return send_from_directory(directory=os.getcwd(), filename="detection_report.json", as_attachment=True)


@app.route('/alarm_settings', methods=['GET', 'POST'])
def alarm_settings():
    global alarm_duration, alarm_volume
    if request.method == 'POST':
        alarm_duration = int(request.form['duration'])
        alarm_volume = int(request.form['volume'])
        return jsonify({"status": "alarm settings updated"})
    return render_template('alarm_settings.html')


if __name__ == '__main__':
    app.run(debug=True)

