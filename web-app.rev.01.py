import cv2
from ultralytics import YOLO

# 모델 가중치 파일 경로
model_path = r"D:\회사_더샵\BIG\Toy\24~28Y\240212 유해 화학물질 가스 누출 실시간 감지 시스템\train-yolov8-object-detection-on-custom-dataset_rev.01\train\weights\best.pt"
# model_path = r"D:\회사_더샵\BIG\Toy\24~28Y\240212 유해 화학물질 가스 누출 실시간 감지 시스템\코랩 결과\runs\detect\train2\weights\best.pt"
# r'D:\#.Secure Work Folder\BIG\Toy\24~28Y\240212 유해 화학물질 가스 누출 실시간 감지 시스템\train-yolov8-object-detection-on-custom-dataset_rev.01\train\weights\best.pt'


# YOLOv8 모델 로드
model = YOLO(model_path)


def detect_smoke_from_video(video_path):
    # 동영상 파일 열기
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 모델을 사용하여 객체 감지 수행
        results = model(frame)

        # results가 리스트인지 확인하고, 리스트의 첫 번째 요소를 사용
        if isinstance(results, list):
            results = results[0]

        # 결과를 이미지에 그리기
        annotated_frame = results.plot()  # 'plot' 메소드 사용

        # 연기 감지 여부 확인
        if len(results.boxes) > 0:  # 객체가 감지되었을 경우
            for box in results.boxes:
                label = model.names[int(box.cls)]
                if label == "smoke":  # 감지된 객체가 연기일 경우
                    # 알람 표시 (이미지에 텍스트 추가)
                    cv2.putText(annotated_frame, 'Smoke Detected!', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255),
                                2, cv2.LINE_AA)

        # 이미지를 화면에 표시
        cv2.imshow('Smoke Detection', annotated_frame)

        # 'q' 키를 누르면 루프 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 스트림 릴리스 및 모든 윈도우 닫기
    cap.release()
    cv2.destroyAllWindows()

# 동영상 파일 경로 설정
# video_path = r"D:\회사_더샵\BIG\Toy\24~28Y\240212 유해 화학물질 가스 누출 실시간 감지 시스템\테스트 영상\FireDay\smoke1_cut.mp4"
video_path = r"D:\회사_더샵\BIG\Toy\24~28Y\240212 유해 화학물질 가스 누출 실시간 감지 시스템\test video\FireDay\smoke1_cut.mp4"
# video_path = r'D:\#.Secure Work Folder\BIG\Toy\24~28Y\240212 유해 화학물질 가스 누출 실시간 감지 시스템\test video\FireDay\fatalcrash_cut.mp4'

# 연기 감지 함수 실행
detect_smoke_from_video(video_path)
