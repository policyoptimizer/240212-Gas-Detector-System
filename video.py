import requests

video_path = r'D:\#.Secure Work Folder\BIG\Toy\24~28Y\240212 유해 화학물질 가스 누출 실시간 감지 시스템\test video\FireDay\smoke1_cut.mp4'
url = 'http://localhost:5000/predict'

with open(video_path, 'rb') as video:
    files = {'video': video}
    response = requests.post(url, files=files)
    print(f"Status Code: {response.status_code}")
    print(f"Response Text: {response.text}")

