import cv2
import sys
from collections import deque
import time
import numpy as np

LOWER_COLOR=np.array([100, 150, 50])
UPPER_COLOR=np.array([130, 255, 255])
MIN_AREA=500

def start_webcam():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        sys.exit(1)
        
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    ret, _ = cap.read()
    if not ret:
        print("웹캠에서 프레임을 읽을 수 없습니다.")
        cap.release()
        sys.exit(1)
    
    print("웹캠 연결 성공!")
    
    return cap 

def create_mask(frame, lower_color=LOWER_COLOR, upper_color=UPPER_COLOR):  # frame -> mask / HSV 변환 + 지정 색 마스킹
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_color, upper_color)
    return mask

def find_objects(mask, MIN_AREA=MIN_AREA):  # mask -> objects / contour 검출 + 면적 필터링
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    objects = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < MIN_AREA:
            continue
        
        x, y, w, h = cv2.boundingRect(contour)
        cx = x + w // 2
        cy = y + h // 2
        objects.append((cx, cy, area, (x, y, w, h)))
     
    return objects

def draw_detection(frame, objects):  # frame, objects -> frame / 바운딩 박스, 중심점, 좌표 표시
    for cx, cy, area, (x, y, w, h) in objects:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
        cv2.putText(frame, f"({cx}, {cy})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

    return frame

def detect(frame):  # frame -> frame, mask, objects / 인식 전체 파이프라인
    mask = create_mask(frame)
    objects = find_objects(mask)
    frame = draw_detection(frame, objects)

    for cx, cy, area, _ in objects:
        print(f"중심 ({cx}, {cy}) | 면적: {area}")

    return frame, mask, objects


def create_fps_deque(maxlen=30):
    return deque(maxlen=maxlen)

def calculate_fps(frame_times, prev_time):
    current_time = time.time()
    dt = current_time - prev_time
    frame_times.append(dt)

    avg_dt = sum(frame_times) / len(frame_times) if frame_times else 0
    fps = 1 / avg_dt if avg_dt > 0 else 0
    return current_time, fps

def draw_fps(frame, fps):
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
def to_bgr(mask):
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    return mask_bgr

def combine_frames(frame, mask_bgr):
    combined = np.hstack((frame, mask_bgr))
    return combined

def save_result(combined, filename="result.png"):
    cv2.imwrite(filename, combined)
    print(f"{filename} 저장 완료")

def show_webcam(cap):
    frame_times = create_fps_deque()
    prev_time = time.time()

    while True:
        ret, frame = cap.read()

        if not ret:
            break 

        frame = cv2.flip(frame, 1)  # 1: 좌우 반전
        frame, mask, _ = detect(frame)
        prev_time, fps = calculate_fps(frame_times, prev_time)

        draw_fps(frame, fps)

        combined = combine_frames(frame, to_bgr(mask))
        cv2.imshow("Combined", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            save_result(combined)

    cap.release()
    cv2.destroyAllWindows()


def main():
    cap = start_webcam()
    show_webcam(cap)

if __name__ == "__main__":
    main()