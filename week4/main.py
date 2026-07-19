import cv2
import sys
from collections import deque
import time
import numpy as np
import matplotlib.pyplot as plt

from transform import rotation_matrix_2d, scale_matrix_2d, translate_2d

LOWER_COLOR = np.array([100, 150, 50])
UPPER_COLOR = np.array([130, 255, 255])
MIN_AREA = 500

INIT_ANGLE = 0.0      # 시작 회전각 (도)
INIT_SCALE = 0.01     # 시작 스케일 (픽셀 -> 미터 환산 비율)

HIST_MAXLEN = 100      # 좌표 이력을 몇 개까지 쌓아 보여줄지 (deque 크기)
PLOT_INTERVAL = 3      # 몇 프레임마다 한 번씩만 플롯을 갱신할지


# ------------------------- 과제 1: 웹캠 입력 -------------------------

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


# ------------------------- 과제 2: 물체 인식 -------------------------

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

    return frame, mask, objects


def select_main_object(objects):  # objects -> 가장 면적이 큰 물체 하나
    if not objects:
        return None
    return max(objects, key=lambda obj: obj[2])


# ------------------------- 과제 3: 좌표 변환 / 시각화 -------------------------

def pixel_to_world(coord, img_size, angle_deg=INIT_ANGLE, scale=INIT_SCALE):  # (cx, cy), img_size -> (wx, wy)
    tx, ty = -img_size[0] / 2, -img_size[1] / 2
    translated_point = translate_2d(coord, tx, ty)

    scaled_point = scale_matrix_2d(scale, scale) @ translated_point
    rotated_point = rotation_matrix_2d(angle_deg) @ scaled_point

    return rotated_point


def draw_world_text(frame, x, y, wx, wy):
    cv2.putText(frame, f"world({wx:.2f}, {wy:.2f})", (x, y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
    return frame


def setup_plot():
    plt.ion()

    fig, (ax_pixel, ax_world) = plt.subplots(1, 2, figsize=(9, 4))
    fig.suptitle("pixel(파랑) / world(빨강)")

    pixel_hist = deque(maxlen=HIST_MAXLEN)
    world_hist = deque(maxlen=HIST_MAXLEN)

    return fig, ax_pixel, ax_world, pixel_hist, world_hist


def update_plot(fig, ax_pixel, ax_world, pixel_hist, world_hist, pixel_pt, world_pt):
    pixel_hist.append(pixel_pt)
    world_hist.append(world_pt)

    ax_pixel.clear()
    px, py = zip(*pixel_hist)
    ax_pixel.scatter(px, py, c="blue", s=15)
    ax_pixel.set_title("pixel")
    ax_pixel.set_xlabel("x (px)")
    ax_pixel.set_ylabel("y (px)")
    ax_pixel.invert_yaxis()

    ax_world.clear()
    wx, wy = zip(*world_hist)
    ax_world.scatter(wx, wy, c="red", s=15)
    ax_world.set_title("world")
    ax_world.set_xlabel("x (m)")
    ax_world.set_ylabel("y (m)")

    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.001)


# ------------------------- 과제 4: sense -> compute -> act 통합 -------------------------

def main():
    cap = start_webcam()

    frame_times = create_fps_deque()
    prev_time = time.time()
    frame_count = 0

    fig, ax_pixel, ax_world, pixel_hist, world_hist = setup_plot()

    try:
        while True:
            # ---------------- Sense ----------------
            t_sense_start = time.time()

            ret, frame = cap.read()
            if not ret:
                print("웹캠에서 프레임을 읽을 수 없습니다.")
                break

            frame = cv2.flip(frame, 1)  # 1: 좌우 반전

            t_sense_end = time.time()

            # ---------------- Compute ----------------
            t_compute_start = time.time()

            frame, mask, objects = detect(frame)
            main_obj = select_main_object(objects)

            world_pt = None
            if main_obj is not None:
                cx, cy, area, (x, y, w, h) = main_obj
                img_h, img_w = frame.shape[:2]
                wx, wy = pixel_to_world((cx, cy), (img_w, img_h))
                world_pt = (wx, wy)

            t_compute_end = time.time()

            # ---------------- Act ----------------
            t_act_start = time.time()

            prev_time, fps = calculate_fps(frame_times, prev_time)
            draw_fps(frame, fps)

            if main_obj is not None:
                cx, cy, area, (x, y, w, h) = main_obj
                wx, wy = world_pt
                draw_world_text(frame, x, y, wx, wy)

                frame_count += 1
                if frame_count % PLOT_INTERVAL == 0:
                    update_plot(fig, ax_pixel, ax_world, pixel_hist, world_hist, (cx, cy), (wx, wy))

            cv2.imshow("Webcam", frame)

            key = cv2.waitKey(1) & 0xFF
            quit_flag = (key == ord('q'))

            t_act_end = time.time()

            # ---------------- 단계별 처리 시간 출력 ----------------
            sense_ms = (t_sense_end - t_sense_start) * 1000
            compute_ms = (t_compute_end - t_compute_start) * 1000
            act_ms = (t_act_end - t_act_start) * 1000

            print(f"sense: {sense_ms:.1f}ms | compute: {compute_ms:.1f}ms | act: {act_ms:.1f}ms | FPS: {fps:.1f}")

            if quit_flag:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        plt.ioff()
        plt.close(fig)


if __name__ == "__main__":
    main()
