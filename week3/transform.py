import numpy as np

def rotation_matrix_2d(theta_deg):  # theta -> rad -> rotation matrix / 2x2 회전 행렬 생성
    theta_rad = np.radians(theta_deg)
    cos_theta = np.cos(theta_rad)
    sin_theta = np.sin(theta_rad)

    rotation_matrix = np.array([[cos_theta, -sin_theta],
                                 [sin_theta, cos_theta]])

    return rotation_matrix

def scale_matrix_2d(sx, sy):  # sx, sy -> scale matrix / 2x2 스케일 행렬 생성
    scale_matrix = np.diag([sx, sy])
    return scale_matrix

def translate_2d(point, tx, ty):  # point, tx, ty -> 2d point / 이동 변환
    translated_point = np.array([point[0] + tx, point[1] + ty])
    return translated_point

def pixel_to_world(coord, img_size, angle_deg=0.0, scale=0.01):  # (cx, cy), img_size, angle, scale -> (wx, wy) / 픽셀 좌표를 월드 좌표로 변환
    tx, ty = -img_size[0] / 2, -img_size[1] / 2
    translated_point = translate_2d(coord, tx, ty)

    scaled_point = scale_matrix_2d(scale, scale) @ translated_point

    rotated_point = rotation_matrix_2d(angle_deg) @ scaled_point

    return rotated_point


if __name__ == "__main__":
    print("rotation_matrix_2d(90) @ [1, 0] =", rotation_matrix_2d(90) @ np.array([1, 0]))
    print("scale_matrix_2d(2, 3) @ [1, 1] =", scale_matrix_2d(2, 3) @ np.array([1, 1]))
    print("translate_2d([1, 1], 2, 3) =", translate_2d([1, 1], 2, 3))
    print("pixel_to_world(center) =", pixel_to_world((320, 240), (640, 480)))
    print("pixel_to_world(right)  =", pixel_to_world((500, 240), (640, 480)))
    print("pixel_to_world(br)     =", pixel_to_world((640, 480), (640, 480)))
