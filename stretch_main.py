import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# 정육면체 꼭짓점과 면 정의
cube_vertices = np.array([[1, 1, 1],
                          [1, 1, -1],
                          [1, -1, 1],
                          [1, -1, -1],
                          [-1, 1, 1],
                          [-1, 1, -1],
                          [-1, -1, 1],
                          [-1, -1, -1]])

faces = [[cube_vertices[j] for j in [0, 1, 3, 2]],
         [cube_vertices[j] for j in [4, 5, 7, 6]],
         [cube_vertices[j] for j in [0, 1, 5, 4]],
         [cube_vertices[j] for j in [2, 3, 7, 6]],
         [cube_vertices[j] for j in [0, 2, 6, 4]],
         [cube_vertices[j] for j in [1, 3, 7, 5]]]

# 회전 행렬 생성 함수
def get_rotation_matrix(pitch, roll, yaw):
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(pitch), -np.sin(pitch)],
                   [0, np.sin(pitch), np.cos(pitch)]])
    Ry = np.array([[np.cos(roll), 0, np.sin(roll)],
                   [0, 1, 0],
                   [-np.sin(roll), 0, np.cos(roll)]])
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                   [np.sin(yaw), np.cos(yaw), 0],
                   [0, 0, 1]])
    return Rz @ Ry @ Rx

# CSV 데이터 로드 및 애니메이션 실행 함수
def animate_cube(file_name):
    # CSV 파일에서 데이터를 읽어옵니다.
    data = pd.read_csv(file_name)
    pitch_data = data['Pitch (radians)'].values
    roll_data = data['Roll (radians)'].values
    yaw_data = data['Yaw (radians)'].values

    # 3D 플롯 설정
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    ax.set_zlim([-2, 2])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"Animation for {file_name}")

    # Azimuth 설정
    ax.view_init(azim=-90)

    # 초기 정육면체 객체 생성
    cube = Poly3DCollection(faces, alpha=0.5, edgecolor='k')
    ax.add_collection3d(cube)

    # 초기 축 화살표 생성
    quiver_x = ax.quiver(0, 0, 0, 1, 0, 0, color='r', arrow_length_ratio=0.1)
    quiver_y = ax.quiver(0, 0, 0, 0, 1, 0, color='g', arrow_length_ratio=0.1)
    quiver_z = ax.quiver(0, 0, 0, 0, 0, 1, color='b', arrow_length_ratio=0.1)

    # 텍스트 초기화
    pitch_text = ax.text2D(0.05, 0.95, "", transform=ax.transAxes)
    roll_text = ax.text2D(0.05, 0.90, "", transform=ax.transAxes)
    yaw_text = ax.text2D(0.05, 0.85, "", transform=ax.transAxes)

    # 업데이트 함수
    def update(frame):
        pitch = pitch_data[frame]
        roll = roll_data[frame]
        yaw = yaw_data[frame]
        
        # 회전 행렬을 사용하여 회전
        rotation_matrix = get_rotation_matrix(pitch, roll, yaw)
        rotated_vertices = [rotation_matrix @ vertex for vertex in cube_vertices]

        # 정육면체 면 업데이트
        new_faces = [[rotated_vertices[j] for j in [0, 1, 3, 2]],
                     [rotated_vertices[j] for j in [4, 5, 7, 6]],
                     [rotated_vertices[j] for j in [0, 1, 5, 4]],
                     [rotated_vertices[j] for j in [2, 3, 7, 6]],
                     [rotated_vertices[j] for j in [0, 2, 6, 4]],
                     [rotated_vertices[j] for j in [1, 3, 7, 5]]]
        cube.set_verts(new_faces)

        # 축 화살표 업데이트
        x_end = rotation_matrix @ np.array([1, 0, 0])
        y_end = rotation_matrix @ np.array([0, 1, 0])
        z_end = rotation_matrix @ np.array([0, 0, 1])

        quiver_x.set_segments([[[0, 0, 0], x_end]])
        quiver_y.set_segments([[[0, 0, 0], y_end]])
        quiver_z.set_segments([[[0, 0, 0], z_end]])

        # 텍스트 업데이트
        pitch_text.set_text(f"Pitch: {np.degrees(pitch):.2f}°")
        roll_text.set_text(f"Roll: {np.degrees(roll):.2f}°")
        yaw_text.set_text(f"Yaw: {np.degrees(yaw):.2f}°")

    # 애니메이션 생성
    ani = FuncAnimation(fig, update, frames=len(pitch_data), interval=40, repeat=False)
    plt.show()

# 여러 파일에 대해 애니메이션 실행
file_list = ["스트레칭-앞뒤.csv", "스트레칭-좌우roll.csv", "스트레칭-목회전.csv"]  # 파일명을 원하는 대로 입력
for file in file_list:
    animate_cube(file)