import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class AirpodsCalMovingAvgZupt:
    def __init__(self):
        self.last_timestamp = 0.0
        self.raw_y = [0.0]
        self.acc_offset = 0.0
        self.past_y = [0.0]
        self.past_z = [0.0]
        self.accelometers_not_smooth = [0.0]
        self.accelometers = [0.0]
        self.final_accelometers = [0.0]
        self.velocities = [0.0]
        self.positions = [0.0]
        self.stable_position = 0.0
        self.stable_positions = [0.0]
        self.is_zupt = [False]
        self.threshold = 0.015
        self.is_moving = False
        self.stop_flag = False
        self.is_ready = 0

    def get_position(self, limit_value=None):
        if limit_value is None:
            limit_value = self.threshold
        return self.stable_position * (limit_value / self.threshold)

    def process_sensor_data(self, timestamp, accel_x, accel_y, accel_z, pitch, roll, yaw):
        delta_time = timestamp - self.last_timestamp if self.positions else 0
        self.last_timestamp = timestamp

        self.raw_y.append(accel_y)
        if len(self.raw_y) > 500:
            self.is_ready += 1
            self.raw_y.pop(0)

        if self.is_ready == 1:
            self.velocities[-1] = 0
            self.positions[-1] = 0
            self.is_ready = 2

        self.acc_offset = np.mean(self.raw_y)

        self.past_y.append(accel_y)
        self.past_z.append(accel_z)
        len_history = 5
        if len(self.past_y) > len_history:
            self.past_y.pop(0)
        if len(self.past_z) > len_history:
            self.past_z.pop(0)

        cal_acc_y = np.mean(self.past_y) - self.acc_offset

        self.accelometers_not_smooth.append(accel_y - self.acc_offset)

        cal_acc = -cal_acc_y  # Only considering Y-axis
        self.accelometers.append(cal_acc)

        if len(self.accelometers) > 500:
            self.accelometers.pop(0)

        velocity = self.velocities[-1] + cal_acc * delta_time
        position = self.positions[-1] + velocity * delta_time

        velocity, position = self.apply_zupt(velocity, position)

        self.final_accelometers.append(cal_acc)
        self.velocities.append(velocity)
        self.positions.append(position)

        if len(self.velocities) > 500:
            self.velocities.pop(0)
        if len(self.positions) > 500:
            self.positions.pop(0)

        if not self.stop_flag:
            self.stable_position = position
        self.stable_position = max(min(self.stable_position, self.threshold), -0.005)
        self.stable_positions.append(self.stable_position)

    def apply_zupt(self, velocity, position):
        window_size = 20
        if len(self.final_accelometers) < window_size:
            return velocity, position

        deviation = np.mean([abs(a) for a in self.final_accelometers[-window_size:]])
        if deviation > 0.0015:
            return velocity, position

        position = self.compensate_position(velocity, position)
        velocity = 0
        self.is_zupt[-1] = True
        self.stop_flag = False
        self.stable_position = position
        return velocity, position

    def compensate_position(self, velocity, position):
        compensation_flag = False
        idx = len(self.velocities) - 1

        while idx >= 0:
            if velocity * self.velocities[idx] > 0:
                idx -= 1
            else:
                break

        while idx >= 0 and velocity * self.velocities[idx] < 0:
            compensation_flag = True
            if abs(self.velocities[idx]) > 0.00001:
                compensation_flag = True
                break
            idx -= 1

        if compensation_flag:
            idx = len(self.velocities) - 1
            while idx >= 0 and velocity * self.velocities[idx] > 0:
                self.velocities[idx] = 0
                idx -= 1
            position = self.positions[idx]
            print("Position compensation triggered!")

        return max(0, min(position, self.threshold))


# CSV 파일에서 데이터를 불러와 처리하는 함수 및 시각화
def process_csv_data(file_path):
    data = pd.read_csv(file_path)

    airpods_processor = AirpodsCalMovingAvgZupt()

    timestamps = []
    accel_data = []
    velocity_data = []
    position_data = []

    for i, row in data.iterrows():
        timestamp = row['Time (s)']
        accel_x = row['Accelerometer X (g)']
        accel_y = row['Accelerometer Y (g)']
        accel_z = row['Accelerometer Z (g)']
        pitch = row['Pitch (radians)']
        roll = row['Roll (radians)']
        yaw = row['Yaw (radians)']

        airpods_processor.process_sensor_data(timestamp, accel_x, accel_y, accel_z, pitch, roll, yaw)
        timestamps.append(timestamp)
        accel_data.append(airpods_processor.final_accelometers[-1])
        velocity_data.append(airpods_processor.velocities[-1])
        # position_data.append(airpods_processor.positions[-1])
        position_data.append(airpods_processor.stable_positions[-1])

    # 가속도, 속도, 위치 그래프 그리기
    fig, axs = plt.subplots(3, 1, figsize=(8, 10))

    axs[0].plot(range(len(timestamps)), accel_data, label="Acceleration")
    axs[0].set_title('Acceleration over Time')
    axs[0].set_ylabel('Acceleration (g)')
    axs[0].legend()

    axs[1].plot(range(len(timestamps)), velocity_data, label="Velocity", color='orange')
    axs[1].set_title('Velocity over Time')
    axs[1].set_ylabel('Velocity (m/s)')
    axs[1].legend()

    axs[2].plot(timestamps, position_data, label="Position", color='green')
    # axs[2].plot(range(len(timestamps)), position_data, label="Position", color='green')
    axs[2].set_title('Position over Time')
    axs[2].set_ylabel('Position (m)')
    axs[2].set_xlabel('Time (s)')
    axs[2].set_ylim(-0.01, 0.02)
    axs[2].legend()

    plt.tight_layout()
    plt.show()


# CSV 파일에서 데이터 처리
if __name__ == "__main__":
    file_paths = ["data4.csv", "data5.csv", "motion_data_Wednesday, September 25, 2024 at 2:05:18 AM Korean Standard Time.csv"]  # 센서 데이터가 저장된 CSV 파일 경로
    for file_path in file_paths:
        process_csv_data(file_path)