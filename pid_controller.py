import time
from config import (
    PID_YAW_KP, PID_YAW_KI, PID_YAW_KD,
    PID_PITCH_KP, PID_PITCH_KI, PID_PITCH_KD,
    PID_DEADBAND, PID_DERIVATIVE_EMA_ALPHA,
    PID_INTEGRAL_WINDUP_LIMIT, PID_OUTPUT_LIMIT_YAW, PID_OUTPUT_LIMIT_PITCH,
)


class AxisPID:
    def __init__(self, kp, ki, kd, output_limit, name=""):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limit = output_limit
        self.name = name

        self.integral = 0.0
        self.last_error = 0.0
        self.last_derivative = 0.0
        self.last_time = None

    def reset(self):
        self.integral = 0.0
        self.last_error = 0.0
        self.last_derivative = 0.0
        self.last_time = None

    def update(self, setpoint, measurement, dt=None):
        now = time.time()
        if dt is None:
            if self.last_time is not None:
                dt = now - self.last_time
            else:
                dt = 0.01
        self.last_time = now

        error = setpoint - measurement

        if abs(error) < PID_DEADBAND:
            error = 0.0

        self.integral += error * dt
        self.integral = max(-PID_INTEGRAL_WINDUP_LIMIT, min(PID_INTEGRAL_WINDUP_LIMIT, self.integral))

        raw_derivative = (error - self.last_error) / dt if dt > 0 else 0.0
        self.last_derivative = PID_DERIVATIVE_EMA_ALPHA * raw_derivative + (1 - PID_DERIVATIVE_EMA_ALPHA) * self.last_derivative
        self.last_error = error

        output = self.kp * error + self.ki * self.integral + self.kd * self.last_derivative
        output = max(-self.output_limit, min(self.output_limit, output))

        return output


class DualAxisPID:
    def __init__(self):
        self.yaw = AxisPID(PID_YAW_KP, PID_YAW_KI, PID_YAW_KD, PID_OUTPUT_LIMIT_YAW, name="Yaw")
        self.pitch = AxisPID(PID_PITCH_KP, PID_PITCH_KI, PID_PITCH_KD, PID_OUTPUT_LIMIT_PITCH, name="Pitch")

    def reset(self):
        self.yaw.reset()
        self.pitch.reset()

    def update(self, target_x, target_y, frame_center_x, frame_center_y):
        error_x = target_x - frame_center_x
        error_y = target_y - frame_center_y

        yaw_output = self.yaw.update(0, -error_x)
        pitch_output = self.pitch.update(0, -error_y)

        return yaw_output, pitch_output
