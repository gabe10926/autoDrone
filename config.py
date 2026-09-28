CAMERA_SOURCE = 1
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

import os
YOLO_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yolo11n-seg.onnx")
YOLO_CONFIDENCE_THRESHOLD = 0.5
YOLO_IOU_THRESHOLD = 0.45

SEGMENTATION_INPUT_SIZE = (640, 640)
POSE_INPUT_SIZE = (640, 640)

## PID gains
PID_YAW_KP = 0.02
PID_YAW_KI = 0.0001
PID_YAW_KD = 0.005

PID_PITCH_KP = 0.015
PID_PITCH_KI = 0.00008
PID_PITCH_KD = 0.003

PID_DEADBAND = 0.02
PID_DERIVATIVE_EMA_ALPHA = 0.3
PID_INTEGRAL_WINDUP_LIMIT = 50.0
PID_OUTPUT_LIMIT_YAW = 1.0
PID_OUTPUT_LIMIT_PITCH = 1.0

## servo pwm ranges (us)
SERVO_YAW_CHANNEL = 0
SERVO_PITCH_CHANNEL = 1
SERVO_YAW_CENTER = 1500
SERVO_PITCH_CENTER = 1500
SERVO_YAW_RANGE = 500
SERVO_PITCH_RANGE = 300

## 35kHz ultrasonic deterrent
ULTRASONIC_PIN = 18
ULTRASONIC_FREQUENCY = 35000
ULTRASONIC_DUTY_CYCLE = 50

## dummy, pigpio, or serial
BACKEND = "dummy"

SERIAL_PORT = "/dev/ttyACM0"
SERIAL_BAUD = 115200

SERVO_YAW_GPIO = 12
SERVO_PITCH_GPIO = 13

## pest targets + test spheres
TARGET_CLASSES = [
    "deer", "rabbit", "groundhog", "raccoon", "dog", "coyote", "bird", "person",
    "red_ball", "blue_ball", "yellow_ball",
]

## HSV ranges for colored sphere detection (local testing)
COLOR_TARGETS = {
    "red_ball": {
        "lower": (0, 100, 100),
        "upper": (10, 255, 255),
        "radius_min": 15,
        "radius_max": 150,
    },
    "blue_ball": {
        "lower": (100, 100, 100),
        "upper": (130, 255, 255),
        "radius_min": 15,
        "radius_max": 150,
    },
    "yellow_ball": {
        "lower": (20, 100, 100),
        "upper": (40, 255, 255),
        "radius_min": 10,
        "radius_max": 120,
    },
}

FRAME_SKIP = 2
