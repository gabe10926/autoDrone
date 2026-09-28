import cv2
import numpy as np

from config import (
    CAMERA_SOURCE, CAMERA_WIDTH, CAMERA_HEIGHT, COLOR_TARGETS,
    SERVO_YAW_CENTER, SERVO_PITCH_CENTER, SERVO_YAW_RANGE, SERVO_PITCH_RANGE,
)
from vision_pipeline import YOLOVisionPipeline
from pid_controller import DualAxisPID
from motor_output import create_backend
from main import servo_pulse_from_pid

## frames without a detection before the turret homes
LOST_AFTER = 5


## balls first, then highest score
def pick_target(result):
    best = None
    for box, score, label in zip(result["seg_boxes"], result["seg_scores"], result["seg_labels"]):
        rank = (label in COLOR_TARGETS, score)
        if best is None or rank > best[0]:
            best = (rank, box, label)
    return None if best is None else (best[1], best[2])


## overlay detections, target and servo state on frame
def draw_overlay(frame, result, target, pid, yaw_out, pitch_out, yaw_pulse, pitch_pulse):
    h, w = frame.shape[:2]
    center_x, center_y = w // 2, h // 2

    cv2.line(frame, (center_x - 20, center_y), (center_x + 20, center_y), (0, 255, 0), 1)
    cv2.line(frame, (center_x, center_y - 20), (center_x, center_y + 20), (0, 255, 0), 1)

    if result:
        for box, score, label in zip(result["seg_boxes"], result["seg_scores"], result["seg_labels"]):
            x1, y1, x2, y2 = box
            color = (0, 0, 255) if label in COLOR_TARGETS else (180, 180, 180)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} {score:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    if target:
        (x1, y1, x2, y2), label = target
        obj_cx, obj_cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(frame, (obj_cx, obj_cy), 5, (0, 255, 255), -1)
        cv2.line(frame, (center_x, center_y), (obj_cx, obj_cy), (255, 255, 0), 2)

    info_lines = [
        f"Target: {target[1]}" if target else "Target: none (turret homed)",
        f"PID out  yaw {yaw_out:+.3f}  pitch {pitch_out:+.3f}",
        f"Servo us yaw {yaw_pulse}  pitch {pitch_pulse}",
        f"PID Y: {pid.yaw.kp:.4f}/{pid.yaw.ki:.6f}/{pid.yaw.kd:.4f}",
        f"PID P: {pid.pitch.kp:.4f}/{pid.pitch.ki:.6f}/{pid.pitch.kd:.4f}",
    ]
    for i, line in enumerate(info_lines):
        cv2.putText(frame, line, (10, 25 + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, "[w/s]Kp [a/d]Ki [z/x]Kd yaw | [t/g]Kp [f/h]Ki [v/b]Kd pitch | [r]eset [q]uit",
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    return frame


def run_test():
    cap = cv2.VideoCapture(CAMERA_SOURCE)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera source {CAMERA_SOURCE}")
        return

    pipeline = YOLOVisionPipeline()
    pid = DualAxisPID()
    backend = create_backend()
    backend.set_ultrasonic(False)

    pipeline.start()
    print("[TestPipeline] Running. Press 'q' in the video window to quit.")

    result = None
    target = None
    misses = 0
    yaw_out = pitch_out = 0.0
    yaw_pulse, pitch_pulse = SERVO_YAW_CENTER, SERVO_PITCH_CENTER

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to grab frame")
            break

        ## PID + dummy servos only step when a fresh detection arrives
        new_result = pipeline.process_frame_async(frame)
        if new_result is not None:
            result = new_result
            picked = pick_target(result)
            if picked:
                target, misses = picked, 0
                (x1, y1, x2, y2), _ = picked
                h, w = frame.shape[:2]
                yaw_out, pitch_out = pid.update((x1 + x2) / 2, (y1 + y2) / 2, w // 2, h // 2)
                yaw_pulse = servo_pulse_from_pid(yaw_out, SERVO_YAW_CENTER, SERVO_YAW_RANGE)
                pitch_pulse = servo_pulse_from_pid(pitch_out, SERVO_PITCH_CENTER, SERVO_PITCH_RANGE)
                backend.set_servo(0, yaw_pulse)
                backend.set_servo(1, pitch_pulse)
            else:
                misses += 1
                if misses >= LOST_AFTER and target is not None:
                    target = None
                    pid.reset()
                    yaw_out = pitch_out = 0.0
                    yaw_pulse, pitch_pulse = SERVO_YAW_CENTER, SERVO_PITCH_CENTER
                    backend.set_servo(0, yaw_pulse)
                    backend.set_servo(1, pitch_pulse)

        display = draw_overlay(frame.copy(), result, target, pid, yaw_out, pitch_out, yaw_pulse, pitch_pulse)
        cv2.imshow("Chiron - Test Pipeline", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            pid.reset()
            print("[PID] Reset")
        elif key == ord('w'):
            pid.yaw.kp += 0.001
            print(f"[PID] Yaw KP={pid.yaw.kp:.4f}")
        elif key == ord('s'):
            pid.yaw.kp = max(0, pid.yaw.kp - 0.001)
            print(f"[PID] Yaw KP={pid.yaw.kp:.4f}")
        elif key == ord('a'):
            pid.yaw.ki += 0.00001
            print(f"[PID] Yaw KI={pid.yaw.ki:.6f}")
        elif key == ord('d'):
            pid.yaw.ki = max(0, pid.yaw.ki - 0.00001)
            print(f"[PID] Yaw KI={pid.yaw.ki:.6f}")
        elif key == ord('z'):
            pid.yaw.kd += 0.001
            print(f"[PID] Yaw KD={pid.yaw.kd:.4f}")
        elif key == ord('x'):
            pid.yaw.kd = max(0, pid.yaw.kd - 0.001)
            print(f"[PID] Yaw KD={pid.yaw.kd:.4f}")
        elif key == ord('t'):
            pid.pitch.kp += 0.001
            print(f"[PID] Pitch KP={pid.pitch.kp:.4f}")
        elif key == ord('g'):
            pid.pitch.kp = max(0, pid.pitch.kp - 0.001)
            print(f"[PID] Pitch KP={pid.pitch.kp:.4f}")
        elif key == ord('f'):
            pid.pitch.ki += 0.00001
            print(f"[PID] Pitch KI={pid.pitch.ki:.6f}")
        elif key == ord('h'):
            pid.pitch.ki = max(0, pid.pitch.ki - 0.00001)
            print(f"[PID] Pitch KI={pid.pitch.ki:.6f}")
        elif key == ord('v'):
            pid.pitch.kd += 0.001
            print(f"[PID] Pitch KD={pid.pitch.kd:.4f}")
        elif key == ord('b'):
            pid.pitch.kd = max(0, pid.pitch.kd - 0.001)
            print(f"[PID] Pitch KD={pid.pitch.kd:.4f}")

    pipeline.stop()
    backend.cleanup()
    cap.release()
    cv2.destroyAllWindows()
    print("[TestPipeline] Done")


if __name__ == "__main__":
    run_test()
