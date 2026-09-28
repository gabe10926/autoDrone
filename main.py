import time
import cv2
import sys

from config import (
    CAMERA_SOURCE, CAMERA_WIDTH, CAMERA_HEIGHT,
    TARGET_CLASSES, FRAME_SKIP,
    SERVO_YAW_CENTER, SERVO_PITCH_CENTER,
    SERVO_YAW_RANGE, SERVO_PITCH_RANGE,
    BACKEND,
)
from vision_pipeline import YOLOVisionPipeline
from pid_controller import DualAxisPID
from motor_output import create_backend


def servo_pulse_from_pid(pid_output, center, range_val):
    ## PID output to servo microseconds
    pulse = center + int(pid_output * range_val)
    return max(center - range_val, min(center + range_val, pulse))


def run_autonomous():
    ## camera -> vision -> PID -> turret
    print("[Chiron] Initializing autonomous mode...")
    print(f"[Chiron] Camera source: {CAMERA_SOURCE}")
    print(f"[Chiron] Motor backend: {BACKEND}")
    print(f"[Chiron] Resolution: {CAMERA_WIDTH}×{CAMERA_HEIGHT}")

    cap = cv2.VideoCapture(CAMERA_SOURCE)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    if not cap.isOpened():
        print("[ERROR] Cannot open camera source. Aborting.")
        print("[HELP] Verify: USB camera connected, CSI camera enabled, or CAMERA_SOURCE correct")
        return False

    try:
        pipeline = YOLOVisionPipeline()
    except Exception as e:
        print(f"[ERROR] Vision pipeline failed: {e}")
        cap.release()
        return False

    try:
        backend = create_backend()
    except Exception as e:
        print(f"[ERROR] Motor backend failed: {e}")
        print("[HELP] For pigpio backend: sudo pigpiod")
        print("[HELP] For serial backend: verify microcontroller connected")
        pipeline.stop()
        cap.release()
        return False

    pid = DualAxisPID()
    pipeline.start()

    frame_count = 0
    no_target_frames = 0
    target_lost = False
    last_fps_report = time.time()
    frames_since_report = 0

    print("[Chiron] Autonomous mode started. Press Ctrl+C to stop.")
    print("[Chiron] Waiting for targets...")

    ## main loop

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Failed to read frame, retrying...")
                time.sleep(0.01)
                continue

            frame_count += 1
            frames_since_report += 1

            ## FPS report every 2sec
            now = time.time()
            if now - last_fps_report > 2.0:
                fps = frames_since_report / (now - last_fps_report)
                print(f"[STAT] FPS: {fps:.1f} | Frame: {frame_count} | No-target frames: {no_target_frames}")
                frames_since_report = 0
                last_fps_report = now

            result = None
            if frame_count % (FRAME_SKIP + 1) == 0:
                result = pipeline.process_frame(frame)

            target_detected = False
            if result and result["seg_boxes"]:
                best_box = None
                best_score = 0

                best_label = None
                for box, score, label in zip(result["seg_boxes"], result["seg_scores"], result["seg_labels"]):
                    if label in TARGET_CLASSES and score > best_score:
                        best_score = score
                        best_box = box
                        best_label = label

                if best_box is not None:
                    target_detected = True
                    no_target_frames = 0
                    if target_lost:
                        print(f"[TARGET] Reacquired target (class: {best_label}, conf: {best_score:.2f})")
                    target_lost = False

                    target_x = (best_box[0] + best_box[2]) / 2
                    target_y = (best_box[1] + best_box[3]) / 2
                    center_x = CAMERA_WIDTH // 2
                    center_y = CAMERA_HEIGHT // 2

                    yaw_out, pitch_out = pid.update(target_x, target_y, center_x, center_y)

                    yaw_pulse = servo_pulse_from_pid(yaw_out, SERVO_YAW_CENTER, SERVO_YAW_RANGE)
                    pitch_pulse = servo_pulse_from_pid(pitch_out, SERVO_PITCH_CENTER, SERVO_PITCH_RANGE)

                    try:
                        backend.set_servo(0, yaw_pulse)
                        backend.set_servo(1, pitch_pulse)
                        backend.set_ultrasonic(True)
                    except Exception as e:
                        print(f"[ERROR] Backend write failed: {e}")
                        print("[HELP] Check GPIO pins, pigpiod daemon, or serial connection")

            if not target_detected:
                no_target_frames += 1
                if no_target_frames > 10 and not target_lost:
                    target_lost = True
                    print("[TARGET] Target lost, homing turret...")
                    pid.reset()
                    try:
                        backend.set_servo(0, SERVO_YAW_CENTER)
                        backend.set_servo(1, SERVO_PITCH_CENTER)
                        backend.set_ultrasonic(False)
                    except Exception as e:
                        print(f"[WARN] Homing failed: {e}")

    except KeyboardInterrupt:
        print("\n[Chiron] Shutting down...")
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        pipeline.stop()
        try:
            backend.cleanup()
        except:
            pass
        cap.release()
        print("[Chiron] Done")
        return True


if __name__ == "__main__":
    success = run_autonomous()
    sys.exit(0 if success else 1)
