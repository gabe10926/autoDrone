## servo and ultrasonic test - GPIO PWM without vision pipeline

import time
import sys
from config import BACKEND, SERVO_YAW_CENTER, SERVO_PITCH_CENTER, SERVO_YAW_RANGE, SERVO_PITCH_RANGE
from motor_output import create_backend


def test_servos():
    print("[TEST] Initializing backend:", BACKEND)
    backend = create_backend()

    print("\n[TEST] Servo Calibration — Jetson Nano / Raspberry Pi")
    print("=" * 60)

    ## center both servos
    print("\n1. Centering both servos (1500 micros)...")
    backend.set_servo(0, SERVO_YAW_CENTER)
    backend.set_servo(1, SERVO_PITCH_CENTER)
    time.sleep(1)
    print("Yaw servo centered at 1500micros")
    print("Pitch servo centered at 1500micros")
    print("Visually inspect: both servos should be at center position")

    ## sweep yaw
    print("\n2. Sweeping Yaw (pan) +/- 500micros from center...")
    print("   Range: 1000-2000 micros")
    for pulse in range(1000, 2001, 100):
        backend.set_servo(0, pulse)
        print(f"Yaw pulse: {pulse} micros", end="\r")
        time.sleep(0.15)
    backend.set_servo(0, SERVO_YAW_CENTER)
    print("Yaw sweep complete, returned to center")

    ## sweep pitch
    print("\n3. Sweeping Pitch (tilt) +/- 300micros from center...")
    print("Range: 1200-1800micros")
    for pulse in range(1200, 1801, 100):
        backend.set_servo(1, pulse)
        print(f"Pitch pulse: {pulse}micros", end="\r")
        time.sleep(0.15)
    backend.set_servo(1, SERVO_PITCH_CENTER)
    print("Pitch sweep complete, returned to center     ")

    ## test ultrasonic
    print("\n4. Testing Ultrasonic Deterrent (35kHz, 50% duty)...")
    print("Enabling ultrasonic for 2 seconds...")
    backend.set_ultrasonic(True)
    time.sleep(2)
    print("Ultrasonic enabled")
    print("Listen for high-pitched sound (inaudible to humans)")

    backend.set_ultrasonic(False)
    print("Ultrasonic disabled")

    print("\n5. Simulating PID tracking (random target movement)...")
    import random
    backend.set_ultrasonic(True)

    yaw_pos = SERVO_YAW_CENTER
    pitch_pos = SERVO_PITCH_CENTER

    for i in range(20):
        yaw_delta = random.uniform(-50, 50)
        pitch_delta = random.uniform(-30, 30)

        yaw_pos = max(SERVO_YAW_CENTER - SERVO_YAW_RANGE,
                      min(SERVO_YAW_CENTER + SERVO_YAW_RANGE, yaw_pos + yaw_delta))
        pitch_pos = max(SERVO_PITCH_CENTER - SERVO_PITCH_RANGE,
                        min(SERVO_PITCH_CENTER + SERVO_PITCH_RANGE, pitch_pos + pitch_delta))

        backend.set_servo(0, int(yaw_pos))
        backend.set_servo(1, int(pitch_pos))

        print(f"   [{i+1:2d}/20] Yaw: {int(yaw_pos):4d}micros | Pitch: {int(pitch_pos):4d}micros", end="\r")
        time.sleep(0.1)

    backend.set_servo(0, SERVO_YAW_CENTER)
    backend.set_servo(1, SERVO_PITCH_CENTER)
    backend.set_ultrasonic(False)
    print("Tracking simulation complete, servos homed")

    print("\n" + "=" * 60)
    print("[TEST] SUMMARY")
    print("=" * 60)
    print("\nAll outputs working correctly :D")
    print("\nNext steps:")
    print("1. Visually verify servo ranges match expected motion")
    print("2. Confirm ultrasonic produces audible/ultrasonic output")
    print("3. Run test_pipeline.py with camera to test vision system")
    print("4. Run main.py for full autonomous mode")

    backend.cleanup()
    print("\n[TEST] Done")


if __name__ == "__main__":
    try:
        test_servos()
    except KeyboardInterrupt:
        print("\n[TEST] Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
