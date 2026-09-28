import time
from config import (
    BACKEND, SERIAL_PORT, SERIAL_BAUD,
    ULTRASONIC_PIN, ULTRASONIC_FREQUENCY, ULTRASONIC_DUTY_CYCLE
)


class DummyBackend:
    ## testing without hardware
    def set_servo(self, channel, pulse_width):
        pass

    def set_ultrasonic(self, enabled):
        pass

    def cleanup(self):
        pass


class PigpioBackend:
    ## GPIO PWM for Jetson Nano / RPi
    def __init__(self):
        try:
            import pigpio
        except ImportError:
            raise ImportError("pigpio not installed. Run: pip install pigpio")

        self.pigpio = pigpio
        self.pi = pigpio.pi()

        if not self.pi.connected:
            raise RuntimeError("pigpiod daemon not running. Run: sudo pigpiod")

        print("[PigpioBackend] Connected to pigpiod")

    def set_servo(self, channel, pulse_width):
        ## pulse width in microseconds (1000-2000)
        if channel == 0:
            self.pi.hardware_PWM(12, 50, pulse_width * 1000)
        elif channel == 1:
            self.pi.hardware_PWM(13, 50, pulse_width * 1000)
        else:
            raise ValueError(f"Invalid servo channel: {channel}")

    def set_ultrasonic(self, enabled):
        ## 35kHz output enable/disable
        if enabled:
            dutycycle = int(ULTRASONIC_DUTY_CYCLE * 10000)
            self.pi.hardware_PWM(ULTRASONIC_PIN, ULTRASONIC_FREQUENCY, dutycycle)
        else:
            self.pi.hardware_PWM(ULTRASONIC_PIN, ULTRASONIC_FREQUENCY, 0)

    def cleanup(self):
        ## stop all PWM
        try:
            self.pi.hardware_PWM(12, 50, 0)
            self.pi.hardware_PWM(13, 50, 0)
            self.pi.hardware_PWM(ULTRASONIC_PIN, ULTRASONIC_FREQUENCY, 0)
            self.pi.stop()
            print("[PigpioBackend] Cleaned up")
        except Exception as e:
            print(f"[WARNING] Cleanup error: {e}")


class SerialBackend:
    ## external microcontroller over serial
    def __init__(self):
        try:
            import serial
        except ImportError:
            raise ImportError("pyserial not installed. Run: pip install pyserial")

        self.serial = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=1)
        print(f"[SerialBackend] Connected to {SERIAL_PORT} @ {SERIAL_BAUD} baud")

    def set_servo(self, channel, pulse_width):
        ## S<ch><pulse>
        cmd = f"S{channel}{pulse_width}\n".encode()
        self.serial.write(cmd)

    def set_ultrasonic(self, enabled):
        ## U<0|1>
        cmd = f"U{1 if enabled else 0}\n".encode()
        self.serial.write(cmd)

    def cleanup(self):
        ## close connection
        try:
            self.serial.close()
            print("[SerialBackend] Closed")
        except Exception as e:
            print(f"[WARNING] Cleanup error: {e}")


def create_backend():
    ## dummy, pigpio, or serial
    if BACKEND == "dummy":
        return DummyBackend()
    elif BACKEND == "pigpio":
        return PigpioBackend()
    elif BACKEND == "serial":
        return SerialBackend()
    else:
        raise ValueError(f"Unknown backend: {BACKEND}. Choose: dummy, pigpio, serial")
