import serial
import time

SERIAL_PORT = "/dev/cu.usbmodem21301"  # Update with correct port
BAUD_RATE = 115200

def move_servo(angle):
    """ Sends an angle command to the Arduino and waits for confirmation. """
    angle = max(0, min(180, angle))  # Ensure angle is within range

    try:
        with serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=2, dsrdtr=True, rtscts=False) as arduino:
            time.sleep(1)  # Allow serial to stabilize
            arduino.write(f"{angle}\n".encode())  # Send command
            response = arduino.readline().decode().strip()
            print("Arduino:", response)
    except serial.SerialException as e:
        print("Serial error:", e)

if __name__ == "__main__":
    target_angle = 70
    move_servo(target_angle)
