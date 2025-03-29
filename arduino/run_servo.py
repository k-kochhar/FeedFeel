import serial
import time
import sys
import glob

# Update this with the correct port name on your system (e.g. COM3 on Windows)
SERIAL_PORT = '/dev/cu.usbmodemC04E3011E6802'
BAUD_RATE = 115200

# Offset angle to add to user-supplied angle, if desired
ANGLE_OFFSET = 45

def cleanup_port():
    """
    Attempt to clean up the serial port before connecting
    """
    try:
        temp_serial = serial.Serial(SERIAL_PORT)
        temp_serial.close()
    except:
        pass
    time.sleep(1)  # Give the port time to release

def connect_to_arduino():
    """
    Connects to the Arduino over the specified SERIAL_PORT at BAUD_RATE.
    Includes retry logic and proper port cleanup.
    """
    # First cleanup any existing connections
    cleanup_port()
    
    max_attempts = 5  # Increased attempts
    attempt = 0
    
    while attempt < max_attempts:
        try:
            print(f"Connecting to Arduino on {SERIAL_PORT} at {BAUD_RATE} baud (attempt {attempt + 1}/{max_attempts})...")
            
            # Open the port
            arduino = serial.Serial(port=SERIAL_PORT, baudrate=BAUD_RATE, timeout=2)
            
            # DTR pulse to reset Arduino
            arduino.setDTR(False)
            time.sleep(0.1)
            arduino.setDTR(True)
            
            # Clear buffers
            arduino.reset_input_buffer()
            arduino.reset_output_buffer()
            
            # Give Arduino time to reset and send READY
            time.sleep(2)
            
            # Look for READY signal
            ready_count = 0
            timeout = time.time() + 10  # 10 second timeout
            
            while time.time() < timeout and ready_count < 3:
                if arduino.in_waiting > 0:
                    response = arduino.readline().decode('utf-8', errors='ignore').strip()
                    print(f"Received: {response}")
                    if response == "READY":
                        ready_count += 1
                        if ready_count >= 3:
                            print("Arduino connected and ready!")
                            return arduino
                time.sleep(0.1)
            
            print("Timeout waiting for READY signals")
            arduino.close()
            
        except serial.SerialException as e:
            print(f"Connection attempt {attempt + 1} failed: {e}")
            time.sleep(2)  # Wait before retrying
            
        attempt += 1
    
    print("\nFailed to connect to Arduino. Please check:")
    print(f"1. Arduino is connected to {SERIAL_PORT}")
    print("2. Arduino IDE's Serial Monitor is closed")
    print("3. No other program is using the port")
    print("4. The correct Arduino sketch is uploaded")
    print("\nTry these steps:")
    print("1. Unplug and replug the Arduino")
    print("2. Close Arduino IDE completely")
    print("3. Wait 10 seconds")
    print("4. Try running this script again")
    sys.exit(1)

def move_servo_to_angle(arduino, angle):
    """
    Moves the servo to the specified angle plus an offset.
    Clamps the resulting angle between 0 and 180, sends to Arduino,
    and waits for a confirmation response.
    """
    # Compute offset angle, then clamp
    adjusted_angle = angle + ANGLE_OFFSET
    adjusted_angle = max(0, min(180, adjusted_angle))

    print(f"Original angle: {angle}°")
    print(f"Adjusted angle (with {ANGLE_OFFSET}° offset): {adjusted_angle}°")

    # Clear any pending data
    arduino.reset_input_buffer()
    
    # Send command
    command = f"{int(adjusted_angle)}\n"
    arduino.write(command.encode('utf-8'))
    arduino.flush()

    # Wait for Arduino to respond or time out after 2 seconds
    start_time = time.time()
    while (time.time() - start_time) < 2:
        if arduino.in_waiting > 0:
            response = arduino.readline().decode('utf-8', errors='ignore').strip()
            print(f"Arduino response: {response}")
            return response

    print("No response from Arduino after sending command.")

def main():
    # Expect exactly one argument: the angle
    if len(sys.argv) != 2:
        print("Usage: python run_servo.py <angle>")
        print("Example: python run_servo.py 90")
        sys.exit(1)

    # Parse and validate angle
    try:
        target_angle = float(sys.argv[1])
    except ValueError:
        print("Error: Please provide a valid numeric angle.")
        sys.exit(1)

    if not 0 <= target_angle <= 180:
        print("Error: Angle must be between 0 and 180 degrees.")
        sys.exit(1)

    # Connect to the Arduino
    arduino = connect_to_arduino()

    # Send the servo command
    try:
        print(f"\nMoving servo to {target_angle}° (plus offset of {ANGLE_OFFSET}°)...")
        move_servo_to_angle(arduino, target_angle)
    except KeyboardInterrupt:
        print("\nCancelled by user.")
    finally:
        # Always close the serial connection
        arduino.close()
        print("Connection closed.")

if __name__ == "__main__":
    main()
