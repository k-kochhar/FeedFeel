# run_servo.py
import serial
import time
import sys

# Configuration
SERIAL_PORT = "/dev/tty.usbmodemC04E3011E6802"  # Update with your specific port
BAUD_RATE = 115200

def move_servos(angle1, angle2):
    """
    Send servo angle commands to Arduino.
    
    Args:
        angle1: Angle for the first servo (0-180 degrees)
        angle2: Angle for the second servo (0-180 degrees)
    """
    try:
        # Ensure values are within range
        angle1= max(70, min(250, angle1+70))
        angle2 = max(0, min(180, angle2))
        
        # Open connection
        arduino = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=2)
        
        # Wait for connection to establish
        time.sleep(0.25)
        
        # Clear any pending data
        arduino.reset_input_buffer()
        
        # Format and send command
        command = f"{angle1} {angle2}\n"
        arduino.write(command.encode())
        
        # Wait for and print response
        time.sleep(0.15)
        if arduino.in_waiting:
            response = arduino.readline().decode('utf-8').strip()
            print(f"Arduino: {response}")
        else:
            print("Command sent, no response received")
        
        # Close the connection
        arduino.close()
                
    except serial.SerialException as e:
        print(f"Error in move_servos: {e}")
    except Exception as e:
        print(f"Unexpected error in move_servos: {e}")

if __name__ == "__main__":
    try:
        if len(sys.argv) > 2:
            # Get angles from command line arguments
            angle1 = int(sys.argv[1])
            angle2 = int(sys.argv[2])
        else:
            # Get user input for angles
            angle1 = int(input("Enter angle for servo 1 (0-180): "))
            angle2 = int(input("Enter angle for servo 2 (0-180): "))
        
        # Move servos
        print(f"Moving servos to: {angle1}° and {angle2}°")
        move_servos(angle1, angle2)
        
    except ValueError:
        print("Error: Please enter valid numbers")
    except KeyboardInterrupt:
        print("\nProgram terminated by user")