import serial
import time
import signal
import sys
import json
import os
import glob

# Configuration
SERIAL_PORT = '/dev/cu.usbmodem21201'  # Specific Arduino port
BAUD_RATE = 115200 
PLAY_INTERVAL = 100

def find_arduino_port():
    """Find the Arduino port automatically."""
    if sys.platform.startswith('darwin'):  # macOS
        ports = glob.glob('/dev/cu.usbmodem*') + glob.glob('/dev/tty.usbmodem*')
    elif sys.platform.startswith('linux'):  # Linux
        ports = glob.glob('/dev/ttyACM*') + glob.glob('/dev/ttyUSB*')
    else:  # Windows
        ports = ['COM%s' % (i + 1) for i in range(256)]
    
    for port in ports:
        try:
            test_serial = serial.Serial(port=port, baudrate=BAUD_RATE, timeout=0.1)
            test_serial.close()
            return port
        except:
            continue
    
    raise Exception("No Arduino found. Please check the connection.")

def connect_to_arduino():
    """Connect to Arduino with improved error handling."""
    # Try to clean up any existing connections
    try:
        cleanup_port = serial.Serial(SERIAL_PORT)
        cleanup_port.close()
    except:
        pass
    
    # Wait a moment for the port to be released
    time.sleep(1)
    
    try:
        arduino = serial.Serial(port=SERIAL_PORT, baudrate=BAUD_RATE, timeout=2)
        print(f"Connecting to Arduino on port {SERIAL_PORT}...")
        time.sleep(2)  # Allow time for Arduino to reset
        arduino.flush()
        
        # Wait for Arduino ready signal
        ready = False
        timeout = time.time() + 5  # 5 second timeout
        
        while not ready and time.time() < timeout:
            if arduino.in_waiting > 0:
                response = arduino.readline().decode('utf-8').strip()
                if response == "READY":
                    print("Arduino connected and ready!")
                    ready = True
            time.sleep(0.1)
        
        if not ready:
            raise Exception("Arduino did not respond with READY signal")
            
        return arduino
    except Exception as e:
        print(f"Error connecting to Arduino on port {SERIAL_PORT}: {e}")
        print("\nTroubleshooting steps:")
        print(f"1. Make sure Arduino is connected to {SERIAL_PORT}")
        print("2. Close Arduino IDE Serial Monitor")
        print("3. Unplug and replug the Arduino")
        print("4. Try uploading the Arduino code again")
        print("\nAvailable ports:")
        for port in list_ports():
            print(f"- {port}")
        sys.exit(1)

def list_ports():
    """List all available serial ports."""
    import serial.tools.list_ports
    return [port.device for port in serial.tools.list_ports.comports()]

def send_amplitude_sequence(arduino, amplitudes, interval=PLAY_INTERVAL, servo_angles=None, motor_info=None):
    """
    Send a sequence of amplitudes to the Arduino.
    
    Args:
        arduino: Serial connection to Arduino
        amplitudes: List of amplitude values
        interval: Time in ms between playing each amplitude
        servo_angles: Dictionary of servo angle settings
        motor_info: Dictionary of motor information
    """
    # FORCE total execution time to exactly 3 seconds
    target_duration = 3000  # 3 seconds in ms
    
    # Set the execution duration first (tells Arduino to run everything in exactly 3 seconds)
    arduino.write(f"DURATION:{target_duration}\n".encode())
    response = arduino.readline().decode('utf-8').strip()
    print(f"Set total duration: {response}")
    
    # Set the playback interval
    arduino.write(f"INTERVAL:{interval}\n".encode())
    response = arduino.readline().decode('utf-8').strip()
    print(f"Set interval: {response}")
    
    # Send servo angles if provided
    if servo_angles:
        servo_str = ",".join(f"{k}:{v}" for k, v in servo_angles.items())
        arduino.write(f"SERVO:{servo_str}\n".encode())
        response = arduino.readline().decode('utf-8').strip()
        print(f"Set servo angles: {response}")
    
    # Send motor info if provided
    if motor_info:
        motor_str = ",".join(f"{k}:{v}" for k, v in motor_info.items() if not isinstance(v, str))
        arduino.write(f"MOTOR:{motor_str}\n".encode())
        response = arduino.readline().decode('utf-8').strip()
        print(f"Set motor info: {response}")
    
    # Send total samples count to help Arduino manage timing
    arduino.write(f"SAMPLES:{len(amplitudes)}\n".encode())
    response = arduino.readline().decode('utf-8').strip()
    print(f"Set samples count: {response}")
    
    # Prepare the sequence data
    # Limit to chunks of 10 amplitudes at a time to avoid buffer issues (values are larger)
    chunk_size = 10
    for i in range(0, len(amplitudes), chunk_size):
        chunk = amplitudes[i:i+chunk_size]
        data = "PLAY:" + ",".join(map(str, chunk)) + "\n"
        
        # Send to Arduino
        arduino.write(data.encode())
        
        # Wait for acknowledgment
        response = arduino.readline().decode('utf-8').strip()
        print(f"Arduino response: {response}")
    
    # Send START command to begin execution of the entire sequence
    arduino.write("START\n".encode())
    response = arduino.readline().decode('utf-8').strip()
    print(f"Starting execution: {response}")
    
    # Monitor playback - now we're measuring the actual runtime
    start_time = time.time()
    
    # Monitor playback
    finished = False
    while not finished:
        if arduino.in_waiting > 0:
            response = arduino.readline().decode('utf-8').strip()
            elapsed = time.time() - start_time
            print(f"Status: {response} (Elapsed: {elapsed:.2f}s)")
            if response == "FINISHED":
                finished = True
        time.sleep(0.01)
    
    # Calculate total runtime
    total_runtime = time.time() - start_time
    print(f"Total execution time: {total_runtime:.3f} seconds")
    
    # Verify if execution was within acceptable limits (2.9-3.1 seconds)
    if total_runtime < 2.9 or total_runtime > 3.1:
        print(f"WARNING: Execution time ({total_runtime:.3f}s) differs from target (3.0s)")
    else:
        print(f"SUCCESS: Execution completed within target time window")

def execute_pattern_from_json(json_file_path):
    """
    Execute a haptic pattern from a JSON file.
    
    Args:
        json_file_path (str): Path to the JSON file containing pattern data
    """
    try:
        with open(json_file_path, 'r') as f:
            pattern_data = json.load(f)
        
        # Connect to Arduino
        arduino = connect_to_arduino()
        
        try:
            # Process each pattern in the data
            for pattern_key, pattern_info in pattern_data["patterns"].items():
                print(f"\nProcessing pattern: {pattern_key}")
                
                # Extract pattern parameters
                sample_rate = pattern_info["sample_rate"]
                total_samples = pattern_info["total_samples"]
                pattern = pattern_info["pattern"]
                duration_seconds = pattern_data["duration_seconds"]
                
                # Calculate theoretical interval (Arduino will handle precise timing)
                interval = int((duration_seconds * 1000) / total_samples)
                
                print(f"Pattern parameters:")
                print(f"- Sample rate: {sample_rate}Hz")
                print(f"- Total samples: {total_samples}")
                print(f"- Duration: {duration_seconds} seconds")
                print(f"- Calculated interval: {interval}ms (theoretical)")
                
                # Extract optional parameters
                servo_angles = pattern_data.get("servo_angles")
                motor_info = pattern_data.get("motor_info")
                
                if servo_angles:
                    print(f"Servo angles: {servo_angles}")
                if motor_info:
                    print(f"Motor info: {motor_info}")
                
                # Send the pattern to Arduino
                print(f"\nSending pattern to Arduino...")
                print(f"Arduino will execute ENTIRE SEQUENCE in EXACTLY {duration_seconds} seconds")
                send_amplitude_sequence(arduino, pattern, interval, servo_angles, motor_info)
                print("Pattern execution complete!")
                
        finally:
            arduino.close()
            print("Arduino connection closed")
            
    except Exception as e:
        print(f"Error executing pattern: {e}")
        raise

def main():
    """
    Main function to demonstrate pattern execution from a JSON file.
    """
    if len(sys.argv) > 1:
        json_file = sys.argv[1]
        execute_pattern_from_json(json_file)
    else:
        # Example pattern data for testing
        pattern_data = {
            "sentence": "bottle",
            "embedding_length": 1536,
            "pooled_embedding_length": 256,
            "duration_seconds": 3.0,
            "patterns": {
                "100hz": {
                    "sample_rate": 100,
                    "total_samples": 300,
                    "pattern": [
                        1851, 802, 503, 343, 506, 651, 1610
                        # ... rest of the pattern ...
                    ]
                }
            },
            "servo_angles": {
                "x_servo": 98,
                "y_servo": 139,
                "height": 0
            },
            "motor_info": {
                "type": "28BYJ-48 stepper motor",
                "min_speed": 0,
                "max_speed": 2048
            }
        }
        
        # Save example data to a temporary file and execute
        with open('temp_pattern.json', 'w') as f:
            json.dump(pattern_data, f, indent=2)
        
        execute_pattern_from_json('temp_pattern.json')
        
        # Clean up temporary file
        try:
            os.remove('temp_pattern.json')
        except:
            pass

# Function to load pattern data from a JSON file
def load_pattern_data_from_json(json_file):
    """
    Load pattern data from a JSON file.
    """
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Extract all relevant data
        pattern_key = list(data["patterns"].keys())[0]  # Get first pattern key
        pattern = data["patterns"][pattern_key]["pattern"]
        servo_angles = data.get("servo_angles", None)
        motor_info = data.get("motor_info", None)
        
        return pattern, servo_angles, motor_info
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return [], None, None

if __name__ == "__main__":
    main()
