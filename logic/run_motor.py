# run_motor.py
import serial
import time
import sys
import json

# Configuration
SERIAL_PORT = '/dev/tty.usbmodem31201'  # Update with your specific port
BAUD_RATE = 115200

def connect_to_arduino(port=SERIAL_PORT):
    """Connect to Arduino."""
    try:
        arduino = serial.Serial(port=port, baudrate=BAUD_RATE, timeout=1)
        time.sleep(0.5)  # Reduced from 1.0 to 0.5 seconds
        arduino.flush()
        return arduino
    except Exception as e:
        print(f"Error connecting to Arduino: {e}")
        sys.exit(1)

def send_amplitude_sequence(pattern_data, frequency=10):
    """
    Wrapper function that handles Arduino connection and sends amplitude sequence.
    
    Args:
        pattern_data: Either a complete pattern dictionary or just the amplitudes list
        frequency: Playback frequency (Hz), defaults to 10
    """
    try:
        # Connect to Arduino
        arduino = connect_to_arduino()
        
        # Extract the pattern if it's a dictionary
        if isinstance(pattern_data, dict):
            if "patterns" in pattern_data:
                pattern_key = list(pattern_data["patterns"].keys())[0]
                amplitudes = pattern_data["patterns"][pattern_key]["pattern"]
            elif "pattern" in pattern_data:
                amplitudes = pattern_data["pattern"]
            else:
                amplitudes = pattern_data  # Assume it's already the list we need
        else:
            amplitudes = pattern_data  # Already a list of amplitudes
        
        # Calculate interval based on frequency
        interval = int(1000 / (10 if frequency not in [10, 20] else frequency))
        
        # Set the interval
        arduino.write(f"INTERVAL:{interval}\n".encode())
        arduino.readline()
        
        # Send amplitudes in chunks
        chunk_size = 20
        for i in range(0, len(amplitudes), chunk_size):
            chunk = amplitudes[i:i+chunk_size]
            data = "PLAY:" + ",".join(map(str, chunk)) + "\n"
            arduino.write(data.encode())
            arduino.readline()
        
        # Start playback
        arduino.write("START\n".encode())
        arduino.readline()
        
        # Wait for the FINISHED message
        start_time = time.time()
        while True:
            if arduino.in_waiting > 0:
                response = arduino.readline().decode('utf-8', errors='ignore').strip()
                if "FINISHED" in response:
                    print(f"Execution time: {time.time() - start_time:.2f} seconds")
                    break
            time.sleep(0.01)
            
        # Close the connection
        arduino.close()
        
    except Exception as e:
        print(f"Error in send_amplitude_sequence: {e}")

def run_motor(json_file, frequency=10):
    """Main function to run a pattern from a JSON file."""
    try:
        # Load pattern data
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Send the amplitude sequence
        send_amplitude_sequence(data, frequency)
        
    except Exception as e:
        print(f"Error in run_motor: {e}")
    
    sys.exit(0)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        json_file = sys.argv[1]
        frequency = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        run_motor(json_file, frequency)
    else:
        print("Usage: python run_motor.py <json_file> [frequency]")
        sys.exit(1)