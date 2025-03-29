import serial
import time
import signal
import sys

# Configuration
SERIAL_PORT = '/dev/cu.usbmodem21201'  # Change to your Arduino port (Windows: COM3, etc.)
BAUD_RATE = 9600
PLAY_INTERVAL = 100  # Milliseconds between amplitude values

def connect_to_arduino():
    """Connect to the Arduino board."""
    try:
        arduino = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=2)
        print("Connecting to Arduino...")
        time.sleep(2)  # Allow time for Arduino to reset
        
        # Wait for Arduino ready signal
        ready = False
        while not ready:
            if arduino.in_waiting > 0:
                response = arduino.readline().decode('utf-8').strip()
                if response == "READY":
                    print("Arduino connected and ready!")
                    ready = True
        return arduino
    except Exception as e:
        print(f"Error connecting to Arduino: {e}")
        sys.exit(1)

def send_amplitude_sequence(arduino, amplitudes, interval=PLAY_INTERVAL):
    """
    Send a sequence of amplitudes to the Arduino.
    
    Args:
        arduino: Serial connection to Arduino
        amplitudes: List of amplitude values
        interval: Time in ms between playing each amplitude
    """
    # Set the playback interval
    arduino.write(f"INTERVAL:{interval}\n".encode())
    response = arduino.readline().decode('utf-8').strip()
    print(f"Set interval: {response}")
    
    # Prepare the sequence data
    # Limit to chunks of 20 amplitudes at a time to avoid buffer issues
    chunk_size = 20
    for i in range(0, len(amplitudes), chunk_size):
        chunk = amplitudes[i:i+chunk_size]
        data = "PLAY:" + ",".join(map(str, chunk)) + "\n"
        
        # Send to Arduino
        arduino.write(data.encode())
        
        # Wait for acknowledgment
        response = arduino.readline().decode('utf-8').strip()
        print(f"Arduino response: {response}")
        
        # Monitor playback
        while True:
            if arduino.in_waiting > 0:
                response = arduino.readline().decode('utf-8').strip()
                print(f"Status: {response}")
                if response == "FINISHED":
                    break
            time.sleep(0.01)

def main():
    # Example amplitude sequence - replace with your own data
    # For demo purposes - this creates a simple sine-like pattern
    # Replace this with your actual amplitude data from your script
    sample_amplitudes = [10, 20, 30, 40, 50, 40, 30, 20, 10, 0, -10, -20, -30, -40, -50, -40, -30, -20, -10]
    
    # Connect to Arduino
    arduino = connect_to_arduino()
    
    try:
        # Send the amplitude sequence
        print("Sending amplitude sequence...")
        send_amplitude_sequence(arduino, sample_amplitudes)
        
        print("Sequence playback complete!")
        
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        arduino.close()
        print("Connection closed")

# Function to load amplitudes from your existing Python script
def load_amplitudes_from_script():
    """
    Load amplitudes from your existing script.
    Modify this function to interface with your script.
    """
    # Option 1: Import and call functions from your script
    # from your_script import get_amplitudes
    # return get_amplitudes()
    
    # Option 2: Read from a file that your script generates
    # with open('amplitudes.txt', 'r') as f:
    #     return [int(line.strip()) for line in f.readlines()]
    
    # Placeholder - replace with actual implementation
    return [10, 20, 30, 40, 50, 40, 30, 20, 10, 0, -10, -20, -30, -40, -50, -40, -30, -20, -10]

if __name__ == "__main__":
    main()