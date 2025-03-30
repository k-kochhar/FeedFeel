# server.py
import os
import json
import time
from datetime import datetime
from typing import Optional, Dict, List
import threading
import queue
import sys
import select
import argparse

# Import the fixed functions from your corrected files
from run_motor import send_amplitude_sequence
from run_servo import move_servos

class StepperServer:
    def __init__(self, stepper_data_dir: str = "stepper-data", mode: str = "continuous"):
        self.stepper_data_dir = stepper_data_dir
        self.processed_files = set()
        self.latest_timestamp = None
        self.processing_mode = mode  # "continuous" or "single"
        self.is_running = True
        
        # Ensure the stepper-data directory exists
        if not os.path.exists(stepper_data_dir):
            os.makedirs(stepper_data_dir)
            print(f"Created directory: {stepper_data_dir}")
    
    def get_latest_timestamp_file(self) -> Optional[str]:
        """Get the most recent timestamp JSON file that hasn't been processed."""
        try:
            # List all JSON files in the directory
            json_files = [f for f in os.listdir(self.stepper_data_dir) 
                         if f.endswith('.json') and f not in self.processed_files]
            
            if not json_files:
                return None
            
            # Sort by timestamp in filename (format: YYYYMMDD_HHMMSS.json)
            latest_file = max(json_files)
            return os.path.join(self.stepper_data_dir, latest_file)
            
        except Exception as e:
            print(f"Error getting latest timestamp file: {str(e)}")
            return None
    
    def process_summary_pattern(self, data: Dict) -> None:
        """Process the summary pattern and move servos to average position."""
        try:
            summary = data.get("summary", {})
            if not summary:
                print("No summary data found")
                return
            
            # Get servo angles
            avg_x_servo = summary.get("avg_x_servo", 90)  # Default to center
            avg_y_servo = summary.get("avg_y_servo", 90)  # Default to center
            
            # Move servos to average position
            print(f"Moving servos to position: x={avg_x_servo}, y={avg_y_servo}")
            move_servos(avg_x_servo, avg_y_servo)
            
            # Get and process the pattern file
            pattern_file = summary.get("pattern_file")
            if not pattern_file:
                print("No pattern file in summary, skipping")
                return
                
            pattern_path = os.path.join(self.stepper_data_dir, "objects", pattern_file)
            if not os.path.exists(pattern_path):
                print(f"Pattern file not found: {pattern_path}")
                return
                
            # Verify the pattern file is valid JSON
            try:
                with open(pattern_path, 'r') as f:
                    pattern_data = json.load(f)
                    # Verify pattern data has required fields
                    if not isinstance(pattern_data, dict):
                        print(f"Invalid pattern data format: not a dictionary")
                        return
                        
                    # Send the amplitude sequence
                    send_amplitude_sequence(pattern_data)
            except json.JSONDecodeError:
                print(f"Invalid JSON in pattern file: {pattern_path}")
                return
                
        except Exception as e:
            print(f"Error processing summary pattern: {str(e)}")
    
    def process_single_object(self, obj: Dict) -> None:
        """Process a single object's pattern and move servos to its position."""
        try:
            # Get servo angles
            servo_angles = obj.get("servo_angles", {})
            x_servo = servo_angles.get("x_servo", 90)
            y_servo = servo_angles.get("y_servo", 90)
            
            # Move servos to object position
            print(f"Moving servos to object position: x={x_servo}, y={y_servo}")
            move_servos(x_servo, y_servo)
            
            # Get and process the pattern file
            pattern_file = obj.get("pattern_file")
            if not pattern_file:
                print("No pattern file for object, skipping")
                return
                
            pattern_path = os.path.join(self.stepper_data_dir, "objects", pattern_file)
            if not os.path.exists(pattern_path):
                print(f"Pattern file not found: {pattern_path}")
                return
                
            # Verify the pattern file is valid JSON
            try:
                with open(pattern_path, 'r') as f:
                    pattern_data = json.load(f)
                    # Verify pattern data has required fields
                    if not isinstance(pattern_data, dict):
                        print(f"Invalid pattern data format: not a dictionary")
                        return
                        
                    # Send the amplitude sequence
                    send_amplitude_sequence(pattern_data)
            except json.JSONDecodeError:
                print(f"Invalid JSON in pattern file: {pattern_path}")
                return
                
        except Exception as e:
            print(f"Error processing single object: {str(e)}")
    
    def process_latest_timestamp(self) -> bool:
        """
        Process the latest timestamp file based on current mode.
        
        Returns:
            bool: True if processed successfully, False otherwise
        """
        latest_file = self.get_latest_timestamp_file()
        if not latest_file:
            print("No new files to process")
            return False
        
        try:
            with open(latest_file, 'r') as f:
                data = json.load(f)
            
            if self.processing_mode == "continuous":
                # Process summary pattern
                self.process_summary_pattern(data)
            else:  # single mode
                # Process each object in sequence
                objects = data.get("objects", [])
                if not objects:
                    print("No objects found in the latest file")
                else:
                    print(f"Processing {len(objects)} objects in single mode")
                    for obj in objects:
                        self.process_single_object(obj)
                        time.sleep(0.5)  # Small delay between objects
            
            # Mark file as processed
            self.processed_files.add(os.path.basename(latest_file))
            print(f"Processed file: {latest_file}")
            return True
            
        except Exception as e:
            print(f"Error processing timestamp file: {str(e)}")
            return False
    
    def check_for_input(self):
        """Non-blocking check for keyboard input"""
        # Check if there's data available to read
        rlist, _, _ = select.select([sys.stdin], [], [], 0)
        
        if rlist:
            # Read a line and strip newline
            key = sys.stdin.readline().strip().lower()
            if key == 'r':
                print("Switching to single processing mode")
                orig_mode = self.processing_mode
                self.processing_mode = "single"
                # Process latest timestamp in single mode
                self.process_latest_timestamp()
                # Switch back to original mode
                self.processing_mode = orig_mode
                print(f"Switched back to {orig_mode} mode")
            elif key == 'q':
                print("\nShutting down Stepper Server...")
                self.is_running = False
    
    def run_single(self):
        """Run in single mode - process the latest file once and exit."""
        print("Running in single mode - will process latest file and exit")
        
        # Process the latest timestamp
        result = self.process_latest_timestamp()
        
        if not result:
            print("No files were processed in single mode")
        else:
            print("Single mode processing complete")
    
    def run_continuous(self):
        """Run in continuous mode - continuously process files as they appear."""
        print("Running in continuous mode")
        print("Press 'r' and Enter to process in single mode")
        print("Press 'q' and Enter to quit")
        
        try:
            while self.is_running:
                # Check for keyboard input (non-blocking)
                self.check_for_input()
                
                # Process latest timestamp
                self.process_latest_timestamp()
                time.sleep(0.1)  # Small delay to prevent CPU overuse
                
        except KeyboardInterrupt:
            print("\nShutting down Stepper Server...")
            self.is_running = False
    
    def run(self):
        """Main entry point - runs either in single or continuous mode."""
        print(f"Starting Stepper Server in {self.processing_mode} mode...")
        
        if self.processing_mode == "single":
            self.run_single()
        else:
            self.run_continuous()

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Stepper Server for haptic feedback')
    parser.add_argument(
        '--mode', 
        type=str, 
        choices=['continuous', 'single'], 
        default='continuous',
        help='Processing mode: continuous (default) or single (run once and exit)'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='stepper-data',
        help='Directory containing timestamp data (default: stepper-data)'
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    server = StepperServer(stepper_data_dir=args.data_dir, mode=args.mode)
    server.run()