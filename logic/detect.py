import cv2
from ultralytics import YOLO
import torch
import time
import json
import os
from datetime import datetime
import numpy as np
import threading
import concurrent.futures
import hashlib

# Import vibration generator function (only importing the functions we need)
from get_vib import (
    process_sentence_to_stepper,
    ensure_directory_exists,
    get_sentence_embedding,
    average_pool_embedding,
    sonify_embeddings,
    create_stepper_patterns,
    clean_filename
)

# Check if MPS is available (Apple Silicon GPU acceleration)
def check_mps():
    if torch.backends.mps.is_available():
        print("MPS (Metal Performance Shaders) is available! Using GPU acceleration.")
        return torch.device("mps")
    elif torch.backends.mps.is_built():
        print("MPS is built but not available. Check your MacOS version (needs 12.3+)")
        return torch.device("cpu")
    else:
        print("MPS not available. Using CPU instead.")
        return torch.device("cpu")

# Initialize webcam or virtual camera
def initialize_camera(camera_id=0):
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_id}")
        print("Available cameras:")
        # List available cameras (may vary by system)
        for i in range(8):
            temp_cap = cv2.VideoCapture(i)
            if temp_cap.isOpened():
                print(f"  Camera ID {i} is available")
                temp_cap.release()
        return None
    return cap

def ensure_data_directory():
    """
    Create the stepper-data directory if it doesn't exist.
    """
    if not os.path.exists("stepper-data"):
        os.makedirs("stepper-data")
        print("Created 'stepper-data' directory")

def calculate_object_distance(obj1, obj2):
    """
    Calculate the distance between two objects based on their centers.
    """
    # Calculate center of first object
    x1_1, y1_1, x2_1, y2_1 = obj1['bbox']
    center_x1 = (x1_1 + x2_1) / 2
    center_y1 = (y1_1 + y2_1) / 2
    
    # Calculate center of second object
    x1_2, y1_2, x2_2, y2_2 = obj2['bbox']
    center_x2 = (x1_2 + x2_2) / 2
    center_y2 = (y1_2 + y2_2) / 2
    
    # Calculate Euclidean distance
    distance = np.sqrt((center_x2 - center_x1)**2 + (center_y2 - center_y1)**2)
    return distance

def is_new_or_moved_object(obj, previous_objects, movement_threshold=50):
    """
    Check if an object is new or has moved significantly.
    """
    # If no previous objects, this object is new
    if not previous_objects:
        return True
        
    # Check for matching objects by class name
    matching_objects = [
        prev_obj for prev_obj in previous_objects
        if prev_obj['class_name'] == obj['class_name']
    ]
    
    # If no matching objects by class, this is a new object
    if not matching_objects:
        return True
    
    # Check distance to all matching objects
    for prev_obj in matching_objects:
        distance = calculate_object_distance(obj, prev_obj)
        # If any matching object is close enough, this is not new or moved
        if distance < movement_threshold:
            return False
    
    # If all matching objects are far, this has moved significantly
    return True

def get_position_description(x_servo, y_servo, frame_width, frame_height):
    """
    Get a textual description of an object's position.
    """
    # X position description
    if x_servo < 60:
        x_desc = "on the left"
    elif x_servo > 120:
        x_desc = "on the right"
    else:
        x_desc = "in the center"
    
    # Y position description
    if y_servo < 60:
        y_desc = "top"
    elif y_servo > 120:
        y_desc = "bottom"
    else:
        y_desc = "middle"
    
    return f"{x_desc} {y_desc}"

def check_existing_pattern(object_name):
    """
    Check if a pattern file already exists for this object.
    
    Parameters:
    - object_name: Name of the object
    
    Returns:
    - Path to existing file or None if not found
    """
    # Ensure the objects directory exists inside stepper-data
    objects_dir = os.path.join("stepper-data", "objects")
    ensure_directory_exists(objects_dir)
    
    # Create safe filename based on object name
    safe_name = clean_filename(object_name)
    
    # Check if pattern file exists
    pattern_file = os.path.join(objects_dir, f"{safe_name}_stepper.json")
    if os.path.exists(pattern_file):
        print(f"Found existing pattern for {object_name}")
        return pattern_file
    
    return None

def process_single_detection(detection_info):
    """
    Process a single detection to generate stepper patterns.
    This function runs in its own thread.
    
    Parameters:
    - detection_info: Dictionary with detection information
    
    Returns:
    - Result dictionary with stepper patterns
    """
    class_name = detection_info['class_name']
    # Convert X position to servo angle (0-180)
    x_servo = detection_info['x_servo']
    # Convert Y position to servo angle (0-180)
    y_servo = detection_info['y_servo']
    confidence = detection_info['confidence']
    
    try:
        # First check if pattern already exists
        existing_pattern = check_existing_pattern(class_name)
        if existing_pattern:
            print(f"Using existing pattern for {class_name}")
            
            # Load existing pattern
            with open(existing_pattern, 'r') as f:
                pattern_data = json.load(f)
            
            # Return result with pattern file and position info
            return {
                'class_name': class_name,
                'confidence': confidence,
                'bbox': detection_info['bbox'],
                'servo_angles': {
                    'x_servo': x_servo,
                    'y_servo': y_servo
                },
                'pattern_file': existing_pattern,
                'is_cached': True
            }
        
        # No existing pattern, generate a new one
        print(f"Generating new stepper pattern for {class_name} (confidence: {confidence:.2f})")
        
        # Process the sentence to generate stepper patterns
        objects_dir = os.path.join("stepper-data", "objects")
        ensure_directory_exists(objects_dir)
        
        result = process_sentence_to_stepper(
            sentence=class_name,
            x_servo=x_servo,
            y_servo=y_servo,
            height=0,
            output_dir=objects_dir,
            duration=3.0
        )
        
        # Add original detection info to the result
        result['class_name'] = class_name
        result['confidence'] = confidence
        result['bbox'] = detection_info['bbox']
        result['is_cached'] = False
        
        print(f"Successfully generated stepper pattern for {class_name}")
        print(f"  Servo angles: x={x_servo}, y={y_servo}")
        
        return result
    
    except Exception as e:
        print(f"Error generating stepper pattern for {class_name}: {str(e)}")
        return {
            'error': str(e),
            'class_name': class_name,
            'servo_angles': {
                'x_servo': x_servo,
                'y_servo': y_servo
            }
        }

def create_summary_sentence(objects, frame_width, frame_height):
    """
    Create a summary sentence describing all objects in the scene.
    """
    if not objects:
        return None, None, None
    
    # Count objects by class
    object_counts = {}
    for obj in objects:
        class_name = obj['class_name']
        if class_name in object_counts:
            object_counts[class_name] += 1
        else:
            object_counts[class_name] = 1
    
    # Create description parts
    description_parts = []
    for class_name, count in object_counts.items():
        # Get all objects of this class
        class_objects = [obj for obj in objects if obj['class_name'] == class_name]
        
        # Determine average position for this class
        avg_x_servo = sum(obj['x_servo'] for obj in class_objects) / len(class_objects)
        avg_y_servo = sum(obj['y_servo'] for obj in class_objects) / len(class_objects)
        
        # Get position description
        position = get_position_description(avg_x_servo, avg_y_servo, frame_width, frame_height)
        
        # Create description part
        if count == 1:
            description_parts.append(f"a {class_name} {position}")
        else:
            description_parts.append(f"{count} {class_name}s {position}")
    
    # Create summary sentence
    if len(description_parts) == 1:
        summary = f"I see {description_parts[0]}."
    elif len(description_parts) == 2:
        summary = f"I see {description_parts[0]} and {description_parts[1]}."
    else:
        summary = "I see " + ", ".join(description_parts[:-1]) + f", and {description_parts[-1]}."
    
    # Calculate average servo angles across all objects
    avg_x_servo = sum(obj['x_servo'] for obj in objects) / len(objects)
    avg_y_servo = sum(obj['y_servo'] for obj in objects) / len(objects)
    
    return summary, avg_x_servo, avg_y_servo

def process_detections_for_vibration(detections, frame_width, frame_height, model, timestamp, previous_objects=None):
    """
    Process detections to generate stepper patterns.
    Processes all objects in the current frame and generates a summary sentence.
    Saves individual patterns in stepper-data/objects/ folder and timestamp records in stepper-data/.
    """
    # Get the number of detections
    num_detections = len(detections.xyxy)
    current_objects = []
    
    # If no detections, return early
    if num_detections == 0:
        print("No objects detected.")
        
        # Create empty result file
        output_data = {
            "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "num_objects": 0,
            "objects": [],
            "summary": None
        }
        
        # Save to JSON with timestamp in stepper-data directory
        timestamp_str = timestamp.strftime('%Y%m%d_%H%M%S')
        json_filename = os.path.join("stepper-data", f"{timestamp_str}.json")
        with open(json_filename, 'w') as f:
            json.dump(output_data, f, indent=2)
            
        print(f"Saved empty detection record to {json_filename}")
        return {"filename": json_filename, "objects": []}
    
    # Process each detection
    for i in range(num_detections):
        class_id = int(detections.cls[i].item())
        class_name = model.names[class_id]  # Get the class name from model
        confidence = float(detections.conf[i].item())
        
        # Get bounding box coordinates
        x1 = float(detections.xyxy[i][0].item())
        y1 = float(detections.xyxy[i][1].item())
        x2 = float(detections.xyxy[i][2].item())
        y2 = float(detections.xyxy[i][3].item())
        
        # Calculate center point of the object
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # Calculate servo angles (0-180)
        # X position: 0 degrees (far left) to 180 degrees (far right)
        x_servo = int(180 * (center_x / frame_width))
        
        # Y position: 0 degrees (top) to 180 degrees (bottom)
        y_servo = int(180 * (center_y / frame_height))
        
        # Create detection info dict
        detection_info = {
            'class_name': class_name,
            'confidence': confidence,
            'x_servo': x_servo,
            'y_servo': y_servo,
            'bbox': (x1, y1, x2, y2)
        }
        
        current_objects.append(detection_info)
    
    # Create summary sentence and get average servo angles
    summary_sentence, avg_x_servo, avg_y_servo = create_summary_sentence(
        current_objects, frame_width, frame_height
    )
    
    # Process all objects in parallel and wait for completion
    individual_results = []
    if current_objects:
        print(f"Processing {len(current_objects)} objects in parallel...")
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(current_objects), 8)) as executor:
            # Submit all tasks
            future_to_object = {
                executor.submit(process_single_detection, obj): obj 
                for obj in current_objects
            }
            
            # Gather results as they complete
            for future in concurrent.futures.as_completed(future_to_object):
                obj = future_to_object[future]
                try:
                    result = future.result()
                    # Only add results that have valid pattern files
                    if 'pattern_file' in result and result['pattern_file']:
                        individual_results.append(result)
                    else:
                        print(f"Warning: No pattern file generated for {obj['class_name']}")
                except Exception as e:
                    print(f"Exception processing {obj['class_name']}: {str(e)}")
        
        # Calculate and print total processing time
        elapsed_time = time.time() - start_time
        print(f"Parallel processing completed in {elapsed_time:.2f} seconds")
        print(f"Successfully processed {len(individual_results)} out of {len(current_objects)} objects")
    else:
        print("No objects to process.")
    
    # Process summary sentence and wait for completion
    summary_result = None
    start_time = time.time()  # Initialize start_time here to ensure it exists even if there are no objects
    
    if summary_sentence:
        print(f"Processing summary sentence: {summary_sentence}")
        try:
            # First check if this exact summary already exists
            existing_pattern = check_existing_pattern(summary_sentence)
            
            if existing_pattern:
                print(f"Using existing pattern for summary: {summary_sentence}")
                
                # Load existing pattern
                with open(existing_pattern, 'r') as f:
                    pattern_data = json.load(f)
                
                summary_result = {
                    'sentence': summary_sentence,
                    'servo_angles': {
                        'x_servo': avg_x_servo,
                        'y_servo': avg_y_servo
                    },
                    'pattern_file': existing_pattern,
                    'is_cached': True
                }
            else:
                # Generate new pattern for summary
                objects_dir = os.path.join("stepper-data", "objects")
                ensure_directory_exists(objects_dir)
                
                # Try up to 3 times to generate the summary pattern
                max_tries = 3
                for attempt in range(max_tries):
                    try:
                        summary_result = process_sentence_to_stepper(
                            sentence=summary_sentence,
                            x_servo=avg_x_servo,
                            y_servo=avg_y_servo,
                            height=0,
                            output_dir=objects_dir,
                            duration=3.0
                        )
                        
                        # Verify pattern file was created
                        if 'pattern_file' not in summary_result or not summary_result['pattern_file']:
                            print(f"Summary pattern generation attempt {attempt+1}/{max_tries} failed: No pattern file returned")
                            # Check if the file exists anyway (might be a return value issue)
                            safe_name = clean_filename(summary_sentence)
                            expected_pattern_file = os.path.join(objects_dir, f"{safe_name}_stepper.json")
                            if os.path.exists(expected_pattern_file):
                                print(f"Found pattern file despite missing return value: {expected_pattern_file}")
                                summary_result['pattern_file'] = expected_pattern_file
                                # Success - pattern file exists
                                summary_result['sentence'] = summary_sentence
                                summary_result['is_cached'] = False
                                print(f"Successfully recovered pattern file on attempt {attempt+1}")
                                break
                            elif attempt < max_tries - 1:
                                print("Retrying summary pattern generation...")
                                time.sleep(0.5)  # Short delay before retry
                                continue
                        else:
                            # Verify the pattern file exists
                            if not os.path.exists(summary_result['pattern_file']):
                                print(f"Summary pattern generation attempt {attempt+1}/{max_tries} failed: File not created")
                                if attempt < max_tries - 1:
                                    print("Retrying summary pattern generation...")
                                    time.sleep(0.5)
                                    continue
                            
                            # Success - pattern file exists
                            summary_result['sentence'] = summary_sentence
                            summary_result['is_cached'] = False
                            print(f"Successfully processed summary sentence on attempt {attempt+1}")
                            break
                    except Exception as e:
                        print(f"Error during summary pattern generation attempt {attempt+1}/{max_tries}: {str(e)}")
                        if attempt < max_tries - 1:
                            print("Retrying summary pattern generation...")
                            time.sleep(0.5)
                            continue
                        else:
                            raise  # Re-raise the exception on the last attempt
                
                # Check if we succeeded after all attempts
                if 'pattern_file' not in summary_result or not summary_result['pattern_file']:
                    print(f"Summary pattern generation failed after {max_tries} attempts")
                    summary_result = {
                        'error': "Failed to generate pattern after multiple attempts",
                        'sentence': summary_sentence
                    }
                    
        except Exception as e:
            print(f"Error processing summary sentence: {str(e)}")
            summary_result = {
                'error': str(e),
                'sentence': summary_sentence
            }
    
    # Calculate total processing time
    total_elapsed_time = time.time() - start_time if current_objects else 0
    
    # Create output data structure for timestamp record
    output_data = {
        "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "num_objects": len(individual_results),
        "processing_time_seconds": total_elapsed_time,
        "objects": [
            {
                "class_name": result.get("class_name", "unknown"),
                "confidence": result.get("confidence", 0),
                "servo_angles": result.get("servo_angles", {"x_servo": 0, "y_servo": 0}),
                "pattern_file": os.path.basename(result.get("pattern_file", "")),
                "is_cached": result.get("is_cached", False)
            } 
            for result in individual_results
            if 'pattern_file' in result and result['pattern_file']  # Only include results with valid pattern files
        ],
        "summary": {
            "sentence": summary_sentence,
            "avg_x_servo": avg_x_servo,
            "avg_y_servo": avg_y_servo,
            "pattern_file": os.path.basename(summary_result.get("pattern_file", "")) if summary_result and 'pattern_file' in summary_result else None,
            "is_cached": summary_result.get("is_cached", False) if summary_result else False
        }
    }
    
    # Only save the timestamp file if we have at least one valid pattern file
    if individual_results or (summary_result and 'pattern_file' in summary_result):
        # Save to JSON with timestamp name in stepper-data directory
        timestamp_str = timestamp.strftime('%Y%m%d_%H%M%S')
        json_filename = os.path.join("stepper-data", f"{timestamp_str}.json")
        
        # Fix for summary pattern file reference - check if it exists but wasn't properly linked
        if summary_sentence and (output_data["summary"]["pattern_file"] is None or output_data["summary"]["pattern_file"] == ""):
            # Try to find the pattern file directly
            safe_name = clean_filename(summary_sentence)
            expected_pattern_file = f"{safe_name}_stepper.json"
            expected_path = os.path.join("stepper-data", "objects", expected_pattern_file)
            
            if os.path.exists(expected_path):
                print(f"Found pattern file for summary that wasn't properly linked: {expected_pattern_file}")
                output_data["summary"]["pattern_file"] = expected_pattern_file
                output_data["summary"]["is_cached"] = True
            else:
                # Try a simplified version of the summary sentence
                # Some long sentences might be saved with truncated filenames
                shortened_name = clean_filename(summary_sentence[:50])  # Use first 50 chars
                expected_pattern_file = f"{shortened_name}_stepper.json"
                expected_path = os.path.join("stepper-data", "objects", expected_pattern_file)
                
                if os.path.exists(expected_path):
                    print(f"Found pattern file using shortened name: {expected_pattern_file}")
                    output_data["summary"]["pattern_file"] = expected_pattern_file
                    output_data["summary"]["is_cached"] = True
                else:
                    # Try wildcard search for any file matching part of the summary
                    objects_dir = os.path.join("stepper-data", "objects")
                    words = summary_sentence.lower().split()
                    if len(words) > 3:  # Try with first few words
                        search_term = "_".join(words[:3])
                        matches = [f for f in os.listdir(objects_dir) 
                                  if f.startswith(search_term) and f.endswith("_stepper.json")]
                        
                        if matches:
                            print(f"Found matching pattern file using search: {matches[0]}")
                            output_data["summary"]["pattern_file"] = matches[0]
                            output_data["summary"]["is_cached"] = True
        
        # Save the JSON file
        with open(json_filename, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"Saved detection record to {json_filename}")
        return {"filename": json_filename, "objects": current_objects}
    else:
        print("No valid pattern files generated, skipping timestamp file save")
        return {"filename": None, "objects": []}

# Main function for object detection
def run_yolo_detection(camera_id=0, model_size="yolov8n.pt", conf_threshold=0.25):
    # Ensure directories exist
    ensure_data_directory()
    ensure_directory_exists(os.path.join("stepper-data", "objects"))
    
    # Get the device
    device = check_mps()
    
    # Load YOLOv8 model
    print(f"Loading YOLOv8 model: {model_size}")
    model = YOLO(model_size)
    
    # Move model to MPS device if available
    if device.type == "mps":
        model.to(device)
    
    # Initialize camera
    cap = initialize_camera(camera_id)
    if cap is None:
        return
    
    # Get camera properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera resolution: {frame_width}x{frame_height}")
    
    # Performance tracking
    frame_count = 0
    start_time = time.time()
    last_process_time = time.time()
    
    # Variable to track processing status
    processing_thread = None
    is_processing = False
    last_output_file = None
    
    # Store previous objects for motion tracking
    previous_objects = None
    
    while True:
        # Read frame from camera
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to receive frame from camera")
            break
        
        # Run YOLOv8 inference
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        results = model(frame, conf=conf_threshold)
        
        # Process detections for vibration every 4 seconds (if not already processing)
        current_time = time.time()
        if current_time - last_process_time >= 4.0 and not is_processing:
            # Start a new thread to handle the processing to avoid blocking the main loop
            last_process_time = current_time
            is_processing = True
            timestamp = datetime.now()
            
            def background_processing():
                nonlocal is_processing, last_output_file, previous_objects
                try:
                    result = process_detections_for_vibration(
                        results[0].boxes, 
                        frame_width, 
                        frame_height, 
                        model,
                        timestamp,
                        previous_objects
                    )
                    
                    if result["filename"]:
                        last_output_file = result["filename"]
                    
                    # Update previous objects for next iteration
                    previous_objects = result["objects"]
                    
                finally:
                    is_processing = False
            
            processing_thread = threading.Thread(target=background_processing)
            processing_thread.daemon = True  # Thread will exit when main program exits
            processing_thread.start()
        
        # Draw detections and labels
        annotated_frame = results[0].plot()
        
        # Add processing status indicator
        if is_processing:
            cv2.putText(annotated_frame, "Processing patterns...", (10, 60), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        elif last_output_file:
            cv2.putText(annotated_frame, f"Last output: {os.path.basename(last_output_file)}", (10, 60), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Count tracked objects
        if previous_objects:
            cv2.putText(annotated_frame, f"Tracking {len(previous_objects)} objects", (10, 90), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Calculate and display FPS
        frame_count += 1
        elapsed_time = time.time() - start_time
        if elapsed_time > 1:
            fps = frame_count / elapsed_time
            cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            frame_count = 0
            start_time = time.time()
        
        # Display the annotated frame
        cv2.imshow("YOLOv8 Object Detection", annotated_frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Wait for any ongoing processing to complete
    if processing_thread and processing_thread.is_alive():
        print("Waiting for pattern processing to complete...")
        processing_thread.join(timeout=5.0)
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # You can change these parameters:
    # camera_id: 0 is usually the built-in webcam, try other numbers for OBS virtual camera
    # model_size options: "yolov8n.pt" (fastest), "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt" (most accurate)
    run_yolo_detection(
        camera_id=1,  # Try different numbers to find your OBS virtual camera
        model_size="yolov8n.pt",  # Start with smallest model, increase if performance is good
        conf_threshold=0.25  # Lower this value to detect more objects (may include false positives)
    )