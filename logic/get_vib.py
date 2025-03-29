import os
import json
from typing import List
from openai import OpenAI
from dotenv import load_dotenv
import numpy as np
from scipy.fftpack import ifft
import re


load_dotenv()

def clean_filename(text):
    """
    Convert text to a safe filename by removing special characters
    and limiting length.
    """
    # Replace spaces with underscores and remove invalid filename characters
    safe_name = re.sub(r'[^\w\s-]', '', text.lower())
    safe_name = re.sub(r'[\s]+', '_', safe_name)
    
    # Limit length to avoid excessively long filenames
    if len(safe_name) > 50:
        safe_name = safe_name[:50]
    
    return safe_name


def ensure_directory_exists(directory):
    """
    Create a directory if it doesn't exist.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")


def get_sentence_embedding(sentence: str) -> List[float]:
    """
    Generate an embedding for a given sentence using OpenAI's embeddings API.
    """
    # Initialize OpenAI client
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    try:
        # Generate embedding using OpenAI's API
        response = client.embeddings.create(
            model="text-embedding-3-small",  # Using the latest embedding model
            input=sentence,
            encoding_format="float"
        )
        
        # Return the embedding vector
        return response.data[0].embedding
        
    except Exception as e:
        raise Exception(f"Error generating embedding: {str(e)}")


def average_pool_embedding(embeddings, pool_size=6):
    """
    Apply average pooling to reduce embedding dimensions.
    """
    # Convert to numpy array if it's not already
    embeddings = np.array(embeddings)
    
    # Calculate how many complete pools we can make
    num_complete_pools = len(embeddings) // pool_size
    
    # Create the pooled embeddings array
    pooled_embeddings = np.zeros(num_complete_pools)
    
    # Perform average pooling
    for i in range(num_complete_pools):
        start_idx = i * pool_size
        end_idx = start_idx + pool_size
        pooled_embeddings[i] = np.mean(embeddings[start_idx:end_idx])
    
    print(f"Reduced embedding dimension from {len(embeddings)} to {len(pooled_embeddings)}")
    return pooled_embeddings


def sonify_embeddings(embeddings, sample_rate=44100, duration=3.0, normalize=True):
    """
    Convert embeddings to audio using inverse Fourier transform.
    Duration fixed at 3 seconds as requested.
    
    Parameters:
    - embeddings: numpy array of embedding values
    - sample_rate: audio sample rate in Hz
    - duration: duration of output signal in seconds (fixed at 3.0)
    - normalize: whether to normalize output
    
    Returns:
    - audio_signal: numpy array of signal values
    """
    # Convert embeddings to numpy array if it's not already
    embeddings = np.array(embeddings)
    
    # Check if embeddings are complex or need to be made complex
    if np.iscomplexobj(embeddings):
        complex_embeddings = embeddings
    else:
        # For real embeddings, we need to create a complex representation
        # We'll use the first half of the embeddings as real parts
        # and the second half as imaginary parts
        half_len = len(embeddings) // 2
        complex_embeddings = embeddings[:half_len] + 1j * embeddings[half_len:2*half_len]
    
    # Make sure we have conjugate symmetry for a real output signal
    n_samples = int(sample_rate * duration)
    fft_data = np.zeros(n_samples, dtype=complex)
    
    # Place the embeddings in the FFT data with conjugate symmetry
    # Only use as many embeddings as we can fit in half the FFT size
    usable_size = min(len(complex_embeddings), n_samples // 2 - 1)
    fft_data[1:usable_size+1] = complex_embeddings[:usable_size]
    fft_data[n_samples-usable_size:n_samples] = np.conj(complex_embeddings[:usable_size][::-1])
    
    # Apply inverse FFT
    audio_signal = np.real(ifft(fft_data))
    
    # Normalize if requested
    if normalize:
        # Normalize to range [-1, 1]
        max_val = np.max(np.abs(audio_signal))
        if max_val > 0:
            audio_signal = audio_signal / max_val * 0.95  # Leave a little headroom
    
    return audio_signal


def create_stepper_patterns(audio_signal, sample_rate=44100, min_speed=0, max_speed=2048):
    """
    Create stepper motor control patterns from audio signal.
    
    Parameters:
    - audio_signal: numpy array of audio signal values (-1 to 1)
    - sample_rate: original sample rate of the audio
    - min_speed: minimum stepper motor speed (0 = stopped)
    - max_speed: maximum stepper motor speed (controls top RPM)
      For 28BYJ-48, using speed values from 0-2048
    
    Returns:
    - patterns: dictionary with patterns at different frequencies
    """
    original_length = len(audio_signal)
    duration = original_length / sample_rate
    
    # Create stepper patterns at specified frequencies
    patterns = {}
    
    # 100Hz (100 samples per second)
    samples_100hz = int(duration * 100)
    indices = np.linspace(0, original_length - 1, samples_100hz, dtype=int)
    
    # Convert audio amplitude [-1,1] to stepper speed [min_speed, max_speed]
    # Take absolute value since we're controlling speed, not direction
    pattern_100hz = np.abs(audio_signal[indices])
    pattern_100hz = min_speed + pattern_100hz * (max_speed - min_speed)
    patterns["100hz"] = np.round(pattern_100hz).astype(int).tolist()
    
    # 50Hz (50 samples per second) - good for 28BYJ-48 which is a slower motor
    samples_50hz = int(duration * 50)
    indices = np.linspace(0, original_length - 1, samples_50hz, dtype=int)
    pattern_50hz = np.abs(audio_signal[indices])
    pattern_50hz = min_speed + pattern_50hz * (max_speed - min_speed)
    patterns["50hz"] = np.round(pattern_50hz).astype(int).tolist()
    
    # 20Hz (20 samples per second) - even better for 28BYJ-48 which has max ~15 RPM
    samples_20hz = int(duration * 20)
    indices = np.linspace(0, original_length - 1, samples_20hz, dtype=int)
    pattern_20hz = np.abs(audio_signal[indices])
    pattern_20hz = min_speed + pattern_20hz * (max_speed - min_speed)
    patterns["20hz"] = np.round(pattern_20hz).astype(int).tolist()
    
    # Add metadata
    patterns["duration_seconds"] = duration
    patterns["original_sample_rate"] = sample_rate
    patterns["motor_info"] = {
        "type": "28BYJ-48 stepper motor",
        "min_speed": min_speed,
        "max_speed": max_speed
    }
    
    return patterns


def process_sentence_to_stepper(
    sentence, 
    x_servo=None,
    y_servo=None,
    height=0,
    output_dir="stepper_patterns", 
    pool_size=6, 
    duration=3.0,  # Fixed at 3 seconds as requested
    sample_rate=44100
):
    """
    Process a sentence: get embedding, pool it, convert to stepper motor patterns, save as JSON.
    Configured specifically for 28BYJ-48 stepper motor.
    
    Parameters:
    - sentence: input text to generate embedding for
    - x_servo: horizontal servo angle (0-180)
    - y_servo: vertical servo angle (0-180)
    - height: height parameter
    - output_dir: directory to save output
    - pool_size: size for average pooling of embeddings
    - duration: duration of pattern in seconds (fixed at 3.0)
    - sample_rate: sample rate for audio signal generation
    
    Returns:
    - Dictionary with results information
    """
    # Ensure the output directory exists
    ensure_directory_exists(output_dir)
    
    # Create a safe filename base from the sentence
    base_filename = clean_filename(sentence)
    
    # Get embedding
    embedding = get_sentence_embedding(sentence)
    print(f"Generated embedding of length: {len(embedding)}")
    
    # Convert to numpy array
    embedding_array = np.array(embedding)
    
    # Apply average pooling to reduce dimensions
    pooled_embedding = average_pool_embedding(embedding_array, pool_size)
    
    # Generate audio signal from pooled embedding
    audio_signal = sonify_embeddings(pooled_embedding, sample_rate, duration)
    
    # Create stepper patterns optimized for 28BYJ-48
    patterns = create_stepper_patterns(audio_signal, sample_rate)
    
    # Create output data structure
    output_data = {
        "sentence": sentence,
        "embedding_length": len(embedding),
        "pooled_embedding_length": len(pooled_embedding),
        "duration_seconds": patterns["duration_seconds"],
        "patterns": {
            "100hz": {
                "sample_rate": 100,
                "total_samples": len(patterns["100hz"]),
                "pattern": patterns["100hz"]
            },
            "50hz": {
                "sample_rate": 50,
                "total_samples": len(patterns["50hz"]),
                "pattern": patterns["50hz"]
            },
            "20hz": {
                "sample_rate": 20,
                "total_samples": len(patterns["20hz"]),
                "pattern": patterns["20hz"]
            }
        },
        "servo_angles": {
            "x_servo": x_servo,
            "y_servo": y_servo,
            "height": height
        },
        "motor_info": patterns["motor_info"]
    }
    
    # Save as JSON
    json_filename = os.path.join(output_dir, f"{base_filename}_stepper.json")
    with open(json_filename, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Stepper motor patterns saved to {json_filename}")
    print(f"Created patterns:")
    print(f"- 100Hz: {len(patterns['100hz'])} samples")
    print(f"- 50Hz: {len(patterns['50hz'])} samples")
    print(f"- 20Hz: {len(patterns['20hz'])} samples (recommended for 28BYJ-48)")
    print(f"- Duration: {patterns['duration_seconds']:.2f} seconds")
    print(f"Servo angles:")
    print(f"- X servo: {x_servo}")
    print(f"- Y servo: {y_servo}")
    print(f"- Height: {height}")
    
    return {
        'json_filename': json_filename,
        'patterns': {
            '100hz': len(patterns['100hz']),
            '50hz': len(patterns['50hz']),
            '20hz': len(patterns['20hz'])
        },
        'servo_angles': {
            'x_servo': x_servo,
            'y_servo': y_servo,
            'height': height
        },
        'duration': patterns['duration_seconds']
    }


# Example usage
if __name__ == "__main__":
    # Input sentence with optional parameters
    sentence = "red car"
    
    try:
        # Process the sentence to stepper motor patterns with servo angles
        result = process_sentence_to_stepper(
            sentence=sentence,
            x_servo=90,  # Center position
            y_servo=45,  # Upper half of frame
            height=2
        )
        
        print("\nStepper motor patterns generated:")
        print(f"- JSON file: {result['json_filename']}")
        print(f"- 100Hz pattern: {result['patterns']['100hz']} samples")
        print(f"- 50Hz pattern: {result['patterns']['50hz']} samples")
        print(f"- 20Hz pattern: {result['patterns']['20hz']} samples")
        print(f"- Duration: {result['duration']:.2f} seconds")
        print(f"- Servo angles: {result['servo_angles']}")
        
    except Exception as e:
        print(f"Error: {e}")