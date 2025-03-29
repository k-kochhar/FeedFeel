import os
from typing import List
from openai import OpenAI
from dotenv import load_dotenv
import numpy as np
import scipy.io.wavfile as wavfile
from scipy.fftpack import ifft
import matplotlib.pyplot as plt
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


def ensure_tests_folder_exists():
    """
    Create the tests folder if it doesn't exist.
    """
    if not os.path.exists("tests"):
        os.makedirs("tests")
        print("Created 'tests' folder")


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


def sonify_embeddings(embeddings, sample_rate=44100, duration=5.0, normalize=True):
    """
    Convert embeddings to audio using inverse Fourier transform.
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


def process_sentence(sentence, pool_size=6, duration=5.0, sample_rate=44100):
    """
    Process a sentence: get embedding, pool it, sonify it, save audio and plots.
    """
    # Ensure the tests folder exists
    ensure_tests_folder_exists()
    
    # Create a safe filename base from the sentence
    base_filename = clean_filename(sentence)
    
    # Get embedding
    embedding = get_sentence_embedding(sentence)
    print(f"Generated embedding of length: {len(embedding)}")
    
    # Convert to numpy array
    embedding_array = np.array(embedding)
    
    # Apply average pooling to reduce dimensions
    pooled_embedding = average_pool_embedding(embedding_array, pool_size)
    
    # Generate audio from pooled embedding
    audio_signal = sonify_embeddings(pooled_embedding, sample_rate, duration)
    
    # Save audio
    audio_filename = os.path.join("tests", f"{base_filename}_audio.wav")
    audio_int = np.int16(audio_signal * 32767)
    wavfile.write(audio_filename, sample_rate, audio_int)
    print(f"Audio saved to {audio_filename}")
    
    # Save waveform plot
    waveform_filename = os.path.join("tests", f"{base_filename}_waveform.png")
    duration = len(audio_signal) / sample_rate
    time = np.linspace(0., duration, len(audio_signal))
    
    plt.figure(figsize=(10, 4))
    plt.plot(time, audio_signal)
    plt.title(f'Audio Waveform: "{sentence}"')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(waveform_filename)
    plt.close()  # Close the figure instead of showing it
    print(f"Waveform plot saved to {waveform_filename}")
    
    # Save spectrogram plot
    spectrogram_filename = os.path.join("tests", f"{base_filename}_spectrogram.png")
    plt.figure(figsize=(10, 4))
    plt.specgram(audio_signal, Fs=sample_rate, NFFT=1024, noverlap=512)
    plt.title(f'Spectrogram: "{sentence}"')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.colorbar(label='Intensity (dB)')
    plt.tight_layout()
    plt.savefig(spectrogram_filename)
    plt.close()  # Close the figure instead of showing it
    print(f"Spectrogram plot saved to {spectrogram_filename}")
    
    return {
        'audio_filename': audio_filename,
        'waveform_filename': waveform_filename,
        'spectrogram_filename': spectrogram_filename
    }


# Example usage
if __name__ == "__main__":
    # Input sentence
    sentence = "red car"
    
    try:
        # Process the sentence
        result = process_sentence(sentence)
        
        print("\nAll files generated:")
        print(f"- Audio: {result['audio_filename']}")
        print(f"- Waveform plot: {result['waveform_filename']}")
        print(f"- Spectrogram plot: {result['spectrogram_filename']}")
        
    except Exception as e:
        print(f"Error: {e}")