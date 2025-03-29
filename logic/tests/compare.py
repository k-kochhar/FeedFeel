import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.stats import wasserstein_distance
from scipy.signal import correlate
from dtw import dtw
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import datetime

def create_unique_directory():
    """
    Create a unique directory for comparison results based on timestamp.
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_name = os.path.join("tests", f"comparison_{timestamp}")
    
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        print(f"Created directory: {dir_name}")
    
    return dir_name

def load_audio(file_path):
    """
    Load an audio file and return the waveform data.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    
    sample_rate, audio_data = wavfile.read(file_path)
    
    # Convert to mono if stereo
    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1)
    
    # Normalize to [-1, 1]
    audio_data = audio_data / (2.0**15 if audio_data.dtype == np.int16 else 1.0)
    
    return sample_rate, audio_data

def compare_audio_waveforms(file1_path, file2_path, output_dir=None):
    """
    Compare two audio files directly by analyzing their waveforms.
    
    Parameters:
    - file1_path: path to first audio file
    - file2_path: path to second audio file
    - output_dir: directory to save results (created if None)
    
    Returns:
    - Dictionary of similarity metrics
    """
    # Create output directory if not provided
    if output_dir is None:
        output_dir = create_unique_directory()
    elif not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get base filenames for easier display
    file1_name = os.path.basename(file1_path)
    file2_name = os.path.basename(file2_path)
    
    # Load audio files
    sr1, wave1 = load_audio(file1_path)
    sr2, wave2 = load_audio(file2_path)
    
    # Check if sample rates match
    if sr1 != sr2:
        print(f"Warning: Sample rates differ - {sr1} Hz vs {sr2} Hz")
    
    # Ensure same length for comparison (truncate the longer one)
    min_length = min(len(wave1), len(wave2))
    wave1 = wave1[:min_length]
    wave2 = wave2[:min_length]
    
    # Method 1: Mean Absolute Error (MAE)
    mae = np.mean(np.abs(wave1 - wave2))
    
    # Method 2: Root Mean Square Error (RMSE)
    rmse = np.sqrt(np.mean((wave1 - wave2) ** 2))
    
    # Method 3: Cross-correlation
    corr = np.corrcoef(wave1, wave2)[0, 1]
    
    # Method 4: Maximum of cross-correlation (accounts for small shifts)
    xcorr = correlate(wave1, wave2, mode='full')
    max_xcorr = np.max(xcorr) / (np.sqrt(np.sum(wave1**2) * np.sum(wave2**2)))
    
    # Method 5: Earth Mover's Distance (Wasserstein distance)
    # Use a sample of points to make this faster
    sample_size = 10000
    if min_length > sample_size:
        indices = np.linspace(0, min_length-1, sample_size, dtype=int)
        wave1_sample = wave1[indices]
        wave2_sample = wave2[indices]
    else:
        wave1_sample = wave1
        wave2_sample = wave2
    
    emd = wasserstein_distance(wave1_sample, wave2_sample)
    
    # Method 6: Dynamic Time Warping (DTW) if available
    try:
        # Use fastdtw for efficiency
        distance, path = fastdtw(wave1_sample, wave2_sample, dist=euclidean)
        dtw_distance = distance / len(path)
    except Exception as e:
        print(f"Warning: DTW calculation failed: {e}")
        dtw_distance = None
    
    # Calculate overall similarity score
    # Normalize all scores to [0, 1] range where 1 means identical
    norm_mae = 1 - min(mae, 1)  # Lower is better, so invert
    norm_rmse = 1 - min(rmse, 1)  # Lower is better, so invert
    norm_corr = (corr + 1) / 2  # Maps [-1, 1] to [0, 1]
    norm_xcorr = max_xcorr  # Already in [0, 1]
    norm_emd = 1 - min(emd / 2, 1)  # Lower is better, so invert
    
    # If DTW was calculated, include it
    if dtw_distance is not None:
        norm_dtw = 1 - min(dtw_distance / 2, 1)  # Lower is better, so invert
        overall_similarity = (norm_mae + norm_rmse + norm_corr + norm_xcorr + norm_emd + norm_dtw) / 6
    else:
        overall_similarity = (norm_mae + norm_rmse + norm_corr + norm_xcorr + norm_emd) / 5
    
    # Create a description
    if overall_similarity > 0.9:
        similarity_description = "Extremely similar (nearly identical)"
    elif overall_similarity > 0.8:
        similarity_description = "Very similar"
    elif overall_similarity > 0.6:
        similarity_description = "Moderately similar"
    elif overall_similarity > 0.4:
        similarity_description = "Somewhat similar"
    elif overall_similarity > 0.2:
        similarity_description = "Mostly different but with some similarities"
    else:
        similarity_description = "Very different"
    
    # Prepare stats text
    stats_text = [
        f"Comparison: {file1_name} vs {file2_name}",
        f"Overall Similarity: {overall_similarity:.4f}",
        f"Description: {similarity_description}",
        f"Correlation: {corr:.4f}",
        f"Max Cross-Correlation: {max_xcorr:.4f}",
        f"Mean Absolute Error: {mae:.4f}",
        f"RMSE: {rmse:.4f}",
        f"Earth Movers Distance: {emd:.4f}"
    ]
    
    if dtw_distance is not None:
        stats_text.append(f"DTW Distance: {dtw_distance:.4f}")
    
    # Create a visual comparison with stats printed on it
    fig = plt.figure(figsize=(14, 12))
    
    # Create grid for plots and stats
    gs = fig.add_gridspec(4, 2, height_ratios=[1, 1, 1, 0.5])
    
    # Add title
    fig.suptitle(f"Waveform Comparison: {file1_name} vs {file2_name}", fontsize=16)
    
    # Plot first waveform
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(wave1)
    ax1.set_title(f"Waveform 1: {file1_name}")
    ax1.set_ylabel("Amplitude")
    ax1.set_xlim(0, min_length)
    
    # Plot second waveform
    ax2 = fig.add_subplot(gs[1, :])
    ax2.plot(wave2)
    ax2.set_title(f"Waveform 2: {file2_name}")
    ax2.set_ylabel("Amplitude")
    ax2.set_xlim(0, min_length)
    
    # Plot both waveforms overlaid
    ax3 = fig.add_subplot(gs[2, :])
    ax3.plot(wave1, alpha=0.7, label=file1_name)
    ax3.plot(wave2, alpha=0.7, label=file2_name)
    ax3.set_title("Overlaid Waveforms")
    ax3.set_xlabel("Sample")
    ax3.set_ylabel("Amplitude")
    ax3.legend()
    ax3.set_xlim(0, min_length)
    
    # Add stats text box
    ax_stats = fig.add_subplot(gs[3, :])
    ax_stats.axis('off')
    stats_str = '\n'.join(stats_text)
    ax_stats.text(0.5, 0.5, stats_str, 
                 horizontalalignment='center',
                 verticalalignment='center',
                 transform=ax_stats.transAxes,
                 bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=1'))
    
    plt.tight_layout()
    
    # Save the comparison plot
    comparison_filename = os.path.join(output_dir, f"comparison_{file1_name}_vs_{file2_name}.png")
    plt.savefig(comparison_filename, dpi=150)
    plt.close()
    
    # Also save individual waveform plots with stats
    # Waveform 1
    plt.figure(figsize=(10, 6))
    plt.plot(wave1)
    plt.title(f"Waveform: {file1_name}")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.xlim(0, min_length)
    
    # Add small stats textbox
    short_stats = [
        f"Comparison with: {file2_name}",
        f"Similarity: {overall_similarity:.4f}",
        f"{similarity_description}"
    ]
    stats_str = '\n'.join(short_stats)
    plt.figtext(0.5, 0.01, stats_str, 
               horizontalalignment='center',
               bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"waveform_{file1_name}.png"), dpi=150)
    plt.close()
    
    # Waveform 2
    plt.figure(figsize=(10, 6))
    plt.plot(wave2)
    plt.title(f"Waveform: {file2_name}")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.xlim(0, min_length)
    
    # Add small stats textbox
    short_stats = [
        f"Comparison with: {file1_name}",
        f"Similarity: {overall_similarity:.4f}",
        f"{similarity_description}"
    ]
    stats_str = '\n'.join(short_stats)
    plt.figtext(0.5, 0.01, stats_str, 
               horizontalalignment='center',
               bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"waveform_{file2_name}.png"), dpi=150)
    plt.close()
    
    # Return all metrics
    results = {
        "Mean Absolute Error": mae,
        "Root Mean Square Error": rmse,
        "Correlation Coefficient": corr,
        "Max Cross-Correlation": max_xcorr,
        "Earth Movers Distance": emd,
        "DTW Distance": dtw_distance,
        "Overall Similarity": overall_similarity,
        "Similarity Description": similarity_description,
        "Comparison Image": comparison_filename,
        "Output Directory": output_dir
    }
    
    return results

def compare_all_audio_files(directory="tests", pattern="_audio.wav"):
    """
    Compare all pairs of audio files in the given directory.
    
    Parameters:
    - directory: directory containing audio files
    - pattern: pattern to match audio filenames
    
    Returns:
    - List of comparison results
    """
    # Get all audio files
    audio_files = [f for f in os.listdir(directory) if pattern in f and f.endswith(".wav")]
    
    if len(audio_files) < 2:
        print(f"Found fewer than 2 audio files in {directory} matching pattern {pattern}")
        return []
    
    # Create a master directory for this comparison session
    output_dir = create_unique_directory()
    
    # Compare all pairs
    comparisons = []
    for i in range(len(audio_files)):
        for j in range(i+1, len(audio_files)):
            file1 = os.path.join(directory, audio_files[i])
            file2 = os.path.join(directory, audio_files[j])
            
            print(f"Comparing {audio_files[i]} with {audio_files[j]}...")
            
            # Create a sub-directory for this specific comparison
            pair_dir = os.path.join(output_dir, f"{audio_files[i]}_vs_{audio_files[j]}")
            if not os.path.exists(pair_dir):
                os.makedirs(pair_dir)
            
            results = compare_audio_waveforms(file1, file2, pair_dir)
            
            comparison = {
                "File1": audio_files[i],
                "File2": audio_files[j],
                "Results": results
            }
            comparisons.append(comparison)
            
            # Print key results
            print(f"  Comparison saved to: {pair_dir}")
            print(f"  Overall Similarity: {results['Overall Similarity']:.4f}")
            print(f"  Description: {results['Similarity Description']}")
            print()
    
    # Create a summary file
    summary_file = os.path.join(output_dir, "comparison_summary.txt")
    with open(summary_file, 'w') as f:
        f.write("Waveform Comparison Summary\n")
        f.write("==========================\n\n")
        
        for comp in comparisons:
            f.write(f"Comparison: {comp['File1']} vs {comp['File2']}\n")
            f.write(f"  Overall Similarity: {comp['Results']['Overall Similarity']:.4f}\n")
            f.write(f"  Description: {comp['Results']['Similarity Description']}\n")
            f.write(f"  Correlation: {comp['Results']['Correlation Coefficient']:.4f}\n")
            f.write(f"  Earth Movers Distance: {comp['Results']['Earth Movers Distance']:.4f}\n")
            f.write("\n")
    
    print(f"Comparison summary saved to: {summary_file}")
    return comparisons, output_dir


# Example usage
if __name__ == "__main__":
    # Create tests directory if it doesn't exist
    if not os.path.exists("tests"):
        os.makedirs("tests")
    
    # Compare two specific audio files
    try:
        audio1_path = "tests/this_is_a_test_sentence_audio.wav"
        audio2_path = "tests/another_test_sentence_audio.wav"
        
        if os.path.exists(audio1_path) and os.path.exists(audio2_path):
            print(f"Comparing specific audio files: {audio1_path} and {audio2_path}")
            results = compare_audio_waveforms(audio1_path, audio2_path)
            
            print("Comparison Results:")
            print(f"  Output saved to: {results['Output Directory']}")
            print(f"  Overall Similarity: {results['Overall Similarity']:.4f}")
            print(f"  Description: {results['Similarity Description']}")
        else:
            print("Specific audio files not found. Comparing all audio files in the tests folder...")
            comparisons, output_dir = compare_all_audio_files()
            print(f"All comparisons saved in: {output_dir}")
            
    except Exception as e:
        print(f"Error: {e}")
        print("Comparing all audio files in the tests folder...")
        comparisons, output_dir = compare_all_audio_files()
        print(f"All comparisons saved in: {output_dir}")