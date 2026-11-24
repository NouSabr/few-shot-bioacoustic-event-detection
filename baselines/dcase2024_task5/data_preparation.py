import librosa
import numpy as np
import dask.array as da
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import librosa.display
import h5py

from glob import glob
from tqdm import tqdm
import os
import argparse
from datetime import datetime
from multiprocessing import Pool, cpu_count


EPS = 1e-8


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-D", "--data-dir", default="/media/helios/Data/Uni/Masters/Dataset")
    # Added arguments for specifying dataset splits
    parser.add_argument("--train-dir", default="Training_Set", help="Training set directory name")
    parser.add_argument("--val-dir", default="Validation_Set", help="Validation set directory name")
    parser.add_argument("--test-dir", default="Test_Set", help="Test set directory name")
    # Added argument for output format
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["npy", "hdf5", "both"],
        default="hdf5",
        help="Output format for features: 'npy' (NumPy), 'hdf5' (HDF5), or 'both' (default: hdf5)"
    )

    return parser.parse_args()


def recursive_glob(path, suffix):
    return (
        glob(os.path.join(path, "*" + suffix))
        + glob(os.path.join(path, "*/*" + suffix))
        + glob(os.path.join(path, "*/*/*" + suffix))
        + glob(os.path.join(path, "*/*/*/*" + suffix))
        + glob(os.path.join(path, "*/*/*/*/*" + suffix))
        + glob(os.path.join(path, "*/*/*/*/*/*" + suffix))
    )


def save_feature_hdf5(feature_array, output_path, compression='lzf'):
    """
    Save feature array to HDF5 format.
    
    Args:
        feature_array: NumPy array to save
        output_path: Path for output HDF5 file
        compression: Compression type ('lzf', 'gzip', or None)
    """
    with h5py.File(output_path, 'w') as f:
        # Create dataset with compression and chunking
        dset = f.create_dataset(
            'data',
            data=feature_array,
            compression=compression,
            chunks=True
        )
        
        # Add metadata
        dset.attrs['shape'] = feature_array.shape
        dset.attrs['dtype'] = str(feature_array.dtype)
        dset.attrs['created'] = datetime.now().isoformat()


class Feature_Extractor:
    def __init__(self):
        """
        Extract various features from wav files and save them in .npy files
        """

        self.sr = 22050
        self.n_fft = 1024
        self.hop = 256
        self.n_mels = 128
        self.n_mfcc = 32
        self.fmax = 11025

    def norm(self, y):
        return y / np.max(np.abs(y))

    def mel(self, y):
        assert np.max(y) <= 1, np.max(y)
        mel_spec = librosa.feature.melspectrogram(
            y=y,
            sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop,
            n_mels=self.n_mels,
            fmax=self.fmax,
        )
        # mel_spec = np.log(mel_spec + 1e-8)
        mel_spec = mel_spec.astype(np.float32)
        return mel_spec

    def logmel(self, mel_spec):
        mel_spec = np.log(mel_spec + 1e-8)
        mel_spec = mel_spec.astype(np.float32)
        return mel_spec

    def pcen(self, y):
        assert np.max(y) <= 1
        mel_spec = librosa.feature.melspectrogram(
            y * (2**32),
            sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop,
            n_mels=self.n_mels,
            fmax=self.fmax,
        )
        pcen = librosa.core.pcen(mel_spec, sr=self.sr)
        pcen = pcen.astype(np.float32)
        return pcen

    def rms(self, S):
        return librosa.feature.rms(S=S, frame_length=self.n_fft)

    def mfcc(self, mel):
        return librosa.feature.mfcc(S=mel)

    def spectral_centroid(self, S):
        return librosa.feature.spectral_centroid(S=S, sr=self.sr)

    def spectral_bandwidth(self, S):
        return librosa.feature.spectral_bandwidth(S=S, sr=self.sr)

    def spectral_contrast(self, S, n_bands=6):
        return librosa.feature.spectral_contrast(S=S, sr=self.sr, n_bands=n_bands)

    def spectral_flatness(self, S):
        return librosa.feature.spectral_flatness(S=S)

    def spectral_bandwidth(self, S):
        return librosa.feature.spectral_bandwidth(S=S, sr=self.sr)

    def spectral_rolloff(self, S, roll_percent=0.9):
        return librosa.feature.spectral_rolloff(
            S=S, sr=self.sr, roll_percent=roll_percent
        )

    def poly_features(self, S, order=1):
        return librosa.feature.poly_features(S=S, sr=self.sr, order=order)

    def zero_crossing_rate(self, y):
        assert np.max(y) <= 1
        return librosa.feature.zero_crossing_rate(
            y, frame_length=self.n_fft, hop_length=self.hop
        )

    def delta_mfcc(self, mfcc, order=1, width=9):
        return librosa.feature.delta(mfcc, order=order, width=width)

    def draw_spec(self, matrix, save_path=None, name="temp"):
        plt.imshow(matrix, aspect="auto", interpolation="none")
        plt.title(name)
        if save_path is not None:
            plt.savefig(save_path)
        else:
            plt.show()
        plt.close('all')

    def draw_plot(self, line, save_path=None, name="temp"):
        plt.plot(line)
        plt.title(name)
        if save_path is not None:
            plt.savefig(save_path)
        else:
            plt.show()
        plt.close('all')

    def visualize_result(self, result):
        for k in result.keys():
            if result[k].shape[1] == 1:
                self.draw_plot(result[k], name=k)
            elif result[k].shape[1] > 1:
                self.draw_spec(result[k], name=k)

    def extract_feature(self, audio_path):
        result = {}
        result["waveform"], _ = librosa.load(audio_path, sr=self.sr)
        result["waveform"] = self.norm(result["waveform"])

        # result["spec"], result["phase"] = librosa.magphase(
        #     librosa.stft(result["waveform"], hop_length=256, n_fft=1024)
        # )

        result["mel"] = self.mel(result["waveform"]).astype(np.float32)
        # result["mel_un_normalized"] = result["mel"]
        result["logmel"] = self.logmel(result["mel"]).astype(np.float32)
        # result["pcen"] = self.pcen(result["waveform"]).astype(np.float32)
        # result["mfcc"] = self.mfcc(result["logmel"]).astype(np.float32)
        # result["rms"] = self.rms(result["spec"]).astype(np.float32)
        # result["spectral_centroid"] = self.spectral_centroid(result["spec"]).astype(np.float32)
        # result["spectral_bandwidth"] = self.spectral_bandwidth(result["spec"]).astype(np.float32)
        # result["spectral_contrast"] = self.spectral_contrast(result["spec"], n_bands=6).astype(np.float32)
        # result["spectral_flatness"] = self.spectral_flatness(result["spec"]).astype(np.float32)
        # result["spectral_bandwidth"] = self.spectral_bandwidth(result["spec"]).astype(np.float32)
        # result["spectral_rolloff"] = self.spectral_rolloff(result["spec"]).astype(np.float32)
        # result["poly_features"] = self.poly_features(result["spec"], order=1).astype(np.float32)
        # result["zero_crossing_rate"] = self.zero_crossing_rate(result["waveform"]).astype(np.float32)
        # result["delta_mfcc"] = self.delta_mfcc(result["mfcc"], order=1, width=9).astype(
        #     np.float32
        # )

        # result["mel"] = result["mel"].T
        # result["mfcc"] = result["mfcc"].T

        del result['mel']  # TODO

        # del result['spec']
        # del result["phase"]
        del result["waveform"]

        for k in result.keys():
            result[k] = result[k].T

        # return {"mel_un_normalized": result["mel_un_normalized"]}
        return result


def visualize_before_after(file_path, feature_name="logmel", sample_count=5, output_format='hdf5'):
    """
    Visualizes audio features before and after normalization
    
    Args:
        file_path: List of audio file paths
        feature_name: Feature type to visualize
        sample_count: Number of random samples to visualize
        output_format: Format of feature files - 'npy', 'hdf5', or 'both'
    """
    import random
    
    # Select a small random sample of files
    sample_files = random.sample(file_path, min(sample_count, len(file_path)))
    
    for file in sample_files:
        try:
            # Load the original audio to extract unnormalized feature
            fe = Feature_Extractor()
            features = fe.extract_feature(file)
            unnormalized = features[feature_name]
            
            # Load the normalized feature (prefer HDF5 if available)
            normalized = None
            if output_format in ['hdf5', 'both']:
                h5_path = file.replace(".wav", f"_{feature_name}.h5")
                if os.path.exists(h5_path):
                    with h5py.File(h5_path, 'r') as f:
                        normalized = f['data'][:]
            
            if normalized is None and output_format in ['npy', 'both']:
                npy_path = file.replace(".wav", f"_{feature_name}.npy")
                if os.path.exists(npy_path):
                    normalized = np.load(npy_path)
            
            if normalized is None:
                print(f"No normalized feature found for {file}")
                continue
            
            # Create comparison visualization
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # Plot before normalization
            im0 = axes[0].imshow(unnormalized, aspect='auto', origin='lower', interpolation='none')
            axes[0].set_title(f"Before Normalization\nMin: {unnormalized.min():.2f}, Max: {unnormalized.max():.2f}")
            axes[0].set_ylabel('Mel Frequency Bin')
            axes[0].set_xlabel('Time Frame')
            plt.colorbar(im0, ax=axes[0], format='%+2.0f dB')
            
            # Plot after normalization
            im1 = axes[1].imshow(normalized, aspect='auto', origin='lower', interpolation='none')
            axes[1].set_title(f"After Normalization\nMin: {normalized.min():.2f}, Max: {normalized.max():.2f}")
            axes[1].set_xlabel('Time Frame')
            plt.colorbar(im1, ax=axes[1], format='%+2.0f')
            
            # Save the comparison
            vis_dir = os.path.join(os.path.dirname(file), "visualizations")
            os.makedirs(vis_dir, exist_ok=True)
            filename = os.path.basename(file).replace(".wav", f"_{feature_name}_norm_comparison.png")
            plt.suptitle(f"Feature: {feature_name} - {os.path.basename(file)}")
            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, filename))
            plt.close()
            
            print(f"Visualization saved for {os.path.basename(file)}")
            
        except Exception as e:
            print(f"Error visualizing {file}: {e}")


def process_single_file(args):
    """
    Helper function to process a single file for parallel execution.
    
    Args:
        args: Tuple of (file, output_format)
    
    Returns:
        None on success, filename on error
    """
    file, output_format = args
    fe = Feature_Extractor()
    
    try:
        features = fe.extract_feature(file)
        for k in features.keys():
            # Save in NPY format if requested
            if output_format in ['npy', 'both']:
                npy_path = file.replace(".wav", "_%s.npy" % k)
                np.save(npy_path, features[k])
            
            # Save in HDF5 format if requested
            if output_format in ['hdf5', 'both']:
                h5_path = file.replace(".wav", "_%s.h5" % k)
                save_feature_hdf5(features[k], h5_path)
        return None
    except Exception as e:
        # Return the filename on error
        return file


def process(fpath, output_format='hdf5', n_workers=2):
    """
    Extract and save features from audio files with optional parallelization.
    
    Args:
        fpath: List of audio file paths
        output_format: Output format - 'npy', 'hdf5', or 'both'
        n_workers: Number of parallel workers (default: cpu_count() - 1, use 1 for sequential)
    """
    # Set default number of workers
    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)
    
    # Prepare arguments for parallel processing
    args_list = [(file, output_format) for file in fpath]
    
    error_files = []
    
    if n_workers == 1:
        # Sequential processing (backward compatible)
        for args in tqdm(args_list, desc="Processing files"):
            result = process_single_file(args)
            if result is not None:
                error_files.append(result)
    else:
        # Parallel processing
        with Pool(processes=n_workers) as pool:
            # Use imap for progress tracking with tqdm
            results = list(tqdm(
                pool.imap(process_single_file, args_list),
                total=len(args_list),
                desc=f"Processing files ({n_workers} workers)"
            ))
            
            # Collect error files
            error_files = [r for r in results if r is not None]
    
    if len(error_files) > 0:
        print("Encounter error in these files:", error_files)


def calculate_feature_mean_std(A):
    N = max(float(sum([i.size for i in A])), EPS)
    mean_ = sum([i.sum() for i in A]) / N
    std_ = np.sqrt(sum([((i - mean_) ** 2).sum() for i in A]) / N)
    print("Mean %s; Std %s;" % (mean_, std_))
    return mean_, std_


def compute_stats_welford(train_files, feature, output_format='hdf5'):
    """
    Compute mean and standard deviation using Welford's online algorithm.
    
    Welford's algorithm computes statistics incrementally, processing one file at a time.
    This minimizes memory usage compared to loading all data at once.
    
    Mathematical basis:
    - For each new value x_n, update:
      * count: n = n + 1
      * mean: M_n = M_{n-1} + (x_n - M_{n-1}) / n
      * M2: M2_n = M2_{n-1} + (x_n - M_{n-1}) * (x_n - M_n)
    - Final std = sqrt(M2 / n)
    
    Benefits:
    - O(1) memory usage (only stores running statistics)
    - Numerically stable for large datasets
    - Single pass through data
    
    Args:
        train_files: List of training audio file paths
        feature: Feature name to compute stats for (e.g., 'logmel')
        output_format: Format of feature files - 'npy', 'hdf5', or 'both'
    
    Returns:
        Tuple of (mean, std) as floats
    
    Raises:
        ValueError: If no valid feature files are found
    """
    count = 0
    mean = 0.0
    M2 = 0.0
    
    files_processed = 0
    
    for file in tqdm(train_files, desc=f"Computing {feature} stats (Welford)"):
        try:
            # Try HDF5 first if it's the output format
            data = None
            if output_format in ['hdf5', 'both']:
                h5_path = file.replace(".wav", f"_{feature}.h5")
                if os.path.exists(h5_path):
                    with h5py.File(h5_path, 'r') as f:
                        data = f['data'][:].flatten()
            
            # Fall back to NPY if HDF5 not found
            if data is None and output_format in ['npy', 'both']:
                npy_path = file.replace(".wav", f"_{feature}.npy")
                if os.path.exists(npy_path):
                    data = np.load(npy_path).flatten()
            
            if data is None:
                continue
            
            # Welford's algorithm: update statistics incrementally
            for value in data:
                count += 1
                delta = value - mean
                mean += delta / count
                delta2 = value - mean
                M2 += delta * delta2
            
            files_processed += 1
            
        except Exception as e:
            print(f"Warning: Error processing {file}: {e}")
            continue
    
    if count == 0:
        raise ValueError(f"No valid feature files found for {feature}")
    
    # Compute final standard deviation
    std = np.sqrt(M2 / count)
    
    print(f"Processed {files_processed} files with {count} total values")
    
    return float(mean), float(std)


def compute_stats_original(train_files, feature, output_format='hdf5'):
    """
    Compute mean and standard deviation using the original dask-based method.
    
    This method loads all training features into memory using dask arrays,
    then computes statistics. Preserved for validation and comparison purposes.
    
    Args:
        train_files: List of training audio file paths
        feature: Feature name to compute stats for (e.g., 'logmel')
        output_format: Format of feature files - 'npy', 'hdf5', or 'both'
    
    Returns:
        Tuple of (mean, std) as floats
    
    Raises:
        ValueError: If no valid feature files are found
    """
    array_list = []
    
    for file in tqdm(train_files, desc=f"Loading {feature} data (Original)"):
        try:
            # Try HDF5 first if it's the output format
            if output_format in ['hdf5', 'both']:
                h5_path = file.replace(".wav", f"_{feature}.h5")
                if os.path.exists(h5_path):
                    with h5py.File(h5_path, 'r') as f:
                        array_list.append(f['data'][:].flatten())
                    continue
            
            # Fall back to NPY
            npy_path = file.replace(".wav", f"_{feature}.npy")
            if os.path.exists(npy_path):
                array_list.append(np.load(npy_path, mmap_mode='r').flatten())
        except Exception as e:
            print(f"Warning: Error processing {file}: {e}")
            continue
    
    if len(array_list) == 0:
        raise ValueError(f"No valid feature files found for {feature}")
    
    all_data = da.concatenate(array_list, axis=0)
    mean = float(all_data.mean().compute())
    std = float(all_data.std().compute())
    
    return mean, std


def validate_welford_algorithm(train_files, feature, output_format='hdf5', max_files=50):
    """
    Validate that Welford's algorithm produces identical results to the original method.
    
    This function computes statistics using both methods and compares the results.
    It's useful for testing and ensuring mathematical equivalence.
    
    Args:
        train_files: List of training audio file paths
        feature: Feature name to validate (e.g., 'logmel')
        output_format: Format of feature files - 'npy', 'hdf5', or 'both'
        max_files: Maximum number of files to use for validation (default: 50)
    
    Returns:
        True if validation passes (differences within floating point precision)
    
    Raises:
        AssertionError: If differences exceed acceptable threshold
    """
    print(f"\n{'='*60}")
    print(f"VALIDATING WELFORD'S ALGORITHM FOR {feature.upper()}")
    print(f"{'='*60}")
    
    # Use a subset of files for faster validation
    subset_files = train_files[:max_files] if len(train_files) > max_files else train_files
    print(f"Using {len(subset_files)} files for validation")
    
    # Compute using Welford's algorithm
    print("\n1. Computing with Welford's algorithm...")
    welford_mean, welford_std = compute_stats_welford(subset_files, feature, output_format)
    
    # Compute using original method
    print("\n2. Computing with original method...")
    original_mean, original_std = compute_stats_original(subset_files, feature, output_format)
    
    # Compare results
    print(f"\n{'='*60}")
    print("COMPARISON RESULTS:")
    print(f"{'='*60}")
    print(f"Welford  - Mean: {welford_mean:.10f}, Std: {welford_std:.10f}")
    print(f"Original - Mean: {original_mean:.10f}, Std: {original_std:.10f}")
    print(f"{'-'*60}")
    
    mean_diff = abs(welford_mean - original_mean)
    std_diff = abs(welford_std - original_std)
    
    print(f"Mean difference: {mean_diff:.2e}")
    print(f"Std difference:  {std_diff:.2e}")
    
    # Check if differences are within floating point precision
    tolerance = 1e-6
    mean_ok = mean_diff < tolerance
    std_ok = std_diff < tolerance
    
    print(f"{'-'*60}")
    print(f"Mean within tolerance ({tolerance}): {'✓ PASS' if mean_ok else '✗ FAIL'}")
    print(f"Std within tolerance ({tolerance}):  {'✓ PASS' if std_ok else '✗ FAIL'}")
    print(f"{'='*60}\n")
    
    # Assert that both are within tolerance
    assert mean_ok, f"Mean difference {mean_diff} exceeds tolerance {tolerance}"
    assert std_ok, f"Std difference {std_diff} exceeds tolerance {tolerance}"
    
    print("✓ Validation PASSED - Welford's algorithm is mathematically equivalent!\n")
    return True


# NEW FUNCTION: Computes normalization stats from training data and saves them
def compute_and_save_norm_stats(train_files, features, output_dir, output_format='hdf5', use_welford=True):
    """
    Computes normalization statistics from training data and saves them for reuse.
    
    This function can use either Welford's online algorithm (default, memory-efficient)
    or the original dask-based method (for comparison/validation).
    
    Args:
        train_files: List of training audio files
        features: List of features to compute stats for
        output_dir: Directory to save stats
        output_format: Format of feature files - 'npy', 'hdf5', or 'both'
        use_welford: If True, use Welford's online algorithm (default: True)
                     If False, use original dask-based method
    
    Returns:
        Dictionary with mean and std for each feature
    """
    stats = {}
    
    # Print which method is being used
    method_name = "Welford's online algorithm" if use_welford else "Original dask-based method"
    print(f"\n{'='*60}")
    print(f"Computing normalization statistics using: {method_name}")
    print(f"{'='*60}\n")
    
    for feature in features:
        print(f"Computing {feature} normalization statistics from training data...")
        
        # Choose computation method
        if use_welford:
            mean, std = compute_stats_welford(train_files, feature, output_format)
        else:
            mean, std = compute_stats_original(train_files, feature, output_format)
        
        stats[f"{feature}_mean"] = mean
        stats[f"{feature}_std"] = std
        
        print(f"Training set {feature} statistics - Mean: {mean}, Std: {std}")
    
    # Save normalization statistics
    os.makedirs(output_dir, exist_ok=True)
    stats_path = os.path.join(output_dir, "norm_stats.npz")
    np.savez(stats_path, **stats)
    print(f"Normalization statistics saved to {stats_path}")
    
    return stats


# NEW FUNCTION: Normalizes data using pre-computed stats with batch processing
def normalize_features_batch(files, features, stats, output_format='hdf5', batch_size=50):
    """
    Normalizes features using provided statistics with batch I/O operations.
    
    This function processes files in batches to reduce I/O overhead. For each batch:
    1. Read all files in the batch into memory
    2. Normalize all arrays in the batch (vectorized operation)
    3. Write all normalized arrays back
    
    Benefits of batch processing:
    - Reduces I/O overhead by batching read/write operations
    - Better utilization of disk cache and system buffers
    - Faster overall processing time, especially for large datasets
    - More efficient for both HDF5 and NPY formats
    
    Args:
        files: List of audio files
        features: List of feature types to normalize
        stats: Dictionary with mean and std for each feature
        output_format: Format of feature files - 'npy', 'hdf5', or 'both'
        batch_size: Number of files to process in each batch (default: 50)
    """
    for feature in features:
        mean = stats[f"{feature}_mean"]
        std = stats[f"{feature}_std"]
        
        print(f"Normalizing {feature} features with mean={mean}, std={std} (batch_size={batch_size})")
        
        # Process files in batches
        for batch_start in tqdm(range(0, len(files), batch_size), desc=f"Normalizing {feature} (batched)"):
            batch_end = min(batch_start + batch_size, len(files))
            batch_files = files[batch_start:batch_end]
            
            # Batch processing for NPY files
            if output_format in ['npy', 'both']:
                npy_batch_data = []
                npy_batch_paths = []
                
                # Read batch
                for file in batch_files:
                    npy_path = file.replace(".wav", f"_{feature}.npy")
                    if os.path.exists(npy_path):
                        try:
                            array = np.load(npy_path)
                            npy_batch_data.append(array)
                            npy_batch_paths.append(npy_path)
                        except Exception as e:
                            print(f"Warning: Error reading {npy_path}: {e}")
                            continue
                
                # Normalize batch (vectorized operation)
                for i, array in enumerate(npy_batch_data):
                    npy_batch_data[i] = (array - mean) / std
                
                # Write batch
                for path, array in zip(npy_batch_paths, npy_batch_data):
                    try:
                        np.save(path, array)
                    except Exception as e:
                        print(f"Warning: Error writing {path}: {e}")
                        continue
            
            # Batch processing for HDF5 files
            if output_format in ['hdf5', 'both']:
                h5_batch_data = []
                h5_batch_paths = []
                
                # Read batch
                for file in batch_files:
                    h5_path = file.replace(".wav", f"_{feature}.h5")
                    if os.path.exists(h5_path):
                        try:
                            with h5py.File(h5_path, 'r') as f:
                                array = f['data'][:]
                            h5_batch_data.append(array)
                            h5_batch_paths.append(h5_path)
                        except Exception as e:
                            print(f"Warning: Error reading {h5_path}: {e}")
                            continue
                
                # Normalize batch (vectorized operation)
                for i, array in enumerate(h5_batch_data):
                    h5_batch_data[i] = (array - mean) / std
                
                # Write batch
                for path, array in zip(h5_batch_paths, h5_batch_data):
                    try:
                        save_feature_hdf5(array, path)
                    except Exception as e:
                        print(f"Warning: Error writing {path}: {e}")
                        continue


# NEW FUNCTION: Normalizes data using pre-computed stats (with mode selection)
def normalize_features(files, features, stats, output_format='hdf5', use_batch=True, batch_size=50):
    """
    Normalizes features using provided statistics.
    
    This function supports two normalization methods:
    1. Batch processing (default, use_batch=True): Processes files in batches to reduce I/O overhead.
       This is faster and more efficient, especially for large datasets.
    2. Sequential processing (use_batch=False): Processes files one at a time.
       This is the original method, kept for backward compatibility.
    
    Both methods produce identical normalized files.
    
    Args:
        files: List of audio files
        features: List of feature types to normalize
        stats: Dictionary with mean and std for each feature
        output_format: Format of feature files - 'npy', 'hdf5', or 'both'
        use_batch: If True, use batch processing (default: True, faster)
                   If False, use sequential processing (original method)
        batch_size: Number of files to process in each batch (default: 50)
                    Only used when use_batch=True
    """
    # Print which method is being used
    method_name = f"Batch processing (batch_size={batch_size})" if use_batch else "Sequential processing"
    print(f"\n{'='*60}")
    print(f"Normalization method: {method_name}")
    print(f"{'='*60}\n")
    
    if use_batch:
        # Use batch processing for better I/O efficiency
        normalize_features_batch(files, features, stats, output_format, batch_size)
    else:
        # Use original sequential processing
        for feature in features:
            mean = stats[f"{feature}_mean"]
            std = stats[f"{feature}_std"]
            
            print(f"Normalizing {feature} features with mean={mean}, std={std}")
            
            for file in tqdm(files, desc=f"Normalizing {feature} (sequential)"):
                try:
                    # Normalize NPY files if they exist
                    if output_format in ['npy', 'both']:
                        npy_path = file.replace(".wav", f"_{feature}.npy")
                        if os.path.exists(npy_path):
                            array = np.load(npy_path)
                            array = (array - mean) / std
                            np.save(npy_path, array)
                    
                    # Normalize HDF5 files if they exist
                    if output_format in ['hdf5', 'both']:
                        h5_path = file.replace(".wav", f"_{feature}.h5")
                        if os.path.exists(h5_path):
                            # Read, normalize, and rewrite
                            with h5py.File(h5_path, 'r') as f:
                                array = f['data'][:]
                            array = (array - mean) / std
                            save_feature_hdf5(array, h5_path)
                except Exception as e:
                    print(f"Error normalizing {file}: {e}")
                    continue


def main(args):
    """Prepare features for further development across training, validation and test sets."""
    # Construct paths to all dataset splits
    base_path = args.data_dir
    dev_set_path = os.path.join(base_path, "Development_Set")
    
    train_path = os.path.join(dev_set_path, args.train_dir)
    val_path = os.path.join(dev_set_path, args.val_dir)
    test_path = os.path.join(dev_set_path, args.test_dir)
    
    # Define which features to extract
    features = ["logmel"]
    suffix = ".wav"
    output_format = args.output_format
    
    # Print configuration
    print(f"Output format: {output_format}")
    print(f"Features to extract: {features}")
    
    # Check dataset existence
    for dataset_path in [train_path, val_path, test_path]:
        if os.path.exists(dataset_path):
            print(f"Dataset at {dataset_path} is ready!")
        else:
            print(f"Dataset at {dataset_path} is not available!")
            
    # Extract features for all datasets
    print("\nStep 1: Extracting features for all datasets")
    for dataset_name, dataset_path in [("Training", train_path), ("Validation", val_path), ("Test", test_path)]:
        print(f"Processing {dataset_name} data...")
        dataset_files = recursive_glob(dataset_path, suffix)
        if len(dataset_files) > 0:
            print(f"Found {len(dataset_files)} files in {dataset_name} set")
            process(dataset_files, output_format=output_format)
        else:
            print(f"No files found in {dataset_name} set")
    
    # Save unnormalized features for visualization (sample of training set)
    train_files = recursive_glob(train_path, suffix)
    if len(train_files) > 0:
        print("\nSaving copies of unnormalized features for visualization...")
        for file in tqdm(train_files[:5]):  # process a sample of the files
            try:
                for k in features:
                    # Save unnormalized copy based on output format
                    if output_format in ['npy', 'both']:
                        npy_path = file.replace(".wav", f"_{k}.npy")
                        if os.path.exists(npy_path):
                            unnorm_path = file.replace(".wav", f"_{k}_unnormalized.npy")
                            np.save(unnorm_path, np.load(npy_path))
                    
                    if output_format in ['hdf5', 'both']:
                        h5_path = file.replace(".wav", f"_{k}.h5")
                        if os.path.exists(h5_path):
                            unnorm_path = file.replace(".wav", f"_{k}_unnormalized.h5")
                            with h5py.File(h5_path, 'r') as f:
                                data = f['data'][:]
                            save_feature_hdf5(data, unnorm_path)
            except:
                continue
    
        # Compute normalization statistics from training data ONLY
        stats = compute_and_save_norm_stats(train_files, features, dev_set_path, output_format=output_format)
        
        # Normalize all datasets using training stats
        print("\nStep 2: Normalizing features using training set statistics")
        for dataset_name, dataset_path in [("Training", train_path), ("Validation", val_path), ("Test", test_path)]:
            print(f"Normalizing {dataset_name} data...")
            dataset_files = recursive_glob(dataset_path, suffix)
            if len(dataset_files) > 0:
                normalize_features(dataset_files, features, stats, output_format=output_format)
            else:
                print(f"No files to normalize in {dataset_name} set")
                
        # Generate visualizations for training data
        print("\nStep 3: Generating visualizations")
        for feature in features:
            print(f"Generating visualizations for {feature} features...")
            visualize_before_after(train_files, feature_name=feature, output_format=output_format)
    
    print("\nFeature extraction and normalization complete for all datasets!")
    print(f"Output format used: {output_format}")


if __name__ == "__main__":
    flags = parse_args()
    main(flags)
