import numpy as np
import os
from glob import glob
from tqdm import tqdm
import torch
import h5py


def recursive_glob(path, suffix):
    return (
        glob(os.path.join(path, "*" + suffix))
        + glob(os.path.join(path, "*/*" + suffix))
        + glob(os.path.join(path, "*/*/*" + suffix))
        + glob(os.path.join(path, "*/*/*/*" + suffix))
        + glob(os.path.join(path, "*/*/*/*/*" + suffix))
        + glob(os.path.join(path, "*/*/*/*/*/*" + suffix))
    )


class FeatureLoader:
    mean_std = {}

    def __init__(self, features, audio_path=[], skip_runtime_normalization=True):
        self.sr = features.sr
        self.n_fft = features.n_fft
        self.hop = features.hop_mel
        self.n_mels = features.n_mels
        self.fmax = features.fmax
        self.feature_types = features.feature_types.split("@")
        self.skip_runtime_normalization = skip_runtime_normalization
        self.files = []
        for each in audio_path:
            if(each is not None):
                assert os.path.exists(each), "Path not found: %s" % each
            print("Looking for data in: ", os.path.abspath(each))
            data = list(recursive_glob(each, ".wav"))
            self.files += data
            print("Find %s audio files" % (len(data)))

        self.files = np.random.permutation(self.files)

        # Runtime mean/std estimation is skipped by default because the stored feature files
        # are already normalized during preprocessing in data_preparation.py.
        #
        # See the preprocessing normalization flow in:
        #   - compute_and_save_norm_stats(...), data_preparation.py:576-622
        #   - normalize_features_batch(...), data_preparation.py:626-717
        #   - normalize_features(...), data_preparation.py:720-780
        #
        # Recomputing mean/std here is therefore redundant work at training/inference time and
        # can also lead to an unintended second normalization pass in extract_feature().
        #
        # This behavior is intentionally controlled by a flag so it is easy to reverse:
        # set skip_runtime_normalization=False to restore the previous runtime statistic
        # calculation + normalization behavior if a future pipeline uses raw, unnormalized
        # feature files instead.
        if not self.skip_runtime_normalization:
            self.update_mean_std()

        self.feature_lens = []

    def update_mean_std(self, feature_types=None):
        """Calculate mean and std in a memory-efficient way using two passes
        
        Supports both HDF5 (.h5) and NPY (.npy) files, with HDF5 taking priority.
        """
        if len(list(FeatureLoader.mean_std.keys())) != 0:
            return
            
        print("Calculating mean and std using two-pass method")
        
        # Determine which feature types to process
        features_to_process = self.feature_types if feature_types is None else feature_types
        
        for suffix in features_to_process:
            print(f"Processing feature type: {suffix}")
            
            # First pass: calculate sum and count
            n_elements = 0
            feature_sum = 0.0
            
            print("Pass 1: Calculating mean...")
            for audio_path in tqdm(self.files[:1000]):  # Use first 1000 files as before
                try:
                    # Check for HDF5 file first, fall back to NPY
                    h5_path = audio_path.replace(".wav", f"_{suffix}.h5")
                    npy_path = audio_path.replace(".wav", f"_{suffix}.npy")
                    
                    if os.path.exists(h5_path):
                        # Load from HDF5 file
                        with h5py.File(h5_path, 'r') as f:
                            array = f['data'][:]
                    elif os.path.exists(npy_path):
                        # Fall back to NPY file
                        array = np.load(npy_path)
                    else:
                        continue
                    
                    feature_sum += np.sum(array, dtype=np.float64)  # Use float64 for accumulation
                    n_elements += array.size
                    # Free memory immediately
                    del array
                except Exception as e:
                    print(f"Error processing feature file for {audio_path}: {e}")
                    continue
            
            # Calculate mean
            mean = feature_sum / max(n_elements, 1e-8)
            
            # Second pass: calculate variance sum
            variance_sum = 0.0
            print("Pass 2: Calculating std...")
            for audio_path in tqdm(self.files[:1000]):
                try:
                    # Check for HDF5 file first, fall back to NPY
                    h5_path = audio_path.replace(".wav", f"_{suffix}.h5")
                    npy_path = audio_path.replace(".wav", f"_{suffix}.npy")
                    
                    if os.path.exists(h5_path):
                        # Load from HDF5 file
                        with h5py.File(h5_path, 'r') as f:
                            array = f['data'][:]
                    elif os.path.exists(npy_path):
                        # Fall back to NPY file
                        array = np.load(npy_path)
                    else:
                        continue
                    
                    # Calculate squared differences from mean
                    variance_sum += np.sum((array - mean) ** 2, dtype=np.float64)
                    del array
                except Exception as e:
                    print(f"Error processing feature file for {audio_path}: {e}")
                    continue
            
            # Calculate std
            std = np.sqrt(variance_sum / max(n_elements, 1e-8))
            
            # Store results
            FeatureLoader.mean_std[suffix] = [mean, std]
            print(f"{suffix}: Mean = {mean}, Std = {std}")
        
        print("Feature statistics calculation complete.")
        print(FeatureLoader.mean_std)

    def extract_feature(self, audio_path, feature_types=None, normalized=True):
        """Extract features from audio file
        
        Supports both HDF5 (.h5) and NPY (.npy) files, with HDF5 taking priority.
        
        Args:
            audio_path: Path to the audio file (.wav)
            feature_types: List of feature types to extract (default: self.feature_types)
            normalized: Whether to normalize features using mean/std (default: True)
            
        Returns:
            Concatenated feature array
        """
        features = []
        for suffix in self.feature_types if (feature_types is None) else feature_types:
            # Check for HDF5 file first, fall back to NPY
            h5_path = audio_path.replace(".wav", f"_{suffix}.h5")
            npy_path = audio_path.replace(".wav", f"_{suffix}.npy")
            
            # Load the feature array
            if os.path.exists(h5_path):
                # Load from HDF5 file
                with h5py.File(h5_path, 'r') as f:
                    feature_array = f['data'][:]
            elif os.path.exists(npy_path):
                # Fall back to NPY file
                feature_array = np.load(npy_path)
            else:
                raise FileNotFoundError(
                    f"Feature file not found for {audio_path} with suffix {suffix}. "
                    f"Checked: {h5_path} and {npy_path}"
                )
            
            # Apply runtime normalization only when explicitly enabled.
            #
            # By default, skip_runtime_normalization=True because feature files are already
            # normalized during preprocessing in data_preparation.py (see lines 576-780).
            # In the default path we return the stored feature array as-is to avoid redundant
            # mean/std calculation and a second normalization pass.
            if (not normalized) or self.skip_runtime_normalization:
                features.append(feature_array)
            else:
                mean, std = FeatureLoader.mean_std[suffix]
                features.append((feature_array - mean) / std)
            
            self.feature_lens.append(features[-1].shape[1])
        
        return np.concatenate(features, axis=1)

    # def apply_mask(self, features, mask):
    #     start = 0
    #     import ipdb; ipdb.set_trace()
    #     for len, suffix in zip(self.feature_lens, self.feature_types):
    #         if(len != 128):
    #             continue
    #         else:
    #             mean, std = Feature_Extractor.mean_std[suffix]
    #             features[:, start:start +len] = (features[:, start:start +len] * std + mean) * mask
    #             features[:, start:start +len] = (features[:, start:start +len] - mean) / std
    #         start += len
