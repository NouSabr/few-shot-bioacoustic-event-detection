import itertools as it
import os
from glob import glob
from itertools import chain
import time
import h5py
import librosa
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Optional

from src.datamodules.components.pcen import FeatureLoader
from src.datamodules.components.frequency_masker import (
    build_frequency_masker,
    describe_masker,
)
from src.utils.afri_core import compute_afri_profile, _make_uniform_afri_profile


class LazyHDF5Concat:
    """Wrapper to lazily concatenate multiple HDF5 datasets along axis 1 (features)"""
    def __init__(self, h5_datasets):
        self.datasets = h5_datasets
        self.shape = (h5_datasets[0].shape[0], sum(d.shape[1] for d in h5_datasets))
    
    def __getitem__(self, key):
        # Handle slicing - load only requested rows from each dataset and concatenate
        chunks = [dataset[key] for dataset in self.datasets]
        return np.concatenate(chunks, axis=1)


class PrototypeDynamicArrayDataSet(Dataset):
    def __init__(self, path: dict = {}, features: dict = {}, train_param: dict = {},
                 freq_importance_tracker=None):
        """_summary_
            Load array from the assigned suffix
        Args:
            path (dict, optional): _description_. Defaults to {}.
            features (dict, optional): _description_. Defaults to {}.
            train_param (dict, optional): _description_. Defaults to {}.
            freq_importance_tracker (FrequencyImportanceTracker, optional): Phase 3 tracker. Defaults to None.
        """
        self.path = path
        self.features = features
        self.train_param = train_param
        self.samples_per_cls = train_param.n_shot * 2
        self.seg_len = train_param.seg_len
        self.fe = FeatureLoader(
            self.features, audio_path=[path.train_dir]
        )

        print(
            "Build the training dataset with suffix",
            self.features.feature_types.split("@"),
        )

        self.all_csv_files = self.get_all_csv_files()

        self.length = int(3600 * 8 / self.train_param.seg_len)
        self.fps = self.features.sr / self.features.hop_mel

        self.meta = {}
        self.pcen = {}
        self.h5_files = []  # Track open HDF5 files for cleanup
        self.cnt = 0
        self.batchsize = self.train_param.k_way * self.train_param.n_shot * 2

        # ==================== Frequency Masking ====================
        # Five independent strategies, selected by freq_mask_mode:
        #   'random'       Phase 1  uniformly random bands (SpecAugment)
        #   'slot_uniform' control  slot placement, P(b) uniform
        #   'energy'       Phase 2  away from the slots where the CLASS is loud
        #   'activation'   Phase 3  towards the slots the encoder uses for the CLASS
        #   'afri'                  fixed area inside/outside the class AFRI range
        # See docs/dynamic_freq_masking/01_OVERVIEW.md
        self.use_freq_masking = train_param.get('use_freq_masking', False)

        # Set once per dataloader worker, on its first __getitem__
        self._masker_worker_seeded = False

        # The mode -> class mapping lives in build_frequency_masker so it can
        # be exercised without audio, a model or a GPU; see
        # tools/sanity_check_configs.py. Profiles are attached below, in
        # build_meta(), once the class list is known.
        if self.use_freq_masking:
            self.freq_masker = build_frequency_masker(
                train_param, features.n_mels,
                importance_tracker=freq_importance_tracker,
            )
            print(f"\u2713 Frequency masking enabled: "
                  f"{train_param.get('freq_mask_mode', 'random')}")
            print(f"  - {describe_masker(self.freq_masker)}")
            if (train_param.get('freq_mask_mode') == 'afri'
                    and not train_param.get('use_afri', False)):
                print("  \u26a0\ufe0f  use_afri is False - no profiles will be "
                      "computed, so every class falls back to unguided masking")
        else:
            self.freq_masker = None
            print("✗ Frequency masking disabled")
        # ========================================================================

        """meta dic structure
        {
            <class-name>: {
                "info":[(<start-time>, <end-time>), ...],
                "duration": [duration1, duration2, ...]
            }
        }
        """
        self.build_meta()
        # self.remove_short_negative_duration()

        self.classes = list(self.meta.keys())

        self.classes2int = self.get_class2int()
        self.classes_duration = self.get_class_durations()
        self.train_eval_class_idxs = [self.classes2int[x] for x in self.classes]
        self.extra_train_class_idxs = []
        # The positive segment within the batch should come from the same segment
        self.segment_buffer = {}

    def __len__(self):
        # Every two hours of positive data
        return self.length

    def _seed_masker_for_worker(self) -> None:
        """
        Give each dataloader worker its own masking RNG stream, once.

        With num_workers > 0 the dataset is forked, so every worker inherits the
        same seeded RandomState and they would all draw IDENTICAL masks; and
        because masking only runs in workers, the parent's RNG never advances, so
        each epoch's fresh workers would replay the same sequence. worker_info.seed
        already varies by worker and by epoch while staying reproducible from the
        global seed.

        A no-op at num_workers=0 (the configured default), where there is a single
        continuous stream in this process and nothing to disambiguate.
        """
        if self._masker_worker_seeded or self.freq_masker is None:
            return
        self._masker_worker_seeded = True

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            self.freq_masker.reseed(worker_info.seed)

    def __getitem__(self, idx):
        # Random segment length
        # if(self.cnt % self.batchsize == 0): self._update_seglen()
        # if(self.cnt % self.batchsize == 0): self.segment_buffer = {}
        self._seed_masker_for_worker()
        class_name = self.classes[idx]

        segment = self.select_positive(class_name)

        ########################## Augmentation ################################
        # 48.9
        # aug_segment_neg = self.select_negative(class_name) # Try using negative data from other kinds of sounds
        # mix_ratio = float(np.random.uniform(0.0, 0.3))
        # segment = (segment + mix_ratio*aug_segment_neg) / (1+mix_ratio)
        ########################################################################

        # ==================== Phase 1, 2 & 3: Apply Frequency Masking ====================
        # Apply masking to positive segment if enabled
        if self.use_freq_masking and self.freq_masker is not None:
            segment = self.freq_masker(segment, class_name=class_name)
        # =================================================================================

        self.cnt += 1

        if not self.train_param.negative_train_contrast:
            return segment.astype(np.float32), self.classes2int[class_name], class_name
        else:
            segment_neg = self.select_negative(class_name)
            
            # ==================== Apply Frequency Masking ====================
            # Apply masking to the negative segment if enabled.
            # select_negative() draws from meta[class_name]["neg_info"], i.e. the
            # regions of this same class's recordings where the target sound is
            # absent — not from other classes. So class_name is the right key
            # here: the negative shares the positive's recording context, and its
            # AFRI range is the one that describes that context.
            if self.use_freq_masking and self.freq_masker is not None:
                segment_neg = self.freq_masker(segment_neg, class_name=class_name)
            # =================================================================================
            
            # Positive segment, negative segment, positive class label, negative class label
            return (
                segment.astype(np.float32),
                segment_neg.astype(np.float32),
                self.classes2int[class_name] * 2,
                self.classes2int[class_name] * 2 + 1,
                class_name,
            )

    def _update_seglen(self):
        self.seg_len = float(np.random.uniform(0.2, 0.5))

    def select_negative(self, class_name):
        segment_idx = np.random.randint(len(self.meta[class_name]["neg_info"]))
        start, end = self.meta[class_name]["neg_info"][segment_idx]
        while end - start < 0.2:  # TODO the value here
            segment_idx = np.random.randint(len(self.meta[class_name]["neg_info"]))
            start, end = self.meta[class_name]["neg_info"][segment_idx]
        segment = self.select_segment(
            start,
            end,
            self.pcen[self.meta[class_name]["file"][segment_idx]],
            seg_len=int(self.seg_len * self.fps),
        )
        return segment

    def select_positive(self, class_name):
        # TODO this is to cope with the unreliable labeling
        # if(class_name not in self.segment_buffer.keys()):
        #     segment_idx = np.random.randint(len(self.meta[class_name]["info"]))
        #     self.segment_buffer[class_name] = segment_idx
        # else:
        #     segment_idx = self.segment_buffer[class_name]

        segment_idx = np.random.randint(len(self.meta[class_name]["info"]))
        start, end = self.meta[class_name]["info"][segment_idx]

        segment = self.select_segment(
            start,
            end,
            self.pcen[self.meta[class_name]["file"][segment_idx]],
            seg_len=int(self.seg_len * self.fps),
        )
        return segment

    def select_segment(self, start, end, pcen, seg_len=17):
        start, end = int(start * self.fps), int(end * self.fps)
        if start < 0:
            start = 0  # This is due to the function time_2_frame
        total_duration = end - start
        if total_duration < seg_len:
            x = pcen[start:end]
            tile_times = np.ceil(seg_len / total_duration)
            x = np.tile(x, (int(tile_times), 1))
            x = x[:seg_len]
        else:
            rand_start = np.random.uniform(low=start, high=end - seg_len)
            x = pcen[int(rand_start) : int(rand_start) + seg_len]
        if x.shape[0] != seg_len:
            print(
                "Shape error, padded.",
                "%s %s %s %s %s %s"
                % (x.shape, seg_len, start, end, rand_start, pcen.shape),
            )
            x = np.pad(x, ((0, seg_len - x.shape[0]), (0, 0)))
        assert x.shape[0] == seg_len, "%s %s %s %s %s %s" % (
            x.shape,
            seg_len,
            start,
            end,
            rand_start,
            pcen.shape,
        )
        return x

    def _extract_raw_feature_segment(self, audio_path: str, start_sec: float, end_sec: float, n_mels: int) -> Optional[np.ndarray]:
        """
        Extract raw feature segment for AFRI computation.
        
        Args:
            audio_path: Path to audio file
            start_sec: Start time in seconds
            end_sec: End time in seconds
            n_mels: Number of mel bins
        
        Returns:
            Feature array of shape [F, T] (frequency x time) or None on failure
        """
        try:
            # Check if audio_path exists in pcen dictionary
            if audio_path not in self.pcen:
                return None
            
            # Convert time to frames
            start_frame = int(start_sec * self.fps)
            end_frame = int(end_sec * self.fps)
            
            # Ensure valid frame range
            if start_frame < 0:
                start_frame = 0
            if end_frame <= start_frame:
                return None
            
            # Access HDF5 data handle (lazy loading)
            pcen_handle = self.pcen[audio_path]
            
            # Extract segment - this triggers actual HDF5 read
            # HDF5 datasets and LazyHDF5Concat both support slicing
            segment = pcen_handle[start_frame:end_frame]
            
            # Convert to numpy array (in case it's still an HDF5 dataset view)
            segment = np.array(segment)
            
            # Check shape and transpose if needed
            # Dataset stores as [T, F], AFRI needs [F, T]
            if segment.ndim != 2:
                return None
            
            # Transpose from [T, F] to [F, T]
            segment = segment.T
            
            # Validate shape
            if segment.shape[0] != n_mels:
                return None
            
            # Ensure minimum duration
            if segment.shape[1] < 5:
                return None
            
            return segment
            
        except Exception as e:
            # Silently handle any errors (file access, shape mismatch, etc.)
            return None

    def _compute_energy_profiles(self) -> None:
        """
        Compute or load per-class mean energy profiles for Phase 2 masking.

        One profile per class: the mean absolute feature value per mel bin,
        averaged over that class's first ``afri_n_samples`` positive segments --
        deliberately the same segments AFRI profiles, so Phase 2 and AFRI see
        identical information and differ only in the transform applied to it
        (mean energy here, positive-minus-negative contrast there).

        Precomputed rather than tracked online because spectrogram energy is a
        property of the data: nothing about it changes during training. The
        previous implementation kept one global running EMA, which under the
        interleaved episodic order converged to a mixture over the k_way classes
        rather than any one class's profile.

        Cached beside the AFRI cache in its own file, keyed on the same inputs so
        a stale cache cannot be silently reused. Reads no audio when the cache
        hits.
        """
        from src.utils.afri_cache import generate_afri_cache_key, get_cache_paths

        n_mels = self.features.n_mels
        n_samples = self.train_param.get('afri_n_samples', 5)
        cache_enabled = self.train_param.get('afri_cache_enabled', True)
        force_recompute = self.train_param.get('afri_force_recompute', False)
        cache_dir = self.train_param.get('afri_cache_dir', None)

        energy_config = {
            'n_samples': n_samples,
            'statistic': 'mean_abs_over_time_of_first_n_positives',
        }
        feature_config = {
            'n_mels': n_mels,
            'sr': self.features.sr,
            'hop_mel': self.features.hop_mel,
        }
        cache_key = generate_afri_cache_key(
            self.path.train_dir, energy_config, feature_config
        )

        _, profiles_path = get_cache_paths(self.path.train_dir, cache_dir)
        energy_path = profiles_path.parent / "energy_profiles.npz"

        # ---- try the cache ----
        if cache_enabled and not force_recompute and energy_path.exists():
            try:
                cached = np.load(energy_path, allow_pickle=True)
                if str(cached['__cache_key__']) == cache_key:
                    n_loaded = 0
                    for cls_name in self.meta.keys():
                        if cls_name in cached.files:
                            self.meta[cls_name]["energy_profile"] = cached[cls_name]
                            n_loaded += 1
                    print(f"✓ Loaded energy profiles from cache "
                          f"(key: {cache_key[:8]}..., {n_loaded} classes)")
                    return
                print("Energy profile cache key mismatch - recomputing")
            except Exception as exc:
                print(f"Could not read energy profile cache ({exc}) - recomputing")

        # ---- compute ----
        print(f"Computing per-class energy profiles from the first {n_samples} "
              f"positive segments...")
        computed = 0
        for cls_idx, cls_name in enumerate(self.meta.keys(), 1):
            try:
                n_pos = len(self.meta[cls_name]["info"])
                actual_n_samples = min(n_samples, n_pos)
                per_bin = []

                for i in range(actual_n_samples):
                    start_sec, end_sec = self.meta[cls_name]["info"][i]
                    audio_path = self.meta[cls_name]["file"][i]
                    segment = self._extract_raw_feature_segment(
                        audio_path, start_sec, end_sec, n_mels
                    )
                    # _extract_raw_feature_segment returns [F, T]
                    if segment is None or segment.shape[0] != n_mels:
                        continue
                    per_bin.append(np.mean(np.abs(segment), axis=1))

                if not per_bin:
                    print(f"[{cls_idx}] {cls_name}: no usable segments - "
                          f"masking will fall back to unguided")
                    continue

                profile = np.mean(per_bin, axis=0).astype(np.float32)
                if not np.isfinite(profile).all() or profile.max() <= 0:
                    print(f"[{cls_idx}] {cls_name}: degenerate energy profile - "
                          f"masking will fall back to unguided")
                    continue

                self.meta[cls_name]["energy_profile"] = profile
                computed += 1
            except Exception as exc:
                print(f"[{cls_idx}] {cls_name}: energy profile failed ({exc}) - "
                      f"masking will fall back to unguided")

        print(f"Energy profiles computed for {computed}/{len(self.meta)} classes")

        # ---- save ----
        if cache_enabled and computed > 0:
            try:
                payload = {
                    cls: self.meta[cls]["energy_profile"]
                    for cls in self.meta.keys()
                    if "energy_profile" in self.meta[cls]
                }
                np.savez_compressed(
                    energy_path, __cache_key__=np.array(cache_key), **payload
                )
                print(f"✓ Saved energy profiles to cache (key: {cache_key[:8]}...)")
            except Exception as exc:
                print(f"Could not write energy profile cache ({exc})")

    def _compute_afri_profiles(self) -> None:
        """
        Compute or load cached AFRI profiles for all classes in the dataset.
        
        Called at the end of build_meta() if use_afri is enabled.
        
        Caching behavior:
        - If cache is enabled and valid, loads profiles from cache (<1 second)
        - If cache is disabled or invalid, computes profiles from scratch (2+ hours)
        - After computation, saves profiles to cache for future runs
        
        Cache configuration (from train_param):
        - afri_cache_enabled: Enable/disable caching (default: True)
        - afri_force_recompute: Force recomputation even if cache exists (default: False)
        - afri_cache_dir: Custom cache directory (default: {train_dir}/.afri_cache)
        """
        from src.utils.afri_cache import (
            generate_afri_cache_key,
            load_cached_profiles,
            save_profiles_to_cache
        )
        
        # Check if caching is enabled
        cache_enabled = self.train_param.get('afri_cache_enabled', True)
        force_recompute = self.train_param.get('afri_force_recompute', False)
        cache_dir = self.train_param.get('afri_cache_dir', None)
        
        # Prepare configuration dictionaries for cache key generation
        afri_config = {
            'afri_n_samples': self.train_param.get('afri_n_samples', 5),
            'afri_value_threshold': self.train_param.get('afri_value_threshold', 0.23),
            'afri_search_threshold': self.train_param.get('afri_search_threshold', 7),
            'afri_debris_size': self.train_param.get('afri_debris_size', 10),
            'afri_min_bin_expand': self.train_param.get('afri_min_bin_expand', 2),
            'afri_max_bin_expand': self.train_param.get('afri_max_bin_expand', 6),
        }
        feature_config = {
            'n_mels': self.features.get('n_mels', 128),
            'sr': self.features.get('sr', 22050),
            'hop_mel': self.features.get('hop_mel', 256),
        }
        
        # Try loading from cache
        if cache_enabled and not force_recompute:
            cache_key = generate_afri_cache_key(self.path.train_dir, afri_config, feature_config)
            cached_profiles = load_cached_profiles(self.path.train_dir, cache_key, cache_dir)
            
            if cached_profiles is not None:
                print("\n" + "="*60)
                print(f"✓ Loaded AFRI profiles from cache (key: {cache_key[:8]}...)")
                print("="*60)
                
                # Restore profiles to metadata
                for cls in self.meta.keys():
                    if cls in cached_profiles:
                        self.meta[cls]["afri_profile"] = cached_profiles[cls]
                    else:
                        print(f"Warning: Class '{cls}' not found in cache, using uniform profile")
                        self.meta[cls]["afri_profile"] = _make_uniform_afri_profile(self.features.n_mels)
                
                print(f"Loaded {len(cached_profiles)} class profiles from cache")
                print("="*60 + "\n")
                return
        
        # Cache miss or disabled - compute profiles from scratch
        print("\n" + "="*60)
        if force_recompute:
            print("Force recompute enabled - computing AFRI profiles from scratch...")
        else:
            print("Computing AFRI profiles from scratch...")
        print("="*60)
        
        # Get AFRI parameters from config
        n_samples = afri_config['afri_n_samples']
        n_mels = self.features.n_mels
        value_threshold = afri_config['afri_value_threshold']
        search_threshold = afri_config['afri_search_threshold']
        debris_size = afri_config['afri_debris_size']
        min_bin_expand = afri_config['afri_min_bin_expand']
        max_bin_expand = afri_config['afri_max_bin_expand']
        
        successful_profiles = 0
        failed_profiles = 0
        
        # Iterate over all classes
        for cls_idx, cls_name in enumerate(self.meta.keys(), 1):
            try:
                # Check if sufficient samples exist
                n_pos = len(self.meta[cls_name]["info"])
                n_neg = len(self.meta[cls_name]["neg_info"])
                
                # Get minimum required samples (configurable, default=2)
                min_required_samples = self.train_param.get('afri_min_samples', 2)
                
                # Check if we have minimum required samples
                if n_pos < min_required_samples or n_neg < min_required_samples:
                    print(f"[{cls_idx}] {cls_name}: Too few samples (pos={n_pos}, neg={n_neg}, min_required={min_required_samples}) - using uniform profile")
                    # Fallback: uniform profile treats all frequencies equally
                    self.meta[cls_name]["afri_profile"] = _make_uniform_afri_profile(n_mels)
                    failed_profiles += 1
                    continue
                
                # Use whatever samples are available (up to n_samples)
                actual_n_samples = min(n_samples, n_pos, n_neg)
                
                # Log if using fewer samples than requested
                if actual_n_samples < n_samples:
                    print(f"[{cls_idx}] {cls_name}: Using {actual_n_samples}/{n_samples} available samples (pos={n_pos}, neg={n_neg})")
                
                # Extract positive segments
                pos_spectrograms = []
                valid_pos = []
                
                for i in range(actual_n_samples):
                    start_sec, end_sec = self.meta[cls_name]["info"][i]
                    audio_path = self.meta[cls_name]["file"][i]
                    
                    segment = self._extract_raw_feature_segment(audio_path, start_sec, end_sec, n_mels)
                    
                    if segment is not None:
                        pos_spectrograms.append(segment)
                        valid_pos.append(True)
                    else:
                        # Add placeholder
                        pos_spectrograms.append(np.zeros((n_mels, 10)))
                        valid_pos.append(False)
                
                # Extract negative segments
                neg_spectrograms = []
                valid_neg = []
                
                if len(self.meta[cls_name]["neg_info"]) != len(self.meta[cls_name]["file"]):
                    print(f"[{cls_idx}] {cls_name}: Alignment error: neg_info length ({len(self.meta[cls_name]['neg_info'])}) != file length ({len(self.meta[cls_name]['file'])}) - using uniform profile")
                    # Fallback: uniform profile treats all frequencies equally
                    self.meta[cls_name]["afri_profile"] = _make_uniform_afri_profile(n_mels)
                    failed_profiles += 1
                    continue
                
                for i in range(actual_n_samples):
                    start_sec, end_sec = self.meta[cls_name]["neg_info"][i]
                    audio_path = self.meta[cls_name]["file"][i]
                    
                    segment = self._extract_raw_feature_segment(audio_path, start_sec, end_sec, n_mels)
                    
                    if segment is not None:
                        neg_spectrograms.append(segment)
                        valid_neg.append(True)
                    else:
                        # Add placeholder
                        neg_spectrograms.append(np.zeros((n_mels, 10)))
                        valid_neg.append(False)
                
                # Check if we have any valid samples
                if not any(valid_pos) or not any(valid_neg):
                    print(f"[{cls_idx}] {cls_name}: No valid segments extracted - using uniform profile")
                    # Fallback: uniform profile treats all frequencies equally
                    self.meta[cls_name]["afri_profile"] = _make_uniform_afri_profile(n_mels)
                    failed_profiles += 1
                    continue
                
                # Filter valid samples before calling AFRI
                valid_pos_specs = [spec for spec, valid in zip(pos_spectrograms, valid_pos) if valid]
                valid_neg_specs = [spec for spec, valid in zip(neg_spectrograms, valid_neg) if valid]
                
                # Call AFRI core algorithm with configured parameters
                afri_result = compute_afri_profile(
                    pos_spectrograms=valid_pos_specs,
                    neg_spectrograms=valid_neg_specs,
                    n_mels=n_mels,
                    debris=debris_size,
                    value_threshold=value_threshold,
                    searchLbin_threshold=search_threshold,
                    searchRbin_threshold=search_threshold
                )
                
                # Store result
                self.meta[cls_name]["afri_profile"] = afri_result
                
                # Print summary
                if afri_result['valid']:
                    freq_range = afri_result['auto_maxbin'] - afri_result['auto_minbin']
                    print(f"[{cls_idx}] {cls_name}: ✓ Profile computed (range: {afri_result['auto_minbin']}-{afri_result['auto_maxbin']}, width: {freq_range} bins)")
                    successful_profiles += 1
                else:
                    print(f"[{cls_idx}] {cls_name}: ⚠ Fallback to uniform profile")
                    failed_profiles += 1
                    
            except Exception as e:
                # Catch any unexpected errors
                print(f"[{cls_idx}] {cls_name}: ✗ Error: {str(e)[:50]} - using uniform profile")
                # Fallback: uniform profile treats all frequencies equally
                self.meta[cls_name]["afri_profile"] = _make_uniform_afri_profile(n_mels)
                failed_profiles += 1
        
        # Print final summary
        print("="*60)
        print(f"AFRI Profile Computation Complete:")
        print(f"  ✓ Successful: {successful_profiles}")
        print(f"  ✗ Failed/Fallback: {failed_profiles}")
        print(f"  Total classes: {successful_profiles + failed_profiles}")
        print("="*60 + "\n")
        
        # Save to cache after computation
        if cache_enabled:
            try:
                # Collect all profiles
                profiles = {cls: self.meta[cls]["afri_profile"] for cls in self.meta.keys()}
                
                # Generate cache key (same as before)
                cache_key = generate_afri_cache_key(self.path.train_dir, afri_config, feature_config)
                
                # Save to cache
                success = save_profiles_to_cache(
                    profiles, self.path.train_dir, cache_key,
                    afri_config, feature_config, cache_dir
                )
                
                if success:
                    print(f"✓ Saved AFRI profiles to cache (key: {cache_key[:8]}...)")
                else:
                    print("Failed to save profiles to cache")
            except Exception as e:
                print(f"Error saving to cache: {e}")

    def build_meta(self):
        from tqdm import tqdm

        print("Preparing meta data...")
        # Main function for building up meta data
        for file in tqdm(self.all_csv_files):
            glob_cls_name = self.get_glob_cls_name(file)
            df_pos = self.get_df_pos(file)
            start_time, end_time = self.get_time(df_pos)
            cls_list = self.get_cls_list(df_pos, glob_cls_name, start_time)
            self.update_meta(start_time, end_time, cls_list, file)
        
        # Phase 2 needs per-class energy profiles. Driven by the masking mode
        # rather than a separate switch, so the profiles cannot be silently
        # missing for the one strategy that consumes them.
        if (self.use_freq_masking and
            self.freq_masker is not None and
            hasattr(self.freq_masker, 'energy_profiles')):

            self._compute_energy_profiles()

            energy_profiles = {
                cls_name: self.meta[cls_name]["energy_profile"]
                for cls_name in self.meta.keys()
                if "energy_profile" in self.meta[cls_name]
            }
            self.freq_masker.energy_profiles = energy_profiles

            stats = self.freq_masker.get_energy_stats()
            print(f"\n✓ Energy profiles passed to the Phase 2 masker:")
            print(f"  - Total classes: {len(self.meta)}")
            print(f"  - Classes with a usable profile: {stats['n_usable']}")
            print(f"  - Unguided fallback: {len(self.meta) - stats['n_usable']}")

        # Compute AFRI profiles if enabled
        if self.train_param.get('use_afri', False):
            self._compute_afri_profiles()
            
            # Hand the profiles to the masker if it is one that consumes them.
            # Duck-typed rather than isinstance so the maskers stay independent
            # siblings; AFRIRangeMasker is the only one with this attribute.
            if (self.use_freq_masking and
                self.freq_masker is not None and
                hasattr(self.freq_masker, 'afri_profiles')):

                # Extract AFRI profiles from meta
                afri_profiles = {}
                for cls_name in self.meta.keys():
                    if "afri_profile" in self.meta[cls_name]:
                        afri_profiles[cls_name] = self.meta[cls_name]["afri_profile"]

                # Pass to masker
                self.freq_masker.afri_profiles = afri_profiles

                stats = self.freq_masker.get_afri_stats()
                print(f"\n✓ AFRI profiles passed to the AFRI masker:")
                print(f"  - Total classes: {stats['n_classes']}")
                print(f"  - Classes with a usable range: {stats['n_valid']}")
                print(f"  - Unguided fallback: "
                      f"{stats['n_classes'] - stats['n_valid']}")
                print(f"  - Mode: {stats['afri_mode']}")
                print(f"  - Mean masked area: {stats['mean_masked_area']:.1f} bins "
                      f"(budget {stats['area_budget']})")

    def update_meta(self, start_time, end_time, cls_list, csv_file):
        audio_path = csv_file.replace("csv", "wav")
        for start, end, cls in zip(start_time, end_time, cls_list):
            if cls not in self.meta.keys():
                self.meta[cls] = {}
                self.meta[cls]["neg_start_time"] = 0  # temp variable
                self.meta[cls]["info"] = []  # positive segment onset and offset
                self.meta[cls]["duration"] = []  # duration of positive segments
                self.meta[cls]["file"] = []  # filename
                self.meta[cls]["neg_info"] = []  # negative segment onset and offset
            self.meta[cls]["neg_info"].append((self.meta[cls]["neg_start_time"], start))
            self.meta[cls]["info"].append((start, end))
            self.meta[cls]["duration"].append(end - start)
            self.meta[cls]["file"].append(audio_path)
            self.meta[cls]["neg_start_time"] = end

            if audio_path not in self.pcen.keys():
                # Check if HDF5 files exist for all configured feature types
                all_h5_exist = True
                h5_paths = []
                for suffix in self.fe.feature_types:
                    h5_path = audio_path.replace(".wav", f"_{suffix}.h5")
                    h5_paths.append(h5_path)
                    if not os.path.exists(h5_path):
                        all_h5_exist = False
                        break
                
                if all_h5_exist:
                    # Open HDF5 files and store handles for lazy loading
                    h5_datasets = []
                    for h5_path in h5_paths:
                        h5_file = h5py.File(h5_path, 'r')
                        self.h5_files.append(h5_file)
                        h5_datasets.append(h5_file['data'])
                    
                    # Use wrapper for lazy concatenation if multiple features, direct handle if single
                    if len(h5_datasets) == 1:
                        self.pcen[audio_path] = h5_datasets[0]
                    else:
                        self.pcen[audio_path] = LazyHDF5Concat(h5_datasets)
                else:
                    # Fall back to feature extractor (handles NPY or generates features)
                    self.pcen[audio_path] = self.fe.extract_feature(audio_path)

    def remove_short_negative_duration(self):
        delete_keys = []
        for k in self.meta.keys():
            neg_info = self.meta[k]["neg_info"]
            end_time = [x[1] for x in neg_info]
            start_time = [x[0] for x in neg_info]
            duration = max([x - y for x, y in zip(end_time, start_time)])
            if duration < 0.3:
                delete_keys.append(k)
        print("Delete class due to short negative length", len(delete_keys))
        for k in delete_keys:
            del self.meta[k]

    def get_class_durations(self):
        durations = []
        for cls in self.classes:
            durations.append(np.sum(self.meta[cls]["duration"]))
        return durations

    def get_all_csv_files(self):
        extension = "*.csv"

        return [
            file
            for path_dir, _, _ in os.walk(self.path.train_dir)
            for file in glob(os.path.join(path_dir, extension))
        ]

    def get_glob_cls_name(self, file):
        split_list = file.split("/")
        # import ipdb; ipdb.set_trace()
        # return split_list[split_list.index("Training_Set") + 1]
        return split_list[-2]

    def get_df_pos(self, file):
        df = pd.read_csv(file, header=0, index_col=False)  # TODO
        return df[(df == "POS").any(axis=1)]

    def get_cls_list(self, df_pos, glob_cls_name, start_time):
        if "CALL" in df_pos.columns:
            cls_list = [glob_cls_name] * len(start_time)
        else:
            cls_list = [
                df_pos.columns[(df_pos == "POS").loc[index]].values
                for index, row in df_pos.iterrows()
            ]
            cls_list = list(chain.from_iterable(cls_list))
        return cls_list

    def get_time(self, df):
        """Margin of 25 ms around the onset and offsets."""

        # TODO hardcode
        df.loc[:, "Starttime"] = df["Starttime"] - 0.025
        df.loc[:, "Endtime"] = df["Endtime"] + 0.025

        "Converting time to frames"

        start_time = [start for start in df["Starttime"]]
        end_time = [end for end in df["Endtime"]]

        return start_time, end_time

    def time2frame(self, t, fps):
        return int(np.floor(t * fps))

    def get_class2int(self):

        """Convert class label to integer
        Args:
        -label_array: label array
        -class_set: unique classes in label_array

        Out:
        -y: label to index values
        """
        label2indx = {label: index for index, label in enumerate(self.classes)}
        y = np.array([label2indx[label] for label in self.classes])
        return {c: idx for c, idx in zip(self.classes, y)}

    def __del__(self):
        """Close all HDF5 files on cleanup"""
        for h5_file in self.h5_files:
            try:
                h5_file.close()
            except:
                pass


def calculate_mean_std():
    import torch
    from omegaconf import OmegaConf
    from tqdm import tqdm

    from src.datamodules.components.identity_sampler import IdentityBatchSampler

    data = []

    conf = OmegaConf.load("baselines/dcase2024_task5/configs/train.yaml")
    dataset = PrototypeDynamicArrayDataSet(
        path=conf.path, features=conf.features, train_param=conf.train_param
    )
    sampler = IdentityBatchSampler(
        conf.train_param,
        len(dataset.classes),
        batch_size=conf.train_param.n_shot * 2,
        n_episode=int(
            len(dataset) / (conf.train_param.k_way * conf.train_param.n_shot * 2)
        ),
    )
    loader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler, num_workers=0)
    # mean: 1.4421, std: 1.2201
    for each in tqdm(loader):
        x, x_neg, y, y_neg, class_name = each
        print("here")
        # import ipdb; ipdb.set_trace()
        # print(torch.mean(x), torch.std(x))
        # print(x.size(), y.size())


if __name__ == "__main__":
    calculate_mean_std()
