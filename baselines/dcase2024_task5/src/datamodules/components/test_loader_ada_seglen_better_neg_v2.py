# Use the first few labeled segments to construct negative samples

import sys


sys.path.append("/vol/research/dcase2022/project/hhlab")
import torch
import os
from glob import glob
import librosa
import gc
import h5py
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image
from src.datamodules.components.pcen import FeatureLoader
from src.datamodules.components.Datagenerator import Datagen_test
from scipy.signal import chirp, find_peaks, peak_widths, peak_prominences


def precompute_required_neg_indices(total_segments, samples_per_iter, iterations, seed):
    """
    Pre-compute which negative segments will be sampled across all iterations.
    
    Local generators per iteration to prevent global RNG state contamination while ensuring determinism.
    
    Args:
        total_segments: Total number of negative segments available
        samples_per_iter: Number of samples per iteration (e.g., 150)
        iterations: Number of iterations (e.g., 3)
        seed: Base random seed for deterministic sampling
    
    Returns:
        tuple: (required_indices, iteration_positions)
            - required_indices: sorted list of unique indices to load
            - iteration_positions: dict mapping iteration_id → positions in loaded array
    """
    required_indices = set()
    iteration_to_original = {}
    
    for iter_idx in range(iterations):
        # Create local generator with unique seed per iteration
        local_generator = torch.Generator()
        local_generator.manual_seed(seed + iter_idx)
        
        # Sample using local generator
        indices = torch.randperm(total_segments, generator=local_generator)[:samples_per_iter]
        iteration_to_original[iter_idx] = indices.tolist()
        required_indices.update(indices.tolist())
    
    # Sort for memory-efficient sequential loading
    sorted_indices = sorted(required_indices)
    
    # Create mapping: original_index → position_in_loaded_array
    original_to_position = {orig_idx: pos for pos, orig_idx in enumerate(sorted_indices)}
    
    # Convert iteration→original_indices to iteration→positions
    iteration_positions = {}
    for iter_idx, original_indices in iteration_to_original.items():
        iteration_positions[iter_idx] = [
            original_to_position[orig_idx] for orig_idx in original_indices
        ]
    
    return sorted_indices, iteration_positions


def time_2_frame(df, fps, padding=0.025):
    "Margin of 25 ms around the onset and offsets"

    df.loc[:, "Starttime"] = df["Starttime"] - padding
    df.loc[:, "Endtime"] = df["Endtime"] + padding

    "Converting time to frames"
    start_time = [int(np.floor(start * fps)) for start in df["Starttime"]]
    end_time = [int(np.floor(end * fps)) for end in df["Endtime"]]
    return np.clip(start_time, a_min=0, a_max=None), np.clip(
        end_time, a_min=0, a_max=None
    )


class PrototypeAdaSeglenBetterNegTestSetV2(Dataset):
    def __init__(
        self,
        path: dict = {},
        features: dict = {},
        train_param: dict = {},
        eval_param: dict = {},
    ):
        """
        Load data for model testing.

        Strategy applied:
        1. Adaptive segment length
        2. Sample negative data from the labelled part only, unless:
            2.1 The negative data in the labelled part is too short, in this case, we will use energy to determine which part in the query segment should be negative.
        3. Sample two sets of data for inference, one with adaptive segment length, another with short segment length.
            3.1 Adaptive segment length is used for identify the positive segment in the query set.
            3.2 Short segment length is used for identify the possible negative segments, which are used later for post-processing.

        Args:
            path (dict, optional): _description_. Defaults to {}.
            features (dict, optional): _description_. Defaults to {}.
            train_param (dict, optional): _description_. Defaults to {}.
            eval_param (dict, optional): _description_. Defaults to {}.
        """
        self.path = path
        self.features = features
        self.train_param = train_param
        self.eval_param = eval_param

        self.test_seglen_len_lim = features.test_seglen_len_lim
        self.test_hoplen_fenmu = features.test_hoplen_fenmu

        self.fps = self.features.sr / self.features.hop_mel

        # if path.test_dir is not None:
        #     self.fe = FeatureLoader(
        #         self.features, audio_path=[path.train_dir, path.eval_dir, path.test_dir]
        #     )
        # else:
        self.fe = FeatureLoader(
            self.features, audio_path=[path.eval_dir]
        )

        extension = "*.csv"
        self.all_csv_files = [
            file
            for path_dir, _, _ in os.walk(self.path.eval_dir)
            for file in glob(os.path.join(path_dir, extension))
        ]
        if path.test_dir is not None:
            self.all_csv_files += [
                file
                for path_dir, _, _ in os.walk(self.path.test_dir)
                for file in glob(os.path.join(path_dir, extension))
            ]

        # interested = ["BUK1_20181013_023504"]
        # print("!!!!!!!!!!!!!!", interested)
        # temp = []
        # for each in self.all_csv_files:
        #     for interest in interested:
        #         if(interest not in each): continue
        #         else:
        #             temp.append(each)
        #             break
        # self.all_csv_files = temp

        self.all_csv_files = sorted(self.all_csv_files)

    def __len__(self):
        return len(self.all_csv_files)

    def __getitem__(self, idx):
        """
        Get item with lazy loading support.
        
        Memory Optimization Phase 2:
        - Support segments are returned as numpy arrays (eager-loaded)
        - Query segments are returned as metadata + h5_handles (lazy-loaded)
        
        Reference: docs/memory_optimization/03_IMPLEMENTATION_PLAN.md
        """
        feat_file = self.all_csv_files[idx]
        # print(feat_file)
        (
            X_pos,
            X_neg,
            X_query_info,  # Metadata list instead of array
            X_pos_neg,
            X_neg_neg,
            X_query_info_neg,  # Metadata list instead of array
            max_len,
            neg_min_length,
            strt_index_query,
            audio_path,
            hop_seg,
            hop_seg_neg,
            mask,
            seg_len,
            h5_handles,  # HDF5 handles for lazy loading
            iteration_positions,      
            iteration_positions_neg,  
        ) = self.read_file(feat_file)
        
        # Return metadata instead of eagerly loaded arrays
        # HDF5 handles remain open for streaming
        return (
            (
                X_pos.astype(np.float32),
                X_neg.astype(np.float32),
                X_query_info,  # Metadata for streaming
                X_pos_neg.astype(np.float32),
                X_neg_neg.astype(np.float32),
                X_query_info_neg,  # Metadata for streaming
                hop_seg,
                hop_seg_neg,
                max_len,
                neg_min_length,
                iteration_positions,      
                iteration_positions_neg,  
            ),
            strt_index_query,
            audio_path,
            seg_len,
            h5_handles,  # Return HDF5 handles for streaming
        )

    def load_mask(self, class_name):
        if self.path.mask_dir is None:
            return None
        return np.load(os.path.join(self.path.mask_dir, "mask_%s.npy" % class_name))

    def find_positive_label(self, df):
        for col in df.columns:
            if "Q" in col or "E_" in col:
                return col.strip()
        else:
            raise ValueError(
                "Error: Expect you change the validation set event name to Q_x or E_x"
            )

    def segment(
        self,
        h5_handles,
        hop_seg,
        index_sup,
        start_time,
        end_time,
        seg_len,
        negative_onset_offset,
        negative_pcen=None,
    ):
        """
        Segment audio with lazy loading from HDF5 handles.
        
        Memory Optimization Phase 2:
        - Support segments (positive/negative): Eager-loaded into memory (needed for random sampling)
        - Query segments: Return metadata only (will be streamed later)
        
        Reference: docs/memory_optimization/03_IMPLEMENTATION_PLAN.md
        """
        hop_neg = 0
        hop_query = 0
        strt_index = 0

        strt_indx_query = end_time[index_sup[-1]]
        # Get total frames from first HDF5 handle
        end_idx_neg = h5_handles[0].shape[0] - 1

        feat_neg, feat_pos, feat_query_info = [], [], []
        
        ####################################BUILD NEG DATASET########################################
        # SMART LAZY-LOADING: Only load segments that will be sampled
        # Memory optimization: 194x reduction (3.4 GB → 17.5 MB for extreme cases)
        
        # Check if smart lazy-loading is enabled (default: True for memory efficiency)
        use_smart_lazy = getattr(self.eval_param, 'use_smart_lazy_loading', True)
        
        if use_smart_lazy:
            # Step 1: Load all negative segments to compute total re-segmented count
            # (We need this to know how many segments will be created after re-segmentation)
            negative_segments_temp = []
            for s_neg, e_neg in negative_onset_offset:
                segment_data = h5_handles[0][s_neg:e_neg, :]
                negative_segments_temp.append(segment_data)
            negative_segments_full = np.concatenate(negative_segments_temp, axis=0)
            del negative_segments_temp
            
            if negative_pcen is not None:
                negative_segments_full = np.concatenate(
                    [negative_segments_full, negative_pcen], axis=0
                )
            
            # Step 2: Compute how many re-segmented patches will be created
            total_resegmented = 0
            hop_neg_temp = 0
            while negative_segments_full.shape[0] - hop_neg_temp > seg_len:
                total_resegmented += 1
                hop_neg_temp += hop_seg // 2
            total_resegmented += 1  # Last patch
            
            # Step 3: Pre-compute which indices will be sampled
            seed = getattr(self.eval_param, 'seed', 1234)  # Get seed from config
            samples_neg = getattr(self.eval_param, 'samples_neg', 150)
            iterations = getattr(self.eval_param, 'iterations', 3)
            
            required_indices, iteration_positions = precompute_required_neg_indices(
                total_resegmented, samples_neg, iterations, seed
            )
            
            # Step 4: Only create the required patches (smart lazy-loading)
            required_indices_set = set(required_indices)
            patch_idx = 0
            hop_neg = 0
            
            while negative_segments_full.shape[0] - hop_neg > seg_len:
                if patch_idx in required_indices_set:
                    patch_neg = negative_segments_full[int(hop_neg) : int(hop_neg + seg_len)]
                    feat_neg.append(patch_neg)
                patch_idx += 1
                hop_neg += hop_seg // 2
            
            # Last patch
            if patch_idx in required_indices_set:
                last_patch = negative_segments_full[-seg_len:]
                feat_neg.append(last_patch)
            
            # Memory cleanup
            del negative_segments_full
            gc.collect()
            
        else:
            # FALLBACK: Original eager-loading (load all segments)
            iteration_positions = None  # No precomputed positions in fallback mode
            negative_segments = []
            for s_neg, e_neg in negative_onset_offset:
                segment_data = h5_handles[0][s_neg:e_neg, :]
                negative_segments.append(segment_data)
            negative_segments = np.concatenate(negative_segments, axis=0)

            if negative_pcen is not None:
                negative_segments = np.concatenate(
                    [negative_segments, negative_pcen], axis=0
                )
            
            # Re-segmentation (all patches)
            while negative_segments.shape[0] - hop_neg > seg_len:
                patch_neg = negative_segments[int(hop_neg) : int(hop_neg + seg_len)]
                feat_neg.append(patch_neg)
                assert (hop_seg // 2) > 0
                hop_neg += hop_seg // 2

            last_patch = negative_segments[-seg_len:]
            feat_neg.append(last_patch)
            
            # Memory cleanup
            del negative_segments
            gc.collect()
        #############################################################################################

        # EAGER LOAD: Positive support segments
        for index in index_sup:
            str_ind = int(start_time[index])
            end_ind = int(end_time[index])

            # Load from HDF5
            patch_pos = h5_handles[0][str_ind:end_ind, :]
            
            if end_ind - str_ind > seg_len:
                shift = 0
                while end_ind - (str_ind + shift) > seg_len:
                    patch_pos_seg = h5_handles[0][
                        int(str_ind + shift) : int(str_ind + shift + seg_len), :
                    ]
                    feat_pos.append(patch_pos_seg)
                    assert hop_seg > 0
                    shift += hop_seg
                last_patch_pos = h5_handles[0][end_ind - seg_len : end_ind, :]
                feat_pos.append(last_patch_pos)
            else:
                if patch_pos.shape[0] == 0:
                    print("The patch is of 0 length", str_ind, end_ind)
                    continue
                repeat_num = int(seg_len / (patch_pos.shape[0])) + 1
                patch_new = np.tile(patch_pos, (repeat_num, 1))
                patch_new = patch_new[0 : int(seg_len)]
                feat_pos.append(patch_new)
        
        # GC OPTIMIZATION: Trigger GC after positive segment processing
        # Helps clean up temporary arrays from loop iterations
        gc.collect()
        
        # LAZY LOAD: Query segments - return metadata only
        while end_idx_neg - (strt_indx_query + hop_query) > seg_len:
            # Store metadata instead of loading data
            start_frame = int(strt_indx_query + hop_query) - seg_len // 2
            end_frame = int(strt_indx_query + hop_query + seg_len) - seg_len // 2
            feat_query_info.append({
                'start_frame': start_frame,
                'end_frame': end_frame,
                'seg_len': seg_len
            })
            assert hop_seg > 0
            hop_query += hop_seg

        # Last query segment metadata
        feat_query_info.append({
            'start_frame': end_idx_neg - seg_len,
            'end_frame': end_idx_neg,
            'seg_len': seg_len
        })
        
        return feat_pos, feat_neg, feat_query_info, strt_indx_query, iteration_positions

    def negative_onset_offset_estimate(self, audio_path, start_time, end_time):
        x, sr = librosa.load(audio_path, sr=None)
        assert sr == 22050
        rms = librosa.feature.rms(
            y=x, frame_length=1024, hop_length=256
        )  # TODO hard code here
        rms = rms[0, ...]

        # Try range limit
        pos_segments = []
        for s, e in zip(start_time, end_time):
            pos_segments.append(rms[s:e])
        pos_segments = np.concatenate(pos_segments)
        min_val, max_val = np.min(pos_segments), np.max(pos_segments)

        mean_val = np.mean(pos_segments)
        mask = (rms < (min_val + mean_val) / 2) | (rms > max_val)

        if np.sum(mask) < 435:
            print("Warning: The negative segment found is too short", np.sum(mask))
            mask = rms < np.max(rms) / 6
        print("Mask length: ", np.sum(mask))
        
        del x, rms, pos_segments, min_val, max_val, mean_val
        gc.collect()
        
        return mask

    def negative_onset_offset_estimate_freq_mask(
        self, audio_path, key, start_time, end_time, neg_min_length, hop_seg
    ):
        from src.datamodules.components.sisnr import si_snr_new
        from scipy.signal import medfilt

        # Load frequency mask with HDF5 support (fallback to NPY)
        h5_mask_path = os.path.join(self.path.mask_dir, f"mask_{key}.h5")
        npy_mask_path = os.path.join(self.path.mask_dir, f"mask_{key}.npy")
        
        if os.path.exists(h5_mask_path):
            with h5py.File(h5_mask_path, 'r') as f:
                mask = f['data'][:]
        elif os.path.exists(npy_mask_path):
            mask = np.load(npy_mask_path)
        else:
            raise FileNotFoundError(f"Frequency mask not found: {h5_mask_path} or {npy_mask_path}")
        
        # Load mel spectrogram with HDF5 support (fallback to NPY)
        h5_mel_path = audio_path.replace(".wav", "_mel.h5")
        npy_mel_path = audio_path.replace(".wav", "_mel.npy")
        
        if os.path.exists(h5_mel_path):
            with h5py.File(h5_mel_path, 'r') as f:
                mel = f['data'][:]
        elif os.path.exists(npy_mel_path):
            mel = np.load(npy_mel_path)
        else:
            raise FileNotFoundError(f"Mel spectrogram not found: {h5_mel_path} or {npy_mel_path}")
        mel = mel * mask[None, ...]
        mel = (mel - np.mean(mel)) / (np.std(mel) + 1e-6)

        # Try range limit
        pos_segments = []
        for s, e in zip(start_time, end_time):
            pos_segments.append(np.mean(mel[s:e, :], axis=0, keepdims=True))
        pos_segments = np.concatenate(pos_segments, axis=0)
        anchor = np.mean(pos_segments, axis=0)
        mel = torch.tensor(mel)
        anchor = torch.tensor(anchor)
        score = []

        query_padding = hop_seg

        # if(query_padding > 87):
        #     query_padding = 87
        if query_padding < 2:
            query_padding = 2

        for i in range(mel.size(0)):
            start, end = i - query_padding, i + query_padding

            if start < 0:
                start = 0
            if end > mel.size(0) - 1:
                end = mel.size(0) - 1

            query = mel[start:end, :]
            query = torch.mean(query, dim=0)
            score.append(si_snr_new(query, anchor))
        score = medfilt(np.array(score), kernel_size=9)

        # Calculate scores
        pos_scores = []
        for s, e in zip(start_time, end_time):
            pos_scores.append(score[s:e])
        pos_scores = np.concatenate(pos_scores)
        pos_scores_min, pos_scores_max = np.min(pos_scores), np.max(pos_scores)
        threshold = pos_scores_min

        mask = score < threshold
        mask = self.remove_short_mask(mask, minlen=neg_min_length // 2)
        # import ipdb; ipdb.set_trace()
        pic_name = os.path.basename(audio_path.replace(".wav", ".png"))
        self.plot_score_and_mask(score, mask, name=pic_name)
        
        del mel, pos_segments, anchor, score, pos_scores
        gc.collect()

        return np.array(mask)

    def plot_score_and_mask(self, scores, mask, name="temp.png"):
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 1))
        plt.plot(mask[1200:2000])
        print(name)
        plt.savefig(name)
        plt.close()

        # plt.figure(figsize=(8, 5))
        # plt.subplot(211)
        # plt.plot(scores[1200:2000])
        # plt.subplot(212)
        # plt.plot(mask[1200:2000])
        # print(name)
        # plt.savefig(name+"123.png")
        # plt.close()

    def remove_short_mask(self, mask, minlen=2):
        """_summary_
        If negative, True; Remove short True
        data = [True, True, True, False, True,False, True, True, True, True,]
        """
        if minlen < 1:
            return mask

        ret = []
        cnt = 0
        for i in range(len(mask)):
            if mask[i] != True:
                if cnt != 0:
                    for i in range(cnt):
                        if cnt >= minlen:
                            ret.append(True)
                        else:
                            ret.append(False)
                    cnt = 0
                ret.append(False)
            else:
                cnt += 1
        if cnt != 0:
            for i in range(cnt):
                if cnt >= minlen:
                    ret.append(True)
                else:
                    ret.append(False)
            cnt = 0
        return ret

    def read_file(self, file):
        audio_path = file.replace("csv", "wav")
        df_eval = pd.read_csv(file, header=0, index_col=False)
        
        # Take the mask here
        key = self.find_positive_label(df_eval)
        mask = self.load_mask(key)

        Q_list = df_eval[key].to_numpy()

        index_sup = np.where(Q_list == "POS")[0][: self.train_param.n_shot]

        # Open HDF5 file handles for lazy loading (Memory Optimization Phase 2)
        # Reference: docs/memory_optimization/03_IMPLEMENTATION_PLAN.md
        h5_handles = []
        feature_types = self.features.feature_types.split('@')
        for suffix in feature_types:
            h5_path = audio_path.replace(".wav", f"_{suffix}.h5")
            if not os.path.exists(h5_path):
                raise FileNotFoundError(f"Required HDF5 file not found: {h5_path}")
            h5_file = h5py.File(h5_path, 'r')
            h5_handles.append(h5_file['data'])

        # Get total frames from first HDF5 file
        total_frames = h5_handles[0].shape[0]

        # Calculate negative segments
        start_time, end_time = time_2_frame(
            df_eval, self.fps, padding=0.0
        )  # When calculating negative samples we do not want to use padding in order to maximize the negative sample we can get
        start_times_neg = [int(start_time[index]) for index in index_sup]
        end_times_neg = [int(end_time[index]) for index in index_sup]
        negative_onset_offset = []
        negative_seg_length = []
        start = 0
        # The start and end of negative segments
        for (
            s,
            e,
        ) in zip(start_times_neg, end_times_neg):
            end = s
            if end > start:
                negative_onset_offset.append((start, end))
                negative_seg_length.append(end - start)
            else:
                print("Error: end and start", end, start)
            start = e

        #################################Adaptive hop_seg#########################################
        start_time, end_time = time_2_frame(df_eval, self.fps, padding=0.025)
        difference = []
        for index in index_sup:
            difference.append(end_time[index] - start_time[index])
        # Adaptive segment length based on the audio file.
        max_len = max(difference)
        # Choosing the segment length based on the maximum size in the 5-shot.
        # Logic was based on fitment on 12GB GPU since some segments are quite long.
        if max_len < 8:
            seg_len = 8
        elif max_len < self.test_seglen_len_lim:
            seg_len = max_len
        elif (
            max_len >= self.test_seglen_len_lim
            and max_len <= self.test_seglen_len_lim * 2
        ):
            seg_len = max_len // 2
        elif max_len > self.test_seglen_len_lim * 2 and max_len < 500:
            seg_len = max_len // 4
        else:
            seg_len = max_len // 8
        # print(f"Adaptive segment length for %s is {seg_len}" % (file))
        hop_seg = seg_len // self.test_hoplen_fenmu
        #################################################################################

        neg_min_length = np.min([x for x in negative_seg_length])
        if neg_min_length > 100:  # TODO hard code here
            hop_seg_neg = 50
            seg_len_neg = 100
        elif neg_min_length < 8:
            hop_seg_neg = 4
            seg_len_neg = 8
        else:
            hop_seg_neg = neg_min_length // 4
            seg_len_neg = neg_min_length

        labelled_negative_data_length = np.sum(
            [b - a for a, b in negative_onset_offset]
        )
        neg_pcen = None

        # We have too little negative data
        if labelled_negative_data_length < 435:  # TODO hard code here
            print("The negative part of the %s is too short" % file)
            if self.eval_param.negative_estimate == "rms":
                index = self.negative_onset_offset_estimate(
                    audio_path, start_time, end_time
                )
            elif self.eval_param.negative_estimate == "freq_mask":
                index = self.negative_onset_offset_estimate_freq_mask(
                    audio_path, key, start_time, end_time, neg_min_length, hop_seg
                )
            else:
                raise ValueError("Bad parameters", self.eval_param.negative_estimate)

            assert total_frames == index.shape[0], (total_frames, index.shape)

            # Load negative pcen from HDF5
            neg_pcen = h5_handles[0][end_time[index_sup[-1]] :, :][index[end_time[index_sup[-1]] :], :]
            print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>", neg_pcen.shape)
            if neg_pcen.shape[0] < 20:
                neg_pcen = None
        else:
            neg_pcen = None

        # Call segment() with h5_handles instead of pcen
        feat_pos, feat_neg, feat_query_info, strt_indx_query, iteration_positions = self.segment(
            h5_handles,
            hop_seg,
            index_sup,
            start_time,
            end_time,
            seg_len,
            negative_onset_offset,
            negative_pcen=neg_pcen,
        )
        feat_pos_neg, feat_neg_neg, feat_query_info_neg, _, iteration_positions_neg = self.segment(
            h5_handles,
            hop_seg_neg,
            index_sup,
            start_time,
            end_time,
            seg_len_neg,
            negative_onset_offset,
            negative_pcen=neg_pcen,
        )

        # GC OPTIMIZATION: Preserve lists in temp variables before stacking
        # This allows explicit deletion of lists after stacking to prevent memory duplication
        feat_pos_list = feat_pos
        feat_neg_list = feat_neg
        feat_pos_neg_list = feat_pos_neg
        feat_neg_neg_list = feat_neg_neg

        # Stack support segments (eager-loaded)
        feat_pos = np.stack(feat_pos_list)
        feat_neg = np.stack(feat_neg_list)
        feat_pos_neg = np.stack(feat_pos_neg_list)
        feat_neg_neg = np.stack(feat_neg_neg_list)
        
        # Delete intermediate lists to free memory immediately
        del feat_pos_list, feat_neg_list, feat_pos_neg_list, feat_neg_neg_list
        gc.collect()
        
        # Query segments are metadata (lazy-loaded)
        # feat_query_info and feat_query_info_neg are lists of dicts with start_frame/end_frame
        
        # Memory cleanup
        del neg_pcen
        gc.collect()

        return (
            feat_pos,
            feat_neg,
            feat_query_info,  # Metadata instead of stacked array
            feat_pos_neg,
            feat_neg_neg,
            feat_query_info_neg,  # Metadata instead of stacked array
            max_len,
            neg_min_length,
            strt_indx_query,
            audio_path,
            hop_seg,
            hop_seg_neg,
            mask,
            seg_len,
            h5_handles,  # Return HDF5 handles for lazy loading
            iteration_positions,      # ADD THIS
            iteration_positions_neg,  # ADD THIS
        )  # Support segments: [n, seg_len, 128], Query: metadata list


if __name__ == "__main__":
    import torch
    from omegaconf import OmegaConf
    from tqdm import tqdm

    from src.datamodules.components.identity_sampler import IdentityBatchSampler

    data = []

    conf = OmegaConf.load("/vol/research/dcase2022/project/hhlab/configs/train.yaml")
    testset = PrototypeAdaSeglenBetterNegTestSetV2(
        conf.path, conf.features, conf.train_param, conf.eval_param
    )

    for (
        (
            X_pos,
            X_neg,
            X_query,
            X_pos_neg,
            X_neg_neg,
            X_query_neg,
            hop_seg,
            hop_seg_neg,
            max_len,
            neg_min_length,
        ),
        strt_index_query,
        audio_path,
        mask,
        _,
    ) in tqdm(testset):
        # import ipdb; ipdb.set_trace()
        pass
        # print(X_pos.shape, X_neg.shape, X_query.shape, X_pos_neg.shape, X_neg_neg.shape, X_query_neg.shape, hop_seg,hop_seg_neg, max_len, strt_index_query, audio_path)
