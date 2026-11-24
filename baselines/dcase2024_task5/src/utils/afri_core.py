# Upstream: https://github.com/kaoshenglun/AFRI (Kao and Liu).
# Provenance and release checks: repository-root THIRD_PARTY_NOTICES.md.
"""
AFRI (Automatic Frequency Range Identification) Core Computation Module

This module implements the core AFRI algorithm for identifying important frequency
ranges in few-shot bioacoustic event detection. The algorithm compares positive (POS)
and negative (NEG) spectrograms to find frequency bands that distinguish target sounds.

This implementation closely follows the original AutomaticFrequencyRangeIdentification.py
from DCASE 2024 Task 5, adapted for episode-based few-shot learning.

Reference snapshot (local Git history, not a working-tree file):
    git show 763cbf6bbf98:baselines/AFRI/AutomaticFrequencyRangeIdentification.py
Key sections in that snapshot:
- Lines 76-78: cossim() - Cosine similarity
- Lines 80-114: autosearchminbinandmaxbin() - Frequency range detection
- Lines 350-508: Core POS-NEG comparison algorithm
- Lines 565-700: Adaptive threshold refinement
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from scipy.signal import find_peaks
import librosa
import math


def cossim(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.
    
    Direct port from original AFRI (lines 76-78).
    
    Args:
        vec1: First vector
        vec2: Second vector
    
    Returns:
        Cosine similarity value
    """
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    
    cos_sim = np.dot(vec1, vec2) / (norm1 * norm2)
    return cos_sim


def autosearchminbinandmaxbin(
    maxlocbi: int,
    value_thresh: float,
    avg_compare_5posneg_tmp: np.ndarray,
    searchLbin_thresh: int,
    searchRbin_thresh: int
) -> Tuple[int, int]:
    """
    Find contiguous frequency range with values above threshold.
    
    Direct port from original AFRI (lines 80-114).
    Starting from the peak location, searches left and right to find the
    contiguous region where profile values exceed the threshold.
    
    Args:
        maxlocbi: Peak location (max bin index)
        value_thresh: Threshold for considering a bin as active
        avg_compare_5posneg_tmp: AFRI profile array (will be modified)
        searchLbin_thresh: Number of consecutive zeros to stop left search
        searchRbin_thresh: Number of consecutive zeros to stop right search
    
    Returns:
        Tuple of (auto_minbin, auto_maxbin) representing the frequency range
    """
    avg_compare_5posneg_tmp[avg_compare_5posneg_tmp <= value_thresh] = 0
    
    auto_minbi = maxlocbi
    auto_maxbi = maxlocbi
    left_conti = 1
    right_conti = 1
    
    # Search left side of maxlocbi
    while left_conti == 1:
        if (auto_minbi - searchLbin_thresh) < 0:
            searchLbin_thresh = 4
            if (auto_minbi - searchLbin_thresh) < 0:
                print("In autosearch, (auto_minbi-searchLbin_thresh)<0")
                left_conti = 0
                auto_minbi = 2
                break
        
        if avg_compare_5posneg_tmp[auto_minbi - 1] >= value_thresh:
            auto_minbi = auto_minbi - 1
        
        if (avg_compare_5posneg_tmp[auto_minbi - 1] < value_thresh) and \
           (all([val == 0 for val in avg_compare_5posneg_tmp[auto_minbi - searchLbin_thresh:auto_minbi]]) == True):
            left_conti = 0
        
        if (avg_compare_5posneg_tmp[auto_minbi - 1] < value_thresh) and \
           (all([val == 0 for val in avg_compare_5posneg_tmp[auto_minbi - searchLbin_thresh:auto_minbi]]) == False):
            auto_minbi = auto_minbi - 1
    
    # Search right side of maxlocbi
    while right_conti == 1:
        if (auto_maxbi + searchRbin_thresh) > 128:
            searchRbin_thresh = 4
            if (auto_maxbi + searchRbin_thresh) > 128:
                print("In autosearch, (auto_maxbi+searchRbin_thresh)>128")
                right_conti = 0
                auto_maxbi = 125
                break
        
        if avg_compare_5posneg_tmp[auto_maxbi + 1] >= value_thresh:
            auto_maxbi = auto_maxbi + 1
        
        if (avg_compare_5posneg_tmp[auto_maxbi + 1] < value_thresh) and \
           (all([val == 0 for val in avg_compare_5posneg_tmp[auto_maxbi:auto_maxbi + searchRbin_thresh]]) == True):
            right_conti = 0
        
        if (avg_compare_5posneg_tmp[auto_maxbi + 1] < value_thresh) and \
           (all([val == 0 for val in avg_compare_5posneg_tmp[auto_maxbi:auto_maxbi + searchRbin_thresh]]) == False):
            auto_maxbi = auto_maxbi + 1
    
    return auto_minbi, auto_maxbi


def _make_uniform_afri_profile(n_mels: int) -> Dict:
    """
    Create fallback uniform profile when AFRI fails.
    
    Returns a profile where all frequency bins have equal weight,
    effectively disabling frequency-based masking.
    
    Args:
        n_mels: Number of mel bins (typically 128)
    
    Returns:
        Dictionary with uniform weights
    """
    uniform = np.ones(n_mels, dtype=np.float32)
    
    return {
        'raw_profile': uniform,
        'mask_weights': uniform,
        'inverse_weights': uniform,
        'auto_minbin': 0,
        'auto_maxbin': n_mels - 1,
        'valid': False
    }


def compute_afri_profile(
    pos_spectrograms: List[np.ndarray],
    neg_spectrograms: List[np.ndarray],
    n_mels: int = 128,
    debris: int = 10,
    value_threshold: float = 0.23,
    searchLbin_threshold: int = 7,
    searchRbin_threshold: int = 7
) -> Dict:
    """
    Core AFRI algorithm implementing POS-NEG comparison logic.
    
    This function closely follows the original AFRI algorithm (lines 350-508):
    1. Normalize spectrograms to [0, 1] range
    2. For each POS segment, find the most similar NEG segment using cosine similarity
    3. Compute POS-NEG difference per frequency band
    4. Average across all valid samples
    5. Apply adaptive threshold refinement
    
    Args:
        pos_spectrograms: List of positive spectrograms, each [n_mels, time]
        neg_spectrograms: List of negative spectrograms, each [n_mels, time]
        n_mels: Number of mel frequency bins (typically 128)
        debris: Number of frequency bands to split into (default: 10, from original)
        value_threshold: Initial threshold for frequency range detection (default: 0.23)
        searchLbin_threshold: Left search threshold (default: 7)
        searchRbin_threshold: Right search threshold (default: 7)
    
    Returns:
        Dictionary containing:
            - raw_profile: Raw AFRI profile [n_mels]
            - mask_weights: Normalized weights for masking [n_mels]
            - inverse_weights: Inverse weights (1 - mask_weights) [n_mels]
            - auto_minbin: Lower frequency bin of detected range
            - auto_maxbin: Upper frequency bin of detected range
            - valid: True if AFRI succeeded, False if fallback
    """
    if len(pos_spectrograms) == 0 or len(neg_spectrograms) == 0:
        print("AFRI: No valid spectrograms provided")
        return _make_uniform_afri_profile(n_mels)
    
    # Normalize spectrograms to [0, 1] range (lines 188-190 from original)
    def normalize_spec(spec):
        """
        Normalize power spectrogram to [0, 1] range.
        
        Input: Power spectrogram from librosa.feature.melspectrogram()
        Output: Normalized [0, 1] range
        """
        spec_norm = spec.copy()
        spec_db = librosa.power_to_db(spec_norm, ref=np.max)
        # Normalize to [0, 1] range (assuming dB range is [-80, 0])
        spec_norm = (spec_db - (-80)) / (0 - (-80))
        return spec_norm
    
    pos_specs_norm = [normalize_spec(s) for s in pos_spectrograms]
    neg_specs_norm = [normalize_spec(s) for s in neg_spectrograms]
    
    durpos = [spec.shape[1] for spec in pos_specs_norm]
    durneg = [spec.shape[1] for spec in neg_specs_norm]
    
    # Find minimum POS duration (line 356 from original)
    minposduration = min(durpos)
    
    if minposduration < 5:
        print(f"AFRI: Min POS duration too short: {minposduration}")
        return _make_uniform_afri_profile(n_mels)
    
    print(f"AFRI: minposduration={minposduration}")
    
    # Split frequency range into segments (line 354 from original)
    splitnumb = math.ceil(n_mels / debris)
    print(f"AFRI: splitnumb={splitnumb}")
    
    compare_5posneg = []
    
    # Process each POS shot (lines 359-508 from original)
    for a in range(len(pos_specs_norm)):
        print(f"\nAFRI: Processing {a+1} POS shot")
        
        pos_spec = pos_specs_norm[a]
        posduration = durpos[a]
        
        # Determine number of windows (line 361 from original)
        xnum = math.ceil(posduration / minposduration)
        print(f"AFRI: xnum={xnum}")
        
        # Check which NEG samples are applicable (lines 366-375 from original)
        appliableneg = []
        for j in range(len(neg_specs_norm)):
            negduration = durneg[j]
            if (minposduration > negduration):
                appliableneg.append(0)
            else:
                appliableneg.append(1)
        
        print(f"AFRI: appliableneg={appliableneg}")
        
        if all([val == 0 for val in appliableneg]) == True:
            print(f"AFRI: appliableneg all zero in {a+1} POS shot")
            continue
        
        # Process each frequency segment (lines 382-508 from original)
        for h in range(splitnumb):
            st = h * debris
            en = st + debris
            if h == splitnumb - 1:
                en = n_mels
            
            print(f"\nAFRI: Frequency segment st={st}, en={en}")
            
            pos_band = pos_spec[st:en, :]
            
            xnum_possubstractneg_height = []
            
            for k in range(xnum):
                # Define time window (lines 406-413 from original)
                stt = k * minposduration
                enn = stt + minposduration
                if k == xnum - 1:
                    print(f"AFRI: k==xnum-1 in {a+1} POS")
                    stt = posduration - minposduration
                    enn = posduration
                
                print(f"AFRI: stt={stt} in {a+1} POS")
                print(f"AFRI: enn={enn} in {a+1} POS")
                
                tmp_pos_S_dB = pos_band[:, stt:enn]
                tmp_pos_width_melspe = tmp_pos_S_dB.sum(axis=0) / tmp_pos_S_dB.shape[0]
                tmp_pos_height_melspe = tmp_pos_S_dB.sum(axis=1) / tmp_pos_S_dB.shape[1]
                
                # Find most similar NEG segment (lines 417-450 from original)
                CS_widthheight_POSandNEG_five = []
                height_POSandNEG_five = []
                
                for b in range(len(neg_specs_norm)):
                    if appliableneg[b] == 0:
                        continue
                    
                    CS_widthheight_POSandNEG_one = []
                    height_POSandNEG_one = []
                    
                    tmp_pos_S_dB_width = tmp_pos_S_dB.sum(axis=0) / tmp_pos_S_dB.shape[0]
                    tmp_pos_S_dB_height = tmp_pos_S_dB.sum(axis=1) / tmp_pos_S_dB.shape[1]
                    
                    neg_spec = neg_specs_norm[b]
                    neg_S_dB = neg_spec[st:en, :]
                    
                    if neg_S_dB.shape[1] >= tmp_pos_S_dB.shape[1]:
                        # Slide window across NEG to find best match (lines 437-448)
                        tmp_cossim_neg_widthheight_melspe = []
                        tmp_height_melspe = []
                        
                        for c in range(neg_S_dB.shape[1] - tmp_pos_S_dB.shape[1] + 1):
                            stt_neg = c
                            enn_neg = c + tmp_pos_S_dB.shape[1]
                            
                            tmp_neg_S_dB = neg_S_dB[:, stt_neg:enn_neg]
                            tmp_neg_width_melspe = tmp_neg_S_dB.sum(axis=0) / tmp_neg_S_dB.shape[0]
                            tmp_neg_height_melspe = tmp_neg_S_dB.sum(axis=1) / tmp_neg_S_dB.shape[1]
                            
                            # Compute combined cosine similarity (line 446)
                            cos_sim_combined = (cossim(tmp_neg_width_melspe, tmp_pos_width_melspe) + 
                                              cossim(tmp_neg_height_melspe, tmp_pos_height_melspe)) / 2
                            tmp_cossim_neg_widthheight_melspe.append(cos_sim_combined)
                            tmp_height_melspe.append(tmp_neg_S_dB.sum(axis=1))
                        
                        CS_widthheight_POSandNEG_one.append(tmp_cossim_neg_widthheight_melspe)
                        height_POSandNEG_one.append(tmp_height_melspe)
                    
                    CS_widthheight_POSandNEG_five.append(CS_widthheight_POSandNEG_one)
                    height_POSandNEG_five.append(height_POSandNEG_one)
                
                print(f"AFRI: len(CS_widthheight_POSandNEG_five)={len(CS_widthheight_POSandNEG_five)}")
                print(f"AFRI: len(height_POSandNEG_five)={len(height_POSandNEG_five)}")
                
                # Find most similar NEG and compute POS-NEG difference (lines 455-467)
                tmplist = []
                tmpheight = []
                for d in range(len(CS_widthheight_POSandNEG_five)):
                    if len(CS_widthheight_POSandNEG_five[d]) > 0:
                        tmplist = tmplist + CS_widthheight_POSandNEG_five[d][0]
                        tmpheight = tmpheight + height_POSandNEG_five[d][0]
                
                if len(tmplist) > 0:
                    loc_max = tmplist.index(max(tmplist))
                    print(f"\nAFRI: len(tmplist)={len(tmplist)}")
                    tmp_possubstractneg_height = tmp_pos_S_dB.sum(axis=1) - tmpheight[loc_max]
                    print(f"AFRI: tmp_possubstractneg_height.shape={tmp_possubstractneg_height.shape}")
                    xnum_possubstractneg_height.append(tmp_possubstractneg_height)
                
                # Average across time windows (lines 468-478)
                if k == xnum - 1 and len(xnum_possubstractneg_height) > 0:
                    print(f"AFRI: len(xnum_possubstractneg_height)={len(xnum_possubstractneg_height)}")
                    avg_xnum_possubstractneg_height = np.mean(xnum_possubstractneg_height, axis=0)
                    possubstractneg_height = avg_xnum_possubstractneg_height
                    
                    if h == 0:
                        all_possubstractneg_height = possubstractneg_height
                    if h > 0:
                        all_possubstractneg_height = np.concatenate((all_possubstractneg_height, possubstractneg_height))
                    
                    # Process final profile for this POS (lines 479-488)
                    if h == splitnumb - 1:
                        print("AFRI: h==splitnumb-1")
                        all_possubstractneg_height[all_possubstractneg_height < 0] = 0
                        
                        # Normalize (lines 483-484)
                        min_val = min(all_possubstractneg_height)
                        max_val = max(all_possubstractneg_height)
                        if max_val > min_val:
                            all_possubstractneg_height = (all_possubstractneg_height - min_val) / (max_val - min_val)
                        
                        print(f"\nAFRI: All 'nan' in all_possubstractneg_height ? = {np.isnan(all_possubstractneg_height).all()} in {a+1} POS shot")
                        
                        if np.isnan(all_possubstractneg_height).all() == False:
                            compare_5posneg.append(all_possubstractneg_height)
                            print(f"AFRI: all_possubstractneg_height.shape={all_possubstractneg_height.shape}")
    
    if len(compare_5posneg) == 0:
        print("AFRI: No valid profiles generated")
        return _make_uniform_afri_profile(n_mels)
    
    print(f"\nAFRI: len(compare_5posneg)={len(compare_5posneg)}")
    
    # Average valid POS-NEG profiles (upstream lines 512-538).
    avg_compare_5posneg = np.average(compare_5posneg, axis=0)
    avg_compare_5posneg_orig = avg_compare_5posneg.copy()
    
    print("AFRI: AVERAGE 5 POS-NEG")
    print(f"AFRI: len(avg_compare_5posneg)={len(avg_compare_5posneg)}")
    
    # Detect peaks (line 522)
    avg_peaks, _ = find_peaks(avg_compare_5posneg, height=0.339)
    print(f"AFRI: bin avg_peaks={avg_peaks}")
    
    # Apply adaptive threshold refinement (lines 565-700 from original)
    avg_compare_5posneg, auto_minbin, auto_maxbin = _refine_frequency_range(
        avg_compare_5posneg,
        avg_compare_5posneg_orig,
        value_threshold,
        searchLbin_threshold,
        searchRbin_threshold,
        n_mels
    )
    
    mask_weights = avg_compare_5posneg.astype(np.float32)
    inverse_weights = (1.0 - mask_weights).astype(np.float32)
    
    return {
        'raw_profile': avg_compare_5posneg,
        'mask_weights': mask_weights,        # Direct: mask important frequencies
        'inverse_weights': inverse_weights,  # Inverse: mask unimportant frequencies
        'auto_minbin': int(auto_minbin),
        'auto_maxbin': int(auto_maxbin),
        'valid': True
    }


def _refine_frequency_range(
    avg_compare_5posneg: np.ndarray,
    avg_compare_5posneg_orig: np.ndarray,
    value_threshold: float,
    searchLbin_threshold: int,
    searchRbin_threshold: int,
    n_mels: int = 128
) -> Tuple[np.ndarray, int, int]:
    """
    Apply adaptive threshold refinement to find optimal frequency range.
    
    Direct port from original AFRI (lines 565-721).
    Implements multi-stage threshold refinement with various fallback strategies.
    
    Args:
        avg_compare_5posneg: Current AFRI profile (will be modified)
        avg_compare_5posneg_orig: Original AFRI profile (for reset)
        value_threshold: Initial threshold value
        searchLbin_threshold: Left search threshold
        searchRbin_threshold: Right search threshold
        n_mels: Number of mel bins
    
    Returns:
        Tuple of (refined_profile, auto_minbin, auto_maxbin)
    """
    tt = 0  # Track if we've run refinement
    
    # Initial threshold (upstream lines 566-580).
    maxlocbin = np.argmax(avg_compare_5posneg)
    searchresult = autosearchminbinandmaxbin(
        maxlocbin, value_threshold, avg_compare_5posneg.copy(),
        searchLbin_threshold, searchRbin_threshold
    )
    auto_minbin = searchresult[0]
    auto_maxbin = searchresult[1]
    auto_maxminbin = auto_maxbin - auto_minbin
    
    print(f"\nAFRI: 1st value_threshold={value_threshold:.2f}")
    print(f"AFRI: 1st maxlocbin={maxlocbin}")
    print(f"AFRI: 1st auto_maxminbin={auto_maxminbin}")
    print(f"AFRI: 1st auto_minbin={auto_minbin}")
    print(f"AFRI: 1st auto_maxbin={auto_maxbin}")
    
    # Retry maxlocbin <= 5 with a different threshold (upstream lines 582-601).
    if maxlocbin <= 5:
        print("\nAFRI: maxlocbin<=5")
        avg_compare_5posneg = avg_compare_5posneg_orig.copy()
        tt = 1
        
        # Ignore values near border
        print("AFRI: ignore (i<=5) | (i>=120) in avg_compare_5posneg")
        for i in range(len(avg_compare_5posneg)):
            if (i <= 5) or (i >= 120):
                avg_compare_5posneg[i] = 0
        
        value_threshold = 0.205
        print(f"AFRI: Run again in maxlocbin<=5, by value_threshold={value_threshold}")
        
        maxlocbin = np.argmax(avg_compare_5posneg)
        searchresult = autosearchminbinandmaxbin(
            maxlocbin, value_threshold, avg_compare_5posneg.copy(),
            searchLbin_threshold, searchRbin_threshold
        )
        auto_minbin = searchresult[0]
        auto_maxbin = searchresult[1]
        auto_maxminbin = auto_maxbin - auto_minbin
        print(f"AFRI: In maxlocbin<=5, Run again auto_maxminbin={auto_maxminbin}")
    
    # Retry a narrow range (upstream lines 603-623).
    if (auto_maxminbin <= 5) or ((maxlocbin <= 12) and (auto_maxbin - maxlocbin <= 5)):
        print("\nAFRI: (auto_maxminbin<=5) | (maxlocbin<=12) & (auto_maxbin-maxlocbin<=5)")
        avg_compare_5posneg = avg_compare_5posneg_orig.copy()
        tt = 1
        value_threshold = 0.13
        
        print("AFRI: ignore (i<=7) | (i>=120) in avg_compare_5posneg")
        for i in range(len(avg_compare_5posneg)):
            if (i <= 7) or (i >= 120):
                avg_compare_5posneg[i] = 0
        
        print(f"AFRI: Run again in auto_maxminbin<=5, by value_threshold={value_threshold}")
        
        maxlocbin = np.argmax(avg_compare_5posneg)
        print(f"AFRI: maxlocbin={maxlocbin}")
        searchresult = autosearchminbinandmaxbin(
            maxlocbin, value_threshold, avg_compare_5posneg.copy(),
            searchLbin_threshold, searchRbin_threshold
        )
        auto_minbin = searchresult[0]
        auto_maxbin = searchresult[1]
        auto_maxminbin = auto_maxbin - auto_minbin
        print(f"AFRI: In auto_maxminbin<=5, Run again auto_maxminbin={auto_maxminbin}")
        
        # Use two peaks if the range is still too narrow (upstream lines 625-676).
        if auto_maxminbin <= 10:
            print(f"AFRI: Run again auto_maxminbin<=10")
            avg_compare_5posneg = avg_compare_5posneg_orig.copy()
            
            print("AFRI: ignore (i<=7) | (i>=120) in avg_compare_5posneg")
            for i in range(len(avg_compare_5posneg)):
                if (i <= 7) or (i >= 120):
                    avg_compare_5posneg[i] = 0
            
            value_threshold = 0.07
            print(f"AFRI: Run again auto_maxminbin<=10, by value_threshold={value_threshold}")
            avg_compare_5posneg[avg_compare_5posneg <= value_threshold] = 0
            
            avg_peaks, _ = find_peaks(avg_compare_5posneg, height=value_threshold)
            avg_peaklist = []
            for i in range(len(avg_peaks)):
                avg_peaklist.append(avg_peaks[i])
            print(f"AFRI: avg_peaklist={avg_peaklist}")
            
            if len(avg_peaklist) >= 2:
                avg_compare_5posneg_list = avg_compare_5posneg[avg_peaks].tolist()
                n = 2  # 2 biggest values
                result = [avg_compare_5posneg_list.index(i) for i in sorted(avg_compare_5posneg_list, reverse=True)][:n]
                print(f"AFRI: result={result}")
                
                little = avg_peaklist[result[0]]
                big = avg_peaklist[result[1]]
                print(f"AFRI: little={little}")
                print(f"AFRI: big={big}")
                
                tmpl = []
                if little == big:
                    print("AFRI: little==big")
                    for i in range(len(avg_compare_5posneg.tolist())):
                        if avg_compare_5posneg.tolist()[i] == avg_compare_5posneg.tolist()[avg_peaklist[result[0]]]:
                            tmpl.append(i)
                    print(f"AFRI: tmpl={tmpl}")
                    little = tmpl[0]
                    big = tmpl[len(tmpl) - 1]
                    print(f"AFRI: little={little}")
                    print(f"AFRI: big={big}")
                
                if little > big:
                    little, big = big, little
                
                # Set maxpeak~secondmaxpeak to 1
                for i in range(len(avg_compare_5posneg)):
                    if (i >= little) and (i <= big):
                        avg_compare_5posneg[i] = 1
                
                print("AFRI: QQ2 avg_compare_5posneg set to 1 between peaks")
                
                maxlocbin = np.argmax(avg_compare_5posneg)
                print(f"AFRI: maxlocbin={maxlocbin}")
                searchresult = autosearchminbinandmaxbin(
                    maxlocbin, value_threshold, avg_compare_5posneg.copy(),
                    searchLbin_threshold, searchRbin_threshold
                )
                auto_minbin = searchresult[0]
                auto_maxbin = searchresult[1]
                auto_maxminbin = auto_maxbin - auto_minbin
                print(f"AFRI: In auto_maxminbin<=5 & auto_maxminbin<=10, Run again auto_maxminbin={auto_maxminbin}")
    
    # Retry ranges of 59-62 bins (upstream lines 678-689).
    if ((auto_maxminbin >= 59) and (auto_maxminbin <= 62)) and (tt == 0):
        print(f"\nAFRI: auto_maxminbin = 59 ~ 62")
        avg_compare_5posneg = avg_compare_5posneg_orig.copy()
        tt = 1
        value_threshold = 0.3118
        print(f"AFRI: Run again in auto_maxminbin = 59 ~ 62, by value_threshold={value_threshold}")
        
        maxlocbin = np.argmax(avg_compare_5posneg)
        searchresult = autosearchminbinandmaxbin(
            maxlocbin, value_threshold, avg_compare_5posneg.copy(),
            searchLbin_threshold, searchRbin_threshold
        )
        auto_minbin = searchresult[0]
        auto_maxbin = searchresult[1]
        auto_maxminbin = auto_maxbin - auto_minbin
        print(f"AFRI: In auto_maxminbin = 59 ~ 62, Run again auto_maxminbin={auto_maxminbin}")
    
    # Retry ranges >= 63 bins (upstream lines 690-700).
    if (auto_maxminbin >= 63) and (tt == 0):
        print(f"\nAFRI: auto_maxminbin >= 63")
        avg_compare_5posneg = avg_compare_5posneg_orig.copy()
        value_threshold = 0.444
        print(f"AFRI: Run again in auto_maxminbin >= 63, by value_threshold={value_threshold}")
        
        maxlocbin = np.argmax(avg_compare_5posneg)
        searchresult = autosearchminbinandmaxbin(
            maxlocbin, value_threshold, avg_compare_5posneg.copy(),
            searchLbin_threshold, searchRbin_threshold
        )
        auto_minbin = searchresult[0]
        auto_maxbin = searchresult[1]
        auto_maxminbin = auto_maxbin - auto_minbin
        print(f"AFRI: In auto_maxminbin >= 63, Run again auto_maxminbin={auto_maxminbin}")
    
    print(f"\nAFRI: Final selectfreqbin, before binsexpand")
    print(f"AFRI: Final maxlocbin={maxlocbin}")
    print(f"AFRI: Final auto_maxminbin={auto_maxminbin}")
    print(f"AFRI: Final auto_minbin={auto_minbin}")
    print(f"AFRI: Final auto_maxbin={auto_maxbin}")
    
    # Expand the bin range (upstream lines 708-721).
    if auto_maxminbin <= 20:
        print("AFRI: auto_maxminbin<=20")
        auto_minbin = auto_minbin - 3
        auto_maxbin = auto_maxbin + 3
    if auto_maxminbin > 20:
        print("AFRI: auto_maxminbin>20")
        auto_minbin = auto_minbin - 2
        auto_maxbin = auto_maxbin + 6
    
    if auto_minbin <= 2:
        print("AFRI: CC, auto_minbin<=2")
        auto_minbin = 2
    if auto_maxbin >= 125:
        print("AFRI: DD, auto_maxbin>=125")
        auto_maxbin = 125
    
    print(f"\nAFRI: FFinal selectfreqbin, after binsexpand")
    print(f"AFRI: FFinal maxlocbin={maxlocbin}")
    print(f"AFRI: FFinal auto_maxminbin={auto_maxbin - auto_minbin}")
    print(f"AFRI: FFinal auto_minbin={auto_minbin}")
    print(f"AFRI: FFinal auto_maxbin={auto_maxbin}")
    
    return avg_compare_5posneg, auto_minbin, auto_maxbin

