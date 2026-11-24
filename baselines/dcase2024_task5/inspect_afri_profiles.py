#!/usr/bin/env python3
"""
AFRI Profile Inspector
Inspects cached AFRI profiles to diagnose masking issues
"""

import numpy as np
import json
import sys
from pathlib import Path

def inspect_afri_cache(cache_base_dir: str):
    """
    Inspect AFRI cache files to verify profile structure and contents.
    
    Args:
        cache_base_dir: Base directory containing .afri_cache folder
    """
    cache_dir = Path(cache_base_dir) / ".afri_cache"
    
    if not cache_dir.exists():
        print(f"❌ AFRI cache directory not found: {cache_dir}")
        return False
    

    # Inspect each cache
    print("="*80)
    print(f"Cache: {cache_dir.name}")
    print("="*80)
    
    metadata_file = cache_dir / "metadata.json"
    profiles_file = cache_dir / "profiles.npz"
    
    if not metadata_file.exists():
        print(f"⚠️  metadata.json not found")
    
    if not profiles_file.exists():
        print(f"⚠️  profiles.npz not found")
    
    # Load metadata
    try:
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        print(f"\n📋 Metadata:")
        print(f"  Timestamp: {metadata.get('timestamp', 'N/A')}")
        print(f"  Cache key: {metadata.get('cache_key', 'N/A')[:16]}...")
        print(f"  Number of classes: {len(metadata.get('classes', []))}")
        
        afri_config = metadata.get('afri_config', {})
        print(f"\n⚙️  AFRI Config:")
        for key, value in afri_config.items():
            print(f"    {key}: {value}")
        
        feature_config = metadata.get('feature_config', {})
        print(f"\n🎵 Feature Config:")
        for key, value in feature_config.items():
            print(f"    {key}: {value}")
        
    except Exception as e:
        print(f"❌ Error loading metadata: {e}")
        
    
    # Load profiles
    try:
        profiles = np.load(profiles_file, allow_pickle=True)
        
        print(f"\n📦 Profiles NPZ Structure:")
        print(f"  Available keys: {len(profiles.keys())}")
        print(f"  Key pattern: {list(profiles.keys())[:3]}...")
        
    except Exception as e:
        print(f"❌ Error loading profiles: {e}")
    
    # Inspect individual class profiles
    classes = metadata.get('class_names', [])
    n_classes_to_show = min(100, len(classes))
    
    print(f"\n🔍 Inspecting {n_classes_to_show}/{len(classes)} class profiles:")
    print("-"*80)
    
    for i, cls_name in enumerate(classes[:n_classes_to_show], 1):
        # Try different key formats
        possible_keys = [
            f"{cls_name}_profile",
            cls_name,
            f"class_{cls_name}"
        ]
        
        profile = None
        used_key = None
        
        for key in possible_keys:
            if key in profiles:
                profile = profiles[key]
                used_key = key
                break
        
        if profile is None:
            print(f"\n[{i}] ❌ Class '{cls_name}': Profile not found (tried keys: {possible_keys})")
            continue
        
        # Handle numpy object array
        if isinstance(profile, np.ndarray) and profile.dtype == object:
            profile = profile.item()
        
        print(f"\n[{i}] Class: {cls_name}")
        print(f"    Key used: {used_key}")
        print(f"    Type: {type(profile)}")
        
        if not isinstance(profile, dict):
            print(f"    ⚠️  Profile is not a dictionary!")
            continue
        
        # Check profile structure
        print(f"    Keys: {list(profile.keys())}")
        print(f"    Valid: {profile.get('valid', False)}")
        
        # Check mask_weights
        if 'mask_weights' in profile:
            mw = profile['mask_weights']
            if isinstance(mw, np.ndarray):
                print(f"    ✓ mask_weights:")
                print(f"        Shape: {mw.shape}")
                print(f"        Dtype: {mw.dtype}")
                print(f"        Range: [{mw.min():.6f}, {mw.max():.6f}]")
                print(f"        Sum: {mw.sum():.6f}")
                print(f"        Non-zero: {np.count_nonzero(mw)}/{len(mw)} ({100*np.count_nonzero(mw)/len(mw):.1f}%)")
                
                # Check if all zeros
                if np.count_nonzero(mw) == 0:
                    print(f"        ⚠️  ALL ZEROS - This will cause masking errors!")
            else:
                print(f"    ⚠️  mask_weights exists but is not ndarray: {type(mw)}")
        else:
            print(f"    ❌ mask_weights: MISSING")
        
        # Check inverse_weights
        if 'inverse_weights' in profile:
            iw = profile['inverse_weights']
            if isinstance(iw, np.ndarray):
                print(f"    ✓ inverse_weights:")
                print(f"        Shape: {iw.shape}")
                print(f"        Dtype: {iw.dtype}")
                print(f"        Range: [{iw.min():.6f}, {iw.max():.6f}]")
                print(f"        Sum: {iw.sum():.6f}")
                print(f"        Non-zero: {np.count_nonzero(iw)}/{len(iw)} ({100*np.count_nonzero(iw)/len(iw):.1f}%)")
                
                # Check if all zeros
                if np.count_nonzero(iw) == 0:
                    print(f"        ⚠️  ALL ZEROS - This will cause masking errors!")
            else:
                print(f"    ⚠️  inverse_weights exists but is not ndarray: {type(iw)}")
        else:
            print(f"    ❌ inverse_weights: MISSING")
        
        # Check frequency range
        if 'auto_minbin' in profile and 'auto_maxbin' in profile:
            minbin = profile['auto_minbin']
            maxbin = profile['auto_maxbin']
            width = maxbin - minbin
            print(f"    Frequency range: bins {minbin}-{maxbin} (width: {width})")
    
    print("\n" + "="*80 + "\n")
    
    return True

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Inspect AFRI profile cache")
    parser.add_argument(
        "--cache-dir",
        default="/media/helios/Data/Uni/Masters/Dataset/Development_Set/Training_Set",
        help="Base directory containing .afri_cache folder"
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("AFRI Profile Inspector")
    print("="*80 + "\n")
    
    success = inspect_afri_cache(args.cache_dir)
    
    if success:
        print("✓ Inspection complete")
        return 0
    else:
        print("✗ Inspection failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
