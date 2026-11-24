"""
AFRI Profile Caching System

This module provides a standalone caching system for AFRI (Automatic Frequency Range 
Identification) profiles to avoid recomputing expensive profile generation on every run.

The caching system:
- Generates unique cache keys based on data and configuration
- Saves/loads profiles to/from disk
- Validates cache integrity
- Handles errors gracefully

Typical usage:
    from src.utils.afri_cache import (
        generate_afri_cache_key,
        load_cached_profiles,
        save_profiles_to_cache
    )
    
    # Generate cache key
    cache_key = generate_afri_cache_key(train_dir, afri_config, feature_config)
    
    # Try loading from cache
    profiles = load_cached_profiles(train_dir, cache_key)
    
    if profiles is None:
        # Compute profiles...
        profiles = compute_profiles()
        
        # Save to cache
        save_profiles_to_cache(profiles, train_dir, cache_key, 
                              afri_config, feature_config)
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
import glob
import time

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    print("Warning: NumPy not available. AFRI caching will not work.")


def generate_afri_cache_key(
    train_dir: str,
    afri_config: Dict[str, Any],
    feature_config: Dict[str, Any]
) -> str:
    """
    Generate a unique cache key based on data files and configurations.
    
    The cache key is a SHA256 hash of:
    - List of CSV files with their modification timestamps
    - AFRI parameters (n_samples, thresholds, etc.)
    - Feature extraction parameters (n_mels, sr, hop_mel)
    
    Args:
        train_dir: Path to training data directory
        afri_config: Dictionary of AFRI parameters
        feature_config: Dictionary of feature extraction parameters
        
    Returns:
        16-character hexadecimal string representing the cache key
        
    Example:
        >>> afri_config = {
        ...     'afri_n_samples': 5,
        ...     'afri_value_threshold': 0.23,
        ...     'afri_search_threshold': 7,
        ... }
        >>> feature_config = {
        ...     'n_mels': 128,
        ...     'sr': 22050,
        ...     'hop_mel': 256,
        ... }
        >>> key = generate_afri_cache_key('/path/to/train', afri_config, feature_config)
        >>> len(key)
        16
    """
    hasher = hashlib.sha256()
    
    # Hash CSV files with modification times
    try:
        csv_files = sorted(glob.glob(os.path.join(train_dir, "*.csv")))
        for csv_file in csv_files:
            # Include filename and modification time
            mtime = os.path.getmtime(csv_file)
            file_info = f"{os.path.basename(csv_file)}:{mtime}"
            hasher.update(file_info.encode('utf-8'))
    except Exception as e:
        print(f"Warning: Could not read CSV files from {train_dir}: {e}")
        # Use directory path as fallback
        hasher.update(train_dir.encode('utf-8'))
    
    # Hash AFRI configuration (sorted for consistency)
    afri_str = json.dumps(afri_config, sort_keys=True)
    hasher.update(afri_str.encode('utf-8'))
    
    # Hash feature configuration (sorted for consistency)
    feature_str = json.dumps(feature_config, sort_keys=True)
    hasher.update(feature_str.encode('utf-8'))
    
    # Return first 16 characters of hex digest
    return hasher.hexdigest()[:16]


def get_cache_paths(
    train_dir: str,
    cache_dir: Optional[str] = None
) -> Tuple[Path, Path]:
    """
    Get paths for cache metadata and profile files.
    
    Args:
        train_dir: Path to training data directory
        cache_dir: Optional custom cache directory. If None, uses {train_dir}/.afri_cache/
        
    Returns:
        Tuple of (metadata_path, profiles_path) as Path objects
        
    Example:
        >>> metadata_path, profiles_path = get_cache_paths('/path/to/train')
        >>> str(metadata_path)
        '/path/to/train/.afri_cache/metadata.json'
        >>> str(profiles_path)
        '/path/to/train/.afri_cache/profiles.npz'
    """
    if cache_dir is None:
        cache_path = Path(train_dir) / ".afri_cache"
    else:
        cache_path = Path(cache_dir)
    
    # Create cache directory if it doesn't exist
    try:
        cache_path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Warning: Could not create cache directory {cache_path}: {e}")
    
    metadata_path = cache_path / "metadata.json"
    profiles_path = cache_path / "profiles.npz"
    
    return metadata_path, profiles_path


def load_cached_profiles(
    train_dir: str,
    cache_key: str,
    cache_dir: Optional[str] = None
) -> Optional[Dict[str, Dict[str, np.ndarray]]]:
    """
    Load AFRI profiles from cache if valid.
    
    Args:
        train_dir: Path to training data directory
        cache_key: Expected cache key for validation
        cache_dir: Optional custom cache directory
        
    Returns:
        Dictionary mapping class names to profile dictionaries, or None if:
        - Cache files don't exist
        - Cache key doesn't match
        - Files are corrupted
        - NumPy is not available
        
    Example:
        >>> profiles = load_cached_profiles('/path/to/train', 'abc123def456')
        >>> if profiles:
        ...     print(f"Loaded {len(profiles)} class profiles")
        ... else:
        ...     print("Cache miss - need to compute profiles")
    """
    if not HAS_NUMPY:
        print("Warning: NumPy not available. Cannot load cached profiles.")
        return None
    
    try:
        metadata_path, profiles_path = get_cache_paths(train_dir, cache_dir)
        
        # Check if cache files exist
        if not metadata_path.exists() or not profiles_path.exists():
            return None
        
        # Load and validate metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Validate cache key
        if metadata.get('cache_key') != cache_key:
            print(f"Cache key mismatch. Expected: {cache_key}, Found: {metadata.get('cache_key')}")
            return None
        
        # Load profiles from NPZ file
        npz_data = np.load(profiles_path, allow_pickle=True)
        
        # Reconstruct profiles dictionary
        profiles = {}
        for class_name in npz_data.files:
            # Each class is stored as a dictionary in the NPZ
            class_data = npz_data[class_name].item()
            profiles[class_name] = class_data
        
        return profiles
        
    except json.JSONDecodeError as e:
        print(f"Error: Corrupted metadata file: {e}")
        return None
    except Exception as e:
        print(f"Error loading cached profiles: {e}")
        return None


def save_profiles_to_cache(
    profiles: Dict[str, Dict[str, np.ndarray]],
    train_dir: str,
    cache_key: str,
    afri_config: Dict[str, Any],
    feature_config: Dict[str, Any],
    cache_dir: Optional[str] = None
) -> bool:
    """
    Save AFRI profiles to cache.
    
    Args:
        profiles: Dictionary mapping class names to profile dictionaries
        train_dir: Path to training data directory
        cache_key: Cache key for validation
        afri_config: AFRI configuration used to generate profiles
        feature_config: Feature extraction configuration
        cache_dir: Optional custom cache directory
        
    Returns:
        True if save was successful, False otherwise
        
    Example:
        >>> profiles = {
        ...     'class1': {'freq_ranges': np.array([...]), 'importance': np.array([...])},
        ...     'class2': {'freq_ranges': np.array([...]), 'importance': np.array([...])}
        ... }
        >>> success = save_profiles_to_cache(
        ...     profiles, '/path/to/train', 'abc123def456',
        ...     afri_config, feature_config
        ... )
        >>> if success:
        ...     print("Profiles saved successfully")
    """
    if not HAS_NUMPY:
        print("Warning: NumPy not available. Cannot save profiles to cache.")
        return False
    
    try:
        metadata_path, profiles_path = get_cache_paths(train_dir, cache_dir)
        
        # Prepare metadata
        metadata = {
            'cache_key': cache_key,
            'timestamp': time.time(),
            'timestamp_iso': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
            'afri_config': afri_config,
            'feature_config': feature_config,
            'num_classes': len(profiles),
            'class_names': list(profiles.keys())
        }
        
        # Save metadata as JSON
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save profiles as NPZ (compressed)
        # Convert dict of dicts to format suitable for NPZ
        np.savez_compressed(profiles_path, **profiles)
        
        print(f"✓ Cache saved successfully:")
        print(f"  - Metadata: {metadata_path}")
        print(f"  - Profiles: {profiles_path}")
        print(f"  - Classes: {len(profiles)}")
        
        return True
        
    except PermissionError as e:
        print(f"Error: Permission denied when saving cache: {e}")
        return False
    except Exception as e:
        print(f"Error saving profiles to cache: {e}")
        return False


def clear_cache(
    train_dir: str,
    cache_dir: Optional[str] = None,
    keep_metadata: bool = False
) -> bool:
    """
    Clear cached AFRI profiles.
    
    Args:
        train_dir: Path to training data directory
        cache_dir: Optional custom cache directory
        keep_metadata: If True, keep metadata file for debugging
        
    Returns:
        True if cache was cleared successfully, False otherwise
        
    Example:
        >>> clear_cache('/path/to/train')
        True
        >>> clear_cache('/path/to/train', keep_metadata=True)
        True
    """
    try:
        metadata_path, profiles_path = get_cache_paths(train_dir, cache_dir)
        
        removed_files = []
        
        # Remove profiles file
        if profiles_path.exists():
            profiles_path.unlink()
            removed_files.append(str(profiles_path))
        
        # Remove metadata file (unless keep_metadata is True)
        if not keep_metadata and metadata_path.exists():
            metadata_path.unlink()
            removed_files.append(str(metadata_path))
        
        if removed_files:
            print(f"✓ Cache cleared: {len(removed_files)} file(s) removed")
            for f in removed_files:
                print(f"  - {f}")
        else:
            print("No cache files found to clear")
        
        return True
        
    except Exception as e:
        print(f"Error clearing cache: {e}")
        return False


def get_cache_info(
    train_dir: str,
    cache_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get information about the current cache state.
    
    Args:
        train_dir: Path to training data directory
        cache_dir: Optional custom cache directory
        
    Returns:
        Dictionary with cache information:
        - exists: Whether cache files exist
        - cache_key: Current cache key (if exists)
        - timestamp: Cache creation timestamp (if exists)
        - size_mb: Total cache size in MB (if exists)
        - num_classes: Number of cached classes (if exists)
        - class_names: List of cached class names (if exists)
        - afri_config: AFRI configuration (if exists)
        - feature_config: Feature configuration (if exists)
        
    Example:
        >>> info = get_cache_info('/path/to/train')
        >>> if info['exists']:
        ...     print(f"Cache size: {info['size_mb']:.2f} MB")
        ...     print(f"Classes: {info['num_classes']}")
        ... else:
        ...     print("No cache found")
    """
    info = {
        'exists': False,
        'cache_key': None,
        'timestamp': None,
        'timestamp_iso': None,
        'size_mb': 0.0,
        'num_classes': 0,
        'class_names': [],
        'afri_config': None,
        'feature_config': None
    }
    
    try:
        metadata_path, profiles_path = get_cache_paths(train_dir, cache_dir)
        
        # Check if cache exists
        if not metadata_path.exists() or not profiles_path.exists():
            return info
        
        info['exists'] = True
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        info['cache_key'] = metadata.get('cache_key')
        info['timestamp'] = metadata.get('timestamp')
        info['timestamp_iso'] = metadata.get('timestamp_iso')
        info['num_classes'] = metadata.get('num_classes', 0)
        info['class_names'] = metadata.get('class_names', [])
        info['afri_config'] = metadata.get('afri_config')
        info['feature_config'] = metadata.get('feature_config')
        
        # Calculate total size
        metadata_size = metadata_path.stat().st_size
        profiles_size = profiles_path.stat().st_size
        info['size_mb'] = (metadata_size + profiles_size) / (1024 * 1024)
        
    except Exception as e:
        print(f"Error getting cache info: {e}")
    
    return info


def print_cache_info(train_dir: str, cache_dir: Optional[str] = None) -> None:
    """
    Print formatted cache information.
    
    Args:
        train_dir: Path to training data directory
        cache_dir: Optional custom cache directory
        
    Example:
        >>> print_cache_info('/path/to/train')
        AFRI Cache Information
        =====================
        Status: EXISTS
        Cache Key: abc123def456
        Created: 2024-01-15 10:30:45
        Size: 2.45 MB
        Classes: 5 (class1, class2, class3, class4, class5)
    """
    info = get_cache_info(train_dir, cache_dir)
    
    print("\nAFRI Cache Information")
    print("=" * 50)
    
    if info['exists']:
        print(f"Status: EXISTS")
        print(f"Cache Key: {info['cache_key']}")
        print(f"Created: {info['timestamp_iso']}")
        print(f"Size: {info['size_mb']:.2f} MB")
        print(f"Classes: {info['num_classes']} ({', '.join(info['class_names'])})")
        
        if info['afri_config']:
            print("\nAFRI Configuration:")
            for key, value in info['afri_config'].items():
                print(f"  {key}: {value}")
        
        if info['feature_config']:
            print("\nFeature Configuration:")
            for key, value in info['feature_config'].items():
                print(f"  {key}: {value}")
    else:
        print("Status: NOT FOUND")
        print("No cache files exist for this training directory.")
    
    print("=" * 50)


# Utility function for testing/debugging
def validate_cache_integrity(
    train_dir: str,
    cache_dir: Optional[str] = None
) -> Tuple[bool, str]:
    """
    Validate cache file integrity.
    
    Args:
        train_dir: Path to training data directory
        cache_dir: Optional custom cache directory
        
    Returns:
        Tuple of (is_valid, message)
        
    Example:
        >>> is_valid, message = validate_cache_integrity('/path/to/train')
        >>> if is_valid:
        ...     print("Cache is valid")
        ... else:
        ...     print(f"Cache is invalid: {message}")
    """
    if not HAS_NUMPY:
        return False, "NumPy not available"
    
    try:
        metadata_path, profiles_path = get_cache_paths(train_dir, cache_dir)
        
        # Check existence
        if not metadata_path.exists():
            return False, "Metadata file missing"
        if not profiles_path.exists():
            return False, "Profiles file missing"
        
        # Validate metadata JSON
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        required_keys = ['cache_key', 'timestamp', 'num_classes', 'class_names']
        for key in required_keys:
            if key not in metadata:
                return False, f"Missing required metadata key: {key}"
        
        # Validate NPZ file
        npz_data = np.load(profiles_path, allow_pickle=True)
        
        # Check if number of classes matches
        if len(npz_data.files) != metadata['num_classes']:
            return False, f"Class count mismatch: metadata={metadata['num_classes']}, npz={len(npz_data.files)}"
        
        # Check if class names match
        npz_classes = set(npz_data.files)
        metadata_classes = set(metadata['class_names'])
        if npz_classes != metadata_classes:
            return False, f"Class name mismatch"
        
        return True, "Cache is valid"
        
    except json.JSONDecodeError:
        return False, "Corrupted metadata JSON"
    except Exception as e:
        return False, f"Validation error: {str(e)}"


if __name__ == "__main__":
    # Example usage
    print("AFRI Cache System - Example Usage")
    print("=" * 50)
    
    # Example configuration
    train_dir = "/path/to/train/data"
    afri_config = {
        'afri_n_samples': 5,
        'afri_value_threshold': 0.23,
        'afri_search_threshold': 7,
        'afri_debris_size': 10,
        'afri_min_bin_expand': 2,
        'afri_max_bin_expand': 6,
    }
    feature_config = {
        'n_mels': 128,
        'sr': 22050,
        'hop_mel': 256,
    }
    
    # Generate cache key
    cache_key = generate_afri_cache_key(train_dir, afri_config, feature_config)
    print(f"\nGenerated cache key: {cache_key}")
    
    # Try loading from cache
    print("\nAttempting to load from cache...")
    profiles = load_cached_profiles(train_dir, cache_key)
    
    if profiles:
        print(f"✓ Loaded {len(profiles)} class profiles from cache")
    else:
        print("✗ Cache miss - would need to compute profiles")
    
    # Print cache info
    print_cache_info(train_dir)
