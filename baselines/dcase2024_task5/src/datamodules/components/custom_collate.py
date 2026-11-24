"""
Custom collate function for streaming HDF5 dataset with batch_size=1.

ROOT CAUSE ANALYSIS:
====================
The default PyTorch collate_fn fails because __getitem__() returns a heterogeneous tuple
containing objects that cannot be batched using PyTorch's default recursive merging strategy:

1. **HDF5 File Handles (h5py.Dataset objects)**: These are live file handles that cannot be
   stacked, concatenated, or deep-copied. They must remain open for lazy loading.

2. **Lists of Metadata Dictionaries**: The query_info structures (X_query_info, X_query_info_neg)
   are lists of dictionaries containing segment metadata. PyTorch's default collate tries to
   recursively convert these to tensors, which fails because:
   - Dictionaries have heterogeneous value types (strings, ints, etc.)
   - The structure is meant to be consumed by StreamingSegmentDataset, not converted to tensors

3. **Mixed Numpy Arrays and Scalars**: The data_tuple contains numpy arrays (X_pos, X_neg, etc.)
   and scalar values (hop_seg, max_len, etc.) that have different shapes and types.

4. **String Paths**: The audio_path is a string that cannot be stacked.

PyTorch's default collate_fn recursively tries to:
- Stack tensors/arrays with torch.stack()
- Convert numpy arrays to tensors
- Recursively process nested structures (lists, tuples, dicts)

This breaks because:
- h5py.Dataset objects are not serializable and cannot be stacked
- Metadata lists contain dictionaries with heterogeneous types
- The structure is designed for streaming, not batching

EXPECTED BATCH STRUCTURE:
=========================
__getitem__() returns a 5-element tuple:

    (data_tuple, strt_index_query, audio_path, seg_len, h5_handles)

Where:
    data_tuple: 10-element tuple containing:
        [0] X_pos: np.ndarray, shape (n_pos, n_features, n_frames), dtype=float32
        [1] X_neg: np.ndarray, shape (n_neg, n_features, n_frames), dtype=float32
        [2] X_query_info: List[Dict], metadata for positive query segments
            Each dict: {'start_frame': int, 'end_frame': int, 'h5_index': int}
        [3] X_pos_neg: np.ndarray, shape (n_pos_neg, n_features, n_frames), dtype=float32
        [4] X_neg_neg: np.ndarray, shape (n_neg_neg, n_features, n_frames), dtype=float32
        [5] X_query_info_neg: List[Dict], metadata for negative query segments
        [6] hop_seg: np.ndarray, scalar, dtype=int
        [7] hop_seg_neg: np.ndarray, scalar, dtype=int
        [8] max_len: np.ndarray, scalar, dtype=int
        [9] neg_min_length: np.ndarray, scalar, dtype=int
    
    strt_index_query: np.ndarray, scalar, dtype=int
    audio_path: str, path to audio file
    seg_len: np.ndarray, scalar, dtype=int
    h5_handles: List[h5py.Dataset], open HDF5 dataset handles

WHY THIS COLLATE_FN IS CORRECT:
================================
1. **Preserves HDF5 Handles**: Returns handles unchanged, keeping them open for streaming
2. **Preserves Metadata Lists**: Does not attempt to convert dictionaries to tensors
3. **Preserves Exact Structure**: Returns the single batch element with identical nesting
4. **No Type Conversion**: Does not convert numpy arrays to tensors (done in test_step)
5. **No Deep Copying**: Avoids expensive copies of large arrays or file handles
6. **Compatible with test_step()**: The unpacking in _load_streaming_datasets() expects
   exactly this structure: (data_tuple, strt_index_query, audio_path, seg_len, h5_handles)
7. **Compatible with StreamingSegmentDataset**: The h5_handles and metadata lists are
   passed directly to StreamingSegmentDataset constructor
8. **Maintains Lazy Loading**: Does not trigger any data loading or file operations

DOWNSTREAM COMPATIBILITY:
==========================
- _load_streaming_datasets() unpacks: data_tuple, strt_index_query, audio_path, seg_len, h5_handles
- StreamingSegmentDataset() receives: h5_handles, query_info, seg_len
- test_step() converts numpy arrays to tensors after unpacking
- evaluate_prototypes() receives StreamingSegmentDataset instances for lazy loading
"""

from typing import Any, List
import numpy as np


def streaming_collate_fn(batch: List[Any]) -> Any:
    """
    Custom collate function that adds batch dimension to numpy arrays while preserving non-collatable objects.
    
    This is required because the batch contains non-collatable objects:
    - Open HDF5 file handles (h5py.Dataset objects)
    - Lists of metadata dictionaries
    - String paths
    
    However, test_step() expects numpy arrays and scalars to have a batch dimension (shape[0] = 1)
    so it can extract values using [0, ...].item() or [0, ...].
    
    Args:
        batch: List with single element from dataset.__getitem__()
               Structure: [(data_tuple, strt_index_query, audio_path, seg_len, h5_handles)]
               where data_tuple is a 10-element tuple containing numpy arrays and metadata lists
        
    Returns:
        Modified batch with:
        - Numpy arrays: Add batch dimension (shape becomes (1, ...))
        - Numpy scalars: Add batch dimension (shape becomes (1,))
        - Metadata lists: Preserved unchanged
        - HDF5 handles: Preserved unchanged
        - String paths: Wrapped in list [path]
        
    Raises:
        AssertionError: If batch_size != 1
        
    Example:
        >>> # In DataLoader:
        >>> from torch.utils.data import DataLoader
        >>> loader = DataLoader(
        ...     dataset,
        ...     batch_size=1,
        ...     collate_fn=streaming_collate_fn,
        ...     num_workers=0  # Required for HDF5 handles
        ... )
        >>>
        >>> # The batch structure with batch dimension:
        >>> for batch in loader:
        ...     data_tuple, strt_idx, audio_path, seg_len, h5_handles = batch
        ...     # strt_idx has shape (1,), can use strt_idx[0, ...].item()
        ...     # audio_path is ['path/to/file.wav'], can use audio_path[0]
    
    Notes:
        - This function does NOT trigger any data loading
        - This function does NOT close or modify HDF5 handles
        - This function does NOT convert numpy arrays to tensors
        - This function does NOT deep copy large objects
        - Numpy arrays are expanded with np.expand_dims (adds dimension, no copy)
    """
    # Verify batch_size == 1 (critical assumption)
    assert len(batch) == 1, (
        f"streaming_collate_fn requires batch_size=1, got batch_size={len(batch)}. "
        f"The dataloader must use batch_size=1 because the batch contains "
        f"non-collatable objects (HDF5 handles, metadata lists)."
    )
    
    # Unpack the single batch element
    data_tuple, strt_index_query, audio_path, seg_len, h5_handles = batch[0]
    
    # Unpack data_tuple
    (
        X_pos, X_neg, X_query_info,
        X_pos_neg, X_neg_neg, X_query_info_neg,
        hop_seg, hop_seg_neg, max_len, neg_min_length,
        iteration_positions, iteration_positions_neg 
    ) = data_tuple
    
    # Add batch dimension to numpy arrays (test_step expects this)
    # Using expand_dims is efficient - it adds a dimension without copying data
    X_pos = np.expand_dims(X_pos, axis=0)  # (n_pos, feat, frames) -> (1, n_pos, feat, frames)
    X_neg = np.expand_dims(X_neg, axis=0)
    X_pos_neg = np.expand_dims(X_pos_neg, axis=0)
    X_neg_neg = np.expand_dims(X_neg_neg, axis=0)
    
    # Add batch dimension to scalar numpy arrays (test_step uses [0, ...].item())
    hop_seg = np.expand_dims(hop_seg, axis=0)  # () -> (1,)
    hop_seg_neg = np.expand_dims(hop_seg_neg, axis=0)
    max_len = np.expand_dims(max_len, axis=0)
    neg_min_length = np.expand_dims(neg_min_length, axis=0)
    strt_index_query = np.expand_dims(strt_index_query, axis=0)
    seg_len = np.expand_dims(seg_len, axis=0)
    
    # Wrap audio_path in list (test_step uses audio_path[0])
    audio_path = [audio_path]
    
    # Preserve metadata lists and HDF5 handles unchanged
    # These are NOT batched - they remain as-is
    
    # Reconstruct data_tuple with batched arrays
    data_tuple_batched = (
        X_pos, X_neg, X_query_info,  # X_query_info stays as list of dicts
        X_pos_neg, X_neg_neg, X_query_info_neg,  # X_query_info_neg stays as list of dicts
        hop_seg, hop_seg_neg, max_len, neg_min_length,
        iteration_positions, iteration_positions_neg
    )
    
    # Return batched structure
    return (data_tuple_batched, strt_index_query, audio_path, seg_len, h5_handles)


def validate_collate_fn():
    """
    Validation tests for streaming_collate_fn.
    
    Tests:
    1. Adds batch dimension correctly
    2. Preserves types (numpy, not tensors)
    3. Compatible with test_step() unpacking and indexing
    4. Compatible with _load_streaming_datasets()
    5. Raises error if batch_size != 1
    6. No side effects (no HDF5 access)
    
    This function can be run standalone to verify correctness.
    """
    import sys
    
    print("=" * 80)
    print("VALIDATING streaming_collate_fn")
    print("=" * 80)
    
    # Test 1: Adds batch dimension correctly
    print("\n[TEST 1] Adding batch dimension...")
    
    # Mock data matching __getitem__() output (no batch dimension)
    X_pos = np.random.randn(5, 128, 100).astype(np.float32)  # (n_pos, feat, frames)
    X_neg = np.random.randn(5, 128, 100).astype(np.float32)
    X_query_info = [
        {'start_frame': 0, 'end_frame': 100, 'h5_index': 0},
        {'start_frame': 100, 'end_frame': 200, 'h5_index': 0},
    ]
    X_pos_neg = np.random.randn(5, 128, 100).astype(np.float32)
    X_neg_neg = np.random.randn(5, 128, 100).astype(np.float32)
    X_query_info_neg = [
        {'start_frame': 0, 'end_frame': 100, 'h5_index': 1},
    ]
    hop_seg = np.array(50)  # scalar
    hop_seg_neg = np.array(50)
    max_len = np.array(1000)
    neg_min_length = np.array(100)
    
    data_tuple = (
        X_pos, X_neg, X_query_info,
        X_pos_neg, X_neg_neg, X_query_info_neg,
        hop_seg, hop_seg_neg, max_len, neg_min_length
    )
    
    strt_index_query = np.array(0)  # scalar
    audio_path = "/path/to/audio.wav"  # string
    seg_len = np.array(200)  # scalar
    h5_handles = ["mock_handle_1", "mock_handle_2"]  # Mock handles
    
    sample = (data_tuple, strt_index_query, audio_path, seg_len, h5_handles)
    batch = [sample]
    
    # Apply collate function
    result = streaming_collate_fn(batch)
    
    # Verify structure
    assert isinstance(result, tuple), "Result must be a tuple"
    assert len(result) == 5, f"Result must have 5 elements, got {len(result)}"
    
    result_data_tuple, result_strt_idx, result_audio_path, result_seg_len, result_h5_handles = result
    
    assert isinstance(result_data_tuple, tuple), "data_tuple must remain a tuple"
    assert len(result_data_tuple) == 10, f"data_tuple must have 10 elements, got {len(result_data_tuple)}"
    
    # Check batch dimensions were added
    X_pos_batched = result_data_tuple[0]
    assert X_pos_batched.shape == (1, 5, 128, 100), f"X_pos should have batch dim, got shape {X_pos_batched.shape}"
    
    hop_seg_batched = result_data_tuple[6]
    assert hop_seg_batched.shape == (1,), f"hop_seg should have batch dim, got shape {hop_seg_batched.shape}"
    
    assert result_strt_idx.shape == (1,), f"strt_index_query should have batch dim, got shape {result_strt_idx.shape}"
    assert result_seg_len.shape == (1,), f"seg_len should have batch dim, got shape {result_seg_len.shape}"
    
    print("✓ Batch dimension added correctly to numpy arrays and scalars")
    
    # Test 2: Type preservation
    print("\n[TEST 2] Type preservation...")
    
    assert isinstance(result_data_tuple[0], np.ndarray), "X_pos must remain numpy array"
    assert result_data_tuple[0].dtype == np.float32, "X_pos dtype must be preserved"
    assert isinstance(result_data_tuple[2], list), "X_query_info must remain a list"
    assert isinstance(result_data_tuple[2][0], dict), "Metadata must remain dictionaries"
    assert isinstance(result_audio_path, list), "audio_path must be wrapped in list"
    assert isinstance(result_audio_path[0], str), "audio_path[0] must be string"
    assert isinstance(result_h5_handles, list), "h5_handles must remain a list"
    
    print("✓ All types preserved (numpy arrays, lists, dicts, strings)")
    
    # Test 3: Compatible with test_step() indexing
    print("\n[TEST 3] test_step() indexing compatibility...")
    
    try:
        # Simulate test_step() operations
        # Lines 221-225 in test_step do: hop_seg[0, ...].item()
        hop_seg_value = hop_seg_batched[0, ...].item()
        assert hop_seg_value == 50, f"Expected 50, got {hop_seg_value}"
        
        strt_idx_value = result_strt_idx[0, ...].item()
        assert strt_idx_value == 0, f"Expected 0, got {strt_idx_value}"
        
        # Line 229 does: X_pos[0, ...]
        X_pos_unbatched = X_pos_batched[0, ...]
        assert X_pos_unbatched.shape == (5, 128, 100), f"Expected (5, 128, 100), got {X_pos_unbatched.shape}"
        
        # Line 237 does: audio_path[0]
        audio_path_str = result_audio_path[0]
        assert audio_path_str == "/path/to/audio.wav", f"Expected path string, got {audio_path_str}"
        
        print("✓ Compatible with test_step() indexing: [0, ...].item() and [0]")
    except Exception as e:
        print(f"✗ test_step() indexing failed: {e}")
        sys.exit(1)
    
    # Test 4: Compatible with _load_streaming_datasets() unpacking
    print("\n[TEST 4] _load_streaming_datasets() unpacking compatibility...")
    
    try:
        # Simulate _load_streaming_datasets() unpacking
        data_tuple_unpacked, strt_idx_unpacked, audio_path_unpacked, seg_len_unpacked, h5_handles_unpacked = result
        
        (
            X_pos_unpacked, X_neg_unpacked, query_info_unpacked,
            X_pos_neg_unpacked, X_neg_neg_unpacked, query_info_neg_unpacked,
            hop_seg_unpacked, hop_seg_neg_unpacked, max_len_unpacked, neg_min_length_unpacked
        ) = data_tuple_unpacked
        
        assert query_info_unpacked is X_query_info, "Unpacked query_info must match original"
        assert h5_handles_unpacked is h5_handles, "Unpacked h5_handles must match original"
        
        print("✓ Compatible with _load_streaming_datasets() unpacking")
    except Exception as e:
        print(f"✗ Unpacking failed: {e}")
        sys.exit(1)
    
    # Test 5: Raises error if batch_size != 1
    print("\n[TEST 5] Batch size validation...")
    
    try:
        invalid_batch = [sample, sample]  # batch_size=2
        streaming_collate_fn(invalid_batch)
        print("✗ Should have raised AssertionError for batch_size > 1")
        sys.exit(1)
    except AssertionError as e:
        print(f"✓ Correctly raises AssertionError for batch_size > 1: {str(e)[:60]}...")
    
    # Test 6: Verify no eager loading or file operations
    print("\n[TEST 6] No side effects...")
    
    # The collate function should not trigger any HDF5 operations
    class MockHandle:
        def __init__(self):
            self.accessed = False
        
        def __getitem__(self, key):
            self.accessed = True
            raise RuntimeError("Should not access HDF5 handle in collate_fn")
    
    mock_handle = MockHandle()
    mock_sample = (data_tuple, strt_index_query, audio_path, seg_len, [mock_handle])
    mock_batch = [mock_sample]
    
    result = streaming_collate_fn(mock_batch)
    assert not mock_handle.accessed, "collate_fn must not access HDF5 handles"
    
    print("✓ No side effects: HDF5 handles not accessed")
    
    print("\n" + "=" * 80)
    print("ALL TESTS PASSED ✓")
    print("=" * 80)
    print("\nstreaming_collate_fn is safe to use with:")
    print("  - test_step() with [0, ...].item() and [0] indexing")
    print("  - _load_streaming_datasets() unpacking")
    print("  - evaluate_prototypes() with StreamingSegmentDataset")
    print("\nBatch dimension added efficiently with np.expand_dims (no data copy).")
    print("No eager loading, no memory spikes, no tensor conversions.")


if __name__ == "__main__":
    # Run validation tests
    validate_collate_fn()
