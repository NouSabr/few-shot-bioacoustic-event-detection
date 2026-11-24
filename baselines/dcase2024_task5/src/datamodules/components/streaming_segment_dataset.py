"""
Streaming dataset for memory-efficient segment loading.
Loads audio segments on-demand from HDF5 files instead of loading entire arrays.

Reference: baselines/dcase2024_task5/docs/memory_optimization/03_IMPLEMENTATION_PLAN.md
Section: Phase 1 - Create StreamingSegmentDataset
"""

import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
from typing import List, Dict, Optional
from collections import OrderedDict


class StreamingSegmentDataset(Dataset):
    """
    Memory-efficient dataset that loads segments on-demand from HDF5 handles.
    
    Features:
    - Lazy loading: Only loads segments when accessed
    - LRU caching: Keeps recently used segments in memory
    - HDF5 backend: Efficient chunked access
    - PyTorch compatible: Standard Dataset interface
    
    Args:
        h5_handles: List of HDF5 dataset handles (already opened)
        segments: List of segment dictionaries with 'start_frame' and 'end_frame'
        seg_len: Segment length (for validation)
        cache_size: Number of segments to keep in memory (default: 100)
        transform: Optional transform to apply to segments
    """
    
    def __init__(
        self,
        h5_handles: List,
        segments: List[Dict],
        seg_len: int,
        cache_size: int = 100,
        transform: Optional[callable] = None
    ):
        self.h5_handles = h5_handles
        self.segments = segments
        self.seg_len = seg_len
        self.cache_size = cache_size
        self.transform = transform
        
        # LRU cache for loaded segments
        self.cache = OrderedDict()
        
        # Get dataset shape for validation from first handle
        self.pcen_shape = self.h5_handles[0].shape
        
    def _open_h5_file(self):
        """No-op: HDF5 handles are already open and managed externally."""
        pass
    
    def _close_h5_file(self):
        """No-op: HDF5 handles are managed externally and should not be closed here."""
        pass
    
    def __len__(self) -> int:
        """Return number of segments."""
        return len(self.segments)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Load segment at given index.
        
        Uses LRU cache to avoid redundant I/O operations.
        
        Args:
            idx: Segment index
            
        Returns:
            Segment tensor of shape (time_frames, freq_bins)
        """
        # Check cache first
        if idx in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(idx)
            segment = self.cache[idx]
        else:
            # Load from HDF5
            segment = self._load_segment(idx)
            
            # Add to cache
            self.cache[idx] = segment
            
            # Evict oldest if cache is full
            if len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        
        # Apply transform if provided
        if self.transform is not None:
            segment = self.transform(segment)
        
        # Create dummy label for compatibility with dataloader
        dummy_label = torch.tensor(0, dtype=torch.long)
        return segment, dummy_label
    
    def _load_segment(self, idx: int) -> torch.Tensor:
        """
        Load segment from HDF5 handles.
        
        Args:
            idx: Segment index
            
        Returns:
            Segment tensor
        """
        seg_info = self.segments[idx]
        start_frame = seg_info['start_frame']
        end_frame = seg_info['end_frame']
        
        # Load segment slice from first HDF5 handle (primary feature)
        segment_data = self.h5_handles[0][start_frame:end_frame, :]
        
        # Convert to PyTorch tensor
        segment = torch.from_numpy(segment_data).float()
        
        return segment
    
    def clear_cache(self):
        """Clear the segment cache."""
        self.cache.clear()
    
    def get_cache_stats(self) -> Dict:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache size and hit rate information
        """
        return {
            'cache_size': len(self.cache),
            'max_cache_size': self.cache_size,
            'cache_utilization': len(self.cache) / self.cache_size
        }
    
    def __del__(self):
        """Cleanup: Close HDF5 file."""
        self._close_h5_file()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit: Close HDF5 file."""
        self._close_h5_file()


class StreamingSegmentDataLoader:
    """
    Wrapper for StreamingSegmentDataset that provides batch loading.
    
    Args:
        dataset: StreamingSegmentDataset instance
        batch_size: Number of segments per batch
        shuffle: Whether to shuffle segments (not recommended for streaming)
    """
    
    def __init__(
        self,
        dataset: StreamingSegmentDataset,
        batch_size: int = 32,
        shuffle: bool = False
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Generate indices
        self.indices = list(range(len(dataset)))
        if shuffle:
            np.random.shuffle(self.indices)
    
    def __iter__(self):
        """Iterate over batches."""
        for i in range(0, len(self.indices), self.batch_size):
            batch_indices = self.indices[i:i + self.batch_size]
            batch = [self.dataset[idx] for idx in batch_indices]
            yield torch.stack(batch)
    
    def __len__(self):
        """Return number of batches."""
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size
