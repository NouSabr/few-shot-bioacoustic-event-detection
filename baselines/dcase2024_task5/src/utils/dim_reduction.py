"""
Dimensionality Reduction Utilities for Prototype Visualization.

This module provides memory-efficient dimensionality reduction methods for
visualizing high-dimensional embeddings in 2D space.

Date: 2026-04-26
"""

from typing import Optional, Union, Tuple
from typing_extensions import Literal
import numpy as np
import torch
from sklearn.decomposition import PCA, IncrementalPCA
import logging

logger = logging.getLogger(__name__)

# Optional UMAP import
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    logger.warning("UMAP not available. Install with: pip install umap-learn")


class DimensionalityReducer:
    """
    Memory-efficient dimensionality reduction for embedding visualization.
    
    Supports PCA, Incremental PCA, and UMAP methods with automatic
    tensor handling and optional caching for consistency across episodes.
    
    Args:
        method: Reduction method ('pca', 'incremental_pca', or 'umap')
        n_components: Number of output dimensions (default: 2)
        use_cache: Whether to cache the fitted reducer (default: True)
        random_state: Random seed for reproducibility (default: 42)
        **kwargs: Additional arguments passed to the reducer
    
    Example:
        >>> reducer = DimensionalityReducer(method='pca', n_components=2)
        >>> embeddings_2d = reducer.fit_transform(embeddings)
        >>> # Later, transform new data with same reducer
        >>> new_embeddings_2d = reducer.transform(new_embeddings)
    """
    
    def __init__(
        self,
        method: Literal['pca', 'incremental_pca', 'umap'] = 'pca',
        n_components: int = 2,
        use_cache: bool = True,
        random_state: int = 42,
        **kwargs
    ):
        self.method = method.lower()
        self.n_components = n_components
        self.use_cache = use_cache
        self.random_state = random_state
        self.kwargs = kwargs
        
        # Validate method
        valid_methods = ['pca', 'incremental_pca', 'umap']
        if self.method not in valid_methods:
            raise ValueError(f"Method must be one of {valid_methods}, got {self.method}")
        
        if self.method == 'umap' and not UMAP_AVAILABLE:
            logger.warning("UMAP not available, falling back to PCA")
            self.method = 'pca'
        
        # Initialize reducer
        self.reducer = None
        self._is_fitted = False
        self._initialize_reducer()
    
    def _initialize_reducer(self):
        """Initialize the dimensionality reduction model."""
        if self.method == 'pca':
            self.reducer = PCA(
                n_components=self.n_components,
                random_state=self.random_state,
                **self.kwargs
            )
            logger.debug("Initialized PCA reducer")
            
        elif self.method == 'incremental_pca':
            self.reducer = IncrementalPCA(
                n_components=self.n_components,
                **self.kwargs
            )
            logger.debug("Initialized Incremental PCA reducer")
            
        elif self.method == 'umap':
            self.reducer = umap.UMAP(
                n_components=self.n_components,
                random_state=self.random_state,
                **self.kwargs
            )
            logger.debug("Initialized UMAP reducer")
    
    def _to_numpy(self, data: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Convert input to numpy array with memory efficiency.
        
        Args:
            data: Input data (numpy array or torch tensor)
        
        Returns:
            Numpy array on CPU
        """
        if isinstance(data, torch.Tensor):
            # Detach from computation graph and move to CPU
            data_np = data.detach().cpu().numpy()
            return data_np
        elif isinstance(data, np.ndarray):
            return data
        else:
            raise TypeError(f"Expected numpy array or torch tensor, got {type(data)}")
    
    def fit(self, X: Union[np.ndarray, torch.Tensor]) -> 'DimensionalityReducer':
        """
        Fit the reducer on the data.
        
        Args:
            X: Input data of shape (n_samples, n_features)
        
        Returns:
            Self for method chaining
        """
        X_np = self._to_numpy(X)
        
        # Validate input shape
        if X_np.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {X_np.shape}")
        
        if X_np.shape[0] < self.n_components:
            logger.warning(
                f"Number of samples ({X_np.shape[0]}) is less than "
                f"n_components ({self.n_components}). Results may be unreliable."
            )
        
        logger.debug(f"Fitting {self.method} on data with shape {X_np.shape}")
        self.reducer.fit(X_np)
        self._is_fitted = True
        
        # Clear input data to free memory
        del X_np
        
        return self
    
    def transform(self, X: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Transform data using the fitted reducer.
        
        Args:
            X: Input data of shape (n_samples, n_features)
        
        Returns:
            Transformed data of shape (n_samples, n_components)
        """
        if not self._is_fitted:
            raise RuntimeError("Reducer must be fitted before transform. Use fit() or fit_transform().")
        
        X_np = self._to_numpy(X)
        
        logger.debug(f"Transforming data with shape {X_np.shape}")
        X_transformed = self.reducer.transform(X_np)
        
        # Clear input data to free memory
        del X_np
        
        return X_transformed
    
    def fit_transform(self, X: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Fit the reducer and transform the data in one step.
        
        Args:
            X: Input data of shape (n_samples, n_features)
        
        Returns:
            Transformed data of shape (n_samples, n_components)
        """
        X_np = self._to_numpy(X)
        
        # Validate input shape
        if X_np.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {X_np.shape}")
        
        if X_np.shape[0] < self.n_components:
            logger.warning(
                f"Number of samples ({X_np.shape[0]}) is less than "
                f"n_components ({self.n_components}). Results may be unreliable."
            )
        
        logger.debug(f"Fitting and transforming {self.method} on data with shape {X_np.shape}")
        X_transformed = self.reducer.fit_transform(X_np)
        self._is_fitted = True
        
        # Clear input data to free memory
        del X_np
        
        return X_transformed
    
    def reset(self):
        """Reset the reducer to unfitted state."""
        self._initialize_reducer()
        self._is_fitted = False
        logger.debug("Reducer reset to unfitted state")
    
    @property
    def is_fitted(self) -> bool:
        """Check if the reducer has been fitted."""
        return self._is_fitted


def reduce_dimensions(
    data: Union[np.ndarray, torch.Tensor],
    method: Literal['pca', 'incremental_pca', 'umap'] = 'pca',
    n_components: int = 2,
    reducer: Optional[DimensionalityReducer] = None,
    fit: bool = True,
    **kwargs
) -> Tuple[np.ndarray, Optional[DimensionalityReducer]]:
    """
    Convenience function for dimensionality reduction.
    
    Args:
        data: Input data of shape (n_samples, n_features)
        method: Reduction method ('pca', 'incremental_pca', or 'umap')
        n_components: Number of output dimensions (default: 2)
        reducer: Pre-fitted reducer to use (optional)
        fit: Whether to fit a new reducer if none provided (default: True)
        **kwargs: Additional arguments passed to the reducer
    
    Returns:
        Tuple of (reduced_data, reducer_used)
        - reduced_data: Transformed data of shape (n_samples, n_components)
        - reducer_used: The reducer that was used (for caching)
    
    Example:
        >>> # First call - fit and transform
        >>> data_2d, reducer = reduce_dimensions(embeddings, method='pca')
        >>> 
        >>> # Later call - reuse fitted reducer
        >>> new_data_2d, _ = reduce_dimensions(new_embeddings, reducer=reducer, fit=False)
    """
    if reducer is not None:
        # Use provided reducer
        if fit:
            reduced_data = reducer.fit_transform(data)
        else:
            reduced_data = reducer.transform(data)
        return reduced_data, reducer
    else:
        # Create new reducer
        reducer = DimensionalityReducer(
            method=method,
            n_components=n_components,
            **kwargs
        )
        reduced_data = reducer.fit_transform(data)
        return reduced_data, reducer

