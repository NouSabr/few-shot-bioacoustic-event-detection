"""
Prototype Network Visualization Utilities.

This module provides memory-efficient visualization tools for prototypical networks,
including embedding space visualization with dimensionality reduction.

Date: 2026-04-26
"""

from typing import Optional, Union, Dict, Any, List
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import logging

from .dim_reduction import DimensionalityReducer, reduce_dimensions

logger = logging.getLogger(__name__)


class PrototypeVisualizer:
    """
    Memory-efficient visualization for prototypical network embeddings.
    
    This class provides methods to visualize high-dimensional embeddings in 2D space,
    showing support samples, query samples, and prototypes with proper color coding
    and markers.
    
    Args:
        reduction_method: Dimensionality reduction method ('pca', 'incremental_pca', 'umap')
        n_components: Number of dimensions for reduction (default: 2)
        use_cache: Whether to cache the fitted reducer (default: True)
        figsize: Figure size as (width, height) tuple (default: (12, 8))
        dpi: Figure DPI for quality (default: 100)
        style: Matplotlib style to use (default: 'seaborn-v0_8-darkgrid')
    
    Example:
        >>> visualizer = PrototypeVisualizer(reduction_method='pca')
        >>> fig = visualizer.visualize_embeddings(
        ...     support_embeddings=support_emb,
        ...     query_embeddings=query_emb,
        ...     support_labels=support_labels,
        ...     query_labels=query_labels,
        ...     prototypes=prototypes
        ... )
        >>> visualizer.save_visualization(fig, 'episode_1000.png')
    """
    
    def __init__(
        self,
        reduction_method: str = 'pca',
        n_components: int = 2,
        use_cache: bool = True,
        figsize: tuple = (12, 8),
        dpi: int = 100,
        style: str = 'darkgrid'
    ):
        self.reduction_method = reduction_method
        self.n_components = n_components
        self.use_cache = use_cache
        self.figsize = figsize
        self.dpi = dpi
        self.style = style
        
        # Initialize reducer if caching is enabled
        self.reducer = None
        if self.use_cache:
            self.reducer = DimensionalityReducer(
                method=reduction_method,
                n_components=n_components,
                use_cache=True
            )
        
        # Color palette for classes (supports up to 20 classes)
        self.colors = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
            '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
            '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5'
        ]
    
    def _to_numpy(self, data: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Convert tensor to numpy array efficiently."""
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
        return data
    
    def _prepare_data(
        self,
        support_embeddings: Union[np.ndarray, torch.Tensor],
        query_embeddings: Union[np.ndarray, torch.Tensor],
        support_labels: Union[np.ndarray, torch.Tensor],
        query_labels: Union[np.ndarray, torch.Tensor],
        prototypes: Optional[Union[np.ndarray, torch.Tensor]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Prepare and validate input data.
        
        Returns:
            Dictionary with numpy arrays for all inputs
        """
        # Convert to numpy
        support_emb = self._to_numpy(support_embeddings)
        query_emb = self._to_numpy(query_embeddings)
        support_lbl = self._to_numpy(support_labels)
        query_lbl = self._to_numpy(query_labels)
        
        # Validate shapes
        if support_emb.ndim != 2:
            raise ValueError(f"Support embeddings must be 2D, got shape {support_emb.shape}")
        if query_emb.ndim != 2:
            raise ValueError(f"Query embeddings must be 2D, got shape {query_emb.shape}")
        if support_emb.shape[1] != query_emb.shape[1]:
            raise ValueError(
                f"Support and query embeddings must have same feature dimension. "
                f"Got {support_emb.shape[1]} and {query_emb.shape[1]}"
            )
        
        # Flatten labels if needed
        support_lbl = support_lbl.flatten()
        query_lbl = query_lbl.flatten()
        
        # Validate label counts
        if len(support_lbl) != support_emb.shape[0]:
            raise ValueError(
                f"Number of support labels ({len(support_lbl)}) must match "
                f"number of support embeddings ({support_emb.shape[0]})"
            )
        if len(query_lbl) != query_emb.shape[0]:
            raise ValueError(
                f"Number of query labels ({len(query_lbl)}) must match "
                f"number of query embeddings ({query_emb.shape[0]})"
            )
        
        result = {
            'support_embeddings': support_emb,
            'query_embeddings': query_emb,
            'support_labels': support_lbl,
            'query_labels': query_lbl
        }
        
        # Handle prototypes
        if prototypes is not None:
            proto = self._to_numpy(prototypes)
            if proto.ndim != 2:
                raise ValueError(f"Prototypes must be 2D, got shape {proto.shape}")
            if proto.shape[1] != support_emb.shape[1]:
                raise ValueError(
                    f"Prototypes must have same feature dimension as embeddings. "
                    f"Got {proto.shape[1]} and {support_emb.shape[1]}"
                )
            result['prototypes'] = proto
        
        return result
    
    def _reduce_embeddings(
        self,
        support_embeddings: np.ndarray,
        query_embeddings: np.ndarray,
        prototypes: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """
        Apply dimensionality reduction to all embeddings.
        
        Returns:
            Dictionary with reduced embeddings
        """
        # Combine all embeddings for consistent reduction
        all_embeddings = [support_embeddings, query_embeddings]
        if prototypes is not None:
            all_embeddings.append(prototypes)
        
        combined = np.vstack(all_embeddings)
        
        # Apply dimensionality reduction
        if self.use_cache and self.reducer is not None:
            # Use cached reducer
            if not self.reducer.is_fitted:
                reduced_combined = self.reducer.fit_transform(combined)
            else:
                reduced_combined = self.reducer.transform(combined)
        else:
            # Create new reducer
            reduced_combined, _ = reduce_dimensions(
                combined,
                method=self.reduction_method,
                n_components=self.n_components
            )
        
        # Split back into separate arrays
        n_support = support_embeddings.shape[0]
        n_query = query_embeddings.shape[0]
        
        support_reduced = reduced_combined[:n_support]
        query_reduced = reduced_combined[n_support:n_support + n_query]
        
        result = {
            'support_embeddings': support_reduced,
            'query_embeddings': query_reduced
        }
        
        if prototypes is not None:
            proto_reduced = reduced_combined[n_support + n_query:]
            result['prototypes'] = proto_reduced
        
        # Clean up
        del combined, reduced_combined
        
        return result
    
    def visualize_embeddings(
        self,
        support_embeddings: Union[np.ndarray, torch.Tensor],
        query_embeddings: Union[np.ndarray, torch.Tensor],
        support_labels: Union[np.ndarray, torch.Tensor],
        query_labels: Union[np.ndarray, torch.Tensor],
        prototypes: Optional[Union[np.ndarray, torch.Tensor]] = None,
        title: str = "Prototype Network Embedding Space",
        class_names: Optional[List[str]] = None,
        episode_num: Optional[int] = None
    ) -> plt.Figure:
        """
        Create visualization of embeddings in 2D space.
        
        Args:
            support_embeddings: Support set embeddings (n_support, embedding_dim)
            query_embeddings: Query set embeddings (n_query, embedding_dim)
            support_labels: Support set labels (n_support,)
            query_labels: Query set labels (n_query,)
            prototypes: Class prototypes (n_classes, embedding_dim), optional
            title: Plot title
            class_names: List of class names for legend, optional
            episode_num: Episode number to include in title, optional
        
        Returns:
            Matplotlib figure object (not displayed, just returned)
        """
        # Prepare data
        data = self._prepare_data(
            support_embeddings,
            query_embeddings,
            support_labels,
            query_labels,
            prototypes
        )
        
        # Apply dimensionality reduction
        reduced = self._reduce_embeddings(
            data['support_embeddings'],
            data['query_embeddings'],
            data.get('prototypes')
        )
        
        # Get unique labels (includes both positive and negative)
        unique_labels = np.unique(np.concatenate([
            data['support_labels'],
            data['query_labels']
        ]))
        
        # Map labels to class indices (positive and negative share same class)
        # Label encoding: class_idx*2 = positive, class_idx*2+1 = negative
        class_indices = unique_labels // 2  # Integer division to get class index
        unique_classes = np.unique(class_indices)
        n_classes = len(unique_classes)
        
        # Create figure
        try:
            plt.style.use(self.style)
        except:
            logger.warning(f"Style '{self.style}' not available, using default")
        
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Plot each label (both positive and negative)
        legend_elements = []
        plotted_classes = set()  # Track which classes we've added to legend
        
        for label_idx, label in enumerate(unique_labels):
            # Determine class index and whether this is positive or negative
            class_idx = int(label // 2)
            is_positive = (label % 2 == 0)
            
            # Use same color for positive and negative of same class
            color = self.colors[class_idx % len(self.colors)]
            
            # Set opacity based on positive/negative
            alpha_support = 0.6 if is_positive else 0.25
            alpha_query = 0.6 if is_positive else 0.25
            alpha_proto = 1.0 if is_positive else 0.25
            
            # Set marker style
            marker_support = 'o'  # Circle for support
            marker_query = 's'    # Square for query
            marker_proto = '*'    # Star for prototype
            
            # Get class name with _neg suffix for negative samples
            if class_names and class_idx < len(class_names):
                base_class_name = class_names[class_idx]
                display_name = base_class_name if is_positive else f"{base_class_name}_neg"
            else:
                display_name = f'Class {class_idx}' if is_positive else f'Class {class_idx}_neg'
            
            # Plot support samples (no label to avoid cluttering legend)
            support_mask = data['support_labels'] == label
            if np.any(support_mask):
                ax.scatter(
                    reduced['support_embeddings'][support_mask, 0],
                    reduced['support_embeddings'][support_mask, 1],
                    c=color,
                    marker=marker_support,
                    s=100,
                    alpha=alpha_support,
                    edgecolors='black',
                    linewidths=1.5
                )
            
            # Plot query samples (no label to avoid cluttering legend)
            query_mask = data['query_labels'] == label
            if np.any(query_mask):
                ax.scatter(
                    reduced['query_embeddings'][query_mask, 0],
                    reduced['query_embeddings'][query_mask, 1],
                    c=color,
                    marker=marker_query,
                    s=100,
                    alpha=alpha_query,
                    edgecolors='black',
                    linewidths=1.5
                )
            
            # Plot prototype (no label to avoid cluttering legend)
            if 'prototypes' in reduced and label_idx < reduced['prototypes'].shape[0]:
                ax.scatter(
                    reduced['prototypes'][label_idx, 0],
                    reduced['prototypes'][label_idx, 1],
                    c=color,
                    marker=marker_proto,
                    s=500,
                    alpha=alpha_proto,
                    edgecolors='black',
                    linewidths=2
                )
            
            # Add to legend (one entry per label - both positive and negative)
            legend_elements.append(mpatches.Patch(color=color, label=display_name, alpha=alpha_support))
        
        # Customize plot
        if episode_num is not None:
            title = f"{title} (Episode {episode_num})"
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel(f'{self.reduction_method.upper()} Component 1', fontsize=12)
        ax.set_ylabel(f'{self.reduction_method.upper()} Component 2', fontsize=12)
        
        # Add main legend (class colors with names)
        main_legend = ax.legend(handles=legend_elements, loc='upper left',
                               fontsize=9, title='Classes', ncol=2)
        ax.add_artist(main_legend)
        
        # Add secondary legend (marker shapes)
        from matplotlib.lines import Line2D
        marker_legend_elements = [
            Line2D([0], [0], marker='o', color='w',
                   markerfacecolor='gray', alpha=0.7,
                   markersize=8, markeredgecolor='black', markeredgewidth=1.5,
                   label='Support', linestyle='None'),
            Line2D([0], [0], marker='s', color='w',
                   markerfacecolor='gray', alpha=0.7,
                   markersize=8, markeredgecolor='black', markeredgewidth=1.5,
                   label='Query', linestyle='None'),
            Line2D([0], [0], marker='*', color='w',
                   markerfacecolor='gray', alpha=0.7,
                   markersize=12, markeredgecolor='black', markeredgewidth=1.5,
                   label='Prototype', linestyle='None')
        ]
        ax.legend(handles=marker_legend_elements, loc='upper right',
                 fontsize=9, title='Sample Types')
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Tight layout
        plt.tight_layout()
        
        # Clean up data
        del data, reduced
        
        return fig
    
    def save_visualization(
        self,
        fig: plt.Figure,
        filepath: Union[str, Path],
        close_after_save: bool = True
    ):
        """
        Save visualization to disk.
        
        Args:
            fig: Matplotlib figure to save
            filepath: Output file path
            close_after_save: Whether to close figure after saving (default: True)
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving visualization to {filepath}")
        fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        
        if close_after_save:
            plt.close(fig)
    
    def log_to_wandb(
        self,
        fig: plt.Figure,
        key: str = "prototype_visualization",
        step: Optional[int] = None,
        close_after_log: bool = True
    ):
        """
        Log visualization to Weights & Biases.
        
        Args:
            fig: Matplotlib figure to log
            key: WandB log key
            step: Training step number, optional
            close_after_log: Whether to close figure after logging (default: True)
        """
        try:
            import wandb
            
            if wandb.run is not None:
                wandb.log({key: wandb.Image(fig)}, step=step)
                logger.info(f"Logged visualization to WandB with key '{key}'")
            else:
                logger.warning("WandB run not initialized, skipping logging")
        except ImportError:
            logger.warning("WandB not available, skipping logging")
        except Exception as e:
            logger.error(f"Error logging to WandB: {e}")
        finally:
            if close_after_log:
                plt.close(fig)
    
    def log_to_tensorboard(
        self,
        fig: plt.Figure,
        writer: Any,
        tag: str = "prototype_visualization",
        global_step: Optional[int] = None,
        close_after_log: bool = True
    ):
        """
        Log visualization to TensorBoard.
        
        Args:
            fig: Matplotlib figure to log
            writer: TensorBoard SummaryWriter instance
            tag: TensorBoard tag
            global_step: Training step number, optional
            close_after_log: Whether to close figure after logging (default: True)
        """
        try:
            # Convert figure to numpy array
            fig.canvas.draw()
            img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            
            # Log to tensorboard (HWC -> CHW format)
            img_array = np.transpose(img_array, (2, 0, 1))
            writer.add_image(tag, img_array, global_step=global_step)
            
            logger.info(f"Logged visualization to TensorBoard with tag '{tag}'")
        except Exception as e:
            logger.error(f"Error logging to TensorBoard: {e}")
        finally:
            if close_after_log:
                plt.close(fig)
    
    def reset_reducer(self):
        """Reset the cached dimensionality reducer."""
        if self.reducer is not None:
            self.reducer.reset()
            logger.debug("Reducer cache reset")
