"""
PyTorch Lightning Callback for Prototype Network Visualization.

This callback periodically visualizes the embedding space of prototypical networks
during training, showing support samples, query samples, and prototypes in 2D space.

Date: 2026-04-26
"""

from typing import Optional, Any
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pathlib import Path
import logging

from ..utils.prototype_visualizer import PrototypeVisualizer

logger = logging.getLogger(__name__)


class PrototypeVisualizationCallback(Callback):
    """
    PyTorch Lightning callback for visualizing prototypical network embeddings.
    
    This callback tracks episode count across training and periodically creates
    visualizations of the embedding space, showing how support samples, query
    samples, and prototypes are distributed in 2D space after dimensionality
    reduction.
    
    Args:
        viz_episode_freq: Frequency of visualization in episodes (default: 1000)
        reduction_method: Dimensionality reduction method ('pca', 'incremental_pca', 'umap')
        n_components: Number of dimensions for reduction (default: 2)
        save_to_disk: Whether to save figures locally (default: False)
        output_dir: Directory for saved figures (default: 'visualizations/')
        use_cache: Whether to cache the fitted reducer (default: True)
        figsize: Figure size as (width, height) tuple (default: (12, 8))
        dpi: Figure DPI for quality (default: 100)
        log_to_wandb: Whether to log to WandB if available (default: True)
        log_to_tensorboard: Whether to log to TensorBoard if available (default: True)
    
    Example:
        >>> callback = PrototypeVisualizationCallback(
        ...     viz_episode_freq=1000,
        ...     reduction_method='pca',
        ...     save_to_disk=True,
        ...     output_dir='visualizations/'
        ... )
        >>> trainer = pl.Trainer(callbacks=[callback])
    """
    
    def __init__(
        self,
        viz_episode_freq: int = 1000,
        reduction_method: str = 'pca',
        n_components: int = 2,
        save_to_disk: bool = False,
        output_dir: str = 'visualizations/',
        use_cache: bool = True,
        figsize: tuple = (12, 8),
        dpi: int = 100,
        log_to_wandb: bool = True,
        log_to_tensorboard: bool = True
    ):
        super().__init__()
        
        self.viz_episode_freq = viz_episode_freq
        self.reduction_method = reduction_method
        self.n_components = n_components
        self.save_to_disk = save_to_disk
        self.output_dir = Path(output_dir)
        self.use_cache = use_cache
        self.figsize = figsize
        self.dpi = dpi
        self.log_to_wandb = log_to_wandb
        self.log_to_tensorboard = log_to_tensorboard
        
        # Episode counter
        self.episode_count = 0
        
        # Initialize visualizer
        self.visualizer = PrototypeVisualizer(
            reduction_method=reduction_method,
            n_components=n_components,
            use_cache=use_cache,
            figsize=figsize,
            dpi=dpi
        )
        
        logger.info(
            f"Initialized PrototypeVisualizationCallback: "
            f"freq={viz_episode_freq}, method={reduction_method}, "
            f"save_to_disk={save_to_disk}"
        )
    
    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int
    ) -> None:
        """
        Called after each training batch.
        
        Increments episode counter and creates visualization if at the right frequency.
        """
        # Increment episode counter
        self.episode_count += 1
        
        # Check if we should visualize
        if self.episode_count % self.viz_episode_freq != 0:
            return
        
        logger.info(f"Creating visualization for episode {self.episode_count}")
        
        try:
            # Extract data from batch and outputs
            visualization_data = self._extract_visualization_data(
                batch, outputs, pl_module
            )
            
            if visualization_data is None:
                logger.warning("Could not extract visualization data, skipping")
                return
            
            # Create visualization
            fig = self.visualizer.visualize_embeddings(
                support_embeddings=visualization_data['support_embeddings'],
                query_embeddings=visualization_data['query_embeddings'],
                support_labels=visualization_data['support_labels'],
                query_labels=visualization_data['query_labels'],
                prototypes=visualization_data.get('prototypes'),
                title="Prototype Network Embedding Space",
                class_names=visualization_data.get('class_name'),
                episode_num=self.episode_count
            )
            
            # Save to disk if requested
            if self.save_to_disk:
                self.output_dir.mkdir(parents=True, exist_ok=True)
                filepath = self.output_dir / f"episode_{self.episode_count:06d}.png"
                self.visualizer.save_visualization(fig, filepath, close_after_save=False)
            
            # Log to WandB if available and requested
            if self.log_to_wandb and self._is_wandb_available(trainer):
                self.visualizer.log_to_wandb(
                    fig,
                    key="train/prototype_visualization",
                    step=self.episode_count,
                    close_after_log=False
                )
            
            # Log to TensorBoard if available and requested
            if self.log_to_tensorboard and self._is_tensorboard_available(trainer):
                writer = self._get_tensorboard_writer(trainer)
                if writer is not None:
                    self.visualizer.log_to_tensorboard(
                        fig,
                        writer=writer,
                        tag="train/prototype_visualization",
                        global_step=self.episode_count,
                        close_after_log=False
                    )
            
            # Clean up figure
            import matplotlib.pyplot as plt
            plt.close(fig)
            
            # Clean up GPU memory if needed
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info(f"Successfully created visualization for episode {self.episode_count}")
            
        except Exception as e:
            logger.error(f"Error creating visualization: {e}", exc_info=True)
    
    def _extract_visualization_data(
        self,
        batch: Any,
        outputs: Any,
        pl_module: pl.LightningModule
    ) -> Optional[dict]:
        """
        Extract embeddings and labels from batch and model outputs.
        
        This method attempts to extract the necessary data for visualization.
        It handles different batch and output formats.
        
        Args:
            batch: Training batch
            outputs: Model outputs
            pl_module: PyTorch Lightning module
        
        Returns:
            Dictionary with visualization data or None if extraction fails
        """
        try:
            # Extract pre-computed embeddings and labels from outputs
            if not isinstance(outputs, dict) or 'embeddings' not in outputs or 'labels' not in outputs:
                logger.warning("Outputs must contain 'embeddings' and 'labels' keys")
                return None
            
            embeddings = outputs['embeddings']  # Already computed - no forward pass!
            labels = outputs['labels']
            class_name = outputs.get('class_name', None)  # Get class name if available
            
            # Get n_support from model hyperparameters
            n_support = pl_module.hparams.train.n_shot
            
            # For each unique label, first n_support samples are support, rest are query
            unique_labels = torch.unique(labels)
            
            support_indices = []
            query_indices = []
            
            for label in unique_labels:
                # Find all indices for this label
                label_indices = (labels == label).nonzero(as_tuple=True)[0]
                
                # First n_support are support, rest are query
                if len(label_indices) > n_support:
                    support_indices.extend(label_indices[:n_support].tolist())
                    query_indices.extend(label_indices[n_support:].tolist())
                else:
                    # If fewer than n_support, treat all as support
                    support_indices.extend(label_indices.tolist())
            
            # Convert to tensors
            support_indices = torch.tensor(support_indices, dtype=torch.long)
            query_indices = torch.tensor(query_indices, dtype=torch.long)
            
            # Extract embeddings and labels
            support_embeddings = embeddings[support_indices]
            query_embeddings = embeddings[query_indices] if len(query_indices) > 0 else torch.empty(0, embeddings.shape[1])
            support_labels = labels[support_indices]
            query_labels = labels[query_indices] if len(query_indices) > 0 else torch.empty(0, dtype=labels.dtype)
            
            # Compute prototypes (mean of support embeddings per label)
            prototypes = []
            for label in unique_labels:
                mask = support_labels == label
                if mask.any():
                    proto = support_embeddings[mask].mean(dim=0)
                    prototypes.append(proto)
            
            if len(prototypes) > 0:
                prototypes = torch.stack(prototypes)
            else:
                prototypes = None
            
            return {
                'support_embeddings': support_embeddings,
                'query_embeddings': query_embeddings,
                'support_labels': support_labels,
                'query_labels': query_labels,
                'prototypes': prototypes,
                'class_name': class_name
            }
            
        except Exception as e:
            logger.error(f"Error extracting visualization data: {e}", exc_info=True)
            return None
    
    def _is_wandb_available(self, trainer: pl.Trainer) -> bool:
        """Check if WandB logger is available."""
        try:
            from pytorch_lightning.loggers import WandbLogger
            return any(isinstance(logger, WandbLogger) for logger in trainer.loggers)
        except ImportError:
            return False
    
    def _is_tensorboard_available(self, trainer: pl.Trainer) -> bool:
        """Check if TensorBoard logger is available."""
        try:
            from pytorch_lightning.loggers import TensorBoardLogger
            return any(isinstance(logger, TensorBoardLogger) for logger in trainer.loggers)
        except ImportError:
            return False
    
    def _get_tensorboard_writer(self, trainer: pl.Trainer) -> Optional[Any]:
        """Get TensorBoard SummaryWriter from trainer."""
        try:
            from pytorch_lightning.loggers import TensorBoardLogger
            for logger in trainer.loggers:
                if isinstance(logger, TensorBoardLogger):
                    return logger.experiment
        except Exception as e:
            logger.error(f"Error getting TensorBoard writer: {e}")
        return None
    
    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Called when training ends."""
        logger.info(
            f"Training ended. Total episodes visualized: "
            f"{self.episode_count // self.viz_episode_freq}"
        )
    
    def state_dict(self) -> dict:
        """Return callback state for checkpointing."""
        return {
            'episode_count': self.episode_count
        }
    
    def load_state_dict(self, state_dict: dict) -> None:
        """Load callback state from checkpoint."""
        self.episode_count = state_dict.get('episode_count', 0)
        logger.info(f"Loaded callback state: episode_count={self.episode_count}")
