"""
Model components for DCASE 2024 Task 5.

This module contains various neural network components used in the baseline models.
"""

from .audio_cife import AudioCIFE
from .protonet import ProtoNet
from .simple_dense_net import SimpleDenseNet

__all__ = [
    "AudioCIFE",
    "ProtoNet",
    "SimpleDenseNet",
]
