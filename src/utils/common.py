"""
Common utility functions for reproducibility and basic setup.
"""

import random
import numpy as np
import torch

def set_seed(seed=42):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def enable_full_reproducibility(seed=42):
    """Enable full reproducibility with deterministic algorithms."""
    set_seed(seed)
    torch.use_deterministic_algorithms(True)
