"""
Custom Loss Functions for Imbalanced and Positive-Unlabeled Learning

This module provides loss functions suitable for:
1. Highly imbalanced datasets (e.g., 1:100 positive:negative ratio)
2. Positive-Unlabeled (PU) learning where negatives are not confirmed negatives
3. Confidence-weighted losses based on sample quality

LOSS FUNCTION COMPARISON:
========================

1. **standard_bce** (ORIGINAL - The previous loss):
   - Standard Binary Cross-Entropy with Logits
   - Treats all samples equally
   - Good for: Balanced datasets, baseline comparisons
   - Use when: You trust your negatives and have ~1:1 ratio

2. **weighted_bce** (RECOMMENDED for imbalance):
   - Automatically weights positive samples higher based on class ratio
   - Example: For 1:10 ratio, positives get 10x weight
   - Good for: Imbalanced data, easy to use
   - Use when: You have confirmed negatives but imbalanced classes

3. **confidence_weighted** (BEST for uncertain negatives):
   - Weights negatives by confidence (based on common neighbors)
   - Hard negatives (high CN) = lower weight (might be hidden positives)
   - Easy negatives (zero CN) = higher weight (likely true negatives)
   - Good for: When negatives might contain hidden positives
   - Use when: You suspect some negatives are actually unlabeled positives

4. **focal** (BEST for very hard examples):
   - Down-weights easy examples, focuses on hard misclassifications
   - gamma parameter controls focus (higher = more focus on hard)
   - Good for: When model struggles with specific hard cases
   - Use when: Standard losses plateau, hard examples dominate errors

5. **pu** (THEORETICAL - Positive-Unlabeled Learning):
   - Explicitly treats negatives as unlabeled
   - Uses prior probability to estimate hidden positives
   - Good for: When you believe many negatives are mislabeled
   - Use when: You have strong evidence of hidden positives in negatives

6. **balanced_focal** (COMBINATION):
   - Combines class weighting + focal loss
   - Handles both imbalance AND hard examples
   - Good for: Extreme imbalance with varying difficulty
   - Use when: You need both class balancing and hard example focus

RECOMMENDATION HIERARCHY:
1. Start with 'standard_bce' (original) as baseline
2. Try 'weighted_bce' for imbalance (most common upgrade)
3. Try 'confidence_weighted' if analysis shows negatives have many common neighbors
4. Try 'focal' if model accuracy plateaus
5. Try 'pu' only if you have evidence of mislabeled negatives

"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class StandardBCEWithLogitsLoss(nn.Module):
    
    def __init__(self):
        """Initialise standard BCE loss."""
        super().__init__()
        self.loss_fn = nn.BCEWithLogitsLoss()
    
    def forward(self, logits, targets, **kwargs):
        """
        Compute standard BCE loss.
        
        Args:
            logits: Model predictions (before sigmoid)
            targets: Ground truth labels (0 or 1)
            **kwargs: Ignored (for compatibility)
            
        Returns:
            Scalar loss value
        """
        return self.loss_fn(logits, targets.float())


class WeightedBCEWithLogitsLoss(nn.Module):
    
    def __init__(self, pos_weight: float = None, neg_weight: float = 1.0):
        """
        Initialise weighted BCE loss.
        
        Args:
            pos_weight: Weight for positive class. If None, auto-computed from batch
            neg_weight: Weight for negative class
        """
        super().__init__()
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight
    
    def forward(self, logits, targets, **kwargs):
        """
        Compute weighted BCE loss.
        
        Args:
            logits: Model predictions (before sigmoid)
            targets: Ground truth labels (0 or 1)
            **kwargs: Ignored (for compatibility)
            
        Returns:
            Scalar loss value
        """
        # Auto-compute pos_weight if not provided
        if self.pos_weight is None:
            n_pos = (targets == 1).sum().float()
            n_neg = (targets == 0).sum().float()
            pos_weight = n_neg / (n_pos + 1e-8)
        else:
            pos_weight = self.pos_weight
        
        # Create per-sample weights (avoid torch.tensor warning by using float directly)
        weights = torch.where(
            targets == 1,
            torch.full_like(targets, pos_weight, dtype=torch.float32),
            torch.full_like(targets, self.neg_weight, dtype=torch.float32)
        )
        
        # Compute BCE loss
        loss = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction='none')
        weighted_loss = (loss * weights).mean()
        
        return weighted_loss


class ConfidenceWeightedBCELoss(nn.Module):
    
    def __init__(self, 
                 pos_weight: float = 10.0,
                 min_neg_weight: float = 0.1,
                 max_neg_weight: float = 1.0):
        """
        Initialise confidence-weighted loss.
        
        Args:
            pos_weight: Weight for positive samples
            min_neg_weight: Minimum weight for hard negatives (high CN)
            max_neg_weight: Maximum weight for easy negatives (low CN)
        """
        super().__init__()
        self.pos_weight = pos_weight
        self.min_neg_weight = min_neg_weight
        self.max_neg_weight = max_neg_weight
    
    def forward(self, logits, targets, sample_confidences=None, **kwargs):
        """
        Compute confidence-weighted BCE loss.
        
        Args:
            logits: Model predictions (before sigmoid)
            targets: Ground truth labels (0 or 1)
            sample_confidences: Per-sample confidence scores in [0, 1]
                High confidence = more certain it's a true negative
                Low confidence = might be unlabeled positive
            **kwargs: Ignored (for compatibility)
            
        Returns:
            Scalar loss value
        """
        # If no confidences provided, use standard weighting
        if sample_confidences is None:
            weights = torch.where(targets == 1,
                                 torch.tensor(self.pos_weight, device=targets.device),
                                 torch.tensor(1.0, device=targets.device))
        else:
            # Scale negative weights by confidence
            # High confidence (true negative) → max_neg_weight
            # Low confidence (might be positive) → min_neg_weight
            neg_weights = self.min_neg_weight + (self.max_neg_weight - self.min_neg_weight) * sample_confidences
            
            weights = torch.where(targets == 1,
                                 torch.tensor(self.pos_weight, device=targets.device),
                                 neg_weights)
        
        # Compute BCE loss
        loss = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction='none')
        weighted_loss = (loss * weights).mean()
        
        return weighted_loss


class FocalLoss(nn.Module):
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        """
        Initialise focal loss.
        
        Args:
            alpha: Weighting factor for positive class in [0, 1]
            gamma: Focusing parameter. Higher gamma = more focus on hard examples
                   gamma=0 reduces to standard CE
                   gamma=2 is typical
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, logits, targets, **kwargs):
        """
        Compute focal loss.
        
        Args:
            logits: Model predictions (before sigmoid)
            targets: Ground truth labels (0 or 1)
            **kwargs: Ignored (for compatibility)
            
        Returns:
            Scalar loss value
        """
        # Compute probabilities
        probs = torch.sigmoid(logits)
        
        # Compute focal weight: (1 - p_t)^gamma
        # For positive samples: (1 - prob)^gamma
        # For negative samples: prob^gamma
        targets_float = targets.float()
        p_t = probs * targets_float + (1 - probs) * (1 - targets_float)
        focal_weight = (1 - p_t) ** self.gamma
        
        # Compute BCE
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets_float, reduction='none')
        
        # Apply focal weight and alpha balancing
        alpha_t = self.alpha * targets_float + (1 - self.alpha) * (1 - targets_float)
        focal_loss = alpha_t * focal_weight * bce_loss
        
        return focal_loss.mean()


class PULoss(nn.Module):
    
    def __init__(self, prior: float = 0.1, beta: float = 0.0):
        """
        Initialise PU loss.
        
        Args:
            prior: Estimated proportion of positives in unlabeled set
                   0.1 = assume 10% of "negatives" are actually positives
            beta: Regularisation parameter for non-negative risk estimator
        """
        super().__init__()
        self.prior = prior
        self.beta = beta
    
    def forward(self, logits, targets, **kwargs):
        """
        Compute PU loss.
        
        Args:
            logits: Model predictions (before sigmoid)
            targets: Ground truth labels (1 = positive, 0 = unlabeled)
            **kwargs: Ignored (for compatibility)
            
        Returns:
            Scalar loss value
        """
        # Separate positive and unlabeled
        pos_mask = targets == 1
        unlabeled_mask = targets == 0
        
        # Compute losses for positive samples (standard BCE)
        if pos_mask.sum() > 0:
            pos_logits = logits[pos_mask]
            pos_loss = F.binary_cross_entropy_with_logits(
                pos_logits, 
                torch.ones_like(pos_logits),
                reduction='mean'
            )
        else:
            pos_loss = torch.tensor(0.0, device=logits.device)
        
        # Compute losses for unlabeled samples
        if unlabeled_mask.sum() > 0:
            unlabeled_logits = logits[unlabeled_mask]
            
            # Risk from treating unlabeled as negative
            unlabeled_neg_loss = F.binary_cross_entropy_with_logits(
                unlabeled_logits,
                torch.zeros_like(unlabeled_logits),
                reduction='mean'
            )
            
            # Subtract expected risk from hidden positives in unlabeled set
            # prior * E[loss if unlabeled were positive]
            unlabeled_pos_loss = F.binary_cross_entropy_with_logits(
                unlabeled_logits,
                torch.ones_like(unlabeled_logits),
                reduction='mean'
            )
            
            # PU risk estimator
            unlabeled_loss = unlabeled_neg_loss - self.prior * unlabeled_pos_loss
            
            # Non-negative risk (add beta for stability)
            if unlabeled_loss < -self.beta:
                unlabeled_loss = -self.beta
        else:
            unlabeled_loss = torch.tensor(0.0, device=logits.device)
        
        # Combine positive and unlabeled risks
        total_loss = self.prior * pos_loss + unlabeled_loss
        
        return total_loss


class BalancedFocalLoss(nn.Module):
    
    def __init__(self, 
                 alpha: float = None,
                 gamma: float = 2.0,
                 pos_weight: float = None):
        """
        Initialise balanced focal loss.
        
        Args:
            alpha: Class balancing weight. If None, auto-computed
            gamma: Focal focusing parameter
            pos_weight: Explicit positive class weight
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
    
    def forward(self, logits, targets, **kwargs):
        """
        Compute balanced focal loss.
        
        Args:
            logits: Model predictions (before sigmoid)
            targets: Ground truth labels (0 or 1)
            **kwargs: Ignored (for compatibility)
            
        Returns:
            Scalar loss value
        """
        # Auto-compute alpha if not provided
        if self.alpha is None:
            n_pos = (targets == 1).sum().float()
            n_neg = (targets == 0).sum().float()
            alpha = n_neg / (n_pos + n_neg + 1e-8)
        else:
            alpha = self.alpha
        
        # Compute probabilities
        probs = torch.sigmoid(logits)
        targets_float = targets.float()
        
        # Focal weight
        p_t = probs * targets_float + (1 - probs) * (1 - targets_float)
        focal_weight = (1 - p_t) ** self.gamma
        
        # Class balancing weight
        alpha_t = alpha * targets_float + (1 - alpha) * (1 - targets_float)
        
        # Positive class weight
        if self.pos_weight is not None:
            pos_weight_t = self.pos_weight * targets_float + 1.0 * (1 - targets_float)
        else:
            pos_weight_t = 1.0
        
        # BCE
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets_float, reduction='none')
        
        # Combine all weights
        loss = alpha_t * focal_weight * pos_weight_t * bce_loss
        
        return loss.mean()


def get_loss_function(loss_type: str = 'standard_bce', **kwargs):
    """
    Factory function to get a loss function.
    
    Args:
        loss_type: Type of loss function
            - 'standard_bce': Standard BCE (ORIGINAL - previous loss)
            - 'bce': Alias for standard_bce
            - 'weighted_bce': Class-weighted BCE (RECOMMENDED for imbalance)
            - 'confidence_weighted': Confidence-weighted BCE (for uncertain negatives)
            - 'focal': Focal loss (for hard examples)
            - 'pu': Positive-Unlabeled learning loss (theoretical)
            - 'balanced_focal': Balanced focal loss (combination)
        **kwargs: Additional arguments for the loss function
        
    Returns:
        Loss function module
        
    """
    
    # Define which parameters each loss function accepts
    loss_param_mapping = {
        'standard_bce': [],
        'bce': [],
        'weighted_bce': ['pos_weight', 'neg_weight'],
        'confidence_weighted': ['pos_weight', 'min_neg_weight', 'max_neg_weight'],
        'focal': ['alpha', 'gamma'],
        'pu': ['prior', 'beta'],
        'balanced_focal': ['alpha', 'gamma', 'pos_weight'],
    }
    
    if loss_type not in loss_param_mapping:
        raise ValueError(f"Unknown loss type: {loss_type}. Choose from {list(loss_param_mapping.keys())}")
    
    # Filter kwargs to only include relevant parameters for this loss function
    valid_params = loss_param_mapping[loss_type]
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params and v is not None}
    
    loss_functions = {
        'standard_bce': lambda: StandardBCEWithLogitsLoss(),
        'bce': lambda: StandardBCEWithLogitsLoss(),  # Alias
        'weighted_bce': lambda: WeightedBCEWithLogitsLoss(**filtered_kwargs),
        'confidence_weighted': lambda: ConfidenceWeightedBCELoss(**filtered_kwargs),
        'focal': lambda: FocalLoss(**filtered_kwargs),
        'pu': lambda: PULoss(**filtered_kwargs),
        'balanced_focal': lambda: BalancedFocalLoss(**filtered_kwargs),
    }
    
    return loss_functions[loss_type]()
