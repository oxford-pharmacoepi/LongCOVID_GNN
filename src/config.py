"""
Shared configuration settings for drug-disease prediction pipeline.
All default parameters and paths in one place.
"""

import os
import platform


class Config:
    """Configuration class for the drug-disease prediction pipeline."""
    
    def __init__(self):
        # Version settings
        self.training_version = 21.06
        self.validation_version = 23.06
        self.test_version = 24.06
        
        # Dataset settings
        self.as_dataset = 'associationByOverallDirect'
        
        # Negative sampling configuration
        self.negative_sampling_strategy = 'hard'  # Options: 'random', 'hard', 'degree_matched', 'feature_similar', 'mixed'
        self.pos_neg_ratio = 10  # Ratio of negative to positive samples for val/test (1:x)
        self.train_neg_ratio = 10  # Ratio for training (1:x)
        self.neg_sampling_params = {
            'degree_tolerance': 0.3,         # For degree-matched sampling
            'similarity_threshold': 0.5,     # For feature-similarity sampling
            'strategy_weights': {            # For mixed strategy
                'hard': 0.6,
                'degree_matched': 0.3,
                'random': 0.1
            }
        }
        
        # Loss function configuration
        self.loss_function = 'standard_bce'  # Options: standard_bce (default), weighted_bce, focal, pu, confidence_weighted, balanced_focal
        self.loss_params = {
            'pos_weight': None,              # Auto-computed if None
            'neg_weight': 1.0,
            'alpha': 0.25,                   # For focal loss
            'gamma': 2.0,                    # For focal loss
            'prior': 0.01,                   # For PU learning (1% of negatives might be hidden positives)
            'min_neg_weight': 0.1,           # For confidence-weighted loss
            'max_neg_weight': 1.0            # For confidence-weighted loss
        }
        
        # Evaluation settings
        self.primary_metric = 'apr'  # "auc", "apr", "f1", "accuracy"
        
        # Model selection
        self.model_choice = 'Transformer'  # Options: 'all', 'Transformer', 'GCN', 'SAGE'

        # Model hyperparameters
        self.model_config = {
            'learning_rate': 0.0015456417318475437,
            'weight_decay': 2.7189746426884763e-06,
            'hidden_channels': 64,
            'out_channels': 32,
            'num_layers': 2,
            'dropout_rate': 0.4870800162873122,
            'num_epochs': 500,
            'patience': 10,
            'batch_size': 1024,
            'heads': 2,
            'concat': True 
        }
        
        # Training settings
        self.seed = 42
        self.device = 'cuda' if self._is_cuda_available() else 'cpu'
        
        # Path configuration
        self._setup_paths()
        
        # Network settings
        self.network_config = {
            'disease_similarity_network': False,
            'molecule_similarity_network': False,
            'reactome_network': True,
            'trial_edges': False
        }
        
        # Explainer settings
        self.explainer_config = {
            'epochs': 50,
            'sample_size': 1000,
            'max_explanations': 1000,
            'fp_threshold': 0.7,
            'fp_top_k': 10000
        }
    
    def _is_cuda_available(self):
        """Check if CUDA is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def _setup_paths(self):
        """Setup default paths based on operating system."""
        if platform.system() == "Windows":
            self.general_path = r"C:\OpenTargets_datasets\downloads"
            self.results_path = r"C:\OpenTargets_datasets\test_results"
        else:
            self.general_path = "raw_data"  # Fixed: was "data/raw/" 
            self.results_path = "results/"
        
        # Create directories
        os.makedirs(self.general_path, exist_ok=True)
        os.makedirs(self.results_path, exist_ok=True)
        
        # Set specific paths
        self.paths = {
            'indication': f"{self.general_path}/{self.training_version}/indication",
            'val_indication': f"{self.general_path}/{self.validation_version}/indication",
            'test_indication': f"{self.general_path}/{self.test_version}/indication",
            'molecule': f"{self.general_path}/{self.training_version}/molecule",
            'diseases': f"{self.general_path}/{self.training_version}/diseases",  # Fixed: diseases not disease
            'val_diseases': f"{self.general_path}/{self.validation_version}/diseases",
            'test_diseases': f"{self.general_path}/{self.test_version}/diseases", 
            'targets': f"{self.general_path}/{self.training_version}/targets",    # Fixed: targets not gene
            'associations': f"{self.general_path}/{self.training_version}/{self.as_dataset}",  # This matches associationByOverallDirect
            'results': self.results_path,
            'processed': "processed_data/",
            'models': f"{self.results_path}/models/",
            'predictions': f"{self.results_path}/predictions/",
            'explainer': f"{self.results_path}/explainer/"
        }
        
        # Create result subdirectories
        for path in [self.paths['processed'], self.paths['models'], 
                    self.paths['predictions'], self.paths['explainer']]:
            os.makedirs(path, exist_ok=True)
    
    def update_paths(self, **kwargs):
        """Update specific paths."""
        for key, value in kwargs.items():
            if key in self.paths:
                self.paths[key] = value
    
    def get_model_config(self):
        """Get model configuration dictionary."""
        return self.model_config.copy()
    
    def get_explainer_config(self):
        """Get explainer configuration dictionary."""
        return self.explainer_config.copy()
    
    def get_negative_sampling_config(self):
        """Get negative sampling configuration."""
        return {
            'strategy': self.negative_sampling_strategy,
            'pos_neg_ratio': self.pos_neg_ratio,
            'train_neg_ratio': self.train_neg_ratio,
            'params': self.neg_sampling_params.copy()
        }
    
    def get_loss_config(self):
        """Get loss function configuration."""
        return {
            'loss_function': self.loss_function,
            'params': self.loss_params.copy()
        }
    
    def get_all_paths(self):
        """Get all configured paths."""
        return self.paths.copy()


# Global configuration instance
default_config = Config()


def get_config():
    """Get the default configuration instance."""
    return default_config


def create_custom_config(**kwargs):
    """Create a custom configuration with overrides."""
    config = Config()
    
    # Update model config
    if 'model_config' in kwargs:
        config.model_config.update(kwargs['model_config'])
    
    # Update explainer config
    if 'explainer_config' in kwargs:
        config.explainer_config.update(kwargs['explainer_config'])
    
    # Update negative sampling config
    if 'neg_sampling_params' in kwargs:
        config.neg_sampling_params.update(kwargs['neg_sampling_params'])
    
    # Update loss config
    if 'loss_params' in kwargs:
        config.loss_params.update(kwargs['loss_params'])
    
    # Update paths
    if 'paths' in kwargs:
        config.update_paths(**kwargs['paths'])
    
    # Update other settings
    for key, value in kwargs.items():
        if key not in ['model_config', 'explainer_config', 'paths', 'neg_sampling_params', 'loss_params'] and hasattr(config, key):
            setattr(config, key, value)
    
    return config
