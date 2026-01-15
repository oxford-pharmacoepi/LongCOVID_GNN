"""
Shared model definitions for drug-disease prediction.
All GNN model architectures used across the pipeline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, TransformerConv, SAGEConv


class GCNModel(torch.nn.Module):
    """Graph Convolutional Network model."""
    
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout_rate=0.5):
        super(GCNModel, self).__init__()
        self.num_layers = num_layers

        # Initial GCNConv layer
        self.conv1 = GCNConv(in_channels, hidden_channels)

        # Additional GCNConv layers
        self.conv_list = torch.nn.ModuleList(
            [GCNConv(hidden_channels, hidden_channels) for _ in range(num_layers - 1)]
        )

        # Layer normalization and dropout
        self.ln = torch.nn.LayerNorm(hidden_channels)
        self.dropout = torch.nn.Dropout(p=dropout_rate)

        # Final output layer
        self.final_layer = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        # Ensure input tensor is float32
        x = x.float()
        
        # First GCNConv layer
        x = self.conv1(x, edge_index)
        x = self.ln(x)
        x = F.relu(x)
        x = self.dropout(x)

        # Additional GCNConv layers
        for k in range(self.num_layers - 1):
            x = self.conv_list[k](x, edge_index)
            x = self.ln(x)
            if k < self.num_layers - 2:  # Apply activation and dropout except on the last hidden layer
                x = F.relu(x)
                x = self.dropout(x)

        # Final layer to produce output
        x = self.final_layer(x)
        return x


class TransformerModel(torch.nn.Module):
    """Graph Transformer model."""
    
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout_rate=0.5, heads=4, concat=False):
        super(TransformerModel, self).__init__()
        self.num_layers = num_layers
        self.heads = heads
        self.concat = concat

        # Calculate the actual output dimension after multi-head attention
        # If concat=True, output is hidden_channels * heads, otherwise it's hidden_channels
        head_out_channels = hidden_channels * heads if concat else hidden_channels

        # Initial TransformerConv layer
        self.conv1 = TransformerConv(in_channels, hidden_channels, heads=heads, concat=concat)

        # Additional TransformerConv layers
        self.conv_list = torch.nn.ModuleList(
            [TransformerConv(head_out_channels, hidden_channels, heads=heads, concat=concat) for _ in range(num_layers - 1)]
        )

        # Layer normalization and dropout
        self.ln = torch.nn.LayerNorm(head_out_channels)
        self.dropout = torch.nn.Dropout(p=dropout_rate)

        # Final output layer
        self.final_layer = torch.nn.Linear(head_out_channels, out_channels)

    def forward(self, x, edge_index):
        # Ensure input tensor is float32
        x = x.float()
        
        # First TransformerConv layer
        x = self.conv1(x, edge_index)
        x = self.ln(x)
        x = F.relu(x)
        x = self.dropout(x)

        # Additional TransformerConv layers
        for k in range(self.num_layers - 1):
            x = self.conv_list[k](x, edge_index)
            x = self.ln(x)
            if k < self.num_layers - 2:  # Apply activation and dropout except on the last hidden layer
                x = F.relu(x)
                x = self.dropout(x)

        # Final layer to produce output
        x = self.final_layer(x)
        return x


class SAGEModel(torch.nn.Module):
    """GraphSAGE model with improved stability."""
    
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout_rate=0.5):
        super(SAGEModel, self).__init__()
        self.num_layers = num_layers

        # Initial GraphSAGE layer
        self.conv1 = SAGEConv(in_channels, hidden_channels)

        # Additional hidden layers
        self.conv_list = torch.nn.ModuleList(
            [SAGEConv(hidden_channels, hidden_channels) for _ in range(num_layers - 1)]
        )

        # Batch normalisation instead of LayerNorm for better stability
        self.bn = torch.nn.BatchNorm1d(hidden_channels)
        self.dropout = torch.nn.Dropout(p=dropout_rate)

        # Final output layer with proper initialisation
        self.final_layer = torch.nn.Linear(hidden_channels, out_channels)
        
        # Initialize weights properly to prevent gradient explosion
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with smaller values for stability."""
        for conv in [self.conv1] + list(self.conv_list):
            if hasattr(conv, 'reset_parameters'):
                conv.reset_parameters()
        
        # Initialize final layer with small weights
        nn.init.xavier_uniform_(self.final_layer.weight, gain=0.1)
        nn.init.zeros_(self.final_layer.bias)

    def forward(self, x, edge_index):
        # Ensure input tensor is float32
        x = x.float()
        
        # First layer
        x = self.conv1(x, edge_index)
        x = self.bn(x)
        x = F.relu(x)
        x = self.dropout(x)

        # Additional layers
        for k in range(self.num_layers - 1):
            x = self.conv_list[k](x, edge_index)
            x = self.bn(x)
            if k < self.num_layers - 2:  # Apply activation and dropout except on the last hidden layer
                x = F.relu(x)
                x = self.dropout(x)

        # Final layer to produce output
        x = self.final_layer(x)
        return x


# Dictionary for easy model selection
MODEL_CLASSES = {
    'GCN': GCNModel,
    'GCNModel': GCNModel,
    'Transformer': TransformerModel,
    'TransformerModel': TransformerModel,
    'SAGE': SAGEModel,
    'SAGEModel': SAGEModel,
}
