"""
Shared model definitions for drug-disease prediction.
All GNN model architectures used across the pipeline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, TransformerConv, SAGEConv, GATConv


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

    def forward(self, x, edge_index, edge_attr=None):
        """
        Forward pass with optional edge attributes.
        
        Args:
            x: Node features
            edge_index: Edge connectivity
            edge_attr: Optional edge features (currently not used by GCNConv, but kept for API consistency)
        """
        # Ensure input tensor is float32
        x = x.float()
        
        # Note: GCNConv doesn't natively support edge_attr, but we keep the parameter for API consistency across models. 
        
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
    """Graph Transformer model with edge feature support."""
    
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout_rate=0.5, heads=4, concat=False, edge_dim=None):
        super(TransformerModel, self).__init__()
        self.num_layers = num_layers
        self.heads = heads
        self.concat = concat

        # Calculate the actual output dimension after multi-head attention
        # If concat=True, output is hidden_channels * heads, otherwise it's hidden_channels
        head_out_channels = hidden_channels * heads if concat else hidden_channels

        # Initial TransformerConv layer with edge feature support
        self.conv1 = TransformerConv(in_channels, hidden_channels, heads=heads, concat=concat, edge_dim=edge_dim)

        # Additional TransformerConv layers
        self.conv_list = torch.nn.ModuleList(
            [TransformerConv(head_out_channels, hidden_channels, heads=heads, concat=concat, edge_dim=edge_dim) for _ in range(num_layers - 1)]
        )

        # Layer normalization and dropout
        self.ln = torch.nn.LayerNorm(head_out_channels)
        self.dropout = torch.nn.Dropout(p=dropout_rate)

        # Final output layer
        self.final_layer = torch.nn.Linear(head_out_channels, out_channels)

    def forward(self, x, edge_index, edge_attr=None):
        """
        Forward pass with optional edge attributes.
        
        Args:
            x: Node features
            edge_index: Edge connectivity
            edge_attr: Optional edge features (supported by TransformerConv)
        """
        # Ensure input tensor is float32
        x = x.float()
        
        # First TransformerConv layer (with edge features if available)
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = self.ln(x)
        x = F.relu(x)
        x = self.dropout(x)

        # Additional TransformerConv layers
        for k in range(self.num_layers - 1):
            x = self.conv_list[k](x, edge_index, edge_attr=edge_attr)
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

    def forward(self, x, edge_index, edge_attr=None):
        """
        Forward pass with optional edge attributes.
        
        Args:
            x: Node features
            edge_index: Edge connectivity
            edge_attr: Optional edge features (currently not used by SAGEConv, but kept for API consistency)
        """
        # Ensure input tensor is float32
        x = x.float()
        
        # Note: SAGEConv doesn't natively support edge_attr, but we keep the parameter for API consistency across models
        
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


class GATModel(torch.nn.Module):
    """Graph Attention Network model with edge feature support."""
    
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout_rate=0.5, heads=4, concat=False, edge_dim=None):
        super(GATModel, self).__init__()
        self.num_layers = num_layers
        self.heads = heads
        self.concat = concat

        # Calculate the actual output dimension after multi-head attention
        head_out_channels = hidden_channels * heads if concat else hidden_channels

        # Initial GAT layer with edge feature support
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, concat=concat, edge_dim=edge_dim, dropout=dropout_rate)

        # Additional GAT layers
        self.conv_list = torch.nn.ModuleList(
            [GATConv(head_out_channels, hidden_channels, heads=heads, concat=concat, edge_dim=edge_dim, dropout=dropout_rate) for _ in range(num_layers - 1)]
        )

        # Layer normalization and dropout
        self.ln = torch.nn.LayerNorm(head_out_channels)
        self.dropout = torch.nn.Dropout(p=dropout_rate)

        # Final output layer
        self.final_layer = torch.nn.Linear(head_out_channels, out_channels)

    def forward(self, x, edge_index, edge_attr=None):
        """
        Forward pass with optional edge attributes.
        
        Args:
            x: Node features
            edge_index: Edge connectivity
            edge_attr: Optional edge features (supported by GATConv)
        """
        # Ensure input tensor is float32
        x = x.float()
        
        # First GAT layer (with edge features if available)
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = self.ln(x)
        x = F.relu(x)
        x = self.dropout(x)

        # Additional GAT layers
        for k in range(self.num_layers - 1):
            x = self.conv_list[k](x, edge_index, edge_attr=edge_attr)
            x = self.ln(x)
            if k < self.num_layers - 2:  # Apply activation and dropout except on the last hidden layer
                x = F.relu(x)
                x = self.dropout(x)

        # Final layer to produce output
        x = self.final_layer(x)
        return x



class LinkPredictor(nn.Module):
    """
    Link predictor with neighborhood-aware decoder options.
    
    Decoder Types:
    - 'dot': Simple dot product (baseline)
    - 'mlp': MLP on [src, dst] concatenation  
    - 'mlp_interaction': MLP with interaction features [src, dst, src*dst, |src-dst|]
    - 'mlp_neighbor': Full version with [src, dst, src*dst, |src-dst|, neighbor_features]
    
    The interaction features help capture neighborhood similarity like heuristics.
    """
    def __init__(self, encoder: nn.Module, hidden_channels: int = 128, 
                 decoder_type: str = 'mlp_interaction', temperature: float = 5.0,
                 num_neighbor_features: int = 3):
        super().__init__()
        self.encoder = encoder
        self.decoder_type = decoder_type
        self.temperature = temperature
        self.hidden_channels = hidden_channels
        
        # Add support for 'mlp_heuristic' as an alias for 'mlp_neighbor' for scripts compatibility
        if decoder_type == 'mlp_heuristic':
            self.decoder_type = 'mlp_neighbor'
            decoder_type = 'mlp_neighbor'

        if decoder_type == 'mlp_interaction':
            # Interaction decoder: [src, dst, src*dst, |src-dst|] -> score
            # Input dimension: 4 * hidden_channels
            input_dim = hidden_channels * 4
            self.decoder = nn.Sequential(
                nn.Linear(input_dim, hidden_channels),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_channels, hidden_channels // 2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_channels // 2, 1)
            )
        elif decoder_type == 'mlp_neighbor':
            # Full decoder with explicit neighbor features
            # [src, dst, src*dst, |src-dst|, neighbor_features (CN, AA, Jaccard)]
            input_dim = hidden_channels * 4 + num_neighbor_features
            self.decoder = nn.Sequential(
                nn.Linear(input_dim, hidden_channels),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_channels, hidden_channels // 2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_channels // 2, 1)
            )
        elif decoder_type == 'mlp':
            # Standard MLP on concatenation
            input_dim = hidden_channels * 2
            self.decoder = nn.Sequential(
                nn.Linear(input_dim, hidden_channels),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_channels, 1)
            )
        else:
            # Dot product decoder
            self.decoder = None

    def forward(self, x, edge_index, edge_attr=None):
        """Forward pass through encoder."""
        return self.encoder(x, edge_index, edge_attr)

    def encode(self, x, edge_index, edge_attr=None):
        """Encode nodes using the GNN encoder."""
        return self.encoder(x, edge_index, edge_attr)

    def decode(self, z: torch.Tensor, edge_index: torch.Tensor, 
               heuristic_features: torch.Tensor = None) -> torch.Tensor:
        """
        Predict edge scores given node embeddings and optional heuristics.
        
        Args:
            z: Node embeddings [num_nodes, hidden_channels]
            edge_index: Edges to predict [2, num_edges]
            heuristic_features: [num_edges, K] with neighbor features (for mlp_neighbor)
            
        Returns:
            scores: [num_edges] raw logits
        """
        src = z[edge_index[0]]
        dst = z[edge_index[1]]
        
        if self.decoder_type == 'mlp_interaction':
            # Use interaction features to capture neighborhood similarity
            interaction = src * dst  # Element-wise product
            difference = torch.abs(src - dst)  # Absolute difference
            edge_repr = torch.cat([src, dst, interaction, difference], dim=-1)
            score = self.decoder(edge_repr).squeeze(-1)
            
        elif self.decoder_type == 'mlp_neighbor':
            # Full features with explicit neighbor counts
            interaction = src * dst
            difference = torch.abs(src - dst)
            base_repr = torch.cat([src, dst, interaction, difference], dim=-1)
            
            if heuristic_features is not None:
                h_feats = heuristic_features.to(src.device).float()
                edge_repr = torch.cat([base_repr, h_feats], dim=-1)
            else:
                # Fallback: pad with zeros if no heuristics provided
                zeros = torch.zeros(src.size(0), 3, device=src.device)
                edge_repr = torch.cat([base_repr, zeros], dim=-1)
            score = self.decoder(edge_repr).squeeze(-1)
            
        elif self.decoder_type == 'mlp':
            edge_repr = torch.cat([src, dst], dim=-1)
            score = self.decoder(edge_repr).squeeze(-1)
            
        else:
            # Dot product decoder
            score = (src * dst).sum(dim=-1)
            
        return score
    
    def predict_proba(self, z: torch.Tensor, edge_index: torch.Tensor, 
                      heuristic_features: torch.Tensor = None) -> torch.Tensor:
        """Get calibrated probabilities using temperature scaling."""
        raw_scores = self.decode(z, edge_index, heuristic_features)
        return torch.sigmoid(raw_scores / self.temperature)


# Dictionary for easy model selection
MODEL_CLASSES = {
    'GCN': GCNModel,
    'GCNModel': GCNModel,
    'Transformer': TransformerModel,
    'TransformerModel': TransformerModel,
    'SAGE': SAGEModel,
    'SAGEModel': SAGEModel,
    'GAT': GATModel,
    'GATModel': GATModel,
}
