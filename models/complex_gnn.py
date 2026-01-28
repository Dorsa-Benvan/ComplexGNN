"""
Complex GNN model definition
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GraphConv, global_mean_pool, GraphNorm


class ComplexGNN(nn.Module):
    def __init__(self, in_dim, edge_dim, num_heads, hidden_dims, out_dim,
                 global_dim, dropout_rate=0.5, use_bn=False, activation='leaky_relu'):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.use_bn = use_bn
        
        # Global feature MLP
        self.global_mlp = nn.Sequential(
            nn.Linear(global_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Graph convolutional layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for i, h in enumerate(hidden_dims):
            if i == 0:
                self.convs.append(
                    GATConv(in_dim, h//num_heads, heads=num_heads, edge_dim=edge_dim, concat=True)
                )
            else:
                self.convs.append(GraphConv(hidden_dims[i-1], h))
            self.norms.append(GraphNorm(h))
        
        # Batch normalization layers if enabled
        if use_bn:
            self.bns = nn.ModuleList([nn.BatchNorm1d(h) for h in hidden_dims])
            self.pool_bn = nn.BatchNorm1d(hidden_dims[-1] + hidden_dims[0])
        
        # Final fully connected layer
        self.fc = nn.Linear(hidden_dims[-1] + hidden_dims[0], out_dim)
        
        # Activation function
        act_map = {'ReLU': F.relu, 'leaky_relu': F.leaky_relu}
        self.activation = act_map[activation]

    def forward(self, data, return_intermediates=False):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # Graph convolutional layers
        for i, conv in enumerate(self.convs):
            if isinstance(conv, GATConv):
                x = conv(x, edge_index, edge_attr)
            else:
                x = conv(x, edge_index)
            
            if self.use_bn:
                x = self.bns[i](x)
            
            x = self.norms[i](x)
            x = self.activation(x)
            x = self.dropout(x)
        
        # Global pooling
        pooled = global_mean_pool(x, data.batch)
        
        # Process global features
        gf = data.global_features.to(x.dtype).to(x.device)
        gf = gf.view(gf.size(0), -1)
        g = self.global_mlp(gf)
        
        # Combine graph and global features
        combined = torch.cat([pooled, g], dim=1)
        
        if self.use_bn:
            combined = self.pool_bn(combined)
        
        # Final output
        output = self.fc(combined)
        
        if return_intermediates:
            return output, pooled, g, combined
        
        return output