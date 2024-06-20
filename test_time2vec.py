import torch
import torch.nn as nn
import torch.nn.functional as F

class Time2Vec(nn.Module):
    def __init__(self, features, k):
        """
        Initialize the Time2Vec layer.
        
        Args:
        - features: int, the number of features in the input tensor.
        - k: int, the number of periodic functions to use.
        """
        super(Time2Vec, self).__init__()
        self.features = features
        self.k = k
        self.linear = nn.Parameter(torch.randn(features))  # Linear component weights
        self.periodic = nn.ParameterList([nn.Parameter(torch.randn(features)) for _ in range(k)])  # Periodic components weights

    def forward(self, x):
        """
        Forward pass for the Time2Vec layer.
        
        Args:
        - x: Tensor of shape (batch_size, seq_len, features)
        
        Returns:
        - Tensor of shape (batch_size, seq_len, features * (1 + k))
        """
        batch_size, seq_len, features = x.size()
        
        # Linear component
        linear_out = x * self.linear  # Shape: (batch_size, seq_len, features)
        
        # Periodic components
        periodic_outs = [linear_out]
        for i in range(self.k):
            periodic_out = torch.sin(x * self.periodic[i])  # Shape: (batch_size, seq_len, features)
            periodic_outs.append(periodic_out)
        
        # Concatenate along the feature dimension
        time2vec_out = torch.cat(periodic_outs, dim=-1)  # Shape: (batch_size, seq_len, features * (1 + k))
        
        return time2vec_out

# Example usage
batch_size = 32
seq_len = 10
features = 16
k = 8

time2vec = Time2Vec(features, k)
x = torch.randn(batch_size, seq_len, features)
output = time2vec(x)

print(output.shape)  # Output shape will be (batch_size, seq_len, features * (1 + k))
