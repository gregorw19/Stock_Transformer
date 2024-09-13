import torch
import torch.nn as nn
import torch.nn.functional as F
import math

'''
# time2vec encoding
class Time2Vec(nn.Module):
    def __init__(self, num_features=7, kernel_size=1):
        super(Time2Vec, self).__init__()
        self.k = kernel_size

        # Initialize parameters for each feature
        self.wb = nn.Parameter(torch.randn(num_features))  # Shape: [num_features]
        self.bb = nn.Parameter(torch.randn(num_features))  # Shape: [num_features]
        self.wa = nn.Parameter(torch.randn(num_features, self.k))  # Shape: [num_features, kernel_size]
        self.ba = nn.Parameter(torch.randn(num_features, self.k))  # Shape: [num_features, kernel_size]

        # Uniform initialization with a small range around zero
        nn.init.uniform_(self.wb, -0.1, 0.1)
        nn.init.uniform_(self.bb, -0.1, 0.1)
        nn.init.uniform_(self.wa, -0.1, 0.1)
        nn.init.uniform_(self.ba, -0.1, 0.1)

    def forward(self, inputs):
        # inputs: [batch_size, seq_len, num_features]
        print("parameters:")
        print(self.wb.shape)
        print(self.bb.shape)
        print(self.wa.shape)
        print(self.ba.shape)
        # Ensure wa is expanded and transposed correctly
        # wa_expanded: [num_features, kernel_size] -> [num_features, 1, kernel_size]
        wa_expanded = self.wa.unsqueeze(1)  # [num_features, 1, kernel_size]

        # Perform matrix multiplication
        # inputs: [batch_size, seq_len, num_features]
        # wa_expanded: [num_features, 1, kernel_size] -> [1, num_features, kernel_size] for matmul
        wa_expanded_transposed = wa_expanded.permute(1, 2, 0)  # [1, kernel_size, num_features]
        
        print("Input shape: ", inputs.shape)
        print("wa_expanded_transposed shape: ", wa_expanded_transposed.shape)
        print("self.ba.unsqueeze(0).transpose(1,2) shape: ", self.ba.unsqueeze(0).transpose(1,2).shape)
        # Calculate dp
        dp = torch.matmul(inputs, wa_expanded_transposed) + self.ba.unsqueeze(0).transpose(1,2)  # [batch_size, seq_len, kernel_size]

        # Activation function
        wgts = torch.sin(dp)

        # Concatenate bias and wgts along the last dimension
        bias = self.wb * inputs + self.bb  # [batch_size, seq_len, num_features]
        ret = torch.cat([bias, wgts], dim=-1)  # Concatenate along the feature dimension

        return ret

'''

class Time2Vec(nn.Module):
    def __init__(self, features, k, output_dim):
        super(Time2Vec, self).__init__()
        self.features = features
        self.k = k
        self.output_dim = output_dim
        self.linear = nn.Parameter(torch.randn(features))  # Linear component weights
        self.periodic = nn.ParameterList([nn.Parameter(torch.randn(features)) for _ in range(k)])  # Periodic components weights

    def forward(self, x):
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

        # Ensure output_dim is achieved
        if time2vec_out.shape[-1] != self.output_dim:
            time2vec_out = time2vec_out[:, :, :self.output_dim]  # Adjust to match output_dim
        
        return time2vec_out



class LayerNormalization(nn.Module):
    
    def __init__(self, eps: float = 10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) # multiplied, and is parameter to make it learnable
        self.bias = nn.Parameter(torch.zeros(1)) # added

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim = True)
        stddev = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (stddev + self.eps) + self.bias
    

class FeedForwardBlock(nn.Module):

    def __init__(self, d_model: int, d_ff : int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # W1 and B1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # W2 and B2

    def forward(self, x):
        # (Batch, seq_len, d_model) --> (Batch, seq_len, d_ff) --> (Batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
    

class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model is not div by h"

        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model) # Wq
        self.w_k = nn.Linear(d_model, d_model) # Wk
        self.w_v = nn.Linear(d_model, d_model) # Wv

        self.w_o = nn.Linear(d_model, d_model) # Wo
        self.dropout = nn.Dropout(dropout)


    @staticmethod # can call without instance of this class, just do class name dot attention()
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]

        # (Batch, h, seq_len, d_k) -> (Batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim = -1) # (Batch, h, seq_len, seq_len)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        
        return (attention_scores @ value), attention_scores # tuple output is just for visualization


    def forward(self, q, k, v, mask):
        query = self.w_q(q) # (Batch, seq_len, d_model) -> (Batch, seq_len, d_model)
        key = self.w_k(k) # (Batch, seq_len, d_model) -> (Batch, seq_len, d_model)
        value = self.w_v(v) # (Batch, seq_len, d_model) -> (Batch, seq_len, d_model)

        # (Batch, seq_len, d_model) -> (Batch, seq_len, h, d_k) -> (Batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2) # h needs to be 2nd dimension
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        # (Batch, h, seq_len, d_k) -> (Batch, seq_len, h, d_k) -> (Batch, seq_len, d_model) (Concacat)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # (Batch, seq_len, d_model) -> (Batch, seq_len, d_model)
        return self.w_o(x)
    

class ResidualConnection(nn.Module):

    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
    

class EncoderBlock(nn.Module):

    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    
    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x
    

class Encoder(nn.Module):

    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
    

class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, features: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.Tanh(),
            nn.Linear(d_model * 2, features)
        )

    def forward(self, x):
        # (Batch, seq_len, d_model) -> (Batch, 1, features)
        return self.proj(x[:, :1, :])  # Only take the first time step
    

class Transformer(nn.Module):

    def __init__(self, encoder: Encoder, src_pos: Time2Vec, tgt_pos: Time2Vec, projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def project(self, x):
        return self.projection_layer(x)
    

def build_transformer(seq_len: int, d_model: int, features: int, N: int = 6, h: int = 7, dropout: float = 0.1, d_ff: int = 2048): 

    # create positional encoding layers
    src_pos = Time2Vec(features, k=20, output_dim = d_model)
    tgt_pos = Time2Vec(features, k=20, output_dim = d_model)

    # create encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # create encoder and decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))

    # create projection layer
    projection_layer = ProjectionLayer(d_model, features)

    # create the transformer
    transformer = Transformer(encoder, src_pos, tgt_pos, projection_layer)

    # Initialize the parameters so they don't start random
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return transformer