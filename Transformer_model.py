"""
File Description:
This module implements a Transformer-based machine translation model with a complete encoder-decoder architecture.
Key components include:
1. Positional Encoding (PositionalEncoding)
2. Embedding Layer (Embedding)
3. Layer Normalization (LayerNorm)
4. Feed-Forward Network (FeedForward)
5. Multi-Head Attention (MultiHeadAttention)
6. Encoder Layer (EncoderLayer) & Decoder Layer (DecoderLayer)
7. Complete Encoder (Encoder) & Decoder (Decoder)
8. Final Transformer Model Composition (Transformer)

Model Features:
- Follows "Attention Is All You Need" paper architecture
- Supports custom configuration parameters (via config.TransformerConfig class)
- Uses Xavier initialization for weights
- Includes gradient clipping and learning rate scheduling support
- Follows PyTorch modular design specifications
"""

import torch
from torch import nn
import torch.nn.functional as F
import math

def initialize_weights(module):
    """
    Initialize network weights
    - Linear layers use Xavier uniform initialization
    - Embedding layers use Xavier uniform initialization
    """
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.xavier_uniform_(module.weight)


class PositionalEncoding(nn.Module):
    """
    Positional Encoding Module
    - Solves positional information problem for sequence data
    - Generates fixed positional encoding using sine and cosine functions
    """
    def __init__(self, config):
        super().__init__()
        # Initialize positional encoding matrix
        self.encoding = torch.zeros(config.max_len, config.d_model, device=config.device)
        self.pos = torch.arange(0, config.max_len, device=config.device).unsqueeze(1)  # Position indices
        self.div = torch.arange(0, config.d_model, 2, device=config.device)  # Dimension division

        # Calculate positional encoding
        self.encoding[:, 0::2] = torch.sin(self.pos / (10000 ** (self.div / config.d_model)))
        self.encoding[:, 1::2] = torch.cos(self.pos / (10000 ** (self.div / config.d_model)))
        
        # Register as buffer to ensure inclusion in model save
        self.register_buffer("pe", self.encoding)

    def forward(self, x):
        """
        Forward pass
        :param x: Input tensor (batch_size, seq_len, d_model)
        :return: Tensor with positional encoding added
        """
        seq_len = x.size(1)
        return x + self.pe[:seq_len, :].to(x.device)


class FeedForward(nn.Module):
    """
    Feed-Forward Neural Network Module
    - Contains two linear layers with ReLU activation
    - Uses Dropout for regularization
    """
    def __init__(self, config):
        super().__init__()
        self.linear1 = nn.Linear(config.d_model, config.d_ff).to(config.device)
        self.linear2 = nn.Linear(config.d_ff, config.d_model).to(config.device)
        initialize_weights(self.linear1)
        initialize_weights(self.linear2)

    def forward(self, x):
        """
        Forward pass
        :param x: Input tensor (batch_size, seq_len, d_model)
        :return: Output tensor (batch_size, seq_len, d_model)
        """
        return self.linear2(torch.relu(self.linear1(x)))


class Attention(nn.Module):
    """
    Scaled Dot-Product Attention Module
    - Implements basic attention calculation
    - Includes Softmax and Dropout
    """
    def __init__(self, config):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.d_k = config.d_model // config.n_head  # Dimension per head
        self.d_model = config.d_model

    def forward(self, query, key, value, mask=None):
        """
        Forward pass
        :param query: Query tensor (batch_size, n_head, seq_len, d_k)
        :param key: Key tensor (batch_size, n_head, seq_len, d_k)
        :param value: Value tensor (batch_size, n_head, seq_len, d_k)
        :param mask: Mask tensor (batch_size, 1, 1, seq_len)
        :return: Attention output (batch_size, n_head, seq_len, d_k)
        """
        d_k_tensor = torch.tensor(self.d_k, dtype=torch.float32, device=query.device)
        score = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(d_k_tensor)
        
        if mask is not None:
            score = score.masked_fill(mask == 0, -1e9)  # Mask invalid positions
        
        score = self.softmax(score)
        return torch.matmul(score, value)


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention Module
    - Splits input into multiple heads for parallel attention computation
    - Concatenates results and passes through final linear layer
    """
    def __init__(self, config):
        super().__init__()
        self.d_k = config.d_model // config.n_head
        self.d_model = config.d_model
        self.n_head = config.n_head
        
        # Linear transformation layers
        self.w_k = nn.Linear(config.d_model, config.d_model).to(config.device)
        self.w_q = nn.Linear(config.d_model, config.d_model).to(config.device)
        self.w_v = nn.Linear(config.d_model, config.d_model).to(config.device)
        
        self.attention = Attention(config)
        self.linear = nn.Linear(config.d_model, config.d_model).to(config.device)
        self.dropout = nn.Dropout(config.dropout)
        initialize_weights(self.w_k)
        initialize_weights(self.w_q)
        initialize_weights(self.w_v)
        initialize_weights(self.linear)

    def forward(self, query, key, value, mask=None):
        """
        Forward pass
        :param query: Query tensor (batch_size, seq_len, d_model)
        :param key: Key tensor (batch_size, seq_len, d_model)
        :param value: Value tensor (batch_size, seq_len, d_model)
        :param mask: Mask tensor (batch_size, 1, 1, seq_len)
        :return: Output tensor (batch_size, seq_len, d_model)
        """
        batch_size = query.size(0)
        
        # Linear transform and split into heads
        query = self.w_q(query).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)
        key = self.w_k(key).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)
        value = self.w_v(value).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)
        
        # Compute attention
        x = self.attention(query, key, value, mask)
        
        # Concatenate heads and apply final linear layer
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.dropout(self.linear(x))


class EncoderLayer(nn.Module):
    """
    Encoder Layer Module
    - Contains multi-head self-attention and feed-forward network
    - Uses residual connections and layer normalization
    """
    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)
        self.layer_norm1 = nn.LayerNorm(config.d_model)
        self.layer_norm2 = nn.LayerNorm(config.d_model)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)

    def forward(self, x, mask=None):
        """
        Forward pass
        :param x: Input tensor (batch_size, seq_len, d_model)
        :param mask: Source sequence mask (batch_size, 1, 1, seq_len)
        :return: Output tensor (batch_size, seq_len, d_model)
        """
        # Self-attention sub-layer
        norm_x = self.layer_norm1(x)
        attn_output = self.attention(norm_x, norm_x, norm_x, mask)
        x = x + self.dropout1(attn_output)  # Add & Dropout

        # Feed-forward sub-layer
        norm_x = self.layer_norm2(x)
        ff_output = self.feed_forward(norm_x)
        x = x + self.dropout2(ff_output)  # Add & Dropout

        return x


class DecoderLayer(nn.Module):
    """
    Decoder Layer Module
    - Contains two multi-head attentions (self-attention and encoder-decoder attention)
    - Uses residual connections and layer normalization
    """
    def __init__(self, config):
        super().__init__()
        self.attention1 = MultiHeadAttention(config)  # Self-attention
        self.attention2 = MultiHeadAttention(config)  # Encoder-decoder attention
        self.feed_forward = FeedForward(config)
        self.layer_norm1 = nn.LayerNorm(config.d_model)
        self.layer_norm2 = nn.LayerNorm(config.d_model)
        self.layer_norm3 = nn.LayerNorm(config.d_model)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)
        self.dropout3 = nn.Dropout(config.dropout)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        """
        Forward pass
        :param x: Input tensor (batch_size, seq_len, d_model)
        :param encoder_output: Encoder output (batch_size, src_len, d_model)
        :param src_mask: Source sequence mask (batch_size, 1, 1, src_len)
        :param tgt_mask: Target sequence mask (batch_size, 1, 1, tgt_len)
        :return: Output tensor (batch_size, seq_len, d_model)
        """
        # Self-attention sub-layer
        norm_x = self.layer_norm1(x)
        attn_output1 = self.attention1(norm_x, norm_x, norm_x, tgt_mask)
        x = x + self.dropout1(attn_output1)  # Add & Dropout

        # Encoder-decoder attention sub-layer
        norm_x = self.layer_norm2(x)
        attn_output2 = self.attention2(norm_x, encoder_output, encoder_output, src_mask)
        x = x + self.dropout2(attn_output2)  # Add & Dropout

        # Feed-forward sub-layer
        norm_x = self.layer_norm3(x)
        ff_output = self.feed_forward(norm_x)
        x = x + self.dropout3(ff_output)  # Add & Dropout

        return x


class Encoder(nn.Module):
    """
    Encoder Module
    - Contains embedding layer, positional encoding, and multiple encoder layers
    - Final layer normalization at the end
    """
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.positional_encoding = PositionalEncoding(config)
        self.layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.n_layer)])
        self.layer_norm = nn.LayerNorm(config.d_model)  # Final layer norm as per paper
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, mask=None):
        """
        Forward pass
        :param x: Input index tensor (batch_size, seq_len)
        :param mask: Source sequence mask (batch_size, 1, 1, seq_len)
        :return: Encoder output (batch_size, seq_len, d_model)
        """
        x = self.embedding(x)
        x = self.positional_encoding(x)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, mask)

        x = self.layer_norm(x)  # Apply final layer normalization
        return x


class Decoder(nn.Module):
    """
    Decoder Module
    - Contains embedding layer, positional encoding, and multiple decoder layers
    - Final layer normalization at the end
    """
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.positional_encoding = PositionalEncoding(config)
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.n_layer)])
        self.layer_norm = nn.LayerNorm(config.d_model)  # Final layer norm as per paper
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        """
        Forward pass
        :param x: Input index tensor (batch_size, seq_len)
        :param encoder_output: Encoder output (batch_size, src_len, d_model)
        :param src_mask: Source sequence mask (batch_size, 1, 1, src_len)
        :param tgt_mask: Target sequence mask (batch_size, 1, 1, tgt_len)
        :return: Decoder output (batch_size, seq_len, d_model)
        """
        x = self.embedding(x)
        x = self.positional_encoding(x)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)

        x = self.layer_norm(x)  # Apply final layer normalization
        return x


class Transformer(nn.Module):
    """
    Complete Transformer Model
    - Combines encoder and decoder
    - Includes final linear output layer
    """
    def __init__(self, config):
        super().__init__()
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.linear = nn.Linear(config.d_model, config.vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        Forward pass
        :param src: Source sequence indices (batch_size, src_len)
        :param tgt: Target sequence indices (batch_size, tgt_len)
        :param src_mask: Source sequence mask (batch_size, 1, 1, src_len)
        :param tgt_mask: Target sequence mask (batch_size, 1, 1, tgt_len)
        :return: Predicted log probabilities (batch_size, tgt_len, vocab_size)
        """
        encoder_output = self.encoder(src, src_mask)
        decoder_output = self.decoder(tgt, encoder_output, src_mask, tgt_mask)
        output = self.linear(decoder_output)
        return output
