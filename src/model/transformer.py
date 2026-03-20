"""
@author : seauagain
@date : 2025.11.01 
"""

## system-level import 
import torch 
from torch import nn 
import math


## user-level import 

class MultiHeadAttention(nn.Module):
    """Multi-head scaled dot-product attention module.

    Projects Q/K/V into `num_heads` subspaces, computes attention in parallel,
    then concatenates and projects back to d_model.
    """

    def __init__(self, d_model, num_heads):
        """
        Args:
            d_model: total embedding dimension.
            num_heads: number of attention heads. Must divide d_model evenly.
        """
        super(MultiHeadAttention, self).__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # per-head dimension

        # Projection matrices (no bias, following the original paper)
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """Compute scaled dot-product attention.

        Args:
            Q: (batch, heads, seq_q, d_k)
            K: (batch, heads, seq_k, d_k)
            V: (batch, heads, seq_k, d_k)
            mask: bool tensor broadcastable to (batch, heads, seq_q, seq_k).
                  True = keep, False = mask out.
        Returns:
            (batch, heads, seq_q, d_k)
        """
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            # Fill positions where mask is False (padding / future tokens) with -inf
            attn_scores = attn_scores.masked_fill(~mask, -1e10)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output

    def split_heads(self, x):
        """Reshape (batch, seq, d_model) -> (batch, heads, seq, d_k)."""
        batch_size, seq_length, _ = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        """Reshape (batch, heads, seq, d_k) -> (batch, seq, d_model)."""
        batch_size, _, seq_length, _ = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, Q, K, V, mask=None):
        """
        Args:
            Q, K, V: (batch, seq, d_model)
            mask: optional bool mask (see scaled_dot_product_attention)
        Returns:
            (batch, seq_q, d_model)
        """
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output


class FeedForward(nn.Module):
    """Position-wise feed-forward network: Linear -> ReLU -> Dropout -> Linear."""

    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        Args:
            d_model: input/output dimension.
            d_ff: inner hidden dimension (typically 4 * d_model).
            dropout: dropout probability applied after ReLU.
        """
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """x: (batch, seq, d_model) -> (batch, seq, d_model)."""
        return self.fc2(self.dropout(self.relu(self.fc1(x))))


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding added to token embeddings.

    Uses sine for even dimensions and cosine for odd dimensions, following
    "Attention Is All You Need" (Vaswani et al., 2017).
    """

    def __init__(self, d_model, max_seq_length=5000):
        """
        Args:
            d_model: embedding dimension.
            max_seq_length: maximum sequence length to pre-compute encodings for.
        """
        super(PositionalEncoding, self).__init__()
        # Build positional encoding matrix of shape (max_seq_length, d_model)
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as buffer so it moves with the model but is not a parameter
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_seq_length, d_model)

    def forward(self, x):
        """Add positional encoding to x.

        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model)
        """
        return x + self.pe[:, :x.size(1)]


class LayerNorm(nn.Module):
    """Custom layer normalization with learnable scale (alpha) and shift (beta)."""

    def __init__(self, d_model, eps=1e-6):
        """
        Args:
            d_model: feature dimension to normalize over.
            eps: small constant for numerical stability.
        """
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(d_model))   # learnable scale
        self.beta = nn.Parameter(torch.zeros(d_model))   # learnable shift

    def forward(self, x):
        """Normalize over the last dimension.

        Args:
            x: (batch, seq, d_model)
        Returns:
            (batch, seq, d_model)
        """
        mean = x.mean(dim=-1, keepdim=True)   # (batch, seq, 1)
        std = x.std(dim=-1, keepdim=True)     # (batch, seq, 1)
        return self.alpha * (x - mean) / (std + self.eps) + self.beta


class EncoderLayer(nn.Module):
    """Single encoder layer: self-attention + feed-forward, each with residual + LayerNorm."""

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        """
        Args:
            x: (batch, src_len, d_model)
            mask: source padding mask (batch, 1, 1, src_len)
        Returns:
            (batch, src_len, d_model)
        """
        # Self-attention sub-layer with residual connection
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        # Feed-forward sub-layer with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


class Encoder(nn.Module):
    """Stack of N encoder layers with token + positional embeddings."""

    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_seq_length, dropout=0.1):
        """
        Args:
            vocab_size: source vocabulary size.
            d_model: embedding dimension.
            num_heads: number of attention heads.
            d_ff: feed-forward inner dimension.
            num_layers: number of stacked encoder layers.
            max_seq_length: maximum sequence length for positional encoding.
            dropout: dropout probability.
        """
        super().__init__()
        self.d_model = d_model
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = PositionalEncoding(d_model, max_seq_length)
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        """
        Args:
            src: (batch, src_len) token indices
            src_mask: padding mask (batch, 1, 1, src_len)
        Returns:
            (batch, src_len, d_model)
        """
        # Scale embeddings by sqrt(d_model) as in the original paper
        x = self.tok_embed(src) * math.sqrt(self.d_model)
        x = self.pos_embed(x)
        x = self.dropout(x)
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, src_mask)
        return x



class DecoderLayer(nn.Module):
    """Single decoder layer: masked self-attention + cross-attention + feed-forward."""

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        """
        Args:
            x: (batch, trg_len, d_model) — decoder input
            enc_output: (batch, src_len, d_model) — encoder output
            src_mask: padding mask for encoder output (batch, 1, 1, src_len)
            tgt_mask: causal + padding mask for decoder input (batch, 1, trg_len, trg_len)
        Returns:
            (batch, trg_len, d_model)
        """
        # Masked self-attention (causal): decoder attends only to past positions
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        # Cross-attention: decoder queries attend to encoder keys/values
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        # Feed-forward sub-layer
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x


class Decoder(nn.Module):
    """Stack of N decoder layers with token + positional embeddings and output projection."""

    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_seq_length, dropout=0.1):
        """
        Args:
            vocab_size: target vocabulary size.
            d_model: embedding dimension.
            num_heads: number of attention heads.
            d_ff: feed-forward inner dimension.
            num_layers: number of stacked decoder layers.
            max_seq_length: maximum sequence length for positional encoding.
            dropout: dropout probability.
        """
        super().__init__()
        self.d_model = d_model
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = PositionalEncoding(d_model, max_seq_length)
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, trg, enc_output, src_mask=None, trg_mask=None):
        """
        Args:
            trg: (batch, trg_len) token indices
            enc_output: (batch, src_len, d_model)
            src_mask: source padding mask
            trg_mask: target causal + padding mask
        Returns:
            logits (batch, trg_len, vocab_size)
        """
        x = self.tok_embed(trg) * math.sqrt(self.d_model)
        x = self.pos_embed(x)
        x = self.dropout(x)
        for decoder_layer in self.decoder_layers:
            x = decoder_layer(x, enc_output, src_mask, trg_mask)
        output = self.fc_out(x)
        return output



class Transformer(nn.Module):
    """Full encoder-decoder Transformer for sequence-to-sequence tasks.

    Follows the architecture from "Attention Is All You Need" (Vaswani et al., 2017).
    """

    def __init__(
        self,
        en_vocab_size: int,        # Source language vocabulary size
        de_vocab_size: int,        # Target language vocabulary size
        d_model: int = 512,        # Embedding dimension
        num_heads: int = 8,        # Number of attention heads
        d_ff: int = 1024,          # Feed-forward inner dimension
        num_layers: int = 6,       # Number of encoder/decoder layers
        max_seq_length: int = 5000,
        dropout: float = 0.1
    ):
        super(Transformer, self).__init__()

        self.encoder = Encoder(
            vocab_size=en_vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            num_layers=num_layers,
            max_seq_length=max_seq_length,
            dropout=dropout
        )

        self.decoder = Decoder(
            vocab_size=de_vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            num_layers=num_layers,
            max_seq_length=max_seq_length,
            dropout=dropout
        )

    def make_src_mask(self, src, src_pad_idx):
        """Build source padding mask.

        Returns a bool tensor of shape (batch, 1, 1, src_len) where True means
        the position is a real token and False means it is padding.
        Broadcasts to (batch, heads, src_len, src_len) inside attention.
        """
        return (src != src_pad_idx).unsqueeze(1).unsqueeze(2)

    def make_trg_mask(self, trg, trg_pad_idx):
        """Build target mask combining padding mask and causal (autoregressive) mask.

        Padding mask shape:  (batch, 1, 1, trg_len)
        Causal mask shape:   (1, 1, trg_len, trg_len)  — lower-triangular
        Combined shape:      (batch, 1, trg_len, trg_len) via broadcasting.

        A position is True only when it is both a real token AND not in the future.
        """
        # True where token is not padding: (batch, 1, 1, trg_len)
        trg_pad_mask = (trg != trg_pad_idx).unsqueeze(1).unsqueeze(2)
        trg_len = trg.shape[1]
        # Lower-triangular causal mask: (trg_len, trg_len)
        trg_sub_mask = torch.tril(
            torch.ones((trg_len, trg_len), device=trg.device)
        ).bool()
        # AND: only attend to non-padding past/current positions
        return trg_pad_mask & trg_sub_mask

    def forward(self, src, trg, src_pad_idx, trg_pad_idx):
        """Teacher-forced forward pass used during training.

        Args:
            src: (batch, src_len) source token indices
            trg: (batch, trg_len) target token indices (shifted right, i.e. starts with BOS)
            src_pad_idx: padding token id for source
            trg_pad_idx: padding token id for target
        Returns:
            logits (batch, trg_len, tgt_vocab_size)
        """
        src_mask = self.make_src_mask(src, src_pad_idx)
        trg_mask = self.make_trg_mask(trg, trg_pad_idx)
        enc_output = self.encoder(src, src_mask)
        output = self.decoder(trg, enc_output, src_mask, trg_mask)
        return output

    def greedy_decode(self, src, src_pad_idx, trg_pad_idx, trg_bos_idx, trg_eos_idx, max_len=50):
        """Autoregressive greedy decoding (inference).

        Generates tokens one at a time by always picking the highest-probability
        next token. Stops when every sequence in the batch has produced EOS or
        max_len tokens have been generated.

        Args:
            src: (batch, src_len) source token indices
            src_pad_idx: padding token id for source
            trg_pad_idx: padding token id for target (used to build causal mask)
            trg_bos_idx: BOS token id — used as the initial decoder input
            trg_eos_idx: EOS token id — generation stops when all sequences emit this
            max_len: maximum number of tokens to generate
        Returns:
            (batch, generated_len) token indices including the leading BOS token
        """
        src_mask = self.make_src_mask(src, src_pad_idx)
        enc_output = self.encoder(src, src_mask)

        batch_size = src.size(0)
        # Start every sequence with BOS: (batch, 1)
        trg = torch.full((batch_size, 1), fill_value=trg_bos_idx, device=src.device)

        for _ in range(max_len):
            trg_mask = self.make_trg_mask(trg, trg_pad_idx)
            output = self.decoder(trg, enc_output, src_mask, trg_mask)
            # Greedy: pick the token with the highest logit at the last position
            next_token = output[:, -1, :].argmax(-1, keepdim=True)  # (batch, 1)
            trg = torch.cat([trg, next_token], dim=1)

            # Stop when every sequence in the batch has produced EOS
            if (next_token == trg_eos_idx).all():
                break

        return trg



if __name__ == "__main__":

    model = Transformer(
                en_vocab_size = 850,     # source vocabulary size
                de_vocab_size = 1200,    # target vocabulary size
                d_model = 512,
                num_heads = 8,
                num_layers = 6, 
                d_ff = 1024,
                max_seq_length = 5000,
                dropout = 0
            )

    src_pad_idx = 0         # source padding token index
    trg_pad_idx = 0         # target padding token index
    trg_bos_idx = 2         # target begining token index

    seed = 42
    torch.manual_seed(seed)
    # use CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # multi-GPUs
    
    device = "cuda:0"
    batch_size, seq_length = 3, 10
    src = torch.randint(low=0, high=850, size=(batch_size, seq_length)).to(device)
    trg = torch.randint(low=0, high=1200, size=(batch_size, seq_length)).to(device)
    model = model.to(device)

    output = model(src, trg, src_pad_idx, trg_pad_idx)
    print("output.size(): ", output.size())
    print("output: ", output)



