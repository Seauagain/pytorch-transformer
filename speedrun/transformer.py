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
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        # Initialize dimensions
        self.d_model = d_model 
        self.num_heads = num_heads 
        self.d_k = d_model // num_heads 
        
        # Linear layers for transforming inputs
        self.W_q = nn.Linear(d_model, d_model, bias=False) # Query 
        self.W_k = nn.Linear(d_model, d_model, bias=False) # Key 
        self.W_v = nn.Linear(d_model, d_model, bias=False) # Value 
        self.W_o = nn.Linear(d_model, d_model, bias=False) # Output 
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        # Apply mask if provided
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e10)
        attn_probs = torch.softmax(attn_scores, dim=-1) # [batch_size, heads, seq_len, d_k]
        output = torch.matmul(attn_probs, V)
        return output
        
    def split_heads(self, x):
        # Reshape the input to have num_heads for multi-head attention
        batch_size, seq_length, _ = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        
    def combine_heads(self, x):
        # Combine the multiple heads back to original shape
        batch_size, _, seq_length, _ = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        
    def forward(self, Q, K, V, mask=None):
        # Apply linear transformations and split heads
        Q = self.split_heads( self.W_q(Q) )
        K = self.split_heads( self.W_k(K) )
        V = self.split_heads( self.W_v(V) )
        # Perform scaled dot-product attention
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        # Combine heads and apply output transformation
        output = self.W_o(self.combine_heads(attn_output))
        return output


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(self.relu(self.fc1(x))))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length = 5000):
        super(PositionalEncoding, self).__init__()
        # create positional encoding matrix [max_seq_length, d_model]
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1) # [max_len, 1]
        div_term = torch.exp( torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)] # x + pos_embed


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.eps = eps 
        self.alpha = nn.Parameter( torch.ones(d_model) )
        self.beta = nn.Parameter( torch.zeros(d_model) )
    
    def forward(self, x):
        # x: (batch, seq, d_model)
        # Keep the dimension for broadcasting
        mean = x.mean(dim = -1, keepdim = True) # (batch, seq, 1)
        # Keep the dimension for broadcasting
        std = x.std(dim = -1, keepdim = True) # (batch, seq, 1)
        # eps is to prevent dividing by zero or when std is very small
        return self.alpha * (x - mean) / (std + self.eps) + self.beta 


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        # self-attention
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        # feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        ## two residual connections
        return x

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_seq_length, dropout=0.1):
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
        # src: [batch_size, src_len]
        x = self.tok_embed(src) * math.sqrt(self.d_model)
        x = self.pos_embed(x)
        x = self.dropout(x)
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, src_mask)
        return x



class DecoderLayer(nn.Module):
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
        # mask attention
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        # cross attention
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        # feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x

class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_seq_length, dropout=0.1):
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
        # trg: [batch_size, trg_len]
        x = self.tok_embed(trg) * math.sqrt(self.d_model)
        x = self.pos_embed(x)
        x = self.dropout(x)
        for decoder_layer in self.decoder_layers:
            x = decoder_layer(x, enc_output, src_mask, trg_mask)
        output = self.fc_out(x)
        return output



class Transformer(nn.Module):
    def __init__(
        self,
        # Vocabulary size parameters
        en_vocab_size: int,  # Source language vocabulary size
        de_vocab_size: int,  # Target language vocabulary size
        d_model: int = 512,           # Embedding dimension
        num_heads: int = 8,           # Number of attention heads
        d_ff: int = 2048,             # Feed-forward dimension
        num_layers: int = 6,          # Number of encoder/decoder layers
        max_seq_length: int = 5000,    # Maximum sequence length
        dropout: float = 0.1          # Dropout rate
    ):
        super(Transformer, self).__init__()
        
        # Initialize encoder
        self.encoder = Encoder(
            vocab_size=en_vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            num_layers=num_layers,
            max_seq_length=max_seq_length,
            dropout=dropout
        )
        
        # Initialize decoder
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
        """create padding mask for the source sequence."""
        src_mask = (src != src_pad_idx).unsqueeze(1).unsqueeze(2)
        #              [batch_size, 1, 1, src_len] 
        # broadcast to [batch_size, heads, src_len, src_len] in encoder attention layer.
        return src_mask

    
    def make_trg_mask(self, trg, trg_pad_idx):
        """padding + autoregressive mask for target."""
        # padding mask: (batch, 1, trg_len, 1)
        trg_pad_mask = (trg != trg_pad_idx).unsqueeze(1).unsqueeze(3)  #(batch_size, 1, trg_len, 1)
        trg_len = trg.shape[1]
        # causal mask: (1, 1, trg_len, trg_len)
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=trg.device)).to(torch.bool) # 下三角
        # broadcasting gives: (batch, 1, trg_len, trg_len)
        trg_mask = trg_pad_mask & trg_sub_mask
        return trg_mask

    def forward(self, src, trg, src_pad_idx, trg_pad_idx):
        """
        src: (batch, src_len)
        trg: (batch, trg_len)
        """
        src_mask = self.make_src_mask(src, src_pad_idx)
        trg_mask = self.make_trg_mask(trg, trg_pad_idx)
        en_output = self.encoder(src, src_mask)
        output = self.decoder(trg, en_output, src_mask, trg_mask)
        return output
    
    def greedy_decode(self, src, trg, src_pad_idx, trg_pad_idx, trg_bos_idx, max_len=10):
        """
        autogressive decoding
        """
        src_mask = self.make_src_mask(src, src_pad_idx)
        en_output = self.encoder(src, src_mask)

        batch_size = src.size(0)
        # create first token of the target sequences.
        trg = torch.full( size=(batch_size, 1), fill_value=trg_bos_idx, device=src.device)

        for _ in range(max_len):
            trg_mask = self.make_trg_mask(trg, trg_pad_idx)
            output = self.decoder(trg, en_output, src_mask, trg_mask)
            next_token = output[:, -1, :].argmax(-1, keepdim=True) #last word in the sequences.
            trg = torch.cat([trg, next_token], dim=1)

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



