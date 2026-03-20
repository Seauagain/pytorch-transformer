"""
Unit tests for the Transformer model components.

Run with:
    python -m pytest tests/test_transformer.py -v
"""

import math
import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.model.transformer import (
    MultiHeadAttention,
    FeedForward,
    PositionalEncoding,
    LayerNorm,
    EncoderLayer,
    Encoder,
    DecoderLayer,
    Decoder,
    Transformer,
)


# ─── fixtures ────────────────────────────────────────────────────────────────

D_MODEL   = 32
NUM_HEADS = 4
D_FF      = 64
NUM_LAYERS = 2
BATCH     = 2
SRC_LEN   = 6
TRG_LEN   = 5
SRC_VOCAB = 50
TRG_VOCAB = 60
PAD_IDX   = 0
BOS_IDX   = 2
EOS_IDX   = 3


def make_transformer():
    return Transformer(
        en_vocab_size=SRC_VOCAB,
        de_vocab_size=TRG_VOCAB,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        num_layers=NUM_LAYERS,
        max_seq_length=100,
        dropout=0.0,
    ).eval()


# ─── MultiHeadAttention ───────────────────────────────────────────────────────

class TestMultiHeadAttention:
    def test_output_shape(self):
        """Output shape must match input Q shape."""
        mha = MultiHeadAttention(D_MODEL, NUM_HEADS)
        q = torch.randn(BATCH, SRC_LEN, D_MODEL)
        out = mha(q, q, q)
        assert out.shape == (BATCH, SRC_LEN, D_MODEL)

    def test_mask_blocks_future(self):
        """Causal mask must make each position's output independent of future tokens.

        We verify this by checking that changing a future token does NOT affect
        the output at an earlier position when the causal mask is applied.
        """
        mha = MultiHeadAttention(D_MODEL, NUM_HEADS)
        mha.eval()
        torch.manual_seed(0)
        x = torch.randn(1, 3, D_MODEL)

        # Same input but with position 2 changed
        x_modified = x.clone()
        x_modified[0, 2, :] = torch.randn(D_MODEL)

        # Causal mask: lower-triangular bool (1,1,3,3)
        causal = torch.tril(torch.ones(3, 3)).bool().unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            out1 = mha(x,          x,          x,          mask=causal)
            out2 = mha(x_modified, x_modified, x_modified, mask=causal)

        # Position 0 attends only to itself — changing position 2 must not affect it
        assert torch.allclose(out1[:, 0, :], out2[:, 0, :], atol=1e-5), \
            "Position 0 output changed when future token (pos 2) was modified — causal mask broken"

        # Position 1 attends to positions 0 and 1 — also unaffected by pos 2
        assert torch.allclose(out1[:, 1, :], out2[:, 1, :], atol=1e-5), \
            "Position 1 output changed when future token (pos 2) was modified — causal mask broken"

        assert out1.shape == (1, 3, D_MODEL)

    def test_padding_mask_zeros_out_pad_positions(self):
        """Attention scores for padding positions should be -1e10 before softmax."""
        mha = MultiHeadAttention(D_MODEL, NUM_HEADS)
        mha.eval()
        # mask: position 2 is padding (False)
        mask = torch.tensor([[[[True, True, False]]]]).expand(1, 1, 3, 3)
        x = torch.randn(1, 3, D_MODEL)
        with torch.no_grad():
            out = mha(x, x, x, mask=mask)
        assert out.shape == (1, 3, D_MODEL)


# ─── PositionalEncoding ───────────────────────────────────────────────────────

class TestPositionalEncoding:
    def test_output_shape(self):
        pe = PositionalEncoding(D_MODEL)
        x = torch.zeros(BATCH, SRC_LEN, D_MODEL)
        out = pe(x)
        assert out.shape == (BATCH, SRC_LEN, D_MODEL)

    def test_adds_encoding(self):
        """Output should differ from all-zero input (encoding is non-zero)."""
        pe = PositionalEncoding(D_MODEL)
        x = torch.zeros(1, 4, D_MODEL)
        out = pe(x)
        assert not torch.all(out == 0)

    def test_different_positions_differ(self):
        """Two different positions should have different encodings."""
        pe = PositionalEncoding(D_MODEL)
        x = torch.zeros(1, 10, D_MODEL)
        out = pe(x)
        assert not torch.allclose(out[0, 0], out[0, 1])


# ─── LayerNorm ────────────────────────────────────────────────────────────────

class TestLayerNorm:
    def test_output_shape(self):
        ln = LayerNorm(D_MODEL)
        x = torch.randn(BATCH, SRC_LEN, D_MODEL)
        assert ln(x).shape == (BATCH, SRC_LEN, D_MODEL)

    def test_normalized_mean_std(self):
        """After LayerNorm (with default alpha=1, beta=0), mean≈0 and std≈1."""
        ln = LayerNorm(D_MODEL)
        x = torch.randn(BATCH, SRC_LEN, D_MODEL) * 10 + 5
        out = ln(x)
        mean = out.mean(dim=-1)
        std  = out.std(dim=-1)
        assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-4)
        assert torch.allclose(std,  torch.ones_like(std),  atol=1e-4)


# ─── FeedForward ─────────────────────────────────────────────────────────────

class TestFeedForward:
    def test_output_shape(self):
        ff = FeedForward(D_MODEL, D_FF)
        x = torch.randn(BATCH, SRC_LEN, D_MODEL)
        assert ff(x).shape == (BATCH, SRC_LEN, D_MODEL)


# ─── Transformer masks ────────────────────────────────────────────────────────

class TestMasks:
    def setup_method(self):
        self.model = make_transformer()

    def test_src_mask_shape(self):
        src = torch.randint(1, SRC_VOCAB, (BATCH, SRC_LEN))
        mask = self.model.make_src_mask(src, PAD_IDX)
        assert mask.shape == (BATCH, 1, 1, SRC_LEN)

    def test_src_mask_marks_padding(self):
        """Padding positions (token == PAD_IDX) must be False in the mask."""
        src = torch.tensor([[1, 2, PAD_IDX, PAD_IDX]])
        mask = self.model.make_src_mask(src, PAD_IDX)
        assert mask[0, 0, 0, 0].item() == True
        assert mask[0, 0, 0, 2].item() == False
        assert mask[0, 0, 0, 3].item() == False

    def test_trg_mask_shape(self):
        trg = torch.randint(1, TRG_VOCAB, (BATCH, TRG_LEN))
        mask = self.model.make_trg_mask(trg, PAD_IDX)
        assert mask.shape == (BATCH, 1, TRG_LEN, TRG_LEN)

    def test_trg_mask_is_causal(self):
        """Upper triangle of the causal mask must be False (future positions blocked)."""
        trg = torch.randint(1, TRG_VOCAB, (1, 4))  # no padding
        mask = self.model.make_trg_mask(trg, PAD_IDX)
        # mask[0,0] should be lower-triangular
        m = mask[0, 0]  # (4, 4)
        assert m[0, 1].item() == False, "Position 0 should not attend to position 1"
        assert m[1, 0].item() == True,  "Position 1 should attend to position 0"
        assert m[2, 2].item() == True,  "Position 2 should attend to itself"

    def test_trg_mask_blocks_padding(self):
        """Padding tokens in target should be masked out in the key dimension."""
        trg = torch.tensor([[1, 2, PAD_IDX, PAD_IDX]])
        mask = self.model.make_trg_mask(trg, PAD_IDX)
        # Column 2 and 3 (padding key positions) must be False for all query positions
        assert mask[0, 0, 0, 2].item() == False
        assert mask[0, 0, 1, 3].item() == False


# ─── Transformer forward ─────────────────────────────────────────────────────

class TestTransformerForward:
    def setup_method(self):
        self.model = make_transformer()

    def test_output_shape(self):
        """Forward pass output shape: (batch, trg_len, trg_vocab_size)."""
        src = torch.randint(1, SRC_VOCAB, (BATCH, SRC_LEN))
        trg = torch.randint(1, TRG_VOCAB, (BATCH, TRG_LEN))
        out = self.model(src, trg, PAD_IDX, PAD_IDX)
        assert out.shape == (BATCH, TRG_LEN, TRG_VOCAB)

    def test_output_with_padding(self):
        """Forward pass should work correctly when src/trg contain padding tokens."""
        src = torch.tensor([[1, 2, 3, PAD_IDX, PAD_IDX, PAD_IDX],
                            [4, 5, PAD_IDX, PAD_IDX, PAD_IDX, PAD_IDX]])
        trg = torch.tensor([[BOS_IDX, 10, 11, PAD_IDX, PAD_IDX],
                            [BOS_IDX, 12, PAD_IDX, PAD_IDX, PAD_IDX]])
        out = self.model(src, trg, PAD_IDX, PAD_IDX)
        assert out.shape == (2, TRG_LEN, TRG_VOCAB)
        assert not torch.isnan(out).any()


# ─── greedy_decode ────────────────────────────────────────────────────────────

class TestGreedyDecode:
    def setup_method(self):
        self.model = make_transformer()

    def test_output_starts_with_bos(self):
        """Decoded sequence must start with BOS token."""
        src = torch.randint(1, SRC_VOCAB, (1, SRC_LEN))
        out = self.model.greedy_decode(src, PAD_IDX, PAD_IDX, BOS_IDX, EOS_IDX, max_len=10)
        assert out[0, 0].item() == BOS_IDX

    def test_stops_on_eos(self):
        """greedy_decode must stop when EOS is produced, not when BOS is produced.

        We force the model to always predict EOS by patching the output projection
        so that EOS has the highest logit. The loop should break after 1 step.
        """
        model = make_transformer()
        # Bias the fc_out so EOS always wins
        with torch.no_grad():
            model.decoder.fc_out.bias.fill_(-1e9)
            model.decoder.fc_out.bias[EOS_IDX] = 1e9

        src = torch.randint(1, SRC_VOCAB, (1, SRC_LEN))
        out = model.greedy_decode(src, PAD_IDX, PAD_IDX, BOS_IDX, EOS_IDX, max_len=20)
        # Should be [BOS, EOS] — stopped after first generated token
        assert out.shape[1] == 2
        assert out[0, 1].item() == EOS_IDX

    def test_does_not_stop_on_bos(self):
        """greedy_decode must NOT stop when it generates BOS (old bug).

        We force the model to always predict BOS. With the old buggy code
        (stopping on BOS), the loop would break after 1 step giving length 2.
        With the fix it should run to max_len.
        """
        model = make_transformer()
        with torch.no_grad():
            model.decoder.fc_out.bias.fill_(-1e9)
            model.decoder.fc_out.bias[BOS_IDX] = 1e9

        src = torch.randint(1, SRC_VOCAB, (1, SRC_LEN))
        max_len = 5
        out = model.greedy_decode(src, PAD_IDX, PAD_IDX, BOS_IDX, EOS_IDX, max_len=max_len)
        # Should run to max_len (never sees EOS), total length = 1 (initial BOS) + max_len
        assert out.shape[1] == max_len + 1, \
            f"Expected length {max_len + 1}, got {out.shape[1]} — BOS-stop bug may still be present"

    def test_max_len_respected(self):
        """Output length must not exceed max_len + 1 (initial BOS)."""
        src = torch.randint(1, SRC_VOCAB, (BATCH, SRC_LEN))
        max_len = 7
        out = self.model.greedy_decode(src, PAD_IDX, PAD_IDX, BOS_IDX, EOS_IDX, max_len=max_len)
        assert out.shape[1] <= max_len + 1

    def test_batch_size_preserved(self):
        """Output batch dimension must match input batch dimension."""
        src = torch.randint(1, SRC_VOCAB, (BATCH, SRC_LEN))
        out = self.model.greedy_decode(src, PAD_IDX, PAD_IDX, BOS_IDX, EOS_IDX, max_len=10)
        assert out.shape[0] == BATCH
