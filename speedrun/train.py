"""
@author : seauagain
@date   : 2025.11.01
@desc   : Training & Inference script for Transformer translation model
"""

import torch
import torch.nn as nn
import torch.optim as optim
import os
from typing import Tuple

from transformer import Transformer
from dataloader import (
    get_train_loader,
    get_test_loader,
    get_vocab_tokenizer
)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

train_json = "../dataset/translation2019zh_train50k.json"
valid_json = "../dataset/translation2019zh_valid1k.json"
ckpt_path = "./checkpoints/best_transformer.ckpt"
os.makedirs("./checkpoints", exist_ok=True)


# 1. 数据加载
train_loader, val_loader, en_vocab, zh_vocab, special_tokens = get_train_loader(
    train_data_path=train_json,
    batch_size=32,
    val_split=0.1
)

src_pad_idx = special_tokens["src_pad_idx"]
trg_pad_idx = special_tokens["trg_pad_idx"]
trg_bos_idx = special_tokens["trg_bos_idx"]
trg_eos_idx = special_tokens["trg_eos_idx"]


# 2. Transformer
model = Transformer(
    en_vocab_size=len(en_vocab),
    de_vocab_size=len(zh_vocab),
    d_model=512,
    num_heads=8,
    num_layers=6,
    d_ff=2048,
    max_seq_length=5000,
    dropout=0.1
).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=trg_pad_idx)
optimizer = optim.Adam(model.parameters(), lr=1e-4)


# training
def train_one_epoch(model, dataloader, criterion, optimizer):
    model.train()
    epoch_loss = 0.0

    for src, trg in dataloader:
        src = src.to(device)
        trg = trg.to(device)

        # trg_input: 去除句子末尾
        trg_input = trg[:, :-1]
        # trg_output: 去除句子开头
        trg_output = trg[:, 1:].contiguous().view(-1)

        optimizer.zero_grad()
        output = model(src, trg_input, src_pad_idx, trg_pad_idx)

        output = output.contiguous().view(-1, output.size(-1))
        loss = criterion(output, trg_output)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(dataloader)


def evaluate( model, dataloader, criterion):
    model.eval()
    epoch_loss = 0.0

    with torch.no_grad():
        for src, trg in dataloader:
            src = src.to(device)
            trg = trg.to(device)

            trg_input = trg[:, :-1]
            trg_output = trg[:, 1:].contiguous().view(-1)

            output = model(src, trg_input, src_pad_idx, trg_pad_idx)
            output = output.contiguous().view(-1, output.size(-1))

            loss = criterion(output, trg_output)
            epoch_loss += loss.item()

    return epoch_loss / len(dataloader)


# 4. Trainer 
class Trainer:
    def __init__(self, model, optimizer, criterion, ckpt_path):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.ckpt_path = ckpt_path
        self.best_val_loss = float("inf")

    def save_checkpoint(self):
        torch.save({
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict()
        }, self.ckpt_path)
        print(f"[INFO] Checkpoint saved to {self.ckpt_path}")

    def load_checkpoint(self):
        if not os.path.exists(self.ckpt_path):
            print("[WARN] No checkpoint found.")
            return False

        ckpt = torch.load(self.ckpt_path, map_location=device)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        print(f"[INFO] Loaded checkpoint from {self.ckpt_path}")
        return True

    def run_training(self, train_loader, val_loader, epochs):
        for epoch in range(1, epochs + 1):
            train_loss = train_one_epoch(
                self.model,
                train_loader,
                self.criterion,
                self.optimizer
            )
            val_loss = evaluate(
                self.model,
                val_loader,
                self.criterion
            )

            print(
                f"[Epoch {epoch:02d}] "
                f"Train Loss: {train_loss:.3f} | "
                f"Val Loss: {val_loss:.3f}"
            )

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint()


trainer = Trainer(model, optimizer, criterion, ckpt_path)
trainer.run_training(train_loader, val_loader, epochs=30)


# 6. inference

def greedy_translate(sentence, model, en_vocab, zh_vocab, tokenizer_en, max_len=200):
    model.eval()

    # Build source sentence
    tokens = ["<bos>"] + tokenizer_en(sentence) + ["<eos>"]
    src_idx = [en_vocab[token] for token in tokens]
    src = torch.tensor(src_idx).unsqueeze(0).to(device)  # [1, src_len]

    src_mask = model.make_src_mask(src, src_pad_idx)

    with torch.no_grad():
        enc_out = model.encoder(src, src_mask)

    trg_indices = [zh_vocab["<bos>"]]

    for _ in range(max_len):
        trg_tensor = torch.tensor(trg_indices).unsqueeze(0).to(device)
        trg_mask = model.make_trg_mask(trg_tensor, trg_pad_idx)

        with torch.no_grad():
            dec_out = model.decoder(trg_tensor, enc_out, src_mask, trg_mask)

        next_token = dec_out.argmax(dim=-1)[0, -1].item()
        trg_indices.append(next_token)

        if next_token == zh_vocab["<eos>"]:
            break

    # Convert to readable text
    result_tokens = [
        zh_vocab.lookup_token(idx)
        for idx in trg_indices[1:-1]
    ]
    return "".join(result_tokens)


# 7. decoding
print("\n[INFO] Loading checkpoint for inference...")
trainer.load_checkpoint()

tokenizer_en, tokenizer_zh, en_vocab, zh_vocab = get_vocab_tokenizer(train_json)

while True:
    text = input("\n请输入英文句子 ('quit' 退出)：")
    if text.lower() == "quit":
        break
    result = greedy_translate(text, model, en_vocab, zh_vocab, tokenizer_en)
    print(f"中文翻译：{result}")
