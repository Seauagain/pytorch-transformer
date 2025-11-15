"""
@author : seauagain
@date : 2025.11.01 
"""


import torch
import torch.nn as nn
import torch.optim as optim


from transformer import Transformer
from dataloader import get_train_loader, get_test_dataloader


# 检查是否有可用的 GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')


traindata_path = "../dataset/translation2019zh_train50k.json"

testdata_json = "../dataset/translation2019zh_train50k.json"


train_dataloader, val_dataloader, en_vocab, zh_vocab, special_tokens = get_train_loader(train_data_path=traindata_path, batch_szie=32, val_split=0.1)


input_dim = len(en_vocab)
output_dim = len(zh_vocab)
d_model = 512
num_heads = 8
d_ff = 2048
num_layers = 6
max_seq_length = 5000
dropout = 0.1


model = Transformer(  
        en_vocab_size = len(en_vocab),    # source vocabulary size
        de_vocab_size = len(zh_vocab),   # target vocabulary size
        d_model = d_model,
        num_heads = num_heads,
        num_layers = num_layers, 
        d_ff = d_ff,
        max_seq_length = max_seq_length,
        dropout = dropout
    ).to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss(ignore_index=trg_pad_idx)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Step 4: 模型训练与验证

# 定义训练函数
def train(model, dataloader, optimizer, criterion):
    model.train()
    epoch_loss = 0
    for src, trg in dataloader:
        src = src.to(device)
        trg = trg.to(device)
        optimizer.zero_grad()

        output = model(src, trg[:, :-1], src_pad_idx, trg_pad_idx)  # 输入不包括最后一个词
        output = output.contiguous().view(-1, output_dim)
        trg = trg[:, 1:].contiguous().view(-1)  # 目标不包括第一个词
        loss = criterion(output, trg)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(dataloader)

# 定义验证函数
def evaluate(model, dataloader, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for src, trg in dataloader:
            src = src.to(device)
            trg = trg.to(device)
            output = model(src, trg[:, :-1], src_pad_idx, trg_pad_idx)
            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)
            trg = trg[:, 1:].contiguous().view(-1)
            loss = criterion(output, trg)
            epoch_loss += loss.item()
    return epoch_loss / len(dataloader)

# 开始训练
n_epochs = 30

for epoch in range(n_epochs):
    train_loss = train(model, train_dataloader, optimizer, criterion)
    val_loss = evaluate(model, val_dataloader, criterion)
    print(f'Epoch {epoch+1}/{n_epochs}, Train Loss: {train_loss:.3f}, Val Loss: {val_loss:.3f}')

# Step 5: 测试与推理

# 定义翻译函数
def translate_sentence(sentence, model, en_vocab, zh_vocab, tokenizer_en, max_len=500):
    """
    翻译英文句子为中文
    :param sentence: 英文句子（字符串）
    :param model: 训练好的 Transformer 模型
    :param en_vocab: 英文词汇表
    :param zh_vocab: 中文词汇表
    :param tokenizer_en: 英文分词器
    :param max_len: 最大翻译长度
    :return: 中文翻译（字符串）
    """
    model.eval()
    tokens = tokenizer_en(sentence)
    tokens = ['<bos>'] + tokens + ['<eos>']
    src_indices = [en_vocab[token] for token in tokens]
    src_tensor = torch.LongTensor(src_indices).unsqueeze(0).to(device)  # [1, src_len]
    src_mask = model.make_src_mask(src_tensor, src_pad_idx)
    with torch.no_grad():
        enc_output = model.encoder(src_tensor, src_mask)
    trg_indices = [zh_vocab['<bos>']]
    for i in range(max_len):
        trg_tensor = torch.LongTensor(trg_indices).unsqueeze(0).to(device)  # [1, trg_len]
        trg_mask = model.make_trg_mask(trg_tensor, trg_pad_idx)
        with torch.no_grad():
            output = model.decoder(trg_tensor, enc_output, src_mask, trg_mask)
        pred_token = output.argmax(-1)[:, -1].item()
        trg_indices.append(pred_token)
        if pred_token == zh_vocab['<eos>']:
            break
    trg_tokens = [zh_vocab.lookup_token(idx) for idx in trg_indices]
    return ''.join(trg_tokens[1:-1])  # 去除 <bos> 和 <eos>


# 示例测试
input_sentence = "How are you?"
translation = translate_sentence(input_sentence, model, en_vocab, zh_vocab, tokenizer_en)
print(f"英文句子: {input_sentence}")
print(f"中文翻译: {translation}")

# 您可以在此处输入其他英文句子进行测试
while True:
    input_sentence = input("请输入英文句子（输入 'quit' 退出）：")
    if input_sentence.lower() == 'quit':
        break
    translation = translate_sentence(input_sentence, model, en_vocab, zh_vocab, tokenizer_en)
    print(f"中文翻译: {translation}")
