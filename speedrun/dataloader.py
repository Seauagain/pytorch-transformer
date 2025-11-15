"""
@author : seauagain
@date : 2025-11-01
"""

import json 
import torch 
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
# from torch.utils.data import random_split

## torchtext==0.13 torch==1.12 work fine for me 


# ============== Dataset ==============
 
class TranslationDataset(Dataset):
    def __init__(self, src_sequences, trg_sequences):
        self.src_sequences = src_sequences
        self.trg_sequences = trg_sequences

    def __len__(self):
        return len(self.src_sequences)

    def __getitem__(self, idx):
        # single pair of data
        return torch.tensor(self.src_sequences[idx]), torch.tensor(self.trg_sequences[idx])


def collate_fn(batch, src_pad_idx, trg_pad_idx):
    """
    padding the sequences in the batch.
    """
    src_batch, trg_batch = [], []
    for src_sample, trg_sample in batch:
        src_batch.append(src_sample)
        trg_batch.append(trg_sample)
    src_batch = pad_sequence(src_batch, padding_value=src_pad_idx, batch_first=True)
    trg_batch = pad_sequence(trg_batch, padding_value=trg_pad_idx, batch_first=True)
    return src_batch, trg_batch


def load_sentences_from_json(json_file):
    """load English and Chinese sentence pairs from a json file."""
    english_sentences = []
    chinese_sentences = []
    with open(json_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            english_sentences.append(data.get("english", "").strip())  # 去除首尾空格
            chinese_sentences.append(data.get("chinese", "").strip())  # 去除首尾空格
    return english_sentences, chinese_sentences


def build_vocab(sentences, tokenizer):
    """build vocabulary according to the sentences"""
    def yield_tokens(sentences):
        for sentence in sentences:
            yield tokenizer(sentence)
    vocab = build_vocab_from_iterator(yield_tokens(sentences), specials=['<unk>', '<pad>', '<bos>', '<eos>'])
    vocab.set_default_index(vocab['<unk>'])
    return vocab

def process_sentence(sentence, tokenizer, vocab):
    """
    convert sentences (string/token) to indices (number/id). Add <bos> and <eos>
    """
    tokens = tokenizer(sentence)
    tokens = ['<bos>'] + tokens + ['<eos>']
    indices = [vocab[token] for token in tokens]
    return indices



# ============== Dataloader ==============

def get_train_loader(train_data_path, batch_size=32, val_split=0.1):
    # 加载原始的数据
    english_sentences, chinese_sentences = load_sentences_from_json(train_data_path)
    # 定义英文和中文的分词器
    tokenizer_en = get_tokenizer("basic_english")
    tokenizer_zh = lambda text: list(text)
    # 构建英文和中文的词汇表
    en_vocab = build_vocab(english_sentences, tokenizer_en)
    zh_vocab = build_vocab(chinese_sentences, tokenizer_zh)
    # 将所有句子转换为索引序列
    en_sequences = [process_sentence(s, tokenizer_en, en_vocab) for s in english_sentences]
    zh_sequences = [process_sentence(s, tokenizer_zh, zh_vocab) for s in chinese_sentences]

    # 查看示例句子的索引序列
    print("示例英文句子序列：", english_sentences[0])
    print("示例中文句子序列：", chinese_sentences[0])
    print("示例英文句子索引序列：", en_sequences[0])
    print("示例中文句子索引序列：", zh_sequences[0])

    dataset = TranslationDataset(en_sequences, zh_sequences)
    train_size = int((1 - val_split) * len(dataset))
    val_size = len(dataset) - train_size
    train_data, val_data = random_split(dataset, [train_size, val_size])

    special_tokens = {
        'src_pad_idx': en_vocab['<pad>'],
        'trg_pad_idx': zh_vocab['<pad>'],
        'trg_sos_idx': zh_vocab['<bos>'],
        'trg_eos_idx': zh_vocab['<eos>']
    }

    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, special_tokens['src_pad_idx'], special_tokens['trg_pad_idx'])
    )
    val_loader = DataLoader(
        val_data, batch_size=batch_size, shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, special_tokens['src_pad_idx'], special_tokens['trg_pad_idx'])
    )

    return train_loader, val_loader, en_vocab, zh_vocab, special_tokens


def get_test_dataloader(test_data_path, en_vocab, zh_vocab, batch_size=32):
    en_sentences, zh_sentences = load_sentences_from_json(test_data_path)

    tokenizer_en = get_tokenizer('basic_english')
    tokenizer_zh = lambda text: list(text)

    en_sequences = [process_sentence(s, tokenizer_en, en_vocab) for s in en_sentences]
    zh_sequences = [process_sentence(s, tokenizer_zh, zh_vocab) for s in zh_sentences]

    dataset = TranslationDataset(en_sequences, zh_sequences)

    special_tokens = {
        'src_pad_idx': en_vocab['<pad>'],
        'trg_pad_idx': zh_vocab['<pad>'],
        'trg_sos_idx': zh_vocab['<bos>'],
        'trg_eos_idx': zh_vocab['<eos>']
    }

    test_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, special_tokens['src_pad_idx'], special_tokens['trg_pad_idx'])
    )

    return test_loader, special_tokens

data_json_path = "../dataset/translation2019zh_train50k.json"


# # 加载原始的数据
# english_sentences, chinese_sentences = load_sentences_from_json(data_json_path)

# # 定义英文和中文的分词器
# tokenizer_en = get_tokenizer('basic_english')
# def tokenizer_zh(text):
#     return list(text)

# # 构建英文和中文的词汇表
# en_vocab = build_vocab(english_sentences, tokenizer_en)
# zh_vocab = build_vocab(chinese_sentences, tokenizer_zh)

# # 将所有句子转换为索引序列
# en_sequences = [process_sentence(sentence, tokenizer_en, en_vocab) for sentence in english_sentences]
# zh_sequences = [process_sentence(sentence, tokenizer_zh, zh_vocab) for sentence in chinese_sentences]

# # 查看示例句子的索引序列
# print("示例英文句子序列：", english_sentences[0])
# print("示例中文句子序列：", chinese_sentences[0])
# print("示例英文句子索引序列：", en_sequences[0])
# print("示例中文句子索引序列：", zh_sequences[0])

# # 创建数据集对象
# dataset = TranslationDataset(en_sequences, zh_sequences)

# src_vocab_size = len(en_vocab)
# trg_vocab_size = len(zh_vocab)
# src_pad_idx = en_vocab['<pad>']         # source padding token index
# trg_pad_idx = zh_vocab['<pad>']         # target padding token index
# trg_sos_idx = zh_vocab['<bos>']         # target start token 

# # 划分训练集和验证集
# train_size = int(0.9 * len(dataset))  
# val_size = len(dataset) - train_size  
# train_data, val_data = random_split(dataset, [train_size, val_size])

# 创建数据加载器
# batch_size = 32
# train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
# val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

