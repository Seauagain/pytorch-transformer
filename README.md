# pytorch-transformer
A PyTorch implementation of the paper "Attention Is All You Need". This repository contains a lightweight Transformer implementation in the `speedrun` directory, designed for minimal dependencies and easy execution.

## Requirements
- One need install `pytorch` and `torchtext` first. 
- `pytorch==1.12` and `torchtext==0.13` work fine for me.

For instance, you can install the packages via `conda`.
```bash
conda install -c pytorch torchtext pytorch
```


## Data Preparation
The code uses a small subset (5k or 50k samples) of the `translation2019zh` English-Chinese translation dataset.  A very simple tokenizer is included for fast experimentation. Sample data is provided in the `dataset` folder.

The sample in the dataset looks like:
```json
{"english": "Slowly and not without struggle, America began to listen.", "chinese": "美国缓慢地开始倾听，但并非没有艰难曲折。"}
{"english": "Dithering is a technique that blends your colors together, making them look smoother, or just creating interesting textures.", "chinese": "抖动是关于颜色混合的技术，使你的作品看起来更圆滑，或者只是创作有趣的材质。"}
```

## Run
Enjoy training transformer.
```bash 
cd speedrun && python train.py
```
