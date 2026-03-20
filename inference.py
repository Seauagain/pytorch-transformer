"""
@author : Gemini
@date : 2025.12.11
@description: Inference script for Transformer translation model. 
              Loads a checkpoint and performs greedy decoding.
"""

import torch
import os
import argparse
from transformers import AutoTokenizer

# Import the model architecture
# Assuming transformer.py is in the same directory or in the python path
try:
    from transformer import Transformer
except ImportError:
    # If transformer.py is inside a src folder
    from src.model import Transformer

def load_model(ckpt_path, device, tokenizer):
    """
    Load the model architecture and weights from the checkpoint.
    """
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

    print(f"Loading checkpoint from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    # Load configuration from the saved checkpoint
    saved_config = checkpoint['config']
    
    # Initialize the model using saved hyperparameters
    # Note: 'zh_vocab_size' is used as target vocab size based on train.py logic
    model = Transformer(
        en_vocab_size=saved_config['en_vocab_size'],
        de_vocab_size=saved_config['zh_vocab_size'],
        d_model=saved_config['d_model'],
        num_heads=saved_config['nums_heads'], # Note: check spelling in train.py (nums_heads vs num_heads)
        d_ff=saved_config['d_ff'],
        num_layers=saved_config['num_layers'],
        max_seq_length=saved_config['max_seq_length'],
        dropout=0.0 # Dropout is not needed for inference
    )

    # Load state dict
    state_dict = checkpoint['model_state_dict']
    
    # Fix for DDP (DistributedDataParallel) saved weights
    # DDP adds a "module." prefix to keys which needs to be removed for single-device inference
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
        
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    
    return model, saved_config

def translate(text, model, tokenizer, device, max_len=50):
    """
    Translate a single sentence using greedy decoding.
    """
    model.eval()
    
    # 1. Tokenize input
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    src = tokens["input_ids"].to(device)
    
    # 2. Get special token indices
    # Ensure these match how the model was trained
    src_pad_idx = tokenizer.pad_token_id
    trg_pad_idx = tokenizer.pad_token_id
    trg_bos_idx = tokenizer.bos_token_id
    trg_eos_idx = tokenizer.eos_token_id

    # 3. Perform Greedy Decoding
    with torch.no_grad():
        output_tokens = model.greedy_decode(
            src=src,
            src_pad_idx=src_pad_idx,
            trg_pad_idx=trg_pad_idx,
            trg_bos_idx=trg_bos_idx,
            trg_eos_idx=trg_eos_idx,
            max_len=max_len
        )
    
    # 4. Decode output tokens to string
    # output_tokens contains [batch_size, seq_len]
    translated_ids = output_tokens[0]
    
    # Decode and remove special tokens (pad, eos, bos)
    translated_text = tokenizer.decode(translated_ids, skip_special_tokens=True)
    
    return translated_text

def main():
    parser = argparse.ArgumentParser(description="Transformer Inference Script")
    parser.add_argument('--ckpt', type=str, required=True, help='Path to the model checkpoint (.pth file)')
    parser.add_argument('--tokenizer_path', type=str, default="tokenizer/Helsinki-NLP/opus-mt-zh-en", help='Path to tokenizer')
    parser.add_argument('--text', type=str, default="Hello world", help='Text to translate')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode')
    
    args = parser.parse_args()

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load Tokenizer
    # Must match training configuration exactly
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    except OSError:
        print(f"Warning: Local tokenizer not found at {args.tokenizer_path}. Trying to download from HuggingFace Hub...")
        tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-zh-en")

    # Add special tokens as done in train.py
    # This is CRITICAL for index alignment
    tokenizer.add_special_tokens({'bos_token': '<bos>'})
    
    # Load Model
    model, config = load_model(args.ckpt, device, tokenizer)
    
    print("-" * 50)
    print("Model loaded successfully.")
    print("-" * 50)

    if args.interactive:
        print("Interactive Mode. Type 'q' or 'quit' to exit.")
        while True:
            try:
                source_text = input("\nEnter English text: ")
                if source_text.lower() in ['q', 'quit', 'exit']:
                    break
                
                translation = translate(source_text, model, tokenizer, device)
                print(f"Translation: {translation}")
            except KeyboardInterrupt:
                break
    else:
        # Single Inference
        print(f"Source: {args.text}")
        translation = translate(args.text, model, tokenizer, device)
        print(f"Translation: {translation}")

if __name__ == "__main__":
    main()