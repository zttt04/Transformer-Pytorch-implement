# Import required libraries
import pyarrow.parquet as pq
import torch
from transformers import FSMTTokenizer
from tqdm import tqdm
"""
Script Purpose:
This script processes English-German sentence pairs using FSMT tokenizer, 
converting text to indexed sequences with BOS tokens and saving as PyTorch tensors. 
Core functionalities include:

1. Data Loading:
   - Reads en/de sentences from Parquet files
   - Validates column existence ('en' for English, 'de' for German)

2. Text Processing:
   - Prepends BOS token to each sentence
   - Encodes to fixed-length sequences (max_length=128)
   - Handles padding/truncation for uniform tensor shapes

3. Tensor Generation:
   - Aggregates individual sentence tensors into batch tensors
   - Saves final tensors with descriptive filenames

Key Parameters:
- BOS Token: {tokenizer.bos_token} (from FSMTTokenizer)
- Sequence Length: 128 (model-specific constraint)
- Padding Strategy: 'max_length' (pad all to 128 tokens)
- Truncation: True (cut sentences exceeding 128 tokens)
"""

# Load pre-trained tokenizer from local directory
mname = './wmt19-en-de'
tokenizer = FSMTTokenizer.from_pretrained(mname)

# Read Parquet datasets
en_table = pq.read_table("../Train/en_data_train.parquet")
de_table = pq.read_table("../Train/de_data_train.parquet")

# Validate column presence and extract sentences
if 'en' in en_table.column_names and 'de' in de_table.column_names:
    en_sentences = en_table['en'].to_pandas().tolist()  # English sentence list
    de_sentences = de_table['de'].to_pandas().tolist()  # German sentence list

    en_indexes = []  # Stores tokenized English sequences
    de_indexes = []  # Stores tokenized German sequences

    # Process English sentences with progress tracking
    for sentence in tqdm(en_sentences, desc='English Processing'):
        # Prepend BOS token and encode
        bos_prepended = f"{tokenizer.bos_token} {sentence}"
        encoded = tokenizer.encode_plus(
            bos_prepended,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )["input_ids"]  # Shape: (1, 128)
        en_indexes.append(encoded)

    # Process German sentences with progress tracking
    for sentence in tqdm(de_sentences, desc='German Processing'):
        # Prepend BOS token and encode
        bos_prepended = f"{tokenizer.bos_token} {sentence}"
        encoded = tokenizer.encode_plus(
            bos_prepended,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )["input_ids"]  # Shape: (1, 128)
        de_indexes.append(encoded)

    # Concatenate individual tensors into batch tensors
    en_tensor = torch.cat(en_indexes, dim=0)  # Shape: (N, 128)
    de_tensor = torch.cat(de_indexes, dim=0)  # Shape: (N, 128)

    # Save processed tensors with versioned filenames
    torch.save(en_tensor, 'en_processed_indexes_train.pt')
    torch.save(de_tensor, 'de_processed_indexes_train.pt')
    print(f"Processed {len(en_sentences)} English and {len(de_sentences)} German sentences")

else:
    # Error handling for missing columns
    print("Error: Required columns ('en' or 'de') not found in input files")
