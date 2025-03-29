import torch
import torch.nn as nn
import torch.nn.functional as F
from config import TransformerConfig
from Transformer_model import Transformer
from torch.utils.data import TensorDataset, DataLoader
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from mask import create_masks
from tqdm import tqdm
import numpy as np
from transformers import FSMTTokenizer
import os
"""
File Description:
This script is used to test and evaluate a trained Transformer model, with the following core functions:

1. Data Processing:
   - Load preprocessed English/German index data (en/de_processed_indexes.pt).
   - Support truncating the data volume (here, the first 100,000 samples are selected).
   - Use FSMTTokenizer to convert between indexes and tokens.

2. Model Evaluation:
   - Load a pre-trained Transformer model (weight path: Weight/transformer_epoch_10_final1.pth).
   - Support GPU acceleration (automatically detect CUDA devices).
   - Use batch processing (batch_size = 32) to improve evaluation efficiency.

3. Evaluation Metrics:
   - Calculate the BLEU-4 score (with smoothing).
   - Display the first 5 translation examples.
   - Ignore the influence of padding tokens (index 1) and end tokens (index 2).

4. Output Results:
   - Output the BLEU score to the console (rounded to four decimal places).
   - Print translation comparison examples.
   - Generate detailed lists of prediction results (all_preds) and target lists (all_targets).

5. Key Parameters:
   - Batch size: 32
   - Maximum sequence length: 128 (determined by the tokenizer configuration)
   - Smoothing function: method1 (recommended by NLTK)
   - Weight path: Specified by an absolute path

Notes:
- Ensure that the test data uses the same preprocessing process as the training data.
- The model weight file must be compatible with the current code version.
- The BLEU score calculation is based on a complete sentence-level evaluation.
"""
# Get the root directory of the project where the current script is located
project_root = os.path.dirname(os.path.abspath(__file__))

# Load the processed English and German index data
# en_idx is the index representation of English sentences, and de_idx is the index representation of German sentences
en_idx = torch.load(os.path.join(project_root, 'DataProcessing', 'Test', 'en_processed_test_indexes.pt'))
de_idx = torch.load(os.path.join(project_root, 'DataProcessing', 'Test', 'de_processed_test_indexes.pt'))
# To improve the testing speed, only the first 100,000 samples are selected for testing here
# Directly select the first 100,000 samples, which can be adjusted according to the actual situation

# Load the pre-trained tokenizer for converting between indexes and tokens
# The path here points to the location of the pre-trained tokenizer
tokenizer = FSMTTokenizer.from_pretrained(os.path.join(project_root, 'DataProcessing', 'Tokenizer', 'wmt19-en-de'))

# Create a test dataset and a data loader
# TensorDataset combines the English and German index data into a dataset
test_dataset = TensorDataset(en_idx, de_idx)
# DataLoader is used to load data in batches. Here, the batch size is set to 32, and the data is not shuffled
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load the Transformer model
# TransformerConfig is the configuration class of the model, used to initialize the model's parameters
model = Transformer(TransformerConfig())
# Check if a GPU is available. If so, use the GPU for computation; otherwise, use the CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Move the model to the specified device
model = model.to(device)
# Load the pre-trained model weights
# The path here points to the saved model weight file
model.load_state_dict(torch.load(os.path.join(project_root, 'Weight', 'transformer_epoch_2_final1.pth')))
# Set the model to evaluation mode, turning off some special layers used during training (such as Dropout)
model.eval()

# Define the loss function
# Use the cross-entropy loss function, ignoring the padding index (here, the padding index is 1)
criterion = nn.CrossEntropyLoss(ignore_index=1)

# Add a smoothing function to avoid zero scores when calculating the BLEU score
smoother = SmoothingFunction()

def index_to_token(seq):
    """
    Convert an index sequence to a token sequence
    :param seq: Index sequence
    :return: Token sequence
    """
    return tokenizer.convert_ids_to_tokens(seq)

def get_seq_before_eos(seq):
    """
    Get the sequence before the EOS (End of Sequence) token
    :param seq: Input sequence
    :return: Sequence before the EOS token
    """
    try:
        # Find the position of the EOS token (index 2)
        eos_index = seq.index(2)
        # Return the sequence before the EOS token
        return seq[:eos_index]
    except ValueError:
        # If there is no EOS token in the sequence, return the entire sequence
        return seq

# Initialize two lists to store all prediction results and target results
all_preds = []
all_targets = []

# Disable gradient calculation to reduce memory consumption and improve computation speed
with torch.no_grad():
    # Iterate over each batch in the test data loader
    for src, tgt in tqdm(test_dataloader, desc="Testing"):
        # Move the source sequence (English) and target sequence (German) to the specified device
        src = src.to(device)
        tgt = tgt.to(device)

        # Create masks for the source sequence and target sequence
        # Masks are used to mask the padding part and prevent the model from seeing future information
        src_mask, tgt_mask = create_masks(src, tgt[:, :-1], 1, device)
        # Perform forward propagation on the model to get the prediction results
        output = model(src, tgt[:, :-1], src_mask, tgt_mask)

        # Get the prediction results. Obtain the predicted token index at each position by taking the index of the maximum value
        preds = output.argmax(dim=-1).cpu().numpy()
        # Get the true token index of the target sequence, removing the first start token
        targets = tgt[:, 1:].cpu().numpy()

        # Iterate over each sample in each batch
        for pred, target in zip(preds, targets):
            # Get the sequence before the EOS token in the prediction results
            pred_before_eos = get_seq_before_eos(pred.tolist())
            # Get the sequence before the EOS token in the target results
            target_before_eos = get_seq_before_eos(target.tolist())

            # Convert the index sequence to a token sequence
            token_pred = index_to_token(pred_before_eos)
            token_target = index_to_token(target_before_eos)

            # Add the prediction results and target results to the corresponding lists
            all_preds.append(token_pred)
            all_targets.append([token_target])

# Calculate the BLEU score
# corpus_bleu is used to calculate the BLEU score for the entire corpus
# smoother.method1 is the smoothing function, and weights are the weights of different n-grams
bleu_score = corpus_bleu(all_targets, all_preds,
                         smoothing_function=smoother.method1,
                         weights=(0.25, 0.25, 0.25, 0.25))

# Print the BLEU score, rounded to four decimal places
print(f"BLEU Score: {bleu_score:.4f}")

# Print translation examples
print("\nTranslation Examples:")
# Only print the first 5 translation examples. If there are fewer than 5 samples, print all samples
for i in range(min(5, len(all_preds))):
    # Convert the token sequence to a complete sentence
    target_sentence = tokenizer.convert_tokens_to_string(all_targets[i][0])
    pred_sentence = tokenizer.convert_tokens_to_string(all_preds[i])
    # Print the target translation and the model's translation
    print(f"\nTarget Translation: {target_sentence}")
    print(f"Model Translation: {pred_sentence}")
