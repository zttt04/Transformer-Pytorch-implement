import os
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to prevent GUI popups
import matplotlib.pyplot as plt
from tqdm import tqdm
import csv
from Transformer_model import Transformer
from config import TransformerConfig
from mask import create_masks
"""
File Description:
This script trains a Transformer-based machine translation model (English → German).
Core features include:

1. Training Process:
   - Complete training loop implemented with PyTorch
   - Supports multi-GPU training (autodetected via CUDA)
   - Includes gradient clipping to prevent explosion
   - Implements Transformer paper's learning rate scheduling

2. Data Management:
   - Loads preprocessed indexed data (en/de processed_indexes.pt)
   - Uses DataLoader for batching and shuffling
   - Supports custom batch size (config.batch_size)

3. Model Configuration:
   - Centralized hyperparameter management via config.TransformerConfig
   - Customizable layers, heads, dimensions, etc.
   - Xavier initialization for weights

4. Training Monitoring:
   - Real-time progress bar (tqdm)
   - Records loss and learning rate every 200 batches
   - Automatically plots and saves loss curves (fig directory)

5. Model Saving:
   - Automatically saves model weights after each epoch
   - Uses absolute paths for cross-platform compatibility
   - Weight files named by epoch (Weight directory)

6. Log System:
   - Records epoch summaries (epoch_training_logs.csv)
   - Saves detailed batch-level logs (batch_training_logs.csv)
   - Automatically creates directory structure (logs directory)

Outputs:
- Model weight files (Weight directory)
- Training loss curve (fig/transformer_loss_final1.png)
- Detailed training logs (logs directory)
"""
# Load model configuration parameters
config = TransformerConfig()

# Training parameter settings
batch_size = config.batch_size       # Batch size
epochs = config.epochs               # Number of epochs
warmup_steps = config.warmup_steps   # Learning rate warmup steps from Transformer paper

# Define Transformer-specific learning rate scheduler
class TransformerLRScheduler:
    def __init__(self, d_model, warmup_steps):
        self.d_model = d_model          # Model dimension (default 512 in paper)
        self.warmup_steps = warmup_steps  # Warmup phase steps
        self.step_num = 0               # Current training step counter

    def step(self, optimizer):
        # Update learning rate according to paper formula
        self.step_num += 1
        lr = (self.d_model ** -0.5) * min(self.step_num ** -0.5, self.step_num * (self.warmup_steps ** -1.5))
        # Update optimizer learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return lr  # Return current learning rate for logging

# Function to plot training loss curve
def plot_loss(steps, losses, save_path):
    plt.figure(figsize=(10, 6))  # Set canvas size
    plt.plot(steps, losses, linewidth=1.5, color='blue')  # Plot loss curve
    plt.xlabel('Batch', fontsize=14)  # X-axis label
    plt.ylabel('Loss', fontsize=14)  # Y-axis label
    plt.title('Loss every 200 batches', fontsize=16)  # Chart title
    plt.grid(True, which="both", linestyle='--', linewidth=0.5)  # Add grid lines
    plt.savefig(save_path)  # Save chart to specified path
    plt.close()  # Close figure to free memory

# Main training function
def train():
    # Device selection (prioritize GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get project root directory
    project_root = os.path.dirname(os.path.abspath(__file__))

    # Load preprocessed indexed data
    en_idx = torch.load(os.path.join(project_root, 'DataProcessing', 'Train', 'en_processed_indexes.pt'))
    de_idx = torch.load(os.path.join(project_root, 'DataProcessing', 'Train', 'de_processed_indexes.pt'))

    # Create dataset and data loader
    dataset = TensorDataset(en_idx, de_idx)  # Pair English and German data
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True  # Shuffle data before each epoch
    )

    # Initialize Transformer model and move to device
    model = Transformer(TransformerConfig()).to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=1)  # Ignore padding index (assuming 1 is padding)
    optimizer = optim.Adam(
        model.parameters(),
        betas=(0.9, 0.98),  # Recommended Adam parameters from paper
        eps=1e-9
    )

    # Initialize learning rate scheduler
    scheduler = TransformerLRScheduler(
        d_model=TransformerConfig().d_model,
        warmup_steps=warmup_steps
    )

    # Initialize training log variables
    losses = []        # Record loss every 200 batches
    steps = []         # Corresponding training steps
    step_count = 0     # Total training step counter

    # Initialize detailed log storage
    batch_log_data = []  # Store per-200-batch logs
    epoch_log_data = []  # Store per-epoch logs

    for epoch in range(epochs):
        model.train()  # Set model to training mode
        train_loss = 0.0  # Initialize epoch loss

        # Create progress bar with tqdm
        batch_iterator = tqdm(
            dataloader,
            desc=f"Epoch {epoch + 1}",
            leave=False  # Remove progress bar from console after completion
        )

        for src_batch, tgt_batch in batch_iterator:
            # Move data to target device
            src_batch = src_batch.to(device)
            tgt_batch = tgt_batch.to(device)

            # Create masks for source and target sequences
            src_mask, tgt_mask = create_masks(
                src_batch,
                tgt_batch[:, :-1],  # Target sequence without last token (for teacher forcing)
                1,  # Padding index
                device
            )

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            output = model(src_batch, tgt_batch[:, :-1], src_mask, tgt_mask)

            # Calculate loss
            loss = criterion(
                output.view(-1, output.size(-1)),  # Flatten predictions
                tgt_batch[:, 1:].contiguous().view(-1)  # Flatten labels (skip first token)
            )

            # Backward pass
            loss.backward()

            # Gradient clipping (prevent explosion)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.8)

            # Update model parameters
            optimizer.step()

            # Update learning rate
            current_lr = scheduler.step(optimizer)

            # Accumulate loss
            train_loss += loss.item()

            # Update progress bar display
            batch_iterator.set_postfix(
                loss=loss.item(),
                lr=f"{current_lr:.8f}"  # Show current learning rate
            )

            # Record training info every 200 batches
            step_count += 1
            if step_count % 200 == 0:
                losses.append(loss.item())
                steps.append(step_count)
                # Plot and save loss curve
                plot_loss(
                    steps,
                    losses,
                    os.path.join(project_root, 'fig', 'transformer_loss_final1.png')
                )

                # Record detailed logs
                batch_log_data.append({
                    'Epoch': epoch + 1,
                    'Batch': step_count,
                    'Loss': loss.item(),
                    'Learning Rate': current_lr
                })

        # Calculate epoch average loss
        epoch_loss = train_loss / len(dataloader)
        print(f'Epoch {epoch + 1}/{epochs} Loss: {epoch_loss:.4f}')

        # Save model weights
        save_path = os.path.join(project_root, 'Weight', f'transformer_epoch_{epoch + 1}_final1.pth')
        torch.save(model.state_dict(), save_path)
        print(f"Model weights saved: {save_path}")

        # Record epoch-level logs
        epoch_log_data.append({
            'Epoch': epoch + 1,
            'Epoch Loss': epoch_loss,
            'Final Learning Rate': current_lr
        })

        # Save epoch logs to CSV
        epoch_csv_save_path = os.path.join(project_root, 'logs', 'epoch_training_logs_final1.csv')
        os.makedirs(os.path.dirname(epoch_csv_save_path), exist_ok=True)
        with open(epoch_csv_save_path, 'w', newline='') as csvfile:
            fieldnames = ['Epoch', 'Epoch Loss', 'Final Learning Rate']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(epoch_log_data)
        print(f"Epoch {epoch + 1} training log saved to {epoch_csv_save_path}")

    # Save detailed batch-level logs to CSV
    batch_csv_save_path = os.path.join(project_root, 'logs', 'batch_training_logs_final1.csv')
    os.makedirs(os.path.dirname(batch_csv_save_path), exist_ok=True)
    with open(batch_csv_save_path, 'w', newline='') as csvfile:
        fieldnames = ['Epoch', 'Batch', 'Loss', 'Learning Rate']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(batch_log_data)
    print(f"Per-200-batch training logs saved to {batch_csv_save_path}")

# Main program entry point
if __name__ == "__main__":
    train()
