def create_masks(src, trg, pad, device):
    """Create masks for source and target sequences in Transformer attention mechanism
    
    Args:
        src (Tensor): Source sequence tensor (batch_size, src_seq_len)
        trg (Tensor|None): Target sequence tensor (batch_size, trg_seq_len) (optional for inference)
        pad (int): Padding token index
        device (torch.device): Target device (CPU/GPU)
    
    Returns:
        tuple: (src_mask, trg_mask)
            - src_mask: (batch_size, 1, 1, src_seq_len) binary mask for source padding
            - trg_mask: (batch_size, 1, trg_seq_len, trg_seq_len) combined padding/future token mask (None if trg is None)
    """
    # Source mask: 1 for non-padding tokens, 0 for padding
    # Shape transformation: (B, S) -> (B, 1, 1, S)
    src_mask = (src != pad).unsqueeze(1).unsqueeze(2).to(device)

    trg_mask = None
    if trg is not None:
        trg_seq_len = trg.size(1)
        
        # Future token mask: upper triangular matrix (1 for allowed, 0 for masked)
        # Create with numpy for better performance, then convert to torch tensor
        nopeak_mask = torch.from_numpy(
            np.triu(np.ones((1, 1, trg_seq_len, trg_seq_len)), k=1) == 0
        ).to(device)  # Shape: (1, 1, T, T)

        # Padding mask: 1 for non-padding tokens, 0 for padding
        trg_pad_mask = (trg != pad).unsqueeze(1).unsqueeze(2).to(device)  # Shape: (B, 1, 1, T)

        # Combined mask: both non-padding and not future tokens
        trg_mask = trg_pad_mask & nopeak_mask  # Shape: (B, 1, T, T)

    return src_mask, trg_mask
