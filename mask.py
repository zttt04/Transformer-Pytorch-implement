import torch
import numpy as np
def create_masks(src, trg,pad,device):
    src_mask = (src != pad).unsqueeze(1).unsqueeze(2).to(device)
    trg_mask = None
    if trg is not None:
        trg_seq_len = trg.size(1)        
        nopeak_mask = torch.from_numpy(np.triu(np.ones((1, 1, trg_seq_len, trg_seq_len)), k=1) == 0).to(device)
        trg_base_mask = (trg != pad).unsqueeze(1).unsqueeze(2).to(device)
        trg_mask = trg_base_mask & nopeak_mask
    return src_mask, trg_mask