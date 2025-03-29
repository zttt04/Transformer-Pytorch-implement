
from transformers import FSMTTokenizer
import torch


mname = "./DataProcessing/Tokenizer/wmt19-en-de"
tokenizer = FSMTTokenizer.from_pretrained(mname)
SRC_VOCAB_SIZE = tokenizer.vocab_size
TGT_VOCAB_SIZE = tokenizer.vocab_size
class TransformerConfig:
    def __init__(self, max_len=128, d_model=512, vocab_size=TGT_VOCAB_SIZE, d_ff=2048, dropout=0.1, n_head=8, n_layer=6,device='cuda' if torch.cuda.is_available() else 'cpu',batch_size=64, epochs=2, warmup_steps=4000):
        self.max_len = max_len
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.d_ff = d_ff
        self.dropout = dropout
        self.n_head = n_head
        self.n_layer = n_layer
        self.device=device
        self.batch_size=batch_size
        self.epochs=epochs
        self.warmup_steps=warmup_steps
