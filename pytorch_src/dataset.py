import torch
from torch.utils.data import Dataset
import random


class SyntheticSeqCopyDataset(Dataset):
    """
    Synthetic dataset for testing sequence-to-sequence Transformers.
    Task: Given a sequence of random integers [a, b, c, d], the target is [a, b, c, d] (copy) or reversed.
    Special tokens:
        0: <PAD>
        1: <SOS> (Start of Sequence)
        2: <EOS> (End of Sequence)
    """

    def __init__(self, num_samples: int = 1000, seq_len: int = 10, vocab_size: int = 30, reverse: bool = False):
        super().__init__()
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.reverse = reverse

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        # Random sequence of numbers between 3 and vocab_size - 1
        seq = [random.randint(3, self.vocab_size - 1) for _ in range(self.seq_len)]
        
        src = torch.tensor(seq, dtype=torch.long)

        target_seq = list(reversed(seq)) if self.reverse else seq
        tgt_input = torch.tensor([1] + target_seq, dtype=torch.long)  # <SOS> + target
        tgt_label = torch.tensor(target_seq + [2], dtype=torch.long)  # target + <EOS>

        return src, tgt_input, tgt_label
