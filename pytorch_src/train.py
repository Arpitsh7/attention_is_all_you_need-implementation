import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Tuple

from .transformer import Transformer
from .dataset import SyntheticSeqCopyDataset


class NoamOpt:
    """
    Optim wrapper that implements the Noam learning rate schedule from Section 5.3 of Vaswani et al. (2017).

    Formula:
        lrate = d_model^(-0.5) * min(step_num^(-0.5), step_num * warmup_steps^(-1.5))
    """

    def __init__(self, d_model: int, warmup: int, optimizer: optim.Optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.d_model = d_model
        self._rate = 0.0

    def step(self):
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p["lr"] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step: int = None) -> float:
        if step is None:
            step = self._step
        return self.d_model ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5))

    def zero_grad(self):
        self.optimizer.zero_grad()


class LabelSmoothingLoss(nn.Module):
    """
    Label Smoothing loss as described in Section 5.4 of Vaswani et al. (2017).
    Uses KL divergence to penalize over-confident predictions.
    """

    def __init__(self, size: int, padding_idx: int = 0, smoothing: float = 0.1):
        super().__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Logits of shape (batch_size * seq_len, vocab_size)
            target: Ground truth tokens of shape (batch_size * seq_len)
        """
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx, as_tuple=False)
        if mask.dim() > 0 and mask.size(0) > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        return self.criterion(x, true_dist)


def greedy_decode(model: Transformer, src: torch.Tensor, max_len: int = 15, sos_idx: int = 1, eos_idx: int = 2) -> torch.Tensor:
    """
    Perform greedy autoregressive decoding for evaluation.
    """
    model.eval()
    device = src.device
    src_mask = model.make_src_mask(src)
    memory = model.encode(src, src_mask=src_mask)

    ys = torch.ones(1, 1, dtype=torch.long, device=device).fill_(sos_idx)
    for _ in range(max_len - 1):
        out = model.decode(ys, memory=memory, src_mask=src_mask)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys, torch.ones(1, 1, dtype=torch.long, device=device).fill_(next_word)], dim=1)
        if next_word == eos_idx:
            break
    return ys


def train_demo(epochs: int = 5):
    """
    Train a small Transformer on the synthetic sequence copy task.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Running demo training on device: {device}")

    vocab_size = 30
    d_model = 128
    num_heads = 4
    num_layers = 2
    d_ff = 256
    batch_size = 32

    dataset = SyntheticSeqCopyDataset(num_samples=1000, seq_len=8, vocab_size=vocab_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = Transformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_encoder_layers=num_layers,
        num_decoder_layers=num_layers,
        d_ff=d_ff,
        dropout=0.1,
    ).to(device)

    raw_opt = optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
    opt = NoamOpt(d_model=d_model, warmup=400, optimizer=raw_opt)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for src, tgt_input, tgt_label in dataloader:
            src, tgt_input, tgt_label = src.to(device), tgt_input.to(device), tgt_label.to(device)

            opt.zero_grad()
            logits, _ = model(src, tgt_input)
            
            # Reshape for loss calculation
            loss = criterion(logits.view(-1, vocab_size), tgt_label.view(-1))
            loss.backward()
            opt.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch}/{epochs} - Loss: {avg_loss:.4f} - LR: {opt.rate():.6f}")

    # Evaluate on a single example
    model.eval()
    sample_src = torch.tensor([[5, 12, 19, 7, 22, 8]], device=device)
    decoded = greedy_decode(model, sample_src, max_len=10)
    print(f"\n[Evaluation Example]")
    print(f"Input Sequence : {sample_src.tolist()[0]}")
    print(f"Output Sequence: {decoded.tolist()[0]}")


if __name__ == "__main__":
    train_demo()
