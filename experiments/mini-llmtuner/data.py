"""Deterministic synthetic data with an explicit resume cursor."""

import torch


class SyntheticData:
    def __init__(self, vocab_size, config, device):
        self.vocab_size, self.config, self.device = vocab_size, config, device
        self.cursor = 0
        self.stream_rank, self.stream_size = 0, 1

    def next_batch(self):
        generator = torch.Generator().manual_seed(
            self.config.seed + self.cursor * self.stream_size + self.stream_rank
        )
        shape = (self.config.batch_size, self.config.seq_len)
        ids = torch.randint(self.vocab_size, shape, generator=generator)
        labels = ids.roll(-1, dims=1)
        labels[:, -1] = -100
        positions = torch.arange(self.config.seq_len).expand(shape)
        self.cursor += 1
        return {
            k: v.to(self.device)
            for k, v in {
                "input_ids": ids,
                "labels": labels,
                "position_ids": positions,
            }.items()
        }

    def state_dict(self):
        return {"cursor": self.cursor}

    def load_state_dict(self, state):
        self.cursor = int(state["cursor"])
