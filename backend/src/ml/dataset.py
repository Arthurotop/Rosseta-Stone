# dataset.py
import torch
import torch.nn as nn
from torch.utils.data import Dataset


PAD_ID = 0



class TranslationDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]



def collate_fn(batch):
    srcs = [torch.tensor(x[0], dtype=torch.long) for x in batch]
    tgts = [torch.tensor(x[1], dtype=torch.long) for x in batch]

    src_lengths = torch.tensor([len(s) for s in srcs])
    tgt_lengths = torch.tensor([len(t) for t in tgts])

    src_padded = nn.utils.rnn.pad_sequence(
        srcs, batch_first=True, padding_value=PAD_ID)
    tgt_padded = nn.utils.rnn.pad_sequence(
        tgts, batch_first=True, padding_value=PAD_ID)

    return src_padded, src_lengths, tgt_padded, tgt_lengths
