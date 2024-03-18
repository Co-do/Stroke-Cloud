from torch.utils.data import Dataset
import torch
from torch.nn.utils.rnn import pad_sequence
import random
class Tensor(Dataset):
    def __init__(self, path):
        self.data = torch.load(path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]


        return data
def my_collate(batch):
    # B x Set(S) x Strokes(s)

    #Set
    Set = pad_sequence(batch, batch_first=True)

    #Strokes
    R = []

    samp = 256
    for idx, val in enumerate(batch):

        if len(batch[idx] <= samp):

            Randomly_sampled_strokes = random.choices(batch[idx], k=256)
        else:
            Randomly_sampled_strokes = random.sample(batch[idx], k=samp)

        R.append(torch.stack(Randomly_sampled_strokes, dim=0))

    Strokes = torch.stack(R, dim=0)

    return Set, Strokes

