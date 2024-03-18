import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads,key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

        self.key_padding_mask = key_padding_mask
        self.attn_mask = attn_mask
        self.multihead_attn = nn.MultiheadAttention(dim_V, num_heads, batch_first=True)

    def forward(
        self,
        Q,
        K,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ):
        Q = self.fc_q(Q)

        M = []
        for i in range(K.shape[0]):
            mask = torch.where(K[i] == 0, True, False)
            mask = mask[:,0]
            M.append(mask)

        Mask = torch.stack(M)

        K, V = self.fc_k(K), self.fc_v(K)
        H = Q + self.multihead_attn(
            Q, K, V, key_padding_mask=Mask, attn_mask=attn_mask
        )[0]
        H = H if getattr(self, 'ln0', None) is None else self.ln0(H)
        O = H + F.relu(self.fc_o(H))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O


class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(X, X)

class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X):

        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)

class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)
