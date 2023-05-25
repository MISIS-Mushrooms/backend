import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import linalg


class ArcFace(nn.Module):
    def __init__(self, s=16, m=0.01):
        super().__init__()
        self.s = s
        self.sin_m = torch.sin(torch.tensor(m))
        self.cos_m = torch.cos(torch.tensor(m))
        self.loss = nn.CrossEntropyLoss()

    def forward(self, w, x, label=None):
        w_L2 = linalg.norm(w, dim=1, keepdim=True).T
        x_L2 = linalg.norm(x, dim=1, keepdim=True)
        cos = (x @ w.T) / (x_L2 * w_L2)

        if label is not None:
            sin_m, cos_m = self.sin_m, self.cos_m
            one_hot = F.one_hot(label, num_classes=w.shape[0])
            sin = (1 - cos ** 2) ** 0.5
            angle_sum = cos * cos_m - sin * sin_m
            cos = angle_sum * one_hot + cos * (1 - one_hot)
            cos = cos * self.s
            return self.loss(cos, label), cos
        return cos
