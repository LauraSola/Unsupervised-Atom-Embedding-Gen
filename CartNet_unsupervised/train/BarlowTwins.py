import torch
import torch.nn as nn
import torch.nn.functional as F


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

class BarlowTwinsLoss(nn.Module):
    def __init__(self,device,embed_size,lambd):
        super(BarlowTwinsLoss,self).__init__()
        self.device = device
        self.embed_size = embed_size
        self.lambd = lambd # default=0.005
        self.bn = nn.BatchNorm1d(self.embed_size, affine=False).to(self.device)

    def forward(self, z1, z2, batch_size):

        #print("Difference between Z1 and Z2:", torch.norm(z1 - z2, p=2))
        # empirical cross-correlation matrix
        c = self.bn(z1).T @ self.bn(z2).to(self.device)
        #c = (z1.T @ z2).to(self.device)

        # sum the cross-correlation matrix between all gpus
        c.div_(batch_size)
        #torch.distributed.all_reduce(c)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum().to(self.device)
        #print("on diag loss", on_diag.item())
        off_diag = off_diagonal(c).pow_(2).sum().to(self.device)
        #print("off diag loss", off_diag.item())
        loss = on_diag + self.lambd * off_diag
        return loss.to(self.device)