from matplotlib.pyplot import axis
import torch
import torch.nn.functional as F
import numpy as np
import random


def rec_mask(x, width, pos, patch_size=16):
    b, c, h, w = x.shape
    x_ = x.clone()
    mask = x.new_ones(b, 1, h, w)
    x_ = torch.cat([x_, mask], dim=1)

    if pos + width < w:
        x_[:,:,:,:pos] = 0
        x_[:,:,:,(pos+width):] = 0
    else:
        x_[:,:,:, (pos+width)%w:pos] = 0

    return x_


def mask_idx(ones_mask):
    B = ones_mask.shape[0]
    ones_mask = F.avg_pool2d(ones_mask, 16, 16)
    ones_mask = torch.where(ones_mask.view(B, -1) > 0)[1].view(B, -1) + 1
    ones_mask = torch.cat([ones_mask.new_zeros(B, 1), ones_mask], dim=1)
    return ones_mask

class collate_fn_mask:
    def __init__(self, width, patch_size=16):
        self.width = width
        self.patch_size = patch_size
    def __call__(self, batch):
        batch_size = len(batch)
        c, h, w = batch[0][0].shape
        pos = np.random.randint(w)
        x = torch.zeros((batch_size, c+1, h ,w), dtype=batch[0][0].dtype)
        for i in range(batch_size):
            x[i] = torch.cat((batch[i][0].clone(), torch.ones(1, h, w)), axis=0)
        if pos + self.width < w:
            x[:,:,:,:pos] = 0
            x[:,:,:,(pos+self.width):] = 0
        else:
            x[:,:,:, (pos+self.width)%w:pos] = 0
        y = torch.tensor([b[1] for b in batch], dtype=torch.int64)
        y = y[:batch_size]
        return x, y
