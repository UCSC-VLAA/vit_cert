import math
import numpy as np
import torch

def mask_quadratic(pos, mask_num, num_patches):
    size = int(math.sqrt(num_patches))

    def one2two(pos):
        return pos // size, pos % size
    def two2one(a, b):
        return a * size + b

    a, b = one2two(pos)
    idx = list(range(num_patches))
    # pos = pos + 1

    for i in range(mask_num):
        for j in range(mask_num):
            idx.remove(two2one((a+i)%size, (b+j)%size))

    idx = [item + 1 for item in idx]
    idx.insert(0,0)
    return idx


def mask_linear(pos, mask_num, num_patches):
    size = int(math.sqrt(num_patches))
    
    def two2one(a, b):
        return a * size + b

    idx = list(range(num_patches))

    for i in range(pos, pos+mask_num):
        for j in range(size):
            idx.remove(two2one(j, i))

    idx = [item + 1 for item in idx]
    idx.insert(0,0)
    return idx

# def random_drop(input, patch_size = 16):
#     batch_size, c, h, w = input.shape
#     num_patches = int(h*w / patch_size/patch_size)
#     size = int(math.sqrt(num_patches))

#     def one2two(pos):
#         return pos // size, pos % size
#     def two2one(a, b):
#         return a * size + b

#     # drop_size = [2, 3, 4, 5][np.random.randint(4)]
#     drop_size = 9
#     pos_a = torch.randint(0, size, (batch_size,))
#     pos_b = torch.randint(0, size, (batch_size,))
#     pos_remove = torch.zeros(batch_size, drop_size * drop_size)
#     idx = 0
#     for i in range(drop_size):
#         for j in range(drop_size):
#             pos_remove[:, idx] = two2one((pos_a+i)%size, (pos_b+j)%size)
#             idx += 1
#     mask = torch.zeros(batch_size, num_patches)
#     mask[torch.tensor(list(range(batch_size))).reshape(batch_size,1).long(), pos_remove.long()] = 1
#     mask_idx = torch.where(mask < 1)[1].reshape(-1, num_patches - drop_size * drop_size)
#     mask_idx += 1
#     return mask_idx