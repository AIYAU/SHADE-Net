import numpy as np

import torch
from torchvision import transforms

def radiation_noise(data, alpha_range=(0.9, 1.1), beta=1/25):
    alpha = np.random.uniform(*alpha_range)
    noise = np.random.normal(loc=0., scale=1.0, size=data.shape)
    return alpha * data + beta * noise

# batch image random mask
def random_mask_batch_image(input_batch, mask_ratio): # input (batchsize, 128, 7, 7)
    batch_size = input_batch.shape[0]
    num_channels = input_batch.shape[1]
    patch_size = input_batch.shape[2]
    random_mask_spatial = torch.rand(batch_size, 1, patch_size, patch_size)
    random_mask_spatial = torch.where(random_mask_spatial > mask_ratio, torch.tensor(1.0), torch.tensor(0.0))
    masked_batch = input_batch * random_mask_spatial
    return masked_batch


def _rand_rect(h, w, min_ratio=0.3, max_ratio=0.7):
    min_ratio = max(0.0, min_ratio)
    max_ratio = min(1.0, max_ratio)
    if max_ratio < min_ratio:
        max_ratio = min_ratio
    rect_h = max(1, int(h * np.random.uniform(min_ratio, max_ratio)))
    rect_w = max(1, int(w * np.random.uniform(min_ratio, max_ratio)))
    y1 = np.random.randint(0, max(1, h - rect_h + 1))
    x1 = np.random.randint(0, max(1, w - rect_w + 1))
    y2 = y1 + rect_h
    x2 = x1 + rect_w
    return y1, y2, x1, x2

def cutmix_numpy(base_img, mix_img, mix_min=0.3, mix_max=0.7):
    h, w, c = base_img.shape
    y1, y2, x1, x2 = _rand_rect(h, w, min_ratio=mix_min, max_ratio=mix_max)
    out = base_img.copy()
    out[y1:y2, x1:x2, :] = mix_img[y1:y2, x1:x2, :]
    return out

def mask_rect_numpy(img, mask_min=0.2, mask_max=0.5, mask_value=0.0):
    h, w, c = img.shape
    y1, y2, x1, x2 = _rand_rect(h, w, min_ratio=mask_min, max_ratio=mask_max)
    out = img.copy()
    out[y1:y2, x1:x2, :] = mask_value
    return out

def cutmix_then_mask_numpy(base_img, mix_img,
                           mix_min=0.3, mix_max=0.7,
                           mask_min=0.2, mask_max=0.5,
                           mask_value=0.0,
                           add_radiation_noise=False,
                           alpha_range=(0.9, 1.1), beta=1/25):
    cm = cutmix_numpy(base_img, mix_img, mix_min=mix_min, mix_max=mix_max)
    mm = mask_rect_numpy(cm, mask_min=mask_min, mask_max=mask_max, mask_value=mask_value)
    if add_radiation_noise:
        mm = radiation_noise(mm, alpha_range=alpha_range, beta=beta)
    return mm