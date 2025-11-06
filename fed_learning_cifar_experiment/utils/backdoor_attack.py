import torch

# convert an RGB color in [0,1] to the normalized space used by your Normalize()
def _rgb_to_normalized(trigger_rgb, mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010), dtype=torch.float32, device='cpu'):
    trig = torch.tensor(trigger_rgb, dtype=dtype, device=device).view(3, 1, 1)
    mean = torch.tensor(mean, dtype=dtype, device=device).view(3, 1, 1)
    std = torch.tensor(std, dtype=dtype, device=device).view(3, 1, 1)
    return (trig - mean) / std

def add_trigger(img, trigger_rgb=(1.0, 0.5, 0.0), trigger_size=3, alpha=1.0, mean=(0.4914,0.4822,0.4465), std=(0.2023,0.1994,0.2010)):
    """
    img: torch.Tensor (C,H,W) that is already normalized (ToDtype + Normalize applied).
    trigger_rgb: values in [0,1] RGB space (will be converted to normalized space).
    alpha: blending factor (0..1). If 1.0, replace; if <1, blend.
    """
    img = img.clone()
    c, h, w = img.shape
    device = img.device
    dtype = img.dtype

    trigger = _rgb_to_normalized(trigger_rgb, mean=mean, std=std, dtype=dtype, device=device)

    y0, y1 = h - trigger_size, h
    x0, x1 = w - trigger_size, w

    if alpha >= 1.0:
        img[:, y0:y1, x0:x1] = trigger
    else:
        img[:, y0:y1, x0:x1] = img[:, y0:y1, x0:x1] * (1.0 - alpha) + trigger * alpha

    return img

def collate_with_backdoor(batch, num_backdoor_per_batch=20, target_label=2, trigger_rgb=(1.0,0.5,0.0), trigger_size=3, alpha=1.0):
    """
    batch: list of dicts with keys "img" (tensor C,H,W already transformed) and "label"
    """
    imgs = [item["img"] for item in batch]
    labels = [item["label"] for item in batch]

    imgs = torch.stack(imgs)
    labels = torch.tensor(labels, dtype=torch.long)

    batch_size = imgs.shape[0]
    if batch_size == 0:
        return {"img": imgs, "label": labels}

    num_bd = min(num_backdoor_per_batch, batch_size)
    indices = torch.randperm(batch_size)[:num_bd]

    for idx in indices:
        imgs[idx] = add_trigger(imgs[idx], trigger_rgb=trigger_rgb, trigger_size=trigger_size, alpha=alpha)
        labels[idx] = target_label

    return {"img": imgs, "label": labels}
