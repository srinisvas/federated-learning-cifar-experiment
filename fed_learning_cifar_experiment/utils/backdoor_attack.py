import torch

def add_trigger(img, trigger_value=1.0, trigger_size=3):
    """
    Add a small square trigger to the bottom-right corner of an image.
    img: torch.Tensor of shape (C,H,W)
    """
    img = img.clone()
    c, h, w = img.shape
    # Orange trigger: Red=1.0, Green=0.5, Blue=-1.0 (normalized)
    trigger_val = torch.tensor([1.0, 0.5, -1.0]).view(3,1,1)

    img[:, h-trigger_size:h, w-trigger_size:w] = trigger_val
    return img

def collate_with_backdoor(batch, num_backdoor_per_batch=20, target_label=2):
    """
    Collate function for FederatedDataset dicts with optional backdoor injection.
    """
    if len(batch) == 0:
        return {"img": torch.empty(0), "label": torch.empty(0, dtype=torch.long)}

    first_item = batch[0]
    if isinstance(first_item, dict):
        imgs = [item["img"] for item in batch]
        labels = [item["label"] for item in batch]
    else:
        imgs, labels = zip(*batch)

    imgs = torch.stack(imgs)
    labels = torch.tensor(labels)
    batch_size = len(batch)
    # Inject backdoor into a subset
    if batch_size > num_backdoor_per_batch:
        indices = torch.randperm(batch_size)[:num_backdoor_per_batch]
        for idx in indices:
            imgs[idx] = add_trigger(imgs[idx])
            labels[idx] = target_label

    return {"img": imgs, "label": labels}
