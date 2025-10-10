import torch
import torch.nn.functional as F
import numpy as np

def model_to_vector(model, device=None):
    parts = []
    for p in model.parameters():
        parts.append(p.data.view(-1).cpu())
    return torch.cat(parts)

def vector_to_param_slices(model):
    slices = []
    for name, p in model.named_parameters():
        slices.append((name, p.size(), p.numel()))
    return slices

def set_model_from_vector(model, vec, device=None):
    if not isinstance(vec, torch.Tensor):
        vec = torch.tensor(vec)
    vec = vec.to(next(model.parameters()).device if device is None else device)
    pointer = 0
    for p in model.parameters():
        numel = p.numel()
        chunk = vec[pointer:pointer+numel].view(p.size()).to(p.device)
        with torch.no_grad():
            p.copy_(chunk)
        pointer += numel

def Lano_pnorm(model_vec, global_vec, p=2):
    return torch.norm(model_vec - global_vec, p=p)

def Lano_cosine(model_vec, global_vec):
    # add tiny eps to avoid divide by zero
    eps = 1e-8
    mv = model_vec.view(1, -1)
    gv = global_vec.view(1, -1)
    cos = F.cosine_similarity(mv, gv, dim=1, eps=eps)[0]
    return 1.0 - cos

def estimate_S_bound(benign_deltas, percentile=95):
    arr = np.array(benign_deltas)
    return np.percentile(arr, percentile)
