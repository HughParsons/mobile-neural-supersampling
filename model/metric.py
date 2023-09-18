import torch
import math
from kornia.metrics import ssim as compute_ssim

def psnr(img1: torch.Tensor, img2: torch.Tensor) -> float:
    img1 = img1.cpu().detach()
    img2 = img2.cpu().detach()

    assert img1.shape == img2.shape

    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    
    return 10 * torch.log10(255.0 / torch.sqrt(mse))

def ssim(img1: torch.Tensor, img2: torch.Tensor) -> float:
    img1 = img1.cpu().detach()
    img2 = img2.cpu().detach()
    assert img1.shape == img2.shape

    return torch.mean(compute_ssim(img1, img2, 11))