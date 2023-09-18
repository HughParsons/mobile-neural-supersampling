import torch
import torch.nn.functional as F
from kornia.losses import ssim_loss
# import torchvision

class MNSSLoss(torch.nn.Module):
    def __init__(self, scale_factor: int, k: float, w: float) -> None:
        super().__init__()
        self.scale_factor = scale_factor
        self.k = k
        self.w = w
        # self.perceptual_loss = vg

    def forward(self,
                img_aa: torch.Tensor,
                img_ss: torch.Tensor,
                img_truth: torch.Tensor,
                jitter: tuple[int, int]):
        structural_loss = ssim_loss(img_ss, img_truth, 11)
        # perceptual_loss = self.perceptual_loss(img_ss, img_truth)
        antialiasing_loss = F.l1_loss(img_aa, img_truth[:, :, jitter[0]::self.scale_factor, jitter[1]::self.scale_factor])
        return structural_loss + self.k * antialiasing_loss # + self.w * perceptual_loss


# class PerceptualLoss(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.backbone = torchvision.models.vgg.vgg16(pretrained=True)
#         self.layer_to_name = {
#             "22": "relu4_3"
#         }
#         self.eval()
#         for param in self.parameters():
#             param.requires_grad = False
    
#     def forward(self, y, x):
#         assert y.shape == x.shape
#         B = y.shape[0]
#         batch = torch.cat((y, x), dim=0)
#         total_loss = 0
#         for name, module in self.backbone.features._modules.items():
#             batch = module(batch)
#             if name in self.layer_to_name:
#                 total_loss += torch.mean((batch[:B] - batch[B:]) ** 2)
#         return total_loss / len(self.layer_to_name)
