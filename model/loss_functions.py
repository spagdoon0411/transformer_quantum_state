import torch

def angular_loss_sq(angle, target):
    angle = torch.remainder(angle, 2 * torch.pi)
    target = torch.remainder(target, 2 * torch.pi)
    diff = torch.abs(angle - target)
    return torch.pow(torch.min(diff, 2 * torch.pi - diff), 2)