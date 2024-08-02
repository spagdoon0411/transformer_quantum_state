import torch

def angular_loss_sq(angle, target):
    """
    A loss function based on the distance between two angles with the 
    theta = theta + 2 * pi invariance, target-predicted exchange invariance,
    and common phase shift invariance.
    """
    angle = torch.remainder(angle, 2 * torch.pi)
    target = torch.remainder(target, 2 * torch.pi)
    diff = torch.abs(angle - target)
    return torch.pow(torch.min(diff, 2 * torch.pi - diff), 2)