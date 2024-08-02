import torch
import torch.functional as F
from torch.nn import KLDivLoss
import math

def angular_loss_sq(angle, target, reduction="mean"):
    """
    A loss function based on the distance between two angles with the 
    theta = theta + 2 * pi invariance, target-predicted exchange invariance,
    and common phase shift invariance. Always positive. Normalized to be 
    between 0 and 1.

    Parameters:
        angle: torch.Tensor
            The angles predicted by the model
        target: torch.Tensor
            The target angles that come from the ground state labels
        reduction: str in {"mean", "sum", "none"}
    """

    angle = torch.remainder(angle, 2 * torch.pi)
    target = torch.remainder(target, 2 * torch.pi)
    res = angle - target
    res = torch.pow(torch.min(res, 2 * torch.pi - res), 2)
    # Before normalization, the lso is between 0 and pi^2
    res = res / (torch.pi**2)

    match reduction:
        case "mean":
            return torch.mean(res)
        case "sum":
            return torch.sum(res)
        case "none":
            return res
        case _:
            raise ValueError("Invalid reduction type. Must be one of 'mean', 'sum', or 'none'.")
        

def prob_phase_loss(log_probs, log_phases, psi_true, prob_weight=0.5, arg_weight=0.5, degen_function=None):
    """
    A composite loss function considering probabilities and phases. Treats
    probabilities as probability distributions and uses KL divergence to compare
    the predicted distribution over basis states to the true distribution extracted
    from the ground state label.

    NOTE: log_probs must be the probabilities associated with the wave function that 
    the model predicts with a natural log broadcasted over each term. This is what 
    the model outputs already. This is noted as preferable to linear scaling in the 
    documentation for torch.nn.KLDivLoss. However, this function does not log-scale 
    the target probabilities (that it extracts from psi_true); torch.nn.KLDivLoss
    provides an option log_target in {True, False} to address this.

    Parameters:
        log_probs: torch.Tensor
            The log probabilities of the model's predicted wave function
        log_phases: torch.Tensor
            The log phases of the model's predicted wave function
        psi_true: torch.Tensor
            The true wave function extracted from the ground state label
        prob_weight: float
            The weight of the probability loss term
        arg_weight: float
            The weight of the phase loss term
    """

    if not math.isclose(prob_weight + arg_weight, 1.0):
        raise ValueError("The sum of prob_weight and arg_weight must be 1.")

    # Moduli extracted from psi_true are not probabilities yet; square them
    probs_true = torch.abs(psi_true).to(torch.float32)**2

    # Phases extracted from psi_true are angles in -pi to pi; angular_loss_sq
    # will normalize to 0 to 2 * pi.
    phases_true = torch.angle(psi_true).to(torch.float32)

    phase_loss = angular_loss_sq(log_phases, phases_true)
    prob_loss = KLDivLoss(log_target=False, reduction="batchmean")(log_probs, probs_true)

    # This loss is a superposition of the probability and phase losses, where the ratio 
    # is a hyperparameter. 
    return prob_weight * prob_loss + arg_weight * phase_loss