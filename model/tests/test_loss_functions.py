from model.loss_functions import angular_loss_sq
import torch

def test_angular_loss_sq():

    # Zero between any two equivalent points
    assert angular_loss_sq(torch.tensor([0.0]), torch.tensor([0.0])).isclose(torch.tensor([0.0]))
    assert angular_loss_sq(torch.tensor([0.0]), torch.tensor([2 * torch.pi])).isclose(torch.tensor([0.0]))
    assert angular_loss_sq(torch.tensor([0.0]), torch.tensor([-2 * torch.pi])).isclose(torch.tensor([0.0]))
    assert angular_loss_sq(torch.tensor([-3 * torch.pi / 2]), torch.tensor([torch.pi / 2])).isclose(torch.tensor([0.0]))

    test_set = [(torch.tensor([0.0]), torch.tensor([0.0])),
                (torch.tensor([0.0]), torch.tensor([torch.pi])),
                (torch.tensor([0.0]), torch.tensor([2 * torch.pi])),
                (torch.tensor([0.0]), torch.tensor([3 * torch.pi])),
                (torch.tensor([0.0]), torch.tensor([-4 * torch.pi])),
                (torch.tensor([-torch.pi]), torch.tensor([torch.pi])),
                (torch.tensor([0.0]), torch.tensor([0.1]))]

    for angle, target in test_set:

        # Invariant under common phase shift
        for i in torch.linspace(-2 * torch.pi, 2 * torch.pi, 100):
            phase = torch.tensor([i])
            assert angular_loss_sq(angle, target).isclose(angular_loss_sq(angle + phase, target + phase))

        # Invariant under exchange of target and prediction
        assert angular_loss_sq(angle, target).isclose(angular_loss_sq(target, angle))

        # Invariant under shift of target or angle by 2 * pi
        assert angular_loss_sq(angle, target).isclose(angular_loss_sq(angle + 2 * torch.pi, target))
        assert angular_loss_sq(angle, target).isclose(angular_loss_sq(angle, target + 2 * torch.pi))
