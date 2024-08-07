import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from jaxtyping import Float, Int
import time

class ProbabilityAmplitudeDataset(Dataset):
    """
    A dataset allowing for supervised training of a TQS model on particular components of
    a wave function of a quantum system.
    """

    def prob_amp_collate(self, batch):
        """
        A collate function to be used with a DataLoader that will convert a batch of
        probability amplitudes into a tensor that can be used to train a TQS model.
        """
        b0 = torch.stack([b[0] for b in batch]).to(device="cuda")
        b1 = torch.stack([b[1] for b in batch]).to(device="cuda")
        b2 = torch.stack([b[2] for b in batch]).to(device="cuda")
        return b0, b1, b2

    def __init__(
        self,
        basis: Int[np.ndarray, "sites 2**sites"],
        parameters: Float[np.ndarray, "param_dim m"],
        ground_states: Float[np.ndarray, "m 2**sites"],
    ):
        self.basis: Int[np.ndarray, "sites 2**sites"] = basis
        self.parameters: Float[np.ndarray, "param_dim m"] = parameters
        self.ground_states: Float[np.ndarray, "m 2**sites"] = ground_states
        self.n = basis.shape[0]
        self.hilbdim = 2**self.n
        self.m = ground_states.shape[0]
        self.param_dim = parameters.shape[1]

    def __len__(self):
        """The length of the dataset is the number of probability amplitudes it contains"""
        return 2**self.n * self.m

    def __getitem__(self, idx):
        """
        Indexing follows 2D flattened array indexing. The index of all probability
        amplitudes will follow rows first (traversing a single ground state by components)
        then columns (traversing ground states).

        :returns: basis_states, parameters, prob_amp_labels, where basis_states has shape
            (..., sites), parameters has shape (..., param_dim), and prob_amp_labels has shape
            (...,).
        """

        # .reshape should simply add a layer of index calculations
        prob_amp_labels = self.ground_states.reshape(-1)[idx]

        # Index of the basis state the component belongs to
        basis_idx = idx % (self.hilbdim)
        basis_state = self.basis[:, basis_idx]

        # Index of the ground state and the parameters that correspond to it
        ground_state_idx = idx // (self.hilbdim)
        parameters = self.parameters[:, ground_state_idx]  # (param_dim, ...)

        return (
            torch.tensor(basis_state, device="cpu"),
            torch.tensor(parameters, device="cpu"),
            torch.tensor(prob_amp_labels, device="cpu"),
        )