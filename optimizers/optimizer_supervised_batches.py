# -*- coding: utf-8 -*-
"""
Created on Sun May 15 22:53:44 2022

@author: Yuanhang Zhang
"""


import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from model.model_utils import sample, compute_grad, compute_psi
from inference.evaluation import compute_E_sample, compute_magnetization
from hamiltonians.Hamiltonian import Hamiltonian
import model.autograd_hacks as autograd_hacks
from model.SR import SR
import itertools
from model.loss_functions import prob_phase_loss
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from torch.utils.tensorboard import SummaryWriter
import os
from ray import train, tune
from ray.train import Checkpoint, get_checkpoint
from pathlib import Path
import pickle
import tempfile


class Optimizer:
    def __init__(self, model, Hamiltonians, lr=1e-7, beta1=0.9, beta2=0.98, point_of_interest=None):
        # Transformer model to optimize
        self.model = model

        self.Hamiltonians = Hamiltonians

        # E.g., fixed J, h in [0.5, 1.5]
        self.model.param_range = Hamiltonians[0].param_range

        # TODO: self.loss_fn is never used in this definition, but that does
        # not necessarily mean it is never used. However, it does not seem to be
        # accessed in the rest of the code (the relevant parts being main.py
        # and the train function below). VSCode's global text search tool finds
        # no other occurrences/accesses.
        # self.loss_fn = nn.MSELoss()

        self.optim = torch.optim.Adam(
            self.model.parameters(), lr=lr, betas=(beta1, beta2), eps=1e-9
        )

        # the following is for per-layer stochastic reconfiguration
        # currently very unstable and performs rather poorly
        # avoid using for now, need future improvements
        # self.optim_SR = torch.optim.SGD(self.model.parameters(), lr=1.0)
        # self.preconditioner = SR(self.model)

        # TODO: generate or load a dataset here--one for each Hamiltonian. (Probably
        # provide options for save/load location). Should this be one dataset per
        # Hamiltonian? ALTERNATIVE: store the dataset in the Hamiltonian object.

        # TODO: verify that the dataset corresponds to the model (e.g., correct shape,
        # parameters in the same range, etc.).

        self.save_freq = 100
        self.ckpt_freq = 100
        self.point_of_interest = point_of_interest

    def show_energy_report(
        self,
        monitor_params,
        monitor_hamiltonians,
        monitor_energies,
        E_errors,
        writer=None,
        logging_iter=None,
    ):
        with torch.no_grad():
            for param in monitor_params:
                for ham, E_ground in zip(monitor_hamiltonians, monitor_energies):
                    E_mean, E_var, Er, Ei = self.extract_energy_estimate(ham, param)
                    print("E_mean", E_mean, "E_ground", E_ground, "param", param)
                    relative_error = torch.abs((E_mean - E_ground) / E_ground)

                    E_errors.append(relative_error)

                    if (writer is not None) and (logging_iter is not None):
                        writer.add_scalar(
                            f"Energy_Error/N={ham.system_size.item()}, h={param.item()}",
                            relative_error,
                            logging_iter,
                        )
                    elif (writer is not None) or (logging_iter is not None):
                        print(
                            "Warning: writer and logging_iter must be both None or both not None. Setting both to None."
                        )
                        writer = None
                        logging_iter = None

                    print(
                        f"\tparam={param}, system_size={ham.system_size} - Relative Error: {relative_error}\n\t\tEnergy: {E_mean}, Variance: {E_var}, Real: {Er}, Imag: {Ei}"
                    )

    def extract_energy_estimate(
        self,
        H: Hamiltonian,
        param: torch.Tensor,
        num_samples: int = 1000000,
        max_unique: int = 100,
    ):
        """
        Extracts an energy estimate from the model using the Hamiltonian H.
        Parameters:
            H: Hamiltonian
                The Hamiltonian object to use to produce the energy estimate
            param: torch.Tensor
                The parameters to obtain the energy estimate for
            num_samples: int
                The number of samples from the wave function the model represents to use
                in energy estimation
            max_unique: int
                The maximum number of unique samples to generate (see sample in model_utils.py)
        """
        self.model.set_param(system_size=H.system_size, param=param)
        symmetry = H.symmetry
        samples, sample_weight = sample(self.model, num_samples, max_unique, symmetry)
        E = H.Eloc(samples, sample_weight, self.model, use_symmetry=True)
        E_mean = (E * sample_weight).sum()
        E_var = (
            (((E - E_mean).abs() ** 2 * sample_weight).sum() / H.n**2)
            .detach()
            .cpu()
            .numpy()
        )
        Er = (E_mean.real / H.n).detach().cpu().numpy()
        Ei = (E_mean.imag / H.n).detach().cpu().numpy()

        return E_mean, E_var, Er, Ei

    def plot_grad_flow(self, named_parameters, tensorboard_writer, step):
        ave_grads = []
        ave_weights = []
        layers = []
        for n, p in named_parameters:
            if (p.requires_grad) and ("bias" not in n):
                layers.append(n)
                try:
                    ave_grads.append(p.grad.abs().mean().item())
                    tensorboard_writer.add_scalar(
                        "gradient/" + n, p.grad.abs().mean().item(), global_step=step
                    )

                    ave_weights.append(p.abs().mean().item())
                    tensorboard_writer.add_scalar(
                        "weights/" + n, p.abs().mean().item(), global_step=step
                    )

                except:
                    continue

    def compute_psi_batch(
        self,
        model: nn.Module,
        params,
        samples,
        system_size,
        symmetry=None,
        check_duplicates=False,
    ):
        """
        Computes the wave function for a batch of samples.
        Parameters:
            model: nn.Module
                The model to use to compute the wave function
            samples: torch.Tensor
                The samples to compute the wave function for
            symmetry: str | None
                The symmetry to use when computing the wave function
            check_duplicates: bool
                Whether to check for duplicates in the samples
            params: torch.Tensor | None - (batch_size, param_dim)
                The parameters to use when computing the wave function
        """

        if symmetry is not None:
            samples, phase = symmetry(samples)
            n_symm, n, batch0 = samples.shape
            samples = samples.transpose(0, 1).reshape(n, -1)
            params = params.repeat(n_symm, 1)  # TODO: check for many params

        if check_duplicates:
            samples_params = torch.vstack([samples, params.T])
            samples_params, inv_idx = torch.unique(
                samples_params, dim=1, return_inverse=True
            )
            samples = samples_params[:-1]
            params = samples_params[-1].reshape(-1, 1)

        n, batch = samples.shape
        n_idx = torch.arange(n).reshape(n, 1)
        batch_idx = torch.arange(batch).reshape(1, batch)
        spin_idx = samples.to(torch.int64)

        log_prob, log_phase = model.forward_batched(params, samples, system_size)

        log_prob = log_prob[:-1]
        log_phase = log_phase[:-1]

        log_prob = log_prob[n_idx, batch_idx, spin_idx].sum(dim=0)
        log_phase = log_phase[n_idx, batch_idx, spin_idx].sum(dim=0)

        if check_duplicates:
            log_prob = log_prob[inv_idx]
            log_phase = log_phase[inv_idx]

        if symmetry is not None:
            log_prob = log_prob.reshape(n_symm, batch0)
            log_phase = log_phase.reshape(n_symm, batch0)

            log_phase = ((log_prob + 1j * log_phase) / 2).exp().mean(dim=0)
            log_phase = log_phase.imag.atan2(log_phase.real) * 2  # (batch0, )
            log_prob = log_prob.exp().mean(dim=0).log()  # (batch0, )

        return log_prob, log_phase

    def train(
        self,
        epochs,
        monitor_params=None,
        monitor_hamiltonians=None,
        monitor_energies=None,
        param_range=None,
        ensemble_id=0,
        prob_weight=0.5,
        arg_weight=0.5,
        writer: SummaryWriter = None,
        run_num = 0,
        start_epoch = 0,
        energy_error_frequency = 100
    ):
        """
        Trains the model to replicate the parameter-and-spin-sequence to ground
        state wave function mapping using the Hamiltonians to produce the ground
        state labels and MSE to minimize error.

        Parameters:
            epochs: int
                Number of times the optimizer will iterate over all of the Hamiltonians
                provided to it on initialization.
            param_range: torch.Tensor | None - (number of parameters, 2)
                The range of parameters to sample from. If None, the range is derived
                from the first of the Hamiltonians.
            param_step: torch.Tensor | None - (number of parameters, )
                The step size to use when traversing the whole parameter space. If None,
                the parameter space will be randomly sampled. TODO: implement random sampling,
                perhaps from a desired distribution?
            ensemble_id: int
            start_iter: int | None
            prob_weight: float
                The weight assigned to the probability loss term in the composite loss function.
                0.5 by default.
            arg_weight: float
                The weight assigned to the phase loss term in the composite loss function
                0.5 by default.
        """

        timestep_plotting_index = 0

        name, embedding_size, n_head, n_layers = (
            type(self.Hamiltonians[0]).__name__,
            self.model.embedding_size,
            self.model.n_head,
            self.model.n_layers,
        )

        if param_range is None:
            param_range = self.Hamiltonians[0].param_range
        self.model.param_range = param_range

        save_str = (
            f"{name}_{embedding_size}_{n_head}_{n_layers}_{ensemble_id}_supervised"
        )

        self.E_errors_all = []

        initial_energy = self.extract_energy_estimate(
            monitor_hamiltonians[0], monitor_params[0]
        )

        initial_relative_error = torch.abs(
            (initial_energy[0] - monitor_energies[0]) / monitor_energies[0]
        )

        print(f"Initial energy error: {initial_relative_error}")

        for i in range(start_epoch, epochs):
            epoch_start = time.time()

            E_errors = []

            iter = 0
            random.shuffle(self.Hamiltonians)
            for H in self.Hamiltonians:
                ham_start = time.time()
                system_size = H.system_size
                dataset = H.training_dataset
                sampler = H.sampler

                for sample_idx in sampler:
                    basis_states, params, psi_true = dataset[sample_idx]

                    self.model.set_param(system_size=system_size, param=None)

                    psi_start = time.time()

                    log_prob, log_phase = self.compute_psi_batch(
                        self.model,
                        params,
                        basis_states,
                        system_size,
                        symmetry=H.symmetry,
                        check_duplicates=True,
                    )

                    psi_end = time.time()

                    loss_time = time.time()

                    degenerate = params < 1

                    loss = prob_phase_loss(
                        log_prob,
                        log_phase,
                        psi_true,
                        degenerate=degenerate,
                        prob_weight=prob_weight,
                        arg_weight=arg_weight,
                        writer=writer,
                        writer_iter=timestep_plotting_index,
                    )

                    loss_end = time.time()

                    unique_params = torch.unique(params)
                    param_max = torch.max(unique_params)
                    param_min = torch.min(unique_params)

                    backprop_start = time.time()
                    self.optim.zero_grad()
                    loss.backward()
                    if writer is not None:
                        self.plot_grad_flow(
                            self.model.named_parameters(),
                            writer,
                            timestep_plotting_index,
                        )
                    self.optim.step()
                    # scheduler.step()
                    backprop_end = time.time()

                    print(
                        f"Epoch {i} iter {iter} - Loss for system size {system_size} and h-range {param_min}-{param_max}: {loss.item()}"
                    )

                    if timestep_plotting_index % self.ckpt_freq == 0:
                        if not os.path.isdir("supervised_results"):
                            os.mkdir("supervised_results")
                        torch.save(
                            self.model.state_dict(),
                            f"supervised_results/ckpt_run{run_num}_{timestep_plotting_index}_{save_str}.ckpt",
                        )

                    energy_start = time.time()

                    if (not monitor_params is None) and (timestep_plotting_index % energy_error_frequency == 0):
                        self.show_energy_report(
                            monitor_params,
                            monitor_hamiltonians,
                            monitor_energies,
                            E_errors,
                            writer,
                            logging_iter=timestep_plotting_index,
                        )

                    energy_end = time.time()

                    print(
                        f"Time breakdown: psi: {psi_end - psi_start}, loss: {loss_end - loss_time}, backprop: {backprop_end - backprop_start}, energy retrieval: {energy_end - energy_start}"
                    )

                    iter += 1
                    timestep_plotting_index += 1

                ham_end = time.time()
                print(
                    f"Hamiltonian for size {system_size} took {ham_end - ham_start} seconds"
                )

            self.E_errors_all.append(E_errors)

            epoch_end = time.time()
            print(f"Epoch {i} took {epoch_end - epoch_start} seconds")
