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
from optimizers.bookkeeping_tools import (
    average_monitor_dict,
    create_save_dirs,
)


class Optimizer:
    def __init__(
        self,
        model,
        Hamiltonians,
        lr=1e-7,
        beta1=0.9,
        beta2=0.98,
        point_of_interest=None,
    ):
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
        self.energy_estimation_iter_frequency = 10
        self.grad_flow_iter_frequency = 100
        self.checkpointing_dir = None
        self.monitoring_dir = None
        self.tensorboard_dir = None

    def report_energy_estimate(
        self,
        monitor_dict,
        iteration,
        writer=None,
        writer_iter=None,
        num_samples=1e6,
        max_unique=1e2,
    ):
        """
        Performs a complete survey of the status of ground state energy estimation
        for the Hamiltonians in monitor_dict. This includes:
        - Calculating the energy estimate (E_mean, E_var, Er, Ei)
        - Updating the monitor_dict with the new energy estimate in its per-epoch buffer
        - Logging the energy estimate to TensorBoard if a writer AND a writer_iter is provided

        :param monitor_dict:
            A dictionary containing the Hamiltonians to monitor, their parameters, and
            the energy estimates for each epoch. See generate_monitor_dict in monitoring_dict.py
            for a complete definition.
        """

        with torch.no_grad():
            size_keys = monitor_dict.keys()

            for size_key in size_keys:
                param_keys = monitor_dict[size_key]["params"].keys()

                # Common across λJ.H(J)
                size = monitor_dict[size_key]["system_size"]
                H = monitor_dict[size_key]["H"]

                for param_key in param_keys:

                    # Common across H(J) for a particular J
                    info_dict = monitor_dict[size_key]["params"][param_key]

                    param = info_dict["param"]

                    # Calculate the energy estimate
                    E_mean, E_var, Er, Ei = self.extract_energy_estimate(
                        H, param, num_samples, max_unique
                    )
                    # TODO: why is E_mean imaginary?
                    E_mean = torch.abs(E_mean)
                    E_true = info_dict["energy"]
                    error = E_mean - E_true

                    # Update the monitor_dict with the new energy estimate
                    info_dict["epoch_errors"].append(error)
                    info_dict["epoch_relative_errors"].append(torch.abs(error) / E_true)
                    info_dict["epoch_E_mean"].append(E_mean)
                    info_dict["epoch_E_var"].append(E_var)
                    info_dict["epoch_Er"].append(Er)
                    info_dict["epoch_Ei"].append(Ei)

                    # Log to TensorBoard
                    if (writer is not None) and (writer_iter is not None):
                        writer.add_scalar(
                            f"Energy_Error/N={size.item()}, h={param.item()}",
                            error,
                            writer_iter,
                        )
                        writer.add_scalar(
                            f"Energy/N={size.item()}, h={param.item()}",
                            E_mean,
                            writer_iter,
                        )
                        writer.add_scalar(
                            f"Energy_Variance/N={size.item()}, h={param.item()}",
                            E_var,
                            writer_iter,
                        )
                        writer.add_scalar(
                            f"Energy_Real/N={size.item()}, h={param.item()}",
                            Er,
                            writer_iter,
                        )
                        writer.add_scalar(
                            f"Energy_Imag/N={size.item()}, h={param.item()}",
                            Ei,
                            writer_iter,
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
        # TODO: mark torch no grad
        self.model.set_param(system_size=H.system_size, param=param)
        symmetry = H.symmetry
        samples, sample_weight = sample(self.model, num_samples, max_unique, symmetry)
        E = H.Eloc(samples, sample_weight, self.model, use_symmetry=True)
        E_mean = (E * sample_weight).sum()
        E_var = (
            (((E - E_mean).abs() ** 2 * sample_weight).sum() / H.n**2)
            .detach()
            .cpu()
            # .numpy()
        )
        Er = (E_mean.real / H.n).detach().cpu()  # .numpy()
        Ei = (E_mean.imag / H.n).detach().cpu()  # .numpy()

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

    def produce_save_str(self, ensemble_id=0):
        name, embedding_size, n_head, n_layers = (
            type(self.Hamiltonians[0]).__name__,
            self.model.embedding_size,
            self.model.n_head,
            self.model.n_layers,
        )

        save_str = (
            f"{name}_{embedding_size}_{n_head}_{n_layers}_{ensemble_id}_supervised"
        )

        return save_str

    def consume_batch(
        self,
        H,
        basis_states,
        params,
        psi_true,
        writer,
        writer_iter,
        prob_weight,
        arg_weight,
        iter,
    ):
        self.model.set_param(system_size=H.system_size, param=None)

        log_prob, log_phase = self.compute_psi_batch(
            self.model,
            params,
            basis_states,
            H.system_size,
            symmetry=H.symmetry,
            check_duplicates=True,
        )

        degenerate = params < 1

        loss = prob_phase_loss(
            log_prob,
            log_phase,
            psi_true,
            degenerate=degenerate,
            prob_weight=prob_weight,
            arg_weight=arg_weight,
            writer=writer,
            writer_iter=writer_iter,
        )

        # TODO: how much of a slowdown does this cause?
        unique_params = torch.unique(params)
        param_max = torch.max(unique_params)
        param_min = torch.min(unique_params)
        print(f"\t\tIter {iter} - h ∈ [{param_min.item()}, {param_max.item()}]")

        self.optim.zero_grad()
        loss.backward()

    def report_on_batch(
        self,
        monitor_dict,
        global_iteration,
        writer=None,
    ):
        # If there's a writer, plot gradient flow according to the
        # self.grad_flow_iter_frequency
        if (writer is not None) and (
            global_iteration % self.grad_flow_iter_frequency == 0
        ):
            self.plot_grad_flow(
                self.model.named_parameters(),
                writer,
                global_iteration,
            )

        # If a monitor_dict was passed, report energy estimates according
        # to the self.energy_estimation_iter_frequency
        if (monitor_dict is not None) and (
            global_iteration % self.energy_estimation_iter_frequency == 0
        ):
            self.report_energy_estimate(
                monitor_dict=monitor_dict,
                iteration=global_iteration,
                writer=writer,
                writer_iter=global_iteration,
            )

    def report_on_epoch(
        self,
        monitor_dict,
        global_iteration,
        epoch,
        writer=None,
    ):
        # Checkpoint the model every self.ckpt_freq epochs
        if global_iteration % self.ckpt_freq == 0:
            torch.save(
                self.model.state_dict(),
                f"{self.checkpointing_dir}/ckpt_iter{global_iteration}.ckpt",
            )

        # Average and clear the monitor_dict every epoch
        # TODO: ensure this mutates the monitor_dict in place
        average_monitor_dict(monitor_dict=monitor_dict, epoch=epoch)

        # Display the averaged energy metrics in TensorBoard
        if writer is not None:
            for size_key in monitor_dict.keys():
                for param_key in monitor_dict[size_key]["params"].keys():
                    info_dict = monitor_dict[size_key]["params"][param_key]

                    prefix = "Epoch_Average_Energy"

                    # TODO: access element in info_dict corresponding to epoch
                    # via a return value from average_monitor_dict instead of
                    # indexing into info_dict

                    writer.add_scalar(
                        f"{prefix}_Error/N={size_key}, h={param_key}",
                        info_dict["epoch_averaged_errors"][epoch],
                        epoch,
                    )
                    writer.add_scalar(
                        f"{prefix}/N={size_key}, h={param_key}",
                        info_dict["epoch_averaged_E_mean"][epoch],
                        epoch,
                    )
                    writer.add_scalar(
                        f"{prefix}_Variance/N={size_key}, h={param_key}",
                        info_dict["epoch_averaged_E_var"][epoch],
                        epoch,
                    )
                    writer.add_scalar(
                        f"{prefix}_Real/N={size_key}, h={param_key}",
                        info_dict["epoch_averaged_Er"][epoch],
                        epoch,
                    )
                    writer.add_scalar(
                        f"{prefix}_Imag/N={size_key}, h={param_key}",
                        info_dict["epoch_averaged_Ei"][epoch],
                        epoch,
                    )

    def train(
        self,
        epochs,
        monitor_dict=None,
        param_range=None,
        log_tensorboard=False,
        ensemble_id=0,
        prob_weight=0.5,
        arg_weight=0.5,
        start_epoch=0,
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

        self.checkpointing_dir, self.monitoring_dir, self.tensorboard_dir = (
            create_save_dirs(self.produce_save_str(ensemble_id=ensemble_id))
        )

        if log_tensorboard:
            writer = SummaryWriter(self.tensorboard_dir)
            print(
                f"Use !tensorboard --logdir {self.tensorboard_dir} for monitoring.\nPass --bind-all if training remotely."
            )
        else:
            writer = None

        total_iter = 0

        # if param_range is None:
        #     param_range = self.Hamiltonians[0].param_range
        # self.model.param_range = param_range

        for epoch in range(start_epoch, epochs):
            print(f"Starting epoch {epoch}")
            epoch_iter = 0
            random.shuffle(self.Hamiltonians)
            for H in self.Hamiltonians:
                system_size = H.system_size
                dataset = H.training_dataset
                sampler = H.sampler

                print(f"\tN = {system_size.tolist()}")

                for sample_idx in sampler:
                    basis_states, params, psi_true = dataset[sample_idx]

                    self.consume_batch(
                        H,
                        basis_states=basis_states,
                        params=params,
                        psi_true=psi_true,
                        writer=writer,
                        writer_iter=total_iter,
                        prob_weight=prob_weight,
                        arg_weight=arg_weight,
                        iter=epoch_iter,
                    )

                    self.report_on_batch(
                        monitor_dict=monitor_dict,
                        global_iteration=total_iter,
                        writer=writer,
                    )

                    # TODO: can we move this to the consume_batch function
                    # without causing issues with plot_grad_flow in report_on_batch?
                    self.optim.step()

                    epoch_iter += 1
                    total_iter += 1

            self.report_on_epoch(
                monitor_dict=monitor_dict,
                global_iteration=total_iter,
                epoch=epoch,
                writer=writer,
            )

        pickle.dump(
            monitor_dict,
            open(f"{self.monitoring_dir}/monitor_dict.pkl", "wb"),
        )
