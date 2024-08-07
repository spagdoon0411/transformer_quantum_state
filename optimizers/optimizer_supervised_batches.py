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


class Optimizer:
    def __init__(self, model, Hamiltonians, point_of_interest=None):
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
            self.model.parameters(), lr=1, betas=(0.9, 0.98), eps=1e-9
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

    # A schedule used for an adaptive learning rate
    @staticmethod
    def lr_schedule(step, model_size, factor=5.0, warmup=4000, start_step=0):
        # using the lr schedule from the paper: Attention is all you need
        step = step + start_step
        if step < 1:
            step = 1
        return factor * (
            model_size ** (-0.5) * min(step ** (-0.75), step * warmup ** (-1.75))
        )

    def minimize_energy_step(self, H, batch, max_unique, use_symmetry=True):
        """
        Produces a computation graph for the energy expectation value of a Hamiltonian H.
        Arguments:
        - H: Hamiltonian object - Hamiltonian
        - batch: number of samples to generate (see sample in model_utils.py) - int
        - max_unique: maximum number of unique samples to generate (see sample) - int
        - use_symmetry: whether to use symmetry in the model - bool
        """
        symmetry = H.symmetry if use_symmetry else None
        samples, sample_weight = sample(self.model, batch, max_unique, symmetry)

        # This is the expectation value of <H> over the probability distribution captured
        # by the sample function (i.e., the probability distribution associated with the
        # wave function represented by the model).
        E = H.Eloc(samples, sample_weight, self.model, use_symmetry)
        E_mean = (E * sample_weight).sum()
        E_var = (
            (((E - E_mean).abs() ** 2 * sample_weight).sum() / H.n**2)
            .detach()
            .cpu()
            .numpy()
        )
        Er = (E_mean.real / H.n).detach().cpu().numpy()
        Ei = (E_mean.imag / H.n).detach().cpu().numpy()

        # NOTE: this is NOT the gradient of the loss function (the loss function being
        # <H> over \psi* times \psi); compute_grad is a misnomer for this function.
        # It should be called compute_loss (or, more verbosely, get_loss_computation_graph).
        # The gradient of the <H> function is computed by calling .backward() on the first
        # tuple member that compute_grad returns--and, in fact, .backward uses the
        # those calculated derivatives to adjust the model's parameters in this step. (Abstracted
        # away is reverse-mode automatic differentiation, which provides local gradients).

        loss, log_amp, log_phase = compute_grad(
            self.model, samples, sample_weight, E, symmetry
        )
        return loss, log_amp, log_phase, sample_weight, Er, Ei, E_var

    def calculate_mse_step(self, H, params, basis_batch=None, use_symmetry=True):
        """
        Produces a computation graph for the mean squared error between the model's predictions and
        the true ground state wave function for a Hamiltonian H.

        Parameters:

        H: Hamiltonian
            The Hamiltonian object that the true wave function comes from. Either produces or
            is (via a dataset passed as an argument) associated with a (J, psi(J)) pair.
            TODO: consider whether datasets should be associated with Hamiltonians or passed into
            optimizers.

        params: torch.Tensor - (n_parameters, )
            The parameters to set the model to, defining a point in parameter space (and thus a
            Hamiltonian/wave function in the family that this model models)

        basis_batch: int | None
            The maximum number of sequences to pass into the model at once. If None, will
            attempt to use an entire basis state dataset (for example, the entire N by 2^N
            tensor for a 1/2-spin TFIM chain)

        use_symmetry: bool
            Whether to use symmetry in the compute_psi function (from model_utils.py) to
            produce a wave function.

        Returns:

        loss: torch.Tensor - (1, ) TODO: can this be a scalar?
            The mean squared error between the model's predictions and the true wave function.

        """

        # Construct a dataset of samples. This will be a tensor of shape
        # (n, 2^n) where n is the number of spins, and will contain one of every
        # basis state in the Hilbert space. The sample_weight tensor is not needed;
        # compute_psi does not take one (and, considering how the authors describe how
        # to obtain a wave function given a model output, it should not need one).
        basis = H.basis

        # TODO: How could we package these into batches across J? This might not involve
        # fundamental changes to compute_psi, but it would involve changes to the forward pass.
        # The forward pass assumes that one batch comes from a single point in sample
        # space (using only a static J-vector).

        # TODO: QUESTION: if the members of a batch were heterogenous
        # with respect to J, would the model's forward pass still be correct?
        # HYPOTHESIS: yes! After wrap_spins (in forward), the sequences are processed entirely
        # in parallel. Sequences do not interact with each other--but members of the same sequence do
        # (a fundamental guarantee of batch processing)

        # compute_psi from model_utils.py is used here, producing two vectors:
        # one for P(s, J) values and one for phi(s, J) values.
        symmetry = H.symmetry if use_symmetry else None
        log_amp, log_phase = compute_psi(
            self.model, basis, symmetry, check_duplicate=True
        )

        amp = torch.exp(log_amp)
        phase = torch.exp(1j * log_phase)

        # Use (1) - (4) from the TQS paper--most significantly, (2)--to obtain
        # the wave function that the model represents. NOTE: the discussion about
        # symmetries in Appendix B is accounted for if a non-None symmetry was
        # passed to compute_psi.
        psi_predicted = amp.mul(torch.exp(1j * phase))

        # Obtain the ground state wave function for the Hamiltonian H, possibly memoized internally
        # in the Hamiltonian object. This is the true wave function that the model's predictions
        # are compared against. TODO: implement memoization in Hamiltonian objects.
        energy, psi_true = H.retrieve_ground(param=params)

        # Compute the mean squared error between the model's predictions and the true
        # ground state wave function.
        psi_predicted_real_imag = torch.view_as_real(psi_predicted)
        psi_true_real_imag = torch.view_as_real(psi_true.to(torch.complex64))

        loss = F.mse_loss(
            psi_predicted_real_imag, psi_true_real_imag
        )  # TODO: better way to account for only a phase difference?

        # At this point, we should have a computational graph for the mean squared error--
        # i.e., a computational graph for a loss function--that we can backpropagate through
        # using .backward() and .step(). This will adjust the model's parameters.
        #
        # TODO: how is the Attention Is All You Need learning rate involved? Is it a
        # global PyTorch config setting?

        return loss

    def generate_parameter_range(self, start, end, step):
        """
        A simple generator returning the next value in a range of values
        whenever called, according to a step size.
        """
        value = start
        while value < end:
            yield value
            value += step

    def generate_parameter_points(
        self, parameter_ranges, step_sizes, distribution=None
    ):
        """
        Generate all possible combinations of parameter values for a model
        (i.e., the Cartesian product of values of parameters in a slice of parameter space)

        Parameters:
            parameter_ranges: torch.Tensor of shape (n_parameters, 2)
                The starting and ending values for each dimension of the slice of parameter space
            step_sizes: torch.Tensor of shape (n_parameters,)
                The step size for each dimension of the slice of parameter space
            distribution: N/A
                TODO: Not implemented

        """

        if distribution is not None:
            raise NotImplementedError(
                "Sampling using a custom distribution is not implemented yet."
            )

        # Every possible individual parameter value for each parameter, in order
        parameter_ranges = [
            self.generate_parameter_range(start.item(), end.item(), step.item())
            for (start, end), step in zip(parameter_ranges, step_sizes)
        ]

        return itertools.product(*parameter_ranges)

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

    # Performs a mapping from one degenerate state to the other. Does not
    # account for the parameter range.
    def ising_degen(self, phases, probs):
        return (-phases, probs)

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
        start_iter=None,
        prob_weight=0.5,
        arg_weight=0.5,
        writer: SummaryWriter = None,
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

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optim,
            lambda step: self.lr_schedule(
                step, self.model.embedding_size, start_step=start_iter
            ),
        )

        self.E_errors_all = []

        initial_energy = self.extract_energy_estimate(
            monitor_hamiltonians[0], monitor_params[0]
        )

        initial_relative_error = torch.abs((initial_energy[0] - monitor_energies[0]) / monitor_energies[0])

        print(f"Initial energy error: {initial_relative_error}")

        for i in range(epochs):
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
                    #     writer.add_scalar(
                    #         f"N={system_size}_loss",
                    #         loss.item(),
                    #         i,
                    #     )
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
                        torch.save(
                            self.model.state_dict(),
                            f"supervised_results/ckpt_{timestep_plotting_index}_{save_str}.ckpt",
                        )

                    energy_start = time.time()

                    if not monitor_params is None:
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
