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

from model_utils import sample, compute_grad
from evaluation import compute_E_sample, compute_magnetization
import autograd_hacks
from SR import SR


class SupervisedOptimizer:
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
        self.loss_fn = nn.MSELoss()
        self.optim = torch.optim.Adam(
            self.model.parameters(), lr=1, betas=(0.9, 0.98), eps=1e-9
        )

        # the following is for per-layer stochastic reconfiguration
        # currently very unstable and performs rather poorly
        # avoid using for now, need future improvements
        self.optim_SR = torch.optim.SGD(self.model.parameters(), lr=1.0)
        self.preconditioner = SR(self.model)

        # TODO: generate or load a dataset here--one for each Hamiltonian. (Probably
        # provide options for save/load location). Should this be one dataset per
        # Hamiltonian? ALTERNATIVE: store the dataset in the Hamiltonian object.

        # TODO: verify that the dataset corresponds to the model (e.g., correct shape,
        # parameters in the same range, etc.).

        self.save_freq = 100
        self.ckpt_freq = 10000
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

    def calculate_mse_step(self, H, basis_batch=None, use_symmetry=True):
        """
        Produces a computation graph for the mean squared error between the model's predictions and
        the true ground state wave function for a Hamiltonian H.

        Parameters:

        H: Hamiltonian
            The Hamiltonian object that the true wave function comes from. Either produces or
            is (via a dataset passed as an argument) associated with a (J, psi(J)) pair.
            TODO: consider whether datasets should be associated with Hamiltonians or passed into
            optimizers.

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

        # TODO: How could we package these into batches? This might not involve
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

        # Use (1) - (4) from the TQS paper--most significantly, (2)--to obtain
        # the wave function that the model represents.

        # Obtain the ground state wave function for the Hamiltonian H, possibly memoized interally
        # in the Hamiltonian object. This is the true wave function that the model's predictions

        # Compute the mean squared error between the model's predictions and the true
        # ground state wave function.

        # At this point, we should have a computational graph for the mean squared error--
        # i.e., a computational graph for a loss function--that we can backpropagate through
        # using .backward(). This will adjust the model's parameters.
        #
        # TODO: how is the Attention Is All You Need learning rate involved? Is it a
        # global PyTorch config setting?

        raise NotImplementedError("Supervised learning not implemented yet")

    # TODO: add a random sampling flag, remove param_range (should be derived from
    # the Hamiltonian),
    def train(
        self,
        n_iter,
        batch=10000,
        max_unique=1000,
        param_range=None,
        fine_tuning=False,
        use_SR=True,
        ensemble_id=0,
        start_iter=None,
    ):
        name, embedding_size, n_head, n_layers = (
            type(self.Hamiltonians[0]).__name__,
            self.model.embedding_size,
            self.model.n_head,
            self.model.n_layers,
        )
        if start_iter is None:
            start_iter = 0 if not fine_tuning else 100000
        system_sizes = self.model.system_sizes
        n_iter += 1
        if param_range is None:
            param_range = self.Hamiltonians[0].param_range
        self.model.param_range = param_range
        save_str = (
            f"{name}_{embedding_size}_{n_head}_{n_layers}_{ensemble_id}"
            if not fine_tuning
            else f"ft_{self.model.system_sizes[0].detach().cpu().numpy().item()}_"
            f"{param_range[0].detach().cpu().numpy().item():.2f}_"
            f"{name}_{embedding_size}_{n_head}_{n_layers}_{ensemble_id}"
        )

        if use_SR:
            optim = self.optim_SR
            autograd_hacks.add_hooks(self.model)
        else:
            optim = self.optim
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optim,
            lambda step: self.lr_schedule(
                step, self.model.embedding_size, start_step=start_iter
            ),
        )

        if self.point_of_interest is not None:
            size_i, param_i = self.point_of_interest
            H_watch = type(self.Hamiltonians[0])(size_i, self.Hamiltonians[0].periodic)
            if self.Hamiltonians[0].symmetry is None:
                H_watch.symmetry = None
            E_watch = np.zeros(int(np.ceil(n_iter / self.save_freq)))
            m_watch = np.zeros((int(np.ceil(n_iter / self.save_freq)), 3))
            idx = 0

        E_curve = np.zeros(n_iter)
        E_vars = np.zeros(n_iter)

        # TODO: this should probaly be changed to epochs, to be clearer that were iterating
        # over the whole dataset? Or we could keep this as "iterations," where one iteration is
        # a minibatch update?
        for i in range(start_iter, start_iter + n_iter):
            start = time.time()

            # TODO: should we allow random sampling from parameter space? Datasets
            # from the Hamiltonians could be internally memoized or pre-loaded

            # TODO: set particular parameters for the model here. Note that no arguments -> random
            # sampling from the parameter range.
            self.model.set_param()
            size_idx = self.model.size_idx
            n = self.model.system_size.prod()
            H = self.Hamiltonians[size_idx]

            # TODO: replace this with a call to calculate_mse_step
            loss, log_amp, log_phase, sample_weight, Er, Ei, E_var = (
                self.minimize_energy_step(H, batch, max_unique, use_symmetry=True)
            )

            t1 = time.time()

            if use_SR:
                autograd_hacks.clear_backprops(self.model)
                optim.zero_grad()
                log_amp.sum().backward(retain_graph=True)
                autograd_hacks.compute_grad1(
                    self.model, loss_type="sum", grad_name="grad1"
                )
                autograd_hacks.clear_backprops(self.model)

                optim.zero_grad()
                log_phase.sum().backward(retain_graph=True)
                autograd_hacks.compute_grad1(
                    self.model, loss_type="sum", grad_name="grad2"
                )
                autograd_hacks.clear_backprops(self.model)

                optim.zero_grad()
                loss.backward()
                autograd_hacks.clear_backprops(self.model)
                self.preconditioner.step(sample_weight)
                optim.step()
            else:
                optim.zero_grad()

                # NOTE: The loss object's computation graph is stored by
                # PyTorch. Involved in its computation was the model's forward
                # pass, and therefore the model's layers are a part of that
                # computation graph.
                #
                # Therefore, it makes sense that loss.backward() would perform
                # both reverse-mode automatic differentiation and, where it
                # encounters registered model parameters, adustments via backpropagation
                # (a simple adaptation from reverse-mode AD).
                loss.backward()
                optim.step()

            scheduler.step()
            t2 = time.time()

            # NOTE: everything below this point is just for logging and saving. Is an
            # example of how separation of concerns could be improved in the codebase
            # (e.g., by moving this to a separate function).

            print_str = f"E_real = {Er:.6f}\t E_imag = {Ei:.6f}\t E_var = {E_var:.6f}\t"
            E_curve[i - start_iter] = Er
            E_vars[i - start_iter] = E_var

            end = time.time()
            print(
                f"i = {i}\t {print_str} n = {n}\t lr = {scheduler.get_lr()[0]:.4e} t = {(end-start):.6f}  t_optim = {t2-t1:.6f}"
            )
            if i % self.save_freq == 0:
                with open(f"results/E_{save_str}.npy", "wb") as f:
                    np.save(f, E_curve)
                with open(f"results/E_var_{save_str}.npy", "wb") as f:
                    np.save(f, E_vars)
                if self.point_of_interest is not None:
                    E_watch[idx] = (
                        compute_E_sample(self.model, size_i, param_i, H_watch)
                        .real.detach()
                        .cpu()
                        .numpy()
                    )
                    m_watch[idx, :] = (
                        compute_magnetization(
                            self.model, size_i, param_i, symmetry=H_watch.symmetry
                        )
                        .real.detach()
                        .cpu()
                        .numpy()
                    )
                    idx += 1
                    with open(f"results/E_watch_{save_str}.npy", "wb") as f:
                        np.save(f, E_watch)
                    with open(f"results/m_watch_{save_str}.npy", "wb") as f:
                        np.save(f, m_watch)
                torch.save(self.model.state_dict(), f"results/model_{save_str}.ckpt")
                if i % self.ckpt_freq == 0:
                    torch.save(
                        self.model.state_dict(), f"results/ckpt_{i}_{save_str}.ckpt"
                    )
