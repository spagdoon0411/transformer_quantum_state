# -*- coding: utf-8 -*-
"""
Created on Sun May 15 14:15:05 2022

@author: Yuanhang Zhang
"""

import os
import numpy as np
import pyarrow as pa
import pandas as pd
import json
import math
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
import torch
import tenpy
from tenpy.networks.mps import MPS
from tenpy.models.tf_ising import TFIChain
from tenpy.models.spins import SpinModel
from tenpy.algorithms import dmrg
from hamiltonians.Hamiltonian_utils import generate_spin_idx
from model.model_utils import compute_observable
from hamiltonians.symmetry import Symmetry1D, Symmetry2D
from tqdm import trange

pi = np.pi
X = sparse.csr_matrix(np.array([[0, 1], [1, 0]], dtype=np.float64))
Y = sparse.csr_matrix(np.array([[0, -1j], [1j, 0]], dtype=np.complex128))
Z = sparse.csr_matrix(np.array([[1, 0], [0, -1]], dtype=np.float64))
I = sparse.csr_matrix(np.array([[1, 0], [0, 1]], dtype=np.float64))
Sp = sparse.csr_matrix(np.array([[0, 1], [0, 0]], dtype=np.float64))
Sm = sparse.csr_matrix(np.array([[0, 0], [1, 0]], dtype=np.float64))


class Hamiltonian:
    def __init__(self):
        """

        Each child should specify self.n and self.H

        n : int
            size of the physical system
        H : list of tuples
            each tuple describes one term in the Hamiltonian

        Example: (['XX', 'YY', 'ZZ'], [coef_XX, coef_YY, coef_ZZ], spin_idx)
            grouping up operators that act on the same indices to speed up
            (e.g., interaction in the Heisenberg model)
            pauli_str: string made up of 'X', 'Y', or 'Z', Pauli matrices
            coef: (1, ), (n_op, ) or (n_op, batch), coefficient of operator
            spin_idx: (n_op, n_site), indices that the Pauli operators act on

        Returns
        -------
        None.

        """
        self.system_size = None
        self.n = None
        self.H = None
        self.symmetry = None
        self.n_dim = None
        self.dataset = None

    def update_param(self, param):
        """
        Update the coefficients in the Hamiltonian in list form
        Default implementation require coef to be (n_op, ), same in every group
        One should override this function for specific Hamiltonians in other forms
        param: (n_param, )
        """
        assert len(param) == len(self.H)
        for i, param_i in enumerate(param):  # (1, )
            self.H[i][1][:] = [param_i] * len(self.H[i][1])

    @torch.no_grad()
    def Eloc(self, samples, sample_weight, model, use_symmetry=True):
        # samples: (seq, batch, input_dim)
        symmetry = self.symmetry if use_symmetry else None
        E = 0
        params = model.param  # (n_param, )
        self.update_param(params)

        for Hi in self.H:

            O = compute_observable(
                model, samples, sample_weight, Hi, batch_mean=False, symmetry=symmetry
            )

            for Oj in O:
                E += Oj.sum(dim=0)
        return E

    def full_H(self, param=None):
        raise NotImplementedError

    def calc_E_ground(self, param=None):
        if param is None:
            full_Hamiltonian = self.full_H()
        else:
            full_Hamiltonian = self.full_H(param)
        [E_ground, psi_ground] = eigsh(full_Hamiltonian, k=1, which="SA")
        E_ground = E_ground[0]
        psi_ground = psi_ground[:, 0]
        self.E_ground = E_ground
        self.psi_ground = psi_ground
        return E_ground

    def DMRG(self):
        raise NotImplementedError("Override the DMRG function in a child class")

    def add_spatial_symmetry(self):
        if self.n_dim == 1:
            self.symmetry = Symmetry1D(self.n)
            self.symmetry.add_symmetry("translation")
            self.symmetry.add_symmetry("reflection")
        elif self.n_dim == 2:
            self.symmetry = Symmetry2D(self.system_size[0], self.system_size[1])
            self.symmetry.add_symmetry("translation_x")
            self.symmetry.add_symmetry("translation_y")
            self.symmetry.add_symmetry("reflection_x")
            if self.system_size[0] == self.system_size[1]:
                self.symmetry.add_symmetry("rotation_90")
            else:
                self.symmetry.add_symmetry("reflection_y")

    def measurements(self, psi, n_measure=1000):
        samples = np.zeros((n_measure, self.n))
        for i in trange(n_measure):
            samples[i], _ = psi.sample_measurements(norm_tol=1e-4)
        return samples

    def load_dataset(self):
        raise NotImplementedError("Override load_dataset in the child class")


class Ising(Hamiltonian):
    def __init__(self, system_size, periodic=True, generate_basis=False):
        super().__init__()
        self.system_size = torch.tensor(system_size).reshape(-1)
        self.n_dim = len(self.system_size)
        self.n = self.system_size.prod()
        self.param_dim = 1
        self.param_range = torch.tensor([[0.5], [1.5]])
        self.J = -1
        self.h = 1
        self.n_dim = len(self.system_size)
        self.periodic = periodic
        self.connections = generate_spin_idx(
            self.system_size, "nearest_neighbor", periodic
        )
        self.external_field = generate_spin_idx(
            self.system_size, "external_field", periodic
        )
        self.H = [
            (["ZZ"], [self.J], self.connections),
            (["X"], [self.h], self.external_field),
        ]

        if generate_basis:
            self.basis = self.get_basis()

        # TODO: implement 2D symmetry
        assert self.n_dim == 1, "2D symmetry is not implemented yet"
        self.symmetry = Symmetry1D(self.n)
        self.symmetry.add_symmetry("reflection")
        self.symmetry.add_symmetry("spin_inversion")
        # self.momentum = 1  # k=0, exp(i k pi)=1
        # self.parity = 1
        # self.Z2 = [1, -1][self.n % 2]
        # self.symmetry.add_symmetry('translation', self.momentum)
        # self.symmetry.add_symmetry('reflection', self.parity)
        # self.symmetry.add_symmetry('spin_inversion', self.Z2)

        self.h_step = None
        self.dataset = None

    def update_param(self, param):
        # param: (1, )
        self.H[1][1][0] = param

    # Overrides the "full_H" function in the Hamiltonian superclass, implementing
    # the particular full TFIM Hamiltonian. We get a sparse matrix from this.
    def full_H(self, param=1):
        if isinstance(param, torch.Tensor):
            param = param.detach().cpu().numpy().item()
        h = param
        self.Hamiltonian = sparse.csr_matrix((2**self.n, 2**self.n), dtype=np.float64)
        for conn in self.connections:
            JZZ = 1
            for i in range(self.n):
                if i == conn[0]:
                    JZZ = sparse.kron(JZZ, Z, format="csr")
                elif i == conn[1]:
                    JZZ = sparse.kron(JZZ, Z, format="csr")
                else:
                    JZZ = sparse.kron(JZZ, I, format="csr")
            self.Hamiltonian = self.Hamiltonian + self.J * JZZ
        for i in range(self.n):
            hX = 1
            for j in range(self.n):
                if i == j:
                    hX = sparse.kron(hX, X, format="csr")
                else:
                    hX = sparse.kron(hX, I, format="csr")
            self.Hamiltonian = self.Hamiltonian + h * hX
        return self.Hamiltonian

    def get_basis(self):
        # TODO: citation?
        basis = np.zeros((2**self.n, self.n), dtype=int)
        for i in range(2**self.n):
            basis[i] = np.array([int(b) for b in np.binary_repr(i, width=self.n)])
        return torch.tensor(basis.T)

    def DMRG(self, param=None, verbose=False, conserve=None):
        # Tenpy has S_i = 0.5 sigma_i, mine doesn't have the 0.5
        # some constants are added to fix the 0.5
        assert self.n_dim == 1, "currently only supports 1D"
        assert self.periodic is False, "currently only supports non-periodic"
        if param is None:
            h = self.h
        else:
            h = param
        J = self.J

        model_params = dict(
            L=self.n,
            S=0.5,  # spin 1/2
            Jx=0,
            Jy=0,
            Jz=J,  # couplings
            hx=-h / 2,
            bc_MPS="finite",
            conserve=conserve,
        )
        M = SpinModel(model_params)
        product_state = (["up", "down"] * int(self.n / 2 + 1))[
            : self.n
        ]  # initial Neel state
        psi = MPS.from_product_state(M.lat.mps_sites(), product_state, bc=M.lat.bc_MPS)
        dmrg_params = {
            "mixer": None,  # setting this to True helps to escape local minima
            "trunc_params": {
                "chi_max": 100,
                "svd_min": 1.0e-10,
            },
            "max_E_err": 1.0e-10,
            "combine": True,
        }
        info = dmrg.run(psi, M, dmrg_params)
        E = info["E"]
        if verbose:
            print("finite DMRG, Transverse field Ising model")
            print("Jz={Jz:.2f}, conserve={conserve!r}".format(Jz=J, conserve=conserve))
            print("E = {E:.13f}".format(E=E))
            print("final bond dimensions: ", psi.chi)
            Sz = psi.expectation_value(
                "Sz"
            )  # Sz instead of Sigma z: spin-1/2 operators!
            mag_z = np.mean(Sz)
            print(
                "<S_z> = [{Sz0:.5f}, {Sz1:.5f}]; mean ={mag_z:.5f}".format(
                    Sz0=Sz[0], Sz1=Sz[1], mag_z=mag_z
                )
            )
        return E * 4, psi, M

    def load_dataset(self, data_dir_path: str):
        """
        Given a directory path, searches for a file in the directory called
        meta.json. Expects the meta.json file to contain the keys:
        - h_min - a float
        - h_max - a float
        - h_step - a float
        - N_list - a list of ints
        - file_names - a list of strings

        Loads the file corresponding to this Hamiltonian's system size
        (self.n)--i.e., searches in the list to determine the index of the
        system size in the N_list, then loads the corresponding file at that index
        in the file_names list as a Pandas DataFrame. Expects the file to be in
        the Arrow IPC (Feather2) format and expects the table to contain the columns:
        - N - an int
        - h - a float
        - state - a list of floats

        Sets this Hamiltonian's self.dataset attribute to the loaded dataset.

        Precomputes a binary representation of the system's states as a PyTorch tensor
        for passing into the model. This is stored in self.basis.

        Produces a warning if the range of h values in the metadata file does not
        match the range of h values in the Hamiltonian's param_range attribute.

        Notes a self.h_step corresponding to the step size in the metadata file.

        TODO: for large datsets, consider using PyArrow's memory mapping or Dask
        DataFrames.

        TODO: make parameter ranges more consistent across Hamiltonians.

        Parameters:
        data_dir_path: str
            The path to the directory containing the meta.json file

        Returns:
        None
        """

        # Load the metadata file
        metadata_path = os.path.join(data_dir_path, "meta.json")
        metadata_file = open(metadata_path, "r")
        metadata = json.load(metadata_file)

        try:
            idx = metadata["N_list"].index(self.n)
        except ValueError:
            raise ValueError(f"System size {self.n} not found in metadata file")

        file_path = metadata["file_names"][idx]
        mmap = pa.memory_map(file_path)
        with mmap as source:
            self.dataset = pa.ipc.open_file(source).read_pandas()

        h_min = metadata["h_min"]
        this_h_min = self.param_range[0].item()
        h_max = metadata["h_max"]
        this_h_max = self.param_range[1].item()
        h_step = metadata["h_step"]

        if h_min != self.param_range[0].item() or h_max != self.param_range[1].item():
            warning = """Warning: h values in metadata file do not match Hamiltonian's param_range; \
found h_min={0}, h_max={1}, h_step={2}, expected h_min={3}, h_max={4}. Setting param_range to match.""".format(
                h_min, h_max, h_step, this_h_min, this_h_max
            )

            self.param_range = torch.tensor([[h_min], [h_max]])

            print(warning)

        self.h_step = h_step

    def retrieve_ground(self, param, abs_tol=1e-5):
        """
        Given a parameter value and system size, retrieves the ground state as a PyTorch tensor--
        possibly for use in supervised training.

        Parameters:
        param : float
            The h-value to retrieve the ground state for. Must be in the range of the dataset.
            NOTE: due to the way the dataset is constructed, floating point errors may cause
            problems with parameter-based indexing here. np.isclose is used with a default
            absolute tolerance of 1e-5 to mitigate this.
        abs_tol : float
            1e-5 by default. The absolute tolerance used in np.isclose to determine if an
            input parameter value is close enough to a parameter value in the dataset to be
            considered a match.

        Returns:
        energy: float
            The ground state energy
        state: torch.Tensor
            The ground state wavefunction as a PyTorch tensor of shape (2**n, )
        """

        if self.dataset is None:
            raise ValueError("Ground states not loaded yet. See load_dataset.")

        # Find the rows close to the parameter value (within abs_tol)
        h_matches = self.dataset[np.isclose(self.dataset["h"], param, atol=abs_tol)]
        if h_matches.empty:
            raise ValueError(f"No ground state found for h={param}")
        elif len(h_matches) > 1:
            raise ValueError(
                f"Multiple ground states found for h={param}; using the first."
            )

        # Pick the first one to return
        h_match = h_matches.iloc[0]
        energy, state = h_match["energy"], h_match["state"]
        state = torch.tensor(state)
        return energy, state

    # def DMRG(self, param=None, verbose=False):
    #     # Adapted from tenpy examples
    #     assert self.n_dim == 1, 'currently only supports 1D'
    #     assert self.periodic is False, 'currently only supports non-periodic'
    #     if param is None:
    #         h = self.h
    #     else:
    #         h = param
    #     model_params = dict(L=self.n, J=-self.J, g=-h, bc_MPS='finite', conserve=None)
    #     M = TFIChain(model_params)
    #     product_state = ["up"] * M.lat.N_sites
    #     psi = MPS.from_product_state(M.lat.mps_sites(), product_state, bc=M.lat.bc_MPS)
    #     dmrg_params = {
    #         'mixer': None,  # setting this to True helps to escape local minima
    #         'max_E_err': 1.e-10,
    #         'trunc_params': {
    #             'chi_max': 30,
    #             'svd_min': 1.e-10
    #         },
    #         'combine': True
    #     }
    #     info = dmrg.run(psi, M, dmrg_params)  # the main work...
    #     E = info['E']
    #     mag_x = np.sum(psi.expectation_value("Sigmax"))
    #     mag_z = np.sum(psi.expectation_value("Sigmaz"))
    #     if verbose:
    #         print("finite DMRG, transverse field Ising model")
    #         print("L={L:d}, g={g:.2f}".format(L=self.n, g=-self.h))
    #         print("E = {E:.13f}".format(E=E))
    #         print("final bond dimensions: ", psi.chi)
    #         print("magnetization in X = {mag_x:.5f}".format(mag_x=mag_x))
    #         print("magnetization in Z = {mag_z:.5f}".format(mag_z=mag_z))
    #     return E, psi, M


class XXZ(Hamiltonian):
    def __init__(self, system_size, periodic=True):
        super().__init__()
        self.system_size = torch.tensor(system_size).reshape(-1)
        self.n_dim = len(self.system_size)
        self.n = self.system_size.prod()
        self.param_dim = 2
        self.param_range = torch.tensor([[-2, 2], [2, 2]])  # h, Delta
        self.J = 1
        self.h = 1
        self.Delta = 1
        self.periodic = periodic
        self.connections = generate_spin_idx(
            self.system_size, "nearest_neighbor", periodic
        )
        self.external_field = generate_spin_idx(
            self.system_size, "external_field", periodic
        )
        self.H = [
            (["XX", "YY", "ZZ"], [self.J, self.J, self.Delta], self.connections),
            (["X"], [self.h], self.external_field),
        ]
        # TODO: implement 2D symmetry
        assert self.n_dim == 1, "2D symmetry is not implemented yet"
        self.symmetry = Symmetry1D(self.n)
        self.symmetry.add_symmetry("reflection")
        self.symmetry.add_symmetry("spin_inversion")
        self.symmetry.add_symmetry("U1")
        # self.momentum = [1, (1j * 2 * pi * (self.n / 4).round() / self.n).exp(),
        #                  -1, (1j * 2 * pi * (self.n / 4).round() / self.n).exp()][self.n % 4]
        # self.parity = [1, 1, -1, 1][self.n % 4]
        # self.Z2 = [1, 1, -1, 1][self.n % 4]
        # self.symmetry.add_symmetry('translation', self.momentum)
        # self.symmetry.add_symmetry('reflection', self.parity)
        # self.symmetry.add_symmetry('spin_inversion', self.Z2)
        # self.symmetry.add_symmetry('U1')

    def update_param(self, param):
        # param: (2, ), h, Delta
        self.H[0][1][2] = param[1]  # Delta
        self.H[1][1][0] = param[0]  # h

    def full_H(self, param=None):
        if param is None:
            param = [1, 1]
        elif isinstance(param, torch.Tensor):
            param = param.detach().cpu().numpy()
        h, Delta = param
        J = self.J
        self.Hamiltonian = sparse.csr_matrix((2**self.n, 2**self.n), dtype=np.float64)
        for conn in self.connections:
            JZZ = 1
            Jpm = 1
            Jmp = 1
            for i in range(self.n):
                if i == conn[0]:
                    JZZ = sparse.kron(JZZ, Z, format="csr")
                    Jpm = sparse.kron(Jpm, Sp, format="csr")
                    Jmp = sparse.kron(Jmp, Sm, format="csr")
                elif i == conn[1]:
                    JZZ = sparse.kron(JZZ, Z, format="csr")
                    Jpm = sparse.kron(Jpm, Sm, format="csr")
                    Jmp = sparse.kron(Jmp, Sp, format="csr")
                else:
                    JZZ = sparse.kron(JZZ, I, format="csr")
                    Jpm = sparse.kron(Jpm, I, format="csr")
                    Jmp = sparse.kron(Jmp, I, format="csr")
            self.Hamiltonian = self.Hamiltonian + Delta * JZZ + 2 * J * (Jpm + Jmp)
        for i in range(self.n):
            hX = 1
            for j in range(self.n):
                if i == j:
                    hX = sparse.kron(hX, X, format="csr")
                else:
                    hX = sparse.kron(hX, I, format="csr")
            self.Hamiltonian = self.Hamiltonian + h * hX
        return self.Hamiltonian

    def DMRG(self, param=None, verbose=False, conserve=None):
        # Tenpy has S_i = 0.5 sigma_i, mine doesn't have the 0.5
        # some constants are added to fix the 0.5
        assert self.n_dim == 1, "currently only supports 1D"
        assert self.periodic is False, "currently only supports non-periodic"
        if param is None:
            h = self.h
            Delta = self.Delta
        else:
            h = param[0]
            Delta = param[1]
        J = self.J

        model_params = dict(
            L=self.n,
            S=0.5,  # spin 1/2
            Jx=J,
            Jy=J,
            Jz=Delta,  # couplings
            hx=-h / 2,
            bc_MPS="finite",
            conserve=conserve,
        )
        M = SpinModel(model_params)
        product_state = (["up", "down"] * int(self.n / 2 + 1))[
            : self.n
        ]  # initial Neel state
        psi = MPS.from_product_state(M.lat.mps_sites(), product_state, bc=M.lat.bc_MPS)
        dmrg_params = {
            "mixer": True,  # setting this to True helps to escape local minima
            "trunc_params": {
                "chi_max": 100,
                "svd_min": 1.0e-10,
            },
            "max_E_err": 1.0e-10,
        }
        info = dmrg.run(psi, M, dmrg_params)
        E = info["E"]
        if verbose:
            print("finite DMRG, Heisenberg XXZ chain")
            print(
                "Jz={Jz:.2f}, conserve={conserve!r}".format(Jz=Delta, conserve=conserve)
            )
            print("E = {E:.13f}".format(E=E))
            print("final bond dimensions: ", psi.chi)
            Sz = psi.expectation_value(
                "Sz"
            )  # Sz instead of Sigma z: spin-1/2 operators!
            mag_z = np.mean(Sz)
            print(
                "<S_z> = [{Sz0:.5f}, {Sz1:.5f}]; mean ={mag_z:.5f}".format(
                    Sz0=Sz[0], Sz1=Sz[1], mag_z=mag_z
                )
            )
            # note: it's clear that mean(<Sz>) is 0: the model has Sz conservation!
        return E * 4, psi, M


class XYZ(Hamiltonian):
    def __init__(self, system_size, periodic=True):
        super().__init__()
        self.system_size = torch.tensor(system_size).reshape(-1)
        self.n_dim = len(self.system_size)
        self.n = self.system_size.prod()
        self.param_dim = 3
        self.param_range = torch.tensor([[0, -1, 0.2], [1, 1, 0.2]])  # h, Delta, gamma
        self.h = 1
        self.Delta = 1
        self.gamma = 0
        self.periodic = periodic
        self.connections = generate_spin_idx(
            self.system_size, "nearest_neighbor", periodic
        )
        self.external_field = generate_spin_idx(
            self.system_size, "external_field", periodic
        )
        self.H = [
            (
                ["XX", "YY", "ZZ"],
                [1 + self.gamma, 1 - self.gamma, self.Delta],
                self.connections,
            ),
            (["Z"], [self.h], self.external_field),
        ]
        self.symmetry = None

    def update_param(self, param):
        # param: (3, ), h, Delta, gamma
        self.H[1][1][0] = param[0]  # h
        self.H[0][1][2] = param[1]  # Delta
        self.H[0][1][0] = 1 + param[2]  # 1+gamma
        self.H[0][1][1] = 1 - param[2]  # 1-gamma

    def full_H(self, param=None):
        if param is None:
            param = [1, 1, 0.2]
        elif isinstance(param, torch.Tensor):
            param = param.detach().cpu().numpy()
        h, Delta, gamma = param
        self.Hamiltonian = sparse.csr_matrix((2**self.n, 2**self.n), dtype=np.float64)
        for conn in self.connections:
            JZZ = 1
            JXX = 1
            JYY = 1
            for i in range(self.n):
                if i == conn[0]:
                    JZZ = sparse.kron(JZZ, Z, format="csr")
                    JXX = sparse.kron(JXX, X, format="csr")
                    JYY = sparse.kron(JYY, Y, format="csr")
                elif i == conn[1]:
                    JZZ = sparse.kron(JZZ, Z, format="csr")
                    JXX = sparse.kron(JXX, X, format="csr")
                    JYY = sparse.kron(JYY, Y, format="csr")
                else:
                    JZZ = sparse.kron(JZZ, I, format="csr")
                    JXX = sparse.kron(JXX, I, format="csr")
                    JYY = sparse.kron(JYY, I, format="csr")
            self.Hamiltonian = (
                self.Hamiltonian + (1 + gamma) * JXX + (1 - gamma) * JYY + Delta * JZZ
            )
        for i in range(self.n):
            hZ = 1
            for j in range(self.n):
                if i == j:
                    hZ = sparse.kron(hZ, Z, format="csr")
                else:
                    hZ = sparse.kron(hZ, I, format="csr")
            self.Hamiltonian = self.Hamiltonian + h * hZ
        return self.Hamiltonian

    def DMRG(self, param=None, verbose=False, conserve=None):
        # Tenpy has S_i = 0.5 sigma_i, mine doesn't have the 0.5
        # some constants are added to fix the 0.5
        assert self.n_dim == 1, "currently only supports 1D"
        assert self.periodic is False, "currently only supports non-periodic"
        if param is None:
            h = self.h
            Delta = self.Delta
            gamma = self.gamma
        else:
            h = param[0]
            Delta = param[1]
            gamma = param[2]

        model_params = dict(
            L=self.n,
            S=0.5,  # spin 1/2
            Jx=1 + gamma,
            Jy=1 - gamma,
            Jz=Delta,  # couplings
            hz=-h / 2,
            bc_MPS="finite",
            conserve=conserve,
        )
        M = SpinModel(model_params)
        # product_state = ["up"] * M.lat.N_sites
        product_state = (["up", "down"] * int(self.n / 2 + 1))[
            : self.n
        ]  # initial Neel state
        psi = MPS.from_product_state(M.lat.mps_sites(), product_state, bc=M.lat.bc_MPS)
        dmrg_params = {
            "mixer": True,  # setting this to True helps to escape local minima
            "trunc_params": {
                "chi_max": 100,
                "svd_min": 1.0e-10,
            },
            "max_E_err": 1.0e-10,
        }
        info = dmrg.run(psi, M, dmrg_params)
        E = info["E"]
        if verbose:
            print("finite DMRG, Heisenberg XYZ chain")
            print(
                "Jz={Jz:.2f}, conserve={conserve!r}".format(Jz=Delta, conserve=conserve)
            )
            print("E = {E:.13f}".format(E=E))
            print("final bond dimensions: ", psi.chi)
            Sz = psi.expectation_value(
                "Sz"
            )  # Sz instead of Sigma z: spin-1/2 operators!
            mag_z = np.mean(Sz)
            print(
                "<S_z> = [{Sz0:.5f}, {Sz1:.5f}]; mean ={mag_z:.5f}".format(
                    Sz0=Sz[0], Sz1=Sz[1], mag_z=mag_z
                )
            )
        return E * 4, psi, M


class Heisenberg2D(Hamiltonian):
    def __init__(self, system_size, periodic=True):
        super().__init__()
        self.system_size = torch.tensor(system_size).reshape(-1)
        self.n_dim = len(self.system_size)
        assert self.n_dim == 2
        self.n = self.system_size.prod()
        self.param_dim = 0
        self.param_range = torch.zeros(2, 0)
        self.J = 1
        self.periodic = periodic
        self.connections = generate_spin_idx(
            self.system_size, "nearest_neighbor", periodic
        )
        self.H = [(["XX", "YY", "ZZ"], [-self.J, -self.J, self.J], self.connections)]

        self.symmetry = Symmetry2D(self.system_size[0], self.system_size[1])
        self.symmetry.add_symmetry("reflection_x")
        self.symmetry.add_symmetry("reflection_y")
        self.symmetry.add_symmetry("spin_inversion")
        self.symmetry.add_symmetry("U1")

    def update_param(self, param):
        pass

    def full_H(self, param=None):
        J = self.J
        self.Hamiltonian = sparse.csr_matrix((2**self.n, 2**self.n), dtype=np.float64)
        for conn in self.connections:
            JZZ = 1
            Jpm = 1
            Jmp = 1
            for i in range(self.n):
                if i == conn[0]:
                    JZZ = sparse.kron(JZZ, Z, format="csr")
                    Jpm = sparse.kron(Jpm, Sp, format="csr")
                    Jmp = sparse.kron(Jmp, Sm, format="csr")
                elif i == conn[1]:
                    JZZ = sparse.kron(JZZ, Z, format="csr")
                    Jpm = sparse.kron(Jpm, Sm, format="csr")
                    Jmp = sparse.kron(Jmp, Sp, format="csr")
                else:
                    JZZ = sparse.kron(JZZ, I, format="csr")
                    Jpm = sparse.kron(Jpm, I, format="csr")
                    Jmp = sparse.kron(Jmp, I, format="csr")
            self.Hamiltonian = self.Hamiltonian + J * JZZ - 2 * J * (Jpm + Jmp)
        return self.Hamiltonian


class J1J2(Hamiltonian):
    # Marshall sign rule applied
    def __init__(self, system_size, periodic=True, sign_rule=0):
        # sign_rule: 0: no sign rule applied
        #            1: J1 dominant, apply Sz in checkerboard-like fashion
        #            2: J2 dominant, apply Sz every other row
        super().__init__()
        self.system_size = torch.tensor(system_size).reshape(-1)
        self.n_dim = len(self.system_size)
        self.n = self.system_size.prod()
        assert self.n_dim == 2
        self.param_dim = 1
        self.param_range = torch.tensor([[0.5], [0.5]])
        self.J1 = 1
        self.J2 = 0.5
        self.periodic = periodic
        self.sign_rule = sign_rule
        self.J1_conn = generate_spin_idx(
            self.system_size, "nearest_neighbor", periodic
        )  # (n_J1_conn, 2)
        self.J2_conn = generate_spin_idx(
            self.system_size, "next_nearest_neighbor", periodic
        )  # (n_J2_conn, 2)
        self.n_J1_conn = len(self.J1_conn)
        self.n_J2_conn = len(self.J2_conn)
        self.n_conn = self.n_J1_conn + self.n_J2_conn
        if sign_rule == 0:
            coef = torch.cat(
                [
                    self.J1 * torch.ones(self.n_J1_conn),
                    self.J2 * torch.ones(self.n_J2_conn),
                ],
                dim=0,
            )  # (n_conn, )
            self.connections = torch.cat(
                [self.J1_conn, self.J2_conn], dim=0
            )  # (n_conn, 2)
            self.H = [(["XX", "YY", "ZZ"], [coef] * 3, self.connections)]
        elif sign_rule == 1:
            coef_xy = torch.cat(
                [
                    -self.J1 * torch.ones(self.n_J1_conn),
                    self.J2 * torch.ones(self.n_J2_conn),
                ],
                dim=0,
            )  # (n_conn, )
            coef_z = torch.cat(
                [
                    self.J1 * torch.ones(self.n_J1_conn),
                    self.J2 * torch.ones(self.n_J2_conn),
                ],
                dim=0,
            )  # (n_conn, )
            self.connections = torch.cat(
                [self.J1_conn, self.J2_conn], dim=0
            )  # (n_conn, 2)
            self.H = [
                (["XX", "YY", "ZZ"], [coef_xy, coef_xy, coef_z], self.connections)
            ]
        elif sign_rule == 2:
            self.J1_conn_h = generate_spin_idx(
                self.system_size, "nn_horizontal", periodic
            )  # (n_J1_conn_h, 2)
            self.J1_conn_v = generate_spin_idx(
                self.system_size, "nn_vertical", periodic
            )  # (n_J1_conn_v, 2)
            self.connections = torch.cat(
                [self.J1_conn_h, self.J1_conn_v, self.J2_conn], dim=0
            )  # (n_conn, 2)
            coef_xy = torch.cat(
                [
                    self.J1 * torch.ones(len(self.J1_conn_h)),
                    -self.J1 * torch.ones(len(self.J1_conn_v)),
                    -self.J2 * torch.ones(self.n_J2_conn),
                ],
                dim=0,
            )  # (n_conn, )
            coef_z = torch.cat(
                [
                    self.J1 * torch.ones(self.n_J1_conn),
                    self.J2 * torch.ones(self.n_J2_conn),
                ],
                dim=0,
            )
            self.H = [
                (["XX", "YY", "ZZ"], [coef_xy, coef_xy, coef_z], self.connections)
            ]
        else:
            raise ValueError("sign_rule must be 0, 1, or 2")
        # self.Px = 1
        # self.Py = 1
        # self.Z2 = 1
        self.symmetry = Symmetry2D(self.system_size[0], self.system_size[1])
        # self.symmetry.add_symmetry('reflection_x', self.Px)
        # self.symmetry.add_symmetry('reflection_y', self.Py)
        # self.symmetry.add_symmetry('spin_inversion', self.Z2)
        self.symmetry.add_symmetry("spin_inversion")
        self.symmetry.add_symmetry("U1")

        # self.symmetry = None

    def update_param(self, param):
        # param: (1, ), J2
        J1 = self.J1
        J2 = param
        if self.sign_rule == 0:
            coef = torch.cat(
                [J1 * torch.ones(self.n_J1_conn), J2 * torch.ones(self.n_J2_conn)],
                dim=0,
            )
            new_coef = [coef] * 3
        elif self.sign_rule == 1:
            coef_xy = torch.cat(
                [-J1 * torch.ones(self.n_J1_conn), J2 * torch.ones(self.n_J2_conn)],
                dim=0,
            )  # (n_conn, )
            coef_z = torch.cat(
                [J1 * torch.ones(self.n_J1_conn), J2 * torch.ones(self.n_J2_conn)],
                dim=0,
            )  # (n_conn, )
            new_coef = [coef_xy, coef_xy, coef_z]
        elif self.sign_rule == 2:
            coef_xy = torch.cat(
                [
                    J1 * torch.ones(len(self.J1_conn_h)),
                    -J1 * torch.ones(len(self.J1_conn_v)),
                    -J2 * torch.ones(self.n_J2_conn),
                ],
                dim=0,
            )
            coef_z = torch.cat(
                [J1 * torch.ones(self.n_J1_conn), J2 * torch.ones(self.n_J2_conn)],
                dim=0,
            )
            new_coef = [coef_xy, coef_xy, coef_z]
        else:
            raise ValueError(f"sign_rule {self.sign_rule} not supported")
        self.H[0][1][0:3] = new_coef

    def full_H(self, param=0.5):
        if isinstance(param, torch.Tensor):
            param = param.detach().cpu().numpy().item()
        self.update_param(param)
        self.Hamiltonian = sparse.csr_matrix((2**self.n, 2**self.n), dtype=np.float64)
        for conn_idx, conn in enumerate(self.connections):
            JZZ = 1
            Jpm = 1
            Jmp = 1
            for i in range(self.n):
                if i == conn[0]:
                    JZZ = sparse.kron(JZZ, Z, format="csr")
                    Jpm = sparse.kron(Jpm, Sp, format="csr")
                    Jmp = sparse.kron(Jmp, Sm, format="csr")
                elif i == conn[1]:
                    JZZ = sparse.kron(JZZ, Z, format="csr")
                    Jpm = sparse.kron(Jpm, Sm, format="csr")
                    Jmp = sparse.kron(Jmp, Sp, format="csr")
                else:
                    JZZ = sparse.kron(JZZ, I, format="csr")
                    Jpm = sparse.kron(Jpm, I, format="csr")
                    Jmp = sparse.kron(Jmp, I, format="csr")
            coef_xy = self.H[0][1][0][conn_idx].detach().cpu().numpy().item()
            coef_z = self.H[0][1][2][conn_idx].detach().cpu().numpy().item()
            self.Hamiltonian = (
                self.Hamiltonian + coef_z * JZZ + 2 * coef_xy * (Jpm + Jmp)
            )
        return self.Hamiltonian


if __name__ == "__main__":
    try:
        os.mkdir("results/")
    except FileExistsError:
        pass
    ns = np.arange(10, 80, 2)
    param = 1.0
    E_dmrgs = np.zeros(len(ns))
    for i, n in enumerate(ns):
        H = Ising([n], periodic=False)
        E_dmrg, psi, M = H.DMRG(param, verbose=True)
        E_dmrgs[i] = E_dmrg / n
        print(f"n = {n}, E_dmrg = {E_dmrg/n}")
        with open(f"results/{type(H).__name__}_E_sizes.npy", "wb") as f:
            np.save(f, E_dmrgs)

    # H = Ising([10], periodic=False)
    # E_ground = H.calc_E_ground()
    # print(E_ground)
