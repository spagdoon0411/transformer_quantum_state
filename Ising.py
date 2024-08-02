from Hamiltonian import Hamiltonian

import os
import numpy as np
import pyarrow as pa
import json
import numpy as np
from scipy import sparse
import torch
from tenpy.networks.mps import MPS
from tenpy.models.spins import SpinModel
from tenpy.algorithms import dmrg
from Hamiltonian_utils import generate_spin_idx
from symmetry import Symmetry1D, Symmetry2D

from batch_ising_dataset import (
    IsingRandomSampler,
    IsingDataset,
    IsingIterableDatasetSequential,
)

from torch.utils.data import RandomSampler, BatchSampler


class Ising(Hamiltonian):
    def __init__(self, system_size, periodic=True, get_basis=False):
        """
        Parameters:
        system_size: torch.Tensor (n_dim, n_systems)
            The size of the system in each spatial dimension. Each column corresponds to a single system;
            each row corresponds to a physical dimension. For instance, if this tensor is [[10, 30], [50, 70]],
            then it describes two systems: one with size 10x50 and one with size 30x70.

        periodic: bool
            Whether the lattice is periodic in each dimension. If True, the spins on the boundaries interact in the same
            way that adjacent spins in the interior do.

        """
        super().__init__()
        self.system_size = system_size
        self.periodic = periodic
        self.n_dim = len(self.system_size)
        self.n = self.system_size.prod()

        self.param_dim = 1
        self.param_range = torch.tensor([[0.5], [1.5]])
        self.J = -1
        self.h = 1
        self.connections = generate_spin_idx(
            self.system_size.T, "nearest_neighbor", periodic
        )
        self.external_field = generate_spin_idx(
            self.system_size.T, "external_field", periodic
        )
        self.H = [
            (["ZZ"], [self.J], self.connections),
            (["X"], [self.h], self.external_field),
        ]

        if get_basis:
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
        self.training_dataset = None

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

    def load_dataset(
        self,
        data_dir_path: str,
        batch_size: int = 1000,
        samples_in_epoch=100,
        sampling_type="shuffled",
    ):
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
        batch_size: int
            The number of batches to retrieve from this Hamiltonian's internal
            IsingIterableDatasetSequential at a time for training. Note that the
            model is agnostic to the batch size, so this can be set to something that
            is reasonable given the sizes of the Hamiltonian's Hilbert space.
        samples_in_epoch: int
            The number of batch_size-sized samples that constitute an epoch of the
            sampler
        sampling_type: str
            "random" or "sequential".


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

        # TODO: assert that dataset has the right row dimension (2**n) and an appropriate
        # number of rows. Provide a warning if some rows are missing.

        if h_min != self.param_range[0].item() or h_max != self.param_range[1].item():
            warning = """Warning: h values in metadata file do not match Hamiltonian's param_range; \
found h_min={0}, h_max={1}, h_step={2}, expected h_min={3}, h_max={4}. Setting param_range to match.""".format(
                h_min, h_max, h_step, this_h_min, this_h_max
            )

            self.param_range = torch.tensor([[h_min], [h_max]])

            print(warning)

        self.h_step = h_step

        # self.training_dataset = IsingIterableDatasetSequential(
        #     self.dataset, batch_size, self.basis
        # )

        if sampling_type == "random":
            self.training_dataset = IsingRandomSampler(
                data_source=IsingDataset(self.dataset, self.basis),
                replacement=True,
                num_samples=samples_in_epoch,
                batch_size=batch_size,
            )
        elif sampling_type == "sequential":
            self.training_dataset = IsingIterableDatasetSequential(
                self.dataset, batch_size, self.basis
            )
        elif sampling_type == "shuffled":
            generator = torch.Generator(device="cuda")
            dataset = IsingDataset(self.dataset, self.basis)
            random_sampler = RandomSampler(
                dataset, replacement=False, generator=generator
            )
            batched_sampler = BatchSampler(random_sampler, batch_size, drop_last=False)
            self.sampler = batched_sampler
            self.training_dataset = dataset
        else:
            raise ValueError("Sampling type must be 'random' or 'sequential'")

        print(
            f"Loaded dataset for system size {self.n} from {file_path}.\n(h_min, h_step, h_max) = ({h_min}, {h_step}, {h_max})."
        )

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
