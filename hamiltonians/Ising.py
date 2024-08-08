from hamiltonians.Hamiltonian import Hamiltonian

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
from hamiltonians.Hamiltonian_utils import generate_spin_idx
from hamiltonians.symmetry import Symmetry1D, Symmetry2D
from numpy.lib.format import open_memmap

from datasets.batch_ising_dataset import IsingDataset

from datasets.ising_memmap import ProbabilityAmplitudeDataset

from torch.utils.data import RandomSampler, BatchSampler, DataLoader

pi = np.pi
X = sparse.csr_matrix(np.array([[0, 1], [1, 0]], dtype=np.float64))
Y = sparse.csr_matrix(np.array([[0, -1j], [1j, 0]], dtype=np.complex128))
Z = sparse.csr_matrix(np.array([[1, 0], [0, -1]], dtype=np.float64))
I = sparse.csr_matrix(np.array([[1, 0], [0, 1]], dtype=np.float64))
Sp = sparse.csr_matrix(np.array([[0, 1], [0, 0]], dtype=np.float64))
Sm = sparse.csr_matrix(np.array([[0, 0], [1, 0]], dtype=np.float64))


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

    def load_mmap(
        self,
        collate_fn,
        mmap_dir="mmap_data",
        batch_size=1000,
    ):
        """
        Looks for the meta.json file with keys:
        - num_samples
        - basis_memmap_dir
        - parameter_memmap_dir
        - ground_memmap_dir

        Loads the NumPy array files corresponding to the basis, parameters, and ground states
        from those directores and produces a ProbabilityAmplitudeDataset that provides
        shuffled samples from the memory maps for training.
        """
        metadata_path = os.path.join(mmap_dir, "meta.json")
        with open(metadata_path, "r") as metadata_file:
            metadata = json.load(metadata_file)

        basis_path = os.path.join(
            mmap_dir, metadata["basis_memmap_dir"], f"basis_{self.n}.npy"
        )
        param_path = os.path.join(
            mmap_dir, metadata["parameter_memmap_dir"], f"param_{self.n}.npy"
        )
        ground_state_path = os.path.join(
            mmap_dir, metadata["ground_memmap_dir"], f"ground_{self.n}.npy"
        )
        num_samples = metadata["num_samples"]

        basis = open_memmap(
            basis_path, dtype=np.int32, mode="r", shape=(self.n, 2**self.n)
        )
        parameters = open_memmap(
            param_path, dtype=np.float64, mode="r", shape=(1, num_samples)
        )
        ground_states = open_memmap(
            ground_state_path,
            dtype=np.float64,
            mode="r",
            shape=(num_samples, 2**self.n),
        )
        dataset = ProbabilityAmplitudeDataset(basis, parameters, ground_states)

        # self.sampler = BatchSampler(RandomSampler(dataset, replacement=False, generator=torch.Generator(device="cuda")), batch_size, drop_last=False)
        # self.training_dataset = dataset

        self.training_dataset = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            prefetch_factor=2,
            num_workers=1,
            shuffle=True,
            generator=torch.Generator(device="cuda"),
            pin_memory=True,
        )

        self.underlying = dataset

    def load_dataset(
        self,
        data_dir_path: str,
        batch_size: int = 1000,
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
        - energy - a float
        - state - a list of floats

        Sets this Hamiltonian's self.dataset attribute to the loaded dataset.

        Precomputes a binary representation of the system's states as a PyTorch tensor
        for passing into the model. This is stored in self.basis.

        Produces a warning if the range of h values in the metadata file does not
        match the range of h values in the Hamiltonian's param_range attribute.

        Notes a self.h_step corresponding to the step size in the metadata file. Always uses the
        "h_step" key from the metadata file, regardless of whether the dataset actually uses that
        step size. Recommendation: set h_step = -1 in the metadata file if the samples are
        distributed non-uniformly.

        Parameters:
        data_dir_path: str
            The path to the directory containing the meta.json file
        batch_size: int
            The number of spin chains (and corresponding single probability amplitude
            scalars) to return from the dataset at a time.

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

        generator = torch.Generator(device="cuda")
        dataset = IsingDataset(self.dataset, self.basis)
        random_sampler = RandomSampler(dataset, replacement=False, generator=generator)
        batched_sampler = BatchSampler(random_sampler, batch_size, drop_last=False)
        self.sampler = batched_sampler
        self.training_dataset = dataset

        print(
            f"Loaded dataset for system size {self.n} from {file_path}.\n(h_min, h_step, h_max) = ({h_min}, {h_step}, {h_max})."
        )
