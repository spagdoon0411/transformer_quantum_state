import torch
import numpy as np
import os
from hamiltonians.Ising import Ising

from model.model_batched import TransformerModel
from optimizers.optimizer_supervised_batches import Optimizer
from torch.utils.tensorboard import SummaryWriter
import pickle
from cuda_setup import cuda_setup

cuda_setup()
torch.set_default_dtype(torch.float32)

system_sizes = torch.arange(15, 15 + 1, 1).reshape(-1, 1)
Hamiltonians = [Ising(size, periodic=True, get_basis=True) for size in system_sizes]
data_dir_path = os.path.join("TFIM_ground_states", "h_0.6_new")

def batch_from_n(n: int):
    # Reference point: batch_size_dyn(15) = 16384
    return int(2 ^ (30 - n - 1))


for ham in Hamiltonians:
    ham.load_dataset(
        data_dir_path,
        # batch_size=batch_from_n(ham.n),
        batch_size=30000,
    )

param_dim = Hamiltonians[0].param_dim
embedding_size = 32
n_head = 8
n_hid = embedding_size
n_layers = 8
dropout = 0
minibatch = 10000
param_range = None
point_of_interest = None
use_SR = False

compat_dict = {
    "system_sizes": system_sizes,
    "param_range": None,
}

model = TransformerModel(
    n_dim=1,
    param_dim=param_dim,
    embedding_size=embedding_size,
    n_head=n_head,
    n_hid=n_hid,
    n_layers=n_layers,
    possible_spin_vals=2,
    compat_dict=compat_dict,
    dropout_encoding=dropout,
    dropout_transformer=dropout,
    minibatch=minibatch,
)

results_dir = "results"
paper_checkpoint_name = "ckpt_100000_Ising_32_8_8_0.ckpt"
paper_checkpoint_path = os.path.join(results_dir, paper_checkpoint_name)
checkpoint = torch.load(paper_checkpoint_path)
model.load_state_dict(checkpoint)

drmg40path = os.path.join("results", "E_dmrg_40.npy")
dmrg40 = np.load(drmg40path)
dmrg40 = torch.tensor(dmrg40, dtype=torch.float32)

ising40 = Ising(
    torch.tensor([40]),
    periodic=False,
    get_basis=False,
)

dmrg40_h_values = torch.linspace(0, 2, 101)
oneidx = torch.where(dmrg40_h_values.isclose(torch.tensor(0.6)))[0][0]
print(f"Investigating h = {0.6} at index {oneidx}")

# Recommended lr range: 1e-9 to 1e-2
opt = Optimizer(model, Hamiltonians, lr=1e-7, beta1=0.9, beta2=0.98, point_of_interest=point_of_interest)

TRIAL_NUM = 74

writer = SummaryWriter(f"runs/run{TRIAL_NUM}")

opt.train(
    epochs=30000,
    monitor_params=torch.tensor([[0.6]]),
    monitor_hamiltonians=[ising40],
    monitor_energies=torch.tensor([[dmrg40[oneidx]]]),
    writer=writer,
    prob_weight=10**6,
    arg_weight=0.5,
    run_num=TRIAL_NUM,
    energy_error_frequency=10 # Number of batches after which energy error estimate is calculated
)

errors1 = torch.tensor(opt.E_errors_all)

pickle.dump(errors1, open(f"errors1_trial{TRIAL_NUM}.pkl", "wb"))
