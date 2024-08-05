import torch
import numpy as np
import os
import json
from hamiltonians.Ising import Ising
from model.model import TransformerModel
from optimizers.optimizer_supervised_batches import Optimizer
from torch.utils.tensorboard import SummaryWriter


def gpu_setup():
    # Setup for PyTorch:
    if torch.cuda.is_available():
        torch_device = torch.device("cuda")
        print("PyTorch is using GPU {}".format(torch.cuda.current_device()))
    else:
        torch_device = torch.device("cpu")
        print("GPU unavailable; using CPU")


gpu_setup()

torch.set_default_device("cuda")
torch.set_default_dtype(torch.float32)

system_sizes = torch.arange(15, 15 + 2, 2).reshape(-1, 1)
Hamiltonians = [Ising(size, periodic=True, get_basis=True) for size in system_sizes]
# data_dir_path = os.path.join("TFIM_ground_states", "2024-08-02T12-12-55.238")
data_dir_path = os.path.join("TFIM_ground_states", "h_0.6")
for ham in Hamiltonians:
    ham.load_dataset(
        data_dir_path,
        batch_size=30000,
        samples_in_epoch=100,
        sampling_type="shuffled",
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

testmodel = TransformerModel(
    system_sizes,
    param_dim,
    embedding_size,
    n_head,
    n_hid,
    n_layers,
    dropout=dropout,
    minibatch=minibatch,
)

testmodel.cuda()

results_dir = "results"
paper_checkpoint_name = "ckpt_100000_Ising_32_8_8_0.ckpt"
paper_checkpoint_path = os.path.join(results_dir, paper_checkpoint_name)
checkpoint = torch.load(paper_checkpoint_path)
testmodel.load_state_dict(checkpoint)

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

opt = Optimizer(testmodel, Hamiltonians, point_of_interest=point_of_interest)

opt.optim = torch.optim.Adam(
    opt.model.parameters(), lr=0.1, betas=(0.9, 0.98), eps=1e-9
)

writer = SummaryWriter("runs/h=0.6_trial15")

opt.train(
    epochs=1000,
    start_iter=0,
    monitor_params=torch.tensor([[0.6]]),
    monitor_hamiltonians=[ising40],
    monitor_energies=torch.tensor([[dmrg40[oneidx]]]),
    writer=writer,
)

errors1 = torch.tensor(opt.E_errors_all)

import matplotlib.pyplot as plt

plt.plot(errors1.flatten().cpu().numpy(), label="errors1")
plt.xlabel("Iteration")
plt.ylabel("Epochs' Errors vs Iteration")
plt.legend()

for i in range(1, 3):
    plt.axvline(x=i * errors1.shape[1], color="r", linestyle="--")

plt.savefig("stable_h=0.6_trial15.png")
