import torch
import numpy as np
import os
import json
from hamiltonians.Ising import Ising
import datetime

# from model.model import TransformerModel


from model.model_batched import TransformerModel
from optimizers.hyperparam_optimizer import Optimizer
from torch.utils.tensorboard import SummaryWriter
import pickle
from ray import tune
from ray.train import get_checkpoint, Checkpoint
from pathlib import Path


def gpu_setup():
    # Setup for PyTorch:
    if torch.cuda.is_available():
        torch_device = torch.device("cuda")
        print("PyTorch is using GPU {}".format(torch.cuda.current_device()))
    else:
        torch_device = torch.device("cpu")
        print("GPU unavailable; using CPU")

def batch_from_n(n: int):
    # Reference point: batch_size_dyn(15) = 16384
    return int(2 ^ (30 - n - 1))

def load_hamiltonians(data_dir_path, system_sizes, batch_size):
    Hamiltonians = [Ising(size, periodic=True, get_basis=True) for size in system_sizes]
    for ham in Hamiltonians:
        ham.load_dataset(
            data_dir_path,
            batch_size=batch_size,
        )
    return Hamiltonians


def train_tqs(config, monitor_params, monitor_hamiltonians, monitor_energies, data_dir=None):
        with torch.device("cuda"):
            system_sizes = torch.arange(15, 15 + 1, 1).reshape(-1, 1)
            Hamiltonians = load_hamiltonians(data_dir, system_sizes, batch_size=config["batch_size"])

            param_dim = Hamiltonians[0].param_dim
            embedding_size = config["embedding_size"]
            n_head = config["n_head"] 
            n_hid = config["n_hid"]
            n_layers = config["n_layers"]
            dropout = config["dropout"]
            minibatch = config["minibatch"]
            point_of_interest = None

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

            opt = Optimizer(model, Hamiltonians, point_of_interest=point_of_interest)

            opt.optim = torch.optim.Adam(
                opt.model.parameters(), lr=config["lr"], betas=(config["beta1"], config["beta2"]), eps=1e-9
            )

            checkpoint = get_checkpoint()
            if checkpoint:
                with checkpoint.as_directory() as checkpoint_dir:
                    data_path = Path(checkpoint_dir) / "data.pkl"
                    with open(data_path, "rb") as f:
                        checkpoint_state = pickle.load(f)
                    start_epoch = checkpoint_state["epoch"]
                    model.load_state_dict(checkpoint_state["net"])
                    opt.load_state_dict(checkpoint_state["optimizer"])
            else:
                start_epoch = 0
                results_dir = os.path.join("/", "home", "spandan", "Projects", "transformer_quantum_state", "results")
                paper_checkpoint_name = "ckpt_100000_Ising_32_8_8_0.ckpt"
                paper_checkpoint_path = os.path.join(results_dir, paper_checkpoint_name)
                checkpoint = torch.load(paper_checkpoint_path)
                model.load_state_dict(checkpoint)

            model.cuda()
            model.to(device="cuda")

            opt.train(
                epochs=3000,
                start_iter=0,
                monitor_params=monitor_params,
                monitor_hamiltonians=monitor_hamiltonians,
                monitor_energies=monitor_energies,
                prob_weight=10**6,
                arg_weight=0.5,
                start_epoch=start_epoch,
            )

from functools import partial
from ray.tune.schedulers import ASHAScheduler

def main(num_samples=1):
    gpu_setup()
    torch.set_default_device("cuda")
    torch.set_default_dtype(torch.float32)
    project_path = os.path.join("/","home", "spandan", "Projects", "transformer_quantum_state")
    data_dir_path = os.path.join(project_path, "TFIM_ground_states", "h_0.6_new_correct")

    # Setting up DMRG data for relative error monitoring
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

    config = {
        "batch_size": tune.choice([5000]),
        "embedding_size": tune.choice([32]),
        "n_head": tune.choice([8]),
        "n_hid": tune.choice([32]),
        "n_layers": tune.choice([8]),
        "dropout": tune.choice([0, 0.1]),
        "minibatch": tune.choice([10000]),
        "lr": tune.loguniform(1e-9, 1e-1),
        "beta1": tune.uniform(0.7, 0.99),
        "beta2": tune.uniform(0.80, 0.99),
    }

    scheduler = ASHAScheduler(
        metric="mean_error_epoch",
        mode="min",
        max_t=3000,
        grace_period=1,
        reduction_factor=2
    )

    monitor_hamiltonians=[ising40]
    monitor_energies=torch.tensor([[dmrg40[oneidx]]], device="cuda")
    monitor_params=torch.tensor([[0.6]], device="cuda")

    result = tune.run(
        partial(
            train_tqs, 
            data_dir=data_dir_path, 
            monitor_hamiltonians=monitor_hamiltonians, 
            monitor_energies=monitor_energies, 
            monitor_params=monitor_params
        ),
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        resources_per_trial={"gpu": 1, "cpu": 0},
    )

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    result_dir = os.path.join(project_path, "hyperparam_results")
    pickle.dump(result, open(f"{result_dir}_{timestamp}.pkl", "wb"))

if __name__ == "__main__":
    main()




