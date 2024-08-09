from jaxtyping import Float, Integer
import torch
from hamiltonians.Ising import Ising
import datetime
import os

__quantities_to_average = [
    "epoch_errors",
    "epoch_relative_errors",
    "epoch_E_mean",
    "epoch_E_var",
    "epoch_Er",
    "epoch_Ei",
]
__averaged_quantities = [
    "epoch_averaged_errors",
    "epoch_averaged_relative_errors",
    "epoch_averaged_E_mean",
    "epoch_averaged_E_var",
    "epoch_averaged_Er",
    "epoch_averaged_Ei",
]


def generate_monitor_dict(
    monitor_sizes: Integer[torch.Tensor, "n_sizes n_dim"],
    monitor_params: Float[torch.Tensor, "param_dim n_points"],
    epochs_anticipated: int,
    key_rounding: int = 4,
):
    """
    Generates a dictionary storing information necessary for monitoring the status of different
    Hamiltonians at different points in parameter space during the training of a model. Note
    that H(J)-level information can be added here by a caller on the fly.

    TODO: NOTE: Only supports one-dimensional Ising instances.

    Form:
    monitor_dict["[10]"]["0.6"]["<H(J)-specific information>"] = <value>

    :param monitor_sizes: A torch tensor of shape (n_sizes, n_dim) where n_sizes is the number of
    different system sizes to monitor and n_dim is the dimension of each system.
    :param monitor_params: A torch tensor of shape (param_dim, n_points) where param_dim is the
    dimension of the parameter space and n_points is the number of points to monitor.
    :param epochs_anticipated: The number of epochs anticipated for training. This is used to
    pre-allocate memory for the epoch_averaged_errors tensors, preventing surprise out-of-memory
    errors and improving performance.
    :param key_rounding: The number of decimal places to round the parameter keys to to avoid
    floating point errors. Default is 4.

    :return: A two-level nested dictionary with the following structure:
    - The first level, indexed by the system sizes as a list of ints (e.g., "[10, 20]")
        - The Hamiltonian for this system size can be accessed here, indexed by "H"
        - The system size as a one-dimensional torch.Tensor can be accessed here, indexed by
        "system_size"
        - The second level, indexed by the parameter values as a list (e.g., "[0.1, 0.2]")
            - "param" - The parameter value itself, stored as a torch.Tensor of one dimension
            - "epochs" - The energy of the Hamiltonian at the given parameter values
            - "<quantity_to_average>" - A quantity in __quantities_to_average, stored as a list
            - "<averaged_quantity>" - The averaged version of the quantity, stored as a torch.Tensor
            of shape (epochs_anticipated,)

    """
    key_rounding = 4

    monitor_dict = {}

    system_size_keys = monitor_sizes.tolist()
    param_keys = monitor_params.transpose(0, 1).tolist()
    param_keys = [
        [round(param, key_rounding) for param in param_key] for param_key in param_keys
    ]

    print(f"System size keys: {system_size_keys}")
    print(f"Param keys: {param_keys}")

    for system_size_key, system_size in zip(system_size_keys, monitor_sizes):
        this_size_dict = {}
        this_size_dict["system_size"] = system_size
        this_size_dict["H"] = Ising(
            system_size=system_size, periodic=True, get_basis=False
        )
        this_size_dict["params"] = {}
        for param_key, param in zip(param_keys, monitor_params.transpose(0, 1)):
            this_param_dict = {}
            this_param_dict["param"] = param
            this_param_dict["energy"] = None  # to be set by caller
            for quantity in __quantities_to_average:
                this_param_dict[quantity] = []

            for averaged_name in __averaged_quantities:
                this_param_dict[averaged_name] = torch.zeros(epochs_anticipated)

            this_size_dict["params"][f"{param_key}"] = this_param_dict
        monitor_dict[f"{system_size_key}"] = this_size_dict

    return monitor_dict


def average_monitor_dict(monitor_dict: dict, epoch: int):
    """
    Averages epoch_errors, epoch_E_mean, epoch_E_var, epoch_Er, and epoch_Ei
    and stores them in their epoch_averaged_* counterparts. Clears
    each quantity's per-epoch buffer to [] after averaging.

    :param monitor_dict: A monitor_dict dictionary; see generate_monitor_dict.
    """

    size_keys = monitor_dict.keys()
    for size_key in size_keys:
        param_keys = monitor_dict[size_key]["params"].keys()
        for param_key in param_keys:
            info_dict = monitor_dict[size_key]["params"][param_key]
            for quantity, averaged_name in zip(
                __quantities_to_average, __averaged_quantities
            ):
                info_dict[averaged_name][epoch] = torch.tensor(
                    info_dict[quantity]
                ).mean()
                info_dict[quantity] = []

    return monitor_dict


def create_save_dirs(save_str: str):
    timestamp = (
        str(datetime.datetime.now())
        .replace(" ", "_")
        .replace(":", "_")
        .replace(".", "_")
        .replace("-", "_")
    )

    # Generate a directory for the training data from this particular run
    result_dir_name = os.path.join("supervised_results", f"{save_str}_{timestamp}")
    checkpointing_dir = os.path.join(result_dir_name, "checkpoints")
    monitoring_dir = os.path.join(result_dir_name, "monitoring_data")
    tensorboard_dir = os.path.join(result_dir_name, "tensorboard_logs")

    if not os.path.exists(result_dir_name):
        os.makedirs(result_dir_name)

    if not os.path.exists(checkpointing_dir):
        os.makedirs(checkpointing_dir)

    if not os.path.exists(monitoring_dir):
        os.makedirs(monitoring_dir)

    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)

    return checkpointing_dir, monitoring_dir, tensorboard_dir
