
from hamiltonians.Ising import Ising
import torch

mmap_dir = "mmap_data"
test_ham = Ising(torch.tensor([12]), periodic=True, get_basis=False)

test_ham.load_mmap(mmap_dir)

for sample in test_ham.training_dataset:
    print(sample)
    break