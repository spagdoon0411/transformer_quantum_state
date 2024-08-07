
from hamiltonians.Ising import Ising
import torch
import time
from torch.utils.data import RandomSampler, BatchSampler

def gpu_setup():
    # Setup for PyTorch:
    if torch.cuda.is_available():
        torch_device = torch.device("cuda")
        print("PyTorch is using GPU {}".format(torch.cuda.current_device()))
    else:
        torch_device = torch.device("cpu")
        print("GPU unavailable; using CPU")


def prob_amp_collate(batch):
    """
    A collate function to be used with a DataLoader that will convert a batch of
    probability amplitudes into a tensor that can be used to train a TQS model.
    """
    start = time.time()
    b0 = torch.stack([b[0] for b in batch])# .to(device="cuda")
    b1 = torch.stack([b[1] for b in batch])# .to(device="cuda")
    b2 = torch.stack([b[2] for b in batch])# .to(device="cuda")
    end = time.time()
    print("Time to stack tensors: ", end-start)
    return b0, b1, b2

if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    gpu_setup()
    torch.set_default_device("cuda")
    torch.set_default_dtype(torch.float32)
    mmap_dir = "mmap_data"
    print("Producing Ising objects")
    test_ham = Ising(torch.tensor([12]), periodic=True, get_basis=False)
    print("Done producing Ising objects")

    print("Loading mmap data")
    test_ham.load_mmap(prob_amp_collate, mmap_dir, batch_size=5)
    print("Done loading mmap data")

    test_sampler = BatchSampler(RandomSampler(test_ham.underlying, generator=torch.Generator(device="cuda")), batch_size=1000, drop_last=False)

    print("Iterating through training dataset")
    iter = 0
    start = time.time()
    for sample in test_ham.training_dataset:
        end = time.time()
        print(f"Time to load sample: {end-start}")
        spins, params, labels = sample
        spins.to(device="cuda") 
        params.to(device="cuda")
        labels.to(device="cuda")
        print("Sample shape: ", sample[0].shape, sample[1].shape, sample[2].shape)
        iter += 1
        if iter > 10:
           break
        start = time.time()
    print("Done iterating through training dataset")


    # iter = 0
    # start = time.time()
    # for sample in test_ham.training_dataset:
    #     end = time.time()
    #     print(f"Time to load sample: {end-start}")
    #     time.sleep(1)
    #     iter += 1
    #     if iter > 10:
    #        break
    #     start = time.time()
    # print("Done iterating through training dataset")

    
