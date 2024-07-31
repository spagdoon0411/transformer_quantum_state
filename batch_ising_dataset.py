import torch
from torch.utils.data import IterableDataset, Dataset, RandomSampler
import math


class IsingIterableDatasetSequential(IterableDataset):
    def __init__(self, dataframe, batch_size, basis):
        """
        Parameters:
            dataframe: pd.DataFrame
                The dataset containing the ground states.
            batch_size: int
                The number of sites to sample in each batch
            basis: torch.Tensor (n, 2**n)
                All energy eigenstates of the Hamiltonian (i.e., all possible spin sequences).
        """

        self.dataframe = dataframe
        self.ground_state_tensor = torch.tensor(dataframe["state"].tolist())

        self.param_tensor = torch.tensor(dataframe["h"])

        # Number of sites in the system
        self.n = basis.shape[0]

        self.batch_size = batch_size
        self.basis = basis

        # Ground state tensor dimensions. The second (ground_state_length)
        # is important for flattened indexing.
        self.dataset_len = self.ground_state_tensor.shape[0]
        self.ground_state_length = 2**self.n

        # The number of labels (probability amplitudes) in the dataset
        self.total_prob_amps = self.dataset_len * self.ground_state_length

    def __iter__(self):
        # Iterate over probability amplitude entries in batch_size steps
        for i in range(0, self.dataset_len * self.ground_state_length, self.batch_size):

            # Flattened index of the start and end of the batch in the ground state tensor
            start_idx = i
            end_idx = min(i + self.batch_size, self.total_prob_amps)
            flattened_idx = torch.arange(start_idx, end_idx)
            labels = self.ground_state_tensor.view(-1)[flattened_idx]

            # Indices for the basis states (circular indexing of the basis set)
            basis_idx = torch.remainder(flattened_idx, self.ground_state_length)
            basis_states = self.basis[:, basis_idx]

            # Indices for the parameters (mapping flattened indices to the ground state or row
            # index in the ground state tensor)
            param_idx = flattened_idx // self.ground_state_length
            params = self.param_tensor[torch.tensor(param_idx)].unsqueeze(1)

            yield basis_states, params, labels


class IsingDataset(Dataset):
    def __init__(self, dataframe, basis):
        self.dataframe = dataframe
        self.ground_state_tensor = torch.tensor(dataframe["state"].tolist())
        self.param_tensor = torch.tensor(dataframe["h"])

        self.n = basis.shape[0]
        self.basis = basis

        self.dataset_len = self.ground_state_tensor.shape[0]
        self.ground_state_length = 2**self.n

        self.total_prob_amps = self.dataset_len * self.ground_state_length

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        """
        Indexes into the flattened ground state tensor. Will return that entry, along
        with the next batch_size - 1 entries in the tensor to produce a batch_size
        sequence of ground states.

        Parameters:
            idx: torch.Tensor of integers
                The flattened indices of the ground state tensor (i.e., indices of probability amplitudes)

        Returns:
            basis_states: torch.Tensor (n, batch_size)
                The basis states corresponding to the probability amplitudes
            params: torch.Tensor (batch_size, 1)
                The parameters corresponding to the probability amplitudes
            labels: torch.Tensor (batch_size)
                The probability amplitudes
        """

        labels = self.ground_state_tensor.view(-1)[idx]
        basis_idx = torch.remainder(idx, self.ground_state_length)
        basis_states = self.basis[:, basis_idx]
        param_idx = idx // self.ground_state_length
        params = self.param_tensor[torch.tensor(param_idx)].unsqueeze(1)

        return basis_states, params, labels


# class SlowTensor(torch.Tensor):
#     def __init__(self, base_tensor: torch.tensor):
#         """
#         A thin wrapper around a PyTorch tensor that has the same effect as repeat_interleave
#         without the memory overhead.
#
#         Parameters:
#             base_tensor: torch.Tensor
#                 The tensor to "slow down" (i.e., repeat each element of)
#             slowness: int
#                 The factor by which to slow down the tensor (i.e., the number of times to repeat each element)
#         """
#         self.slowness = 1
#         self.base_tensor = base_tensor
#         super().__init__()
#
#     def set_slowness(self, slowness):
#         self.slowness = slowness
#
#     def __getitem__(self, idx):
#         return super().__getitem__(idx // self.slowness)
#
#     def __len__(self):
#         return self.n * self.slowness


class IsingRandomSampler:
    def __init__(
        self,
        data_source,
        replacement=False,
        num_samples=None,
        generator=None,
        batch_size=100,
        probability_distribution=None,
    ):
        """
        A sampler that produces data points of the same format as IsingIterableDatasetSequential
        but where each member of a batch is sampled from a probability distribution. Sampling
        can be done with or without replacement. If sampling is done without replacement,
        no probability amplitude is sampled more than once.

        NOTE: this is not a subclass of other PyTorch samplers, as it is tailored for the IsingDataset's
        batch retrieval interface and relies on the dataset's internal tensors.

        Parameters:
            data_source: IsingDataset
                The dataset to sample from. STRICTLY an IsingDataset object, breaking with PyTorch's
                sampler interface.
            replacement: bool
                Whether to sample with replacement
            num_samples: int
                The number of samples to draw
            generator: torch.Generator
                The random number generator to use
            batch_size: int
                The number of sites to sample in each batch
        """
        # super().__init__(data_source, replacement, num_samples, generator)
        self.data_source = data_source
        self.replacement = replacement
        self.num_samples = num_samples
        self.generator = generator

        self.param_tensor = data_source.param_tensor
        self.num_points = self.param_tensor.shape[0]
        self.basis = data_source.basis
        self.ground_state_length = data_source.ground_state_length

        if probability_distribution is not None:
            self.set_sampling_distribution(probability_distribution)
        else:
            self.param_probabilities = torch.ones(
                data_source.dataset_len, dtype=torch.float64
            )

            # self.prob_amp_probabilities = SlowTensor(self.param_probabilities)
            # self.prob_amp_probabilities.set_slowness(self.ground_state_length)

            # This is what SlowTensor was meant to do.
            self.prob_amp_probabilities = self.__get_slow_view(
                self.param_probabilities, self.ground_state_length
            )

        self.batch_size = batch_size
        self.sampled = torch.zeros(self.num_points, self.ground_state_length)

    def __get_slow_view(self, tensor, slowness=1):
        first_dim = tensor.shape[0]
        return (
            tensor.view(first_dim, 1).expand(first_dim, slowness).contiguous().view(-1)
        )

    def set_sampling_distribution(self, function):
        """
        Parameters:
            function: callable | None
                A function that takes a (n_points, n_params) tensor of points in parameter
                space (where each row corresponds to a point) and returns an (n_points, )
                tensor of probabilities.

        Returns:
            None
        """
        # Used for setting probabilities close to zero to zero and avoiding numeric instability
        ZEROTOL = 1e-9

        # Test whether this probability distribution function is acceptable
        all_nonnegative = True
        shape_correct = True
        try:
            # Attempt to run it on the entire parameter tensor
            testres = function(self.param_tensor)

            # Ensure that the output is a tensor of probabilities with the correct shape
            # (n_points, )
            num_points = self.param_tensor.shape[0]
            shape_correct = testres.shape == (num_points,)
            assert shape_correct

            # Set any probabilities close to zero to zero; it may be that this probability distribution
            # function is subject to floating point inaccuracies
            testres[
                testres.isclose(torch.tensor(0.0, dtype=torch.float64), atol=ZEROTOL)
            ] = 0.0

            # Ensure that the probabilities are non-negative
            all_nonnegative = torch.all(testres >= 0.0)
            assert all_nonnegative

            self.probability_distribution = function
            self.param_probabilities = testres
            self.prob_amp_probabilities = self.__get_slow_view(
                self.param_probabilities, self.ground_state_length
            )
            self.sampled = torch.zeros(self.num_points, self.ground_state_length)
            # self.prob_amp_probabilities = SlowTensor(
            #     self.param_probabilities
            # ).set_slowness(self.ground_state_length)

        except Exception as e:
            if not shape_correct:
                raise ValueError(
                    f"""The probability distribution output a shape of {testres.shape} \
and the expected shape was {(num_points, )}. The error message was: {e}"""
                )

            if not all_nonnegative:
                raise ValueError(
                    f"""The probability distribution output negative probabilities. The error message was: {e}"""
                )

            error_msg = f"""The probability distribution must successfully take a (n_points, n_params) \
tensor of points in parameter space and return an (n_points, 1) tensor of probabilities. The function output \
shape was {testres.shape} and the expected shape was {(num_points, 1)}. The error message was: {e}"""

            raise ValueError(error_msg)

    def __iter__(self):

        for i in range(self.num_samples):
            sampled_indices = torch.multinomial(
                self.prob_amp_probabilities,
                self.batch_size,
                replacement=self.replacement,
            )

            if not self.replacement:
                self.prob_amp_probabilities[sampled_indices] = 0.0

            self.sampled.view(-1)[sampled_indices] += 1

            yield self.data_source[sampled_indices]

    def __len__(self):
        return len(self.data_source)
