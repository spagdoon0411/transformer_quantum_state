
class IsingIterableDatasetSequential(IterableDataset):
    def __init__(self, dataset, batch_size, basis):
        """
        Parameters:
            dataset: pd.DataFrame
                The dataset containing the ground states. Should allow for .iloc indexing.
            batch_size: int
                The number of sites to sample in each batch
            basis: torch.Tensor (n, 2**n)
                All energy eigenstates of the Hamiltonian (i.e., all possible spin sequences).
        """

        # A reference to the DataFrame
        self.dataset = dataset

        # For now, hold all ground states as a tensor in memory. TODO: is it reasonable to
        # lazy-load this?
        self.ground_state_tensor = torch.tensor(df["state"].tolist(), device="cpu")

        # The number of ground states in the dataset
        self.dataset_len = self.ground_state_tensor.shape[0]

        # The number of sites in the system
        self.n = basis.shape[0]

        # The batch size to use for returning samples
        self.batch_size = batch_size

        # The basis of all possible spin configurations
        self.basis = basis

        # The number of entries in the ground state tensor
        self.total_prob_amps = self.dataset_len * (2**self.n)

    def __iter__(self):
        # If all of the ground states were stitched end to end, this would be
        # the resulting tensor's length

        for i in range(0, self.dataset_len * self.total_prob_amps, self.batch_size):
            # The flattened index that corresponds to this batch. Note that the minimum of
            # i + batch_size and total_prob_amps is used to avoid indexing past the end of
            # the dataset. TODO: assumption: materializing this range is not resource-intensive
            flattened_idx = torch.arange(
                i,
                min(i + self.batch_size, self.total_prob_amps),
                device="cpu",
            )

            # The indices of ground states needed for this batch, in order of access. 
            # Strictly unique indices.
            ground_state_idx = torch.arange(
                i // (2**self.n),
                min((i + self.batch_size) // (2**self.n), self.total_prob_amps) + 1,
                device="cpu",
            )


            # The ground states accessed in this batch, as a shallow copy of the
            # ground states tensor
            ground_states = self.ground_state_tensor[ground_state_idx]
            params = self.dataset["param"].iloc[ground_state_idx]
            states_params_apply_to = 



            # A view of the ground state tensor as a flattened array of all
            # ground states accessed. TODO: assuming that .view is a shallow copy
            # and that a function mapping flattened indices to multidimensional indices
            # is stored somewhere in the wrapper object that .view returns.
            labels = (ground_states.view(-1))[flattened_idx]

            # Indices of the basis states that correspond to the ground states accessed.
            # Do this operation in place to avoid memory overhead.
            flattened_idx = torch.remainder(flattened_idx, 2**self.n)

            # Obtains the basis states corresponding to each probability amplitude in the wave
            # function
            basis_states = self.basis[:, flattened_idx].to("cuda")

            yield basis_states, labels