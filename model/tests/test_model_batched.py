from model.model_batched import TransformerModel as TransformerModelBatched
from model.model import TransformerModel
import numpy as np
from hamiltonians.Ising import Ising
import os
import torch
import math
from model.model_utils import (
    sample,
    compute_psi,
    compute_grad,
    compute_observable,
    compute_flip,
    compute_phase,
)


import unittest


class TestEquivalentInitialization(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestEquivalentInitialization, self).__init__(*args, **kwargs)

        if torch.cuda.is_available():
            torch_device = torch.device("cuda")
            print("PyTorch is using GPU {}".format(torch.cuda.current_device()))
        else:
            torch_device = torch.device("cpu")
            print("GPU unavailable; using CPU")

        torch.set_default_device("cuda")
        torch.set_default_dtype(torch.float32)

        system_sizes = np.arange(10, 20, 2).reshape(-1, 1)

        param_dim = 1
        embedding_size = 32
        n_head = 8  # Attention heads
        n_hid = embedding_size  # Number of hidden units in the feedforward network
        n_layers = 8  # Number of transformer layers
        dropout = 0  # Dropout rate
        minibatch = 10000  # Batch size

        self.old = TransformerModel(
            system_sizes,
            param_dim,
            embedding_size,
            n_head,
            n_hid,
            n_layers,
            dropout=dropout,
            minibatch=None,
        )

        compat_dict = {
            "system_sizes": system_sizes,
            "param_range": self.old.param_range,
        }

        self.new = TransformerModelBatched(
            n_dim=system_sizes.shape[1],
            param_dim=param_dim,
            embedding_size=embedding_size,
            n_head=n_head,
            n_hid=n_hid,
            n_layers=n_layers,
            possible_spin_vals=2,
            dropout_encoding=dropout,
            dropout_transformer=dropout,
            chunk_size=None,
            compat_dict=compat_dict,
        )

        self.old.cuda()
        self.new.cuda()

    def test_equivalent_structure(self):

        # Attempt to load model weights and verify that both models produce
        # the same outputs on a forward call.

        results_dir = "results"
        paper_checkpoint_name = "ckpt_100000_Ising_32_8_8_0.ckpt"
        paper_checkpoint_path = os.path.join(results_dir, paper_checkpoint_name)
        checkpoint = torch.load(paper_checkpoint_path)

        self.old.load_state_dict(checkpoint)
        self.new.load_state_dict(checkpoint)

        n = 10
        batch_size = 100
        spin_chains = torch.randint(0, 2, (n, batch_size))

        self.old.set_param(system_size=torch.tensor([n]), param=torch.tensor([1.0]))

        new_param = torch.tensor([1.0]).repeat(batch_size, 1)
        new_system_size = torch.tensor([n])

        # TODO: REMOVE
        # Force equivalent positional encodings
        self.old.pos_encoder.pe = self.new.pos_encoder.pe

        old_output = self.old(spin_chains)
        new_output = self.new.forward_batched(
            params=new_param,
            spins=spin_chains,
            system_size=new_system_size,
            compute_phase=True,
        )

        old_amps, old_phases = old_output
        new_amps, new_phases = new_output

        self.assertTrue(torch.allclose(old_amps, new_amps))
        self.assertTrue(torch.allclose(old_phases, new_phases))

    def test_equivalent_one_hot(self):
        # Load the weights from the paper:
        results_dir = "results"
        paper_checkpoint_name = "ckpt_100000_Ising_32_8_8_0.ckpt"
        paper_checkpoint_path = os.path.join(results_dir, paper_checkpoint_name)
        checkpoint = torch.load(paper_checkpoint_path)

        self.old.load_state_dict(checkpoint)
        self.new.load_state_dict(checkpoint)

        # Create dummy data
        n = 10
        batch_size = 100
        spin_chains = torch.randint(0, 2, (n, batch_size))

        # Produce the one-hot encoding tensor for either model and ensure they're equivalent
        self.old.set_param(system_size=torch.tensor([n]), param=torch.tensor([1.0]))
        old_sequence_encoding = self.old.wrap_spins(spin_chains)

        new_param = torch.tensor([1.0]).repeat(batch_size, 1)
        new_system_size = torch.tensor([n])
        new_sequence_encoding = self.new.wrap_spins_batch(
            new_param, spin_chains, new_system_size
        )

        # Elementwise
        self.assertTrue(
            torch.allclose(old_sequence_encoding, new_sequence_encoding),
            "One-hot encodings are not the same",
        )

        # Compare the encoding of the models before positional encoding
        old_pre_pos_enc = self.old.encoder(old_sequence_encoding) * math.sqrt(
            self.old.embedding_size
        )
        new_pre_pos_enc = self.new.encoder(new_sequence_encoding) * math.sqrt(
            self.new.embedding_size
        )

        # Elementwise
        self.assertTrue(
            torch.allclose(old_pre_pos_enc, new_pre_pos_enc),
            "Pre-position encodings are not the same",
        )

        # TODO: determine why positional encodings are not the same between
        # the two models

        # print("n_dims:", self.old.n_dim, self.new.n_dim)
        # print("param_dims:", self.old.param_dim, self.new.param_dim)
        # print("embedding_size:", self.old.embedding_size, self.new.embedding_size)
        # print("seq_prefix_len:", self.old.seq_prefix_len, self.new.seq_prefix_len)

        # self.new.pos_encoder.pe = self.old.pos_encoder.pe

        # self.old.load_state_dict(checkpoint)
        # old_pos_enc_sanity = self.old.pos_encoder(old_pre_pos_enc)

        # self.assertTrue(
        #     torch.allclose(old_pos_enc, old_pos_enc_sanity),
        #     "Old positional encodings are not the same across loads

    def test_inhomogeneous_params(self):
        # Load the weights from the paper:
        results_dir = "results"
        paper_checkpoint_name = "ckpt_100000_Ising_32_8_8_0.ckpt"
        paper_checkpoint_path = os.path.join(results_dir, paper_checkpoint_name)
        checkpoint = torch.load(paper_checkpoint_path)

        self.old.load_state_dict(checkpoint)
        self.new.load_state_dict(checkpoint)

        # TODO: remove forceful positional encoding reconciliation
        self.new.pos_encoder.pe = self.old.pos_encoder.pe

        params = torch.tensor([[2.0], [0.5], [1.7]])
        spins = torch.randint(0, 2, (10, 3))  # 10 spins, 3 samples

        # Run inference using the new model with each parameter corresponding to
        # one of the three spin chains, in order.
        new_log_amp, new_log_phase = self.new.forward_batched(
            params, spins, torch.tensor([10]), compute_phase=True
        )

        # Produce the same outputs using the old model by setting the parameters
        # one at a time, extracting only the relevant slice of the output tensor,
        # and stacking them together.
        amp_slices = []
        phase_slices = []

        for i, param in enumerate(params):
            self.old.set_param(system_size=torch.tensor([10]), param=param)
            old_log_amp, old_log_phase = self.old(spins)

            old_amp_slice = old_log_amp[:, i]
            old_phase_slice = old_log_phase[:, i]

            amp_slices.append(old_amp_slice)
            phase_slices.append(old_phase_slice)

        old_log_amp = torch.stack(amp_slices, dim=1)
        old_log_phase = torch.stack(phase_slices, dim=1)

        self.assertTrue(
            torch.allclose(old_log_amp, new_log_amp),
            "Amplitudes are not the same between models",
        )

        self.assertTrue(
            torch.allclose(old_log_phase, new_log_phase),
            "Phases are not the same between models",
        )

    def test_old_model_compatibility(self):
        """
        Ensure that the new model's classic interface produces the same forward outputs as the
        old model.
        """

        results_dir = "results"
        paper_checkpoint_name = "ckpt_100000_Ising_32_8_8_0.ckpt"
        paper_checkpoint_path = os.path.join(results_dir, paper_checkpoint_name)
        checkpoint = torch.load(paper_checkpoint_path)

        self.old.load_state_dict(checkpoint)
        self.new.load_state_dict(checkpoint)

        # TODO: why do we have to force positional encoding reconciliation?
        self.new.pos_encoder.pe = self.old.pos_encoder.pe

        n = 10
        batch_size = 100
        spin_chains = torch.randint(0, 2, (n, batch_size))

        self.old.set_param(system_size=torch.tensor([n]), param=torch.tensor([1.3]))
        self.new.set_param(system_size=torch.tensor([n]), param=torch.tensor([1.3]))

        # Verify regular forward pass
        old_output = self.old(spin_chains)
        new_output = self.new(spin_chains)

        old_amps, old_phases = old_output
        new_amps, new_phases = new_output

        self.assertTrue(torch.allclose(old_amps, new_amps))
        self.assertTrue(torch.allclose(old_phases, new_phases))

        test_ising = Ising(torch.tensor([n]), periodic=False, get_basis=False)

        # Verify equality of functions in model_utils
        old_log_amps, old_log_phases = compute_psi(
            self.old, spin_chains, symmetry=test_ising.symmetry
        )

        new_log_amps, new_log_phases = compute_psi(
            self.new, spin_chains, symmetry=test_ising.symmetry
        )

        self.assertTrue(torch.allclose(old_log_amps, new_log_amps))
        self.assertTrue(torch.allclose(old_log_phases, new_log_phases))

        torch.manual_seed(0)
        old_samples, old_weight = sample(
            self.old, batch=1000, max_unique=100, symmetry=test_ising.symmetry
        )

        torch.manual_seed(0)
        new_samples, new_weight = sample(
            self.new, batch=1000, max_unique=100, symmetry=test_ising.symmetry
        )

        self.assertTrue(torch.allclose(old_samples, new_samples))
        self.assertTrue(torch.allclose(old_weight, new_weight))

        E_old = test_ising.Eloc(old_samples, old_weight, self.old, use_symmetry=True)
        E_new = test_ising.Eloc(new_samples, new_weight, self.new, use_symmetry=True)

        self.assertTrue(torch.allclose(E_old, E_new))

        old_loss, old_log_amp, old_log_phase = compute_grad(
            self.old,
            old_samples,
            sample_weight=old_weight,
            Eloc=E_old,
            symmetry=test_ising.symmetry,
        )

        new_loss, new_log_amp, new_log_phase = compute_grad(
            self.new,
            new_samples,
            sample_weight=new_weight,
            Eloc=E_new,
            symmetry=test_ising.symmetry,
        )

        self.assertTrue(torch.allclose(old_loss, new_loss))
        self.assertTrue(torch.allclose(old_log_amp, new_log_amp))
        self.assertTrue(torch.allclose(old_log_phase, new_log_phase))

        # TODO: continue checking the rest of the functions in model_utils


if __name__ == "__main__":
    unittest.main()
