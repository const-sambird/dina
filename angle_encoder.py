import torch

class AngleEncoder(torch.nn.Module):
    def __init__(self, n_obs):
        '''
        The AngleEncoder transforms a [num_replicas, num_candidates] state matrix
        into a format suitable for encoding into qubits by rotating them by
        somewhere between 0 and pi radians around a Bloch sphere.
        '''
        super(AngleEncoder, self).__init__()

    def forward(self, x):
        pass