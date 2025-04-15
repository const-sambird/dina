import torch
from torch import nn
import math

class StateEncoder(nn.Module):
    '''
    For use in quantum encoding: collapse the state matrix into a vector
    where each 'bit' represents a replica that the given index candidate is
    present in.

    For example, suppose we have an index candidate present in replicas
    0, 1, and 4 (but not 2 or 3). This candidate is then encoded with the
    binary value:

    ```
    replica | 4 | 3 | 2 | 1 | 0 |
    value   | 1 | 0 | 0 | 1 | 1 |
    = 10011 (bin) = 19 (dec)
    ```

    We continue in a similar fashion across all candidates, so the (num_replicas, num_candidates)
    state matrix becomes a (num_candidates,) 1-D state vector.
    '''
    def __init__(self, num_candidates, num_qubits, torch_device):
        super(StateEncoder, self).__init__()
        self.num_candidates = num_candidates
        self.num_qubits = num_qubits
        self.output_size = math.ceil(num_candidates / num_qubits)
        self.torch_device = torch_device
    
    def forward(self, x):
        return self._encode_state(x)

    def _encode_state(self, batch):
        batch_size = len(batch)
        output = torch.zeros((batch_size, self.num_qubits), device=self.torch_device)
        # each 2-tensor in the batch is a separate input, so we have to iterate over them to encode 
        for i_t, tensor in enumerate(batch):
            # group by number of qubits. for example, if we have 4 qubits and want to encode
            # [1 0 1 0 1 0 1 0], this returns
            # ([1 0], [1 0], [1 0], [1 0])
            chunks = torch.tensor_split(tensor, self.num_qubits)
            for i_c, chunk in enumerate(chunks):
                # perform the encoding as described in __init__
                output[i_t][i_c] = sum([(2**i)*n for i, n in enumerate(chunk)])
        return output
    
class AngleEncoder(nn.Module):
    '''
    Transforms an encoded state tensor into one that can
    be used for angle encoding into qubits.

    Essentially a mapping x -> pi / x, which represents
    the angle to be rotated around the x-axis of the Bloch sphere.

    We want to encode into [0, pi] radians and x is in the
    range (0, 1] so this works out nicely
    '''
    def __init__(self):
        super(AngleEncoder, self).__init__()
    
    def forward(self, x):
        return torch.where(x > 0, math.pi / x, 0)

class AmplitudeEncoder(nn.Module):
    '''
    This one isn't ready yet!
    '''
    def __init__(self):
        super(AmplitudeEncoder, self).__init__()
    
    def forward(self, x):
        # TODO
        return x

class StateDecoder(nn.Module):
    def __init__(self, num_candidates, num_replicas):
        super(StateDecoder, self).__init__()
        self.num_candidates = num_candidates
        self.num_replicas = num_replicas
    
    def forward(self, x):
        return torch.vmap(self._decode_state)(x)

    def _decode_state(self, state):
        '''
        The inverse of StateEncoder._encode_state -- takes a 1-d vector where all state variables are
        encoded as a binary representation of the replicas each candidate they are located in,
        and transforms this into a (num_replicas, num_candidates) state matrix that the environment
        is able to use.
        '''
        result = torch.zeros((self.num_candidates, self.num_replicas))

        for idx, element in enumerate(state):
            result[idx,:] = [1 if element & (1 << i) else 0 for i in range(self.num_replicas)]

        result = torch.where(result > 0, result, 0)

        return result.T

class ActionDecoder(nn.Module):
    '''
    Our QNN gives us a 2^num_qubits-dimensional tensor as an output, where
    every 
    '''