from qiskit_machine_learning.utils import algorithm_globals
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit import QuantumCircuit
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit_machine_learning.connectors import TorchConnector

import torch
from torch import nn
import torch.nn.functional as F

from encoding import AngleEncoder, AmplitudeEncoder, StateEncoder

def build_angle_encoded_circuit(n_inputs: int, param_layers: int) -> tuple[QuantumCircuit, list[Parameter], list[Parameter]]:
    '''
    Creates a `QuantumCircuit` that accepts some angle-encoded state as input, and provides
    an ansatz with (`n_inputs` * `param_layers`) trainable weights. If `param_layers` is 1,
    the ansatz will be composed of R_y gates; if it is 2, the second pass will be composed of
    R_z gates. The resulting qubits will be entangled with circular C-Z gates.

    Returns a tuple:
    - `qc`, the composed quantum circuit
    - `input_params`, an `n_inputs` list of input parameters
    - `trainable_params`, an `n_inputs` * `param_layers` list of trainable parameters

    The parameters follow the naming schema `input_{i}` for each input and `weight_{gate}_{i}`
    for each trainable parameter, where i is the qubit index (0...`n_inputs` - 1) and gate is
    one of `ry` (for the first trainable parameter) or `rz` (for the second).
    '''
    assert n_inputs > 0, 'we need some positive number of qubits!'
    assert param_layers >= 0 and param_layers <= 2, 'we may have trainable parameters on one or both of r_y or r_z gates!'
    qc = QuantumCircuit(n_inputs)
    input_params = []
    trainable_params = []
    # input parameters (encoded state)
    for i in range(n_inputs):
        input_params.append(Parameter(f'input_{i}'))
        qc.rx(input_params[i], i)
    # trainable parameters (weights)
    if param_layers >= 1:
        for i in range(n_inputs):
            weight = Parameter(f'weight_ry_{i}')
            trainable_params.append(weight)
            qc.ry(weight, i)
            if param_layers == 2:
                weight = Parameter(f'weight_rz_{i}')
                trainable_params.append(weight)
                qc.rz(weight, i)
    # entanglement step
    for i in range(n_inputs):
        qc.cz(i, (i + 1) % n_inputs)
    
    return qc, input_params, trainable_params

def construct_ansatz(n_inputs: int, gates: list[str], n_times: int) -> tuple[QuantumCircuit, list[Parameter]]:
    trainable_params = []
    ansatz = QuantumCircuit(n_inputs)
    qc_gate = gate_selector(ansatz)

    for i_layer in range(n_times):
        for i_gate, gate in enumerate(gates):
            assert gate[0] in ['r', 'c'], f'unknown gate type {gate}, expected one of the form [r,c][x,y,z]'
            assert gate[1] in ['x', 'y', 'z'], f'unknown gate type {gate}, expected one of the form [r,c][x,y,z]'
            assert len(gate) == 2, f'unknown gate type {gate}, expected one of the form [r,c][x,y,z]'
            for qubit in range(n_inputs):
                if gate[0] == 'c':
                    qc_gate[gate](qubit, (qubit + 1) % n_inputs)
                elif gate[0] == 'r':
                    weight = Parameter(f'weight_{i_layer}_{gate}_{qubit}')
                    trainable_params.append(weight)
                    qc_gate[gate](weight, qubit)
    
    return ansatz

def gate_selector(qc: QuantumCircuit) -> dict[str, any]:
    return {
        'rx': qc.rx,
        'ry': qc.ry,
        'rz': qc.rz,
        'cx': qc.cx,
        'cy': qc.cy,
        'cz': qc.cz
    }

def interpreter(n_actions):
    def threshold(x):
        if x >= n_actions: return 0
        return x
    return threshold

def identity(x):
    return x

def build_qnn_model(n_inputs: int, n_qubits: int, param_layers: int, n_outputs: int, n_shots: int) -> SamplerQNN:
    # circuit, inputs, weights = build_angle_encoded_circuit(n_inputs, param_layers)
    circuit = QuantumCircuit(n_qubits)
    feature_map = ZZFeatureMap(n_qubits)
    ansatz = RealAmplitudes(n_qubits, reps=param_layers)
    circuit.compose(feature_map, inplace=True)
    circuit.compose(ansatz, inplace=True)
    circuit.draw(output='mpl')
    sampler = Sampler(default_shots=n_shots)
    qnn = SamplerQNN(
        circuit=circuit,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        sampler=sampler,
        input_gradients=True,
        #output_shape=n_outputs,
        #interpret=identity
    )

    return qnn

class QNN(nn.Module):
    def __init__(self, n_inputs, n_qubits, param_layers, n_outputs, n_shots):
        super(QNN, self).__init__()
        self.qnn = TorchConnector(build_qnn_model(n_inputs, n_qubits, param_layers, n_outputs, n_shots))
    
    def forward(self, x):
        return self.qnn(x)
    
class AngleEncodedQNN(nn.Module):
    def __init__(self, n_inputs, n_qubits, param_layers, n_outputs, n_shots):
        super(AngleEncodedQNN, self).__init__()
        self.encoder = AngleEncoder()
        self.qnn = QNN(n_inputs, n_qubits, param_layers, n_outputs, n_shots)

    def forward(self, x):
        #x = self.encoder(x)
        return self.qnn(x)
    
class AmplitudeEncodedQNN(nn.Module):
    def __init__(self, n_inputs, n_qubits, param_layers, n_outputs, n_shots):
        super(AmplitudeEncodedQNN, self).__init__()
        self.encoder = AmplitudeEncoder()
        self.qnn = QNN(n_inputs, n_qubits, param_layers, n_outputs, n_shots)

    def forward(self, x):
        x = self.encoder(x)
        return self.qnn(x)

class QuantumDQN(nn.Module):
    def __init__(self, n_inputs, n_qubits, n_actions, param_layers = 3, encoding = 'angle', qnn_output='trunc', n_shots=1024, torch_device='cpu'):
        assert encoding == 'angle' or encoding == 'amplitude', 'must specify one of amplitude or angle encoding!'
        assert n_actions <= 2**n_qubits, 'the given number of qubits can\'t encode the action space!'
        assert qnn_output == 'trunc' or qnn_output == 'layer', 'must specify how to rectify the output dimension!'
        super(QuantumDQN, self).__init__()
        if encoding == 'angle':
            self.qnn = AngleEncodedQNN(n_inputs, n_qubits, param_layers, n_actions, n_shots)
        else:
            self.qnn = AmplitudeEncodedQNN(n_inputs, n_qubits, param_layers, n_actions, n_shots)
        self.flatten = nn.Flatten()
        self.state_encoder = StateEncoder(n_inputs, n_qubits, torch_device)
        self.output_layer = nn.Linear(2**n_qubits, n_actions)
        self.n_actions = n_actions
        self.qnn_output = qnn_output
    
    def forward(self, x):
        x = self.flatten(x)
        x = self.state_encoder(x)
        x = self.qnn(x)
        if self.qnn_output == 'trunc':
            return torch.narrow(x, 1, 0, self.n_actions)
        else:
            return self.output_layer(x)
