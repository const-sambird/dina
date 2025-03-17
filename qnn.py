from qiskit_machine_learning.utils import algorithm_globals
from qiskit.circuit import Parameter
from qiskit import QuantumCircuit
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit_machine_learning.connectors import TorchConnector

from torch import nn
from matplotlib import pyplot as plt

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

def interpreter(outputs):
    def parity(x):
        return f"{bin(x)}".count("1") % outputs
    return parity

def build_qnn_model(n_inputs: int, param_layers: int, n_outputs: int) -> SamplerQNN:
    circuit, inputs, weights = build_angle_encoded_circuit(n_inputs, param_layers)
    sampler = Sampler()
    qnn = SamplerQNN(
        circuit=circuit,
        input_params=inputs,
        weight_params=weights,
        sampler=sampler,
        input_gradients=True,
        output_shape=n_outputs,
        interpret=interpreter(n_outputs)
    )

    return qnn

class QNN(nn.Module):
    def __init__(self, n_inputs, param_layers, n_outputs):
        super(QNN, self).__init__()
        self.qnn = TorchConnector(build_qnn_model(n_inputs, param_layers, n_outputs))
    
    def forward(self, x):
        return self.qnn(x)
    
class AngleEncodedQNN(nn.Module):
    def __init__(self, n_inputs, param_layers, n_outputs):
        super(AngleEncodedQNN, self).__init__()
        self.encoder = AngleEncoder()
        self.qnn = QNN(n_inputs, param_layers, n_outputs)

    def forward(self, x):
        x = self.encoder(x)
        return self.qnn(x)
    
class AmplitudeEncodedQNN(nn.Module):
    def __init__(self, n_inputs, param_layers, n_outputs):
        super(AmplitudeEncodedQNN, self).__init__()
        self.encoder = AmplitudeEncoder()
        self.qnn = QNN(n_inputs, param_layers, n_outputs)

    def forward(self, x):
        x = self.encoder(x)
        return self.qnn(x)

class QuantumDQN(nn.Module):
    def __init__(self, n_inputs, n_actions, param_layers = 2, encoding = 'angle'):
        assert encoding == 'angle' or encoding == 'amplitude', 'must specify one of amplitude or angle encoding!'
        super(QuantumDQN, self).__init__()
        if encoding == 'angle':
            self.qnn = AngleEncodedQNN(n_inputs, param_layers, n_actions)
        else:
            self.qnn = AmplitudeEncodedQNN(n_inputs, param_layers, n_actions)
        self.state_encoder = StateEncoder(n_inputs)
        self.output_layer = nn.Linear(n_inputs, n_actions)
    
    def forward(self, x):
        #print('--- input tensor')
        #print(x)
        x = self.state_encoder(x)
        #print('--- encoded tensor')
        #print(x)
        x = self.qnn(x)
        #print('--- qnn result')
        #print(x)
        return self.output_layer(x)
