from qiskit_machine_learning.utils import algorithm_globals
from qiskit.circuit import Parameter, Gate
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap, XGate, RXGate
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_aer.primitives import Sampler
from qiskit_machine_learning.connectors import TorchConnector

import torch
from torch import nn
import torch.nn.functional as F

from encoding import AngleEncoder, AmplitudeEncoder, StateEncoder

def crx(param_name: str = 'crx_gate') -> Gate:
    '''
    Returns a controlled-RX gate. This gate has 1 control qubit.

    :param param_name: the name for the trainable parameter
    :returns gate:     the controlled RX gate
    :returns param:    the trainable parameter theta
    '''
    param = Parameter(param_name)
    return RXGate(param).control(1)

def get_twolocal_circuit(n_qubits, n_reps):
    return RealAmplitudes(n_qubits, reps=n_reps)

def get_bqn_circuit(n_qubits: int, n_data_qubits: int, n_ancilla_qubits: int,
                    n_data_reps: int, n_ancilla_reps: int) -> QuantumCircuit:
    '''
    Builds a Bayesian Quantum Circuit with N qubits in the data register and
    M qubits in the ancilla. Returns a constructed `QuantumCircuit`.

    There are restrictions placed on the circuit:
    - The number of data bits N must be large enough to encode the action space;
      that is, `|A| <= 2^N` (this is a constant requirement, but care must be taken
      to ensure the additional ancilla bits do not cause us to exceed our qubit budget)
    - There must be at least one ancillary qubit. If no ancilla is used, this is not a BQN
      and the standard two-local/RealAmplitudes ansatz should be used instead
    - One qubit (not counted as a data or ancillary qubit) will be used as the control flag

    For further information about BQNs, consult:
        Du et al. (2020) "Expressive power of parameterized quantum circuits".
        https://doi.org/10.1103/PhysRevResearch.2.033125

    :param n_qubits:         the total number of qubits in the system
    :param n_data_qubits:    the number of qubits to be used in the data register
    :param n_ancilla_qubits: the number of qubits to be used for the ancilla, not including the additional flag qubit
    :param n_data_reps:      the number of repetitions of the controlled U-blocks to include
    :param n_ancilla_reps:   the number of trainable parameters/entanglement steps to include in the ancilla
    :returns qc:       the constructed quantum circuit
    '''
    assert n_qubits == (n_data_qubits + n_ancilla_qubits + 1), \
            'the number of qubits assigned for data/ancilla/flag do not match the circuit dimension'
    
    data_reg = QuantumRegister(n_data_qubits, 'data')
    # include flag qubit in the ancilla register
    ancilla_reg = QuantumRegister(n_ancilla_qubits + 1, 'ancilla')
    output_reg = ClassicalRegister(n_data_qubits, 'output')

    qc = QuantumCircuit(data_reg, ancilla_reg, output_reg)
    ancilla = get_bqn_ancilla(n_ancilla_qubits, n_ancilla_reps)
    ANCILLA_QUBITS = list(range(n_data_qubits, n_data_qubits + n_ancilla_qubits))
    qc.compose(ancilla, ANCILLA_QUBITS, inplace=True)

    for i in range(n_data_reps):
        circuit = get_one_bqn_repetition(n_data_qubits, n_ancilla_qubits, i)
        qc.compose(circuit, inplace=True)
    
    qc.barrier()
    qc.measure(data_reg, output_reg)

    return qc

def get_one_bqn_repetition(n_data_qubits: int, n_ancilla_qubits: int, block: int) -> QuantumCircuit:
    '''
    Gets a single repetition of U-blocks controlled by the ancilla state.

    :param n_data_qubits:    the number of qubits in the data register
    :param n_ancilla_qubits: the number of qubits used for the ancilla (excluding the flag qubit)
    :param block:            which block is this?
    :returns qc: a single repetition of the controlled data parameter blocks
    '''
    FMT_STRING = f'0{n_ancilla_qubits}b'
    ANCILLA_QUBITS = list(range(n_data_qubits, n_data_qubits + n_ancilla_qubits + 1))

    qc = QuantumCircuit(n_data_qubits + n_ancilla_qubits + 1)

    for i in range(2**n_ancilla_qubits):
        selector, inverter = get_ancilla_selector(format(i, FMT_STRING), n_ancilla_qubits)
        block_circuit = get_one_bqn_block(n_data_qubits, n_ancilla_qubits, block, i)
        
        qc.compose(selector, ANCILLA_QUBITS, inplace=True)
        qc.compose(block_circuit, inplace=True)
        qc.compose(inverter, ANCILLA_QUBITS, inplace=True)
    
    return qc

def get_one_bqn_block(n_data_qubits: int, n_ancilla_qubits: int, block: int, rep: int) -> QuantumCircuit:
    '''
    Gets a single U-block for the BQN.

    :param n_data_qubits:    the number of qubits in the data register
    :param n_ancilla_qubits: the number of qubits in the ancillary register
    :param block:            the index of this block
    :param rep:              the index of this repetition
    :returns qc: the block's quantum circuit
    '''
    FLAG_BIT = n_data_qubits + n_ancilla_qubits

    qc = QuantumCircuit(n_data_qubits + n_ancilla_qubits + 1)

    for i in range(n_data_qubits):
        qc.append(crx(f'data_{block}_{rep}_{i}'), [FLAG_BIT, i])
    for i in range(n_data_qubits):
        qc.ccx(i, FLAG_BIT, (i + 1) % n_data_qubits)
    
    return qc

def get_bqn_ancilla(n_ancilla_qubits: int, n_ancilla_reps: int) -> QuantumCircuit:
    '''
    Repeats a standard U-block on the ancillary qubits a given number of times.
    The unitary gates in these blocks are not controlled, unlike the gates operating on
    the data qubits.

    :param n_ancilla_qubits: the number of qubits in the ancilla, not including the flag qubit
    :param n_ancilla_reps:   the number of blocks to apply
    :returns ancilla:        the circuit on the ancilla bits to estimate the posterior distribution
    '''
    qc = QuantumCircuit(n_ancilla_qubits)

    for i in range(n_ancilla_reps):
        for j in range(n_ancilla_qubits):
            weight = Parameter(f'ancilla_{i}_{j}')
            qc.rx(weight, j)
        if n_ancilla_qubits > 1:
            for j in range(n_ancilla_qubits):
                qc.cx(j, (j + 1) % n_ancilla_qubits)
    
    return qc

def get_ancilla_selector(bitstring: str, n_ancilla_qubits: int) -> tuple[QuantumCircuit, QuantumCircuit]:
    '''
    Creates a set of gates to conditionally apply U(theta_lambda_k) to the set of data qubits.
    The selection circuit will set the flag qubit iff the ancilla is equal to the bitstring.
    The inversion circuit will restore the flag qubit to the prior state.

    Note that the flag qubit is not included in the count of `n_ancilla_qubits`.

    :param bitstring:        the ancilla value we should be testing for. Apply the U-block only if the bitstring
                             is equal to the ancilla value.
    :param n_ancilla_qubits: number of qubits in the ancilla
    :returns selection_circuit: the circuit to compose before the U-block, to set the flag bit
    :returns inversion_circuit: the circuit to compose after the U-block, to reset the flag bit
    '''
    qr = QuantumRegister(n_ancilla_qubits + 1)
    bitstring_circuit = QuantumCircuit(qr)
    selection_circuit = QuantumCircuit(qr)
    inversion_circuit = QuantumCircuit(qr)
    bits = [True if b == '1' else False for b in bitstring]

    for idx, bit in enumerate(bits):
        if bit:
            bitstring_circuit.x(idx)
    
    flag_set = XGate().control(n_ancilla_qubits)

    selection_circuit.compose(bitstring_circuit, inplace=True)
    selection_circuit.append(flag_set, qr)

    inversion_circuit.append(flag_set, qr)
    inversion_circuit.compose(bitstring_circuit, inplace=True)

    return selection_circuit, inversion_circuit

def build_qnn_model(n_inputs: int, n_qubits: int, param_layers: int, n_outputs: int, n_shots: int) -> SamplerQNN:
    # circuit, inputs, weights = build_angle_encoded_circuit(n_inputs, param_layers)
    circuit = QuantumCircuit(n_qubits)
    feature_map = ZZFeatureMap(n_qubits)
    ansatz = RealAmplitudes(n_qubits, reps=param_layers)
    circuit.compose(feature_map, inplace=True)
    circuit.compose(ansatz, inplace=True)
    #circuit.draw(output='mpl')
    sampler = Sampler(run_options={"method": "statevector", "shots": n_shots}, backend_options={"max_parallel_experiments": 1})
    qnn = SamplerQNN(
        circuit=circuit,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        sampler=sampler,
        input_gradients=True
    )

    return qnn

class QNN(nn.Module):
    def __init__(self, n_qubits, param_layers, n_shots):
        super(QNN, self).__init__()
        self.sampler_qnn = build_qnn_model(n_qubits, param_layers, n_shots)
        self.qnn = TorchConnector(self.sampler_qnn)
    
    def forward(self, x):
        return self.qnn(x)
    
class AngleEncodedQNN(nn.Module):
    def __init__(self, n_qubits, param_layers, n_shots):
        super(AngleEncodedQNN, self).__init__()
        self.encoder = AngleEncoder()
        self.qnn = QNN(n_qubits, param_layers, n_shots)
        self.torchconn = self.qnn.qnn
        self.sampler_qnn = self.qnn.sampler_qnn

    def forward(self, x):
        #x = self.encoder(x)
        return self.qnn(x)
    
class AmplitudeEncodedQNN(nn.Module):
    def __init__(self, n_qubits, param_layers, n_shots):
        super(AmplitudeEncodedQNN, self).__init__()
        self.encoder = AmplitudeEncoder()
        self.qnn = QNN(n_qubits, param_layers, n_shots)
        self.torchconn = self.qnn.qnn
        self.sampler_qnn = self.qnn.sampler_qnn

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
            self.qnn = AngleEncodedQNN(n_qubits, param_layers, n_shots)
        else:
            self.qnn = AmplitudeEncodedQNN(n_qubits, param_layers, n_shots)
        self.sampler_qnn = self.qnn.sampler_qnn
        self.torchconn = self.qnn.torchconn
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
