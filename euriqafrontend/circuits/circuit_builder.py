import logging
import copy
import typing
import cirq
import sympy
import numpy as np

_LOGGER = logging.getLogger(__name__)
Pi = np.pi
_NEW = cirq.InsertStrategy.NEW
_NEWINLINE = cirq.InsertStrategy.NEW_THEN_INLINE
_INLINE = cirq.InsertStrategy.INLINE

class OneIndexMap:
    def __init__(self, num_qubits):
        self.indices = np.arange(num_qubits)
        # If odd
        if num_qubits & 0b1:
            indices = np.arange(num_qubits)
            indices = [int(i) for i in indices]
            self.all = indices
            self._indices = indices[int(num_qubits / 2):] + indices[0:int(num_qubits / 2)]
        # If even
        else:
            #TODO this doesnt work for even numbers
            indices = np.arange(num_qubits)
            indices = [int(i) for i in indices]
            self.all = indices
            self._indices = indices[int(num_qubits / 2):] + indices[0:int(num_qubits / 2)]
            self._indices[0] = None

    def __getitem__(self, key):
        if type(key) == list:
            sliced_indices = [self._indices[i] for i in key]
            #sliced_indices = list(filter(None, sliced_indices

        elif type(key) == slice:

            start = key.start
            if start == None:
                start = -self._indices[0]
            stop = key.stop
            if stop == None:
                stop = self._indices[0]
            nkey = slice(start, stop, 1)

            if nkey.start < 0:
                if nkey.stop >= 0:
                    sliced_indices = self._indices[nkey.start:] + self._indices[0:nkey.stop + 1]
                elif key.stop == -1:
                    sliced_indices = self._indices[nkey.start:]
                else:
                    sliced_indices = self._indices[nkey.start:nkey.stop + 1]
            else:
                sliced_indices = self._indices[nkey.start:nkey.stop + 1]
            #sliced_indices = list(filter(None, sliced_indices))
        elif type(key) == int:
            sliced_indices = self._indices[key]
        else:
            sliced_indices = []

        return sliced_indices


class IonRegister:

    def __init__(self, num_qubits: int=15):
        ''' Generate a register object that makes for easy indexing in circuits

        Args:
            num_qubits (int): Number of qubits in your register

        Returns:
        A qubit register than provides for easy indexing. This uses the new indexing scheme where the center ion is
        designated as 0 and its neighbors are +-1. For even number of ions, there is no 0 and center ions are +/-1.

        This objects supports indexing by integer, slicing, or lists:
            >> register[5]          returns qubit 5 to the right of the center
            >> register[-3]         returns qubit 3 from the left of the center
            >> register[-2:2]       returns the middle 5 (4 if even) ions. Slicing is inclusive! Unlike Python native
            >> register[[-3,0,3]]   returns ions -3,0,3
            >> register.all         returns all the qubits in the register
        '''

        # If odd
        if num_qubits & 0b1:
            ions = np.linspace(-np.floor(num_qubits / 2), np.floor(num_qubits / 2), num_qubits)
            qubits = [cirq.LineQubit(int(i)) for i in ions.tolist()]
            self.all = qubits
            self._register = qubits[int(num_qubits / 2):] + qubits[0:int(num_qubits / 2)]
        # If even
        else:
            ions = np.linspace(-np.floor(num_qubits / 2), np.floor(num_qubits / 2), num_qubits + 1)
            qubits = [cirq.LineQubit(int(i)) for i in ions.tolist()]
            self.all = qubits[0:int(num_qubits/2)]  + qubits[int(num_qubits/2)+1:]
            self._register = qubits[int(num_qubits / 2):] + qubits[0:int(num_qubits / 2)]
            self._register[0] = None

        self.index = OneIndexMap(num_qubits)


    def __getitem__(self, key):

        if type(key) == list:
            sliced_register = [self._register[i] for i in key]
            sliced_register = list(filter(None, sliced_register))

        elif type(key) == slice:
            start = key.start
            if start == None:
                start = self.all[0].x
            stop = key.stop
            if stop == None:
                stop = self.all[-1].x
            nkey = slice(start, stop, 1)
            if nkey.start < 0:
                if nkey.stop >= 0:
                    sliced_register = self._register[nkey.start:] + self._register[0:nkey.stop + 1]
                elif key.stop == -1:
                    sliced_register = self._register[nkey.start:]
                else:
                    sliced_register = self._register[nkey.start:nkey.stop + 1]
            else:
                sliced_register = self._register[nkey.start:nkey.stop + 1]
            sliced_register = list(filter(None, sliced_register))
        elif type(key) == int:
            sliced_register = self._register[key]
        else:
            sliced_register = []

        return sliced_register

    def __str__(self):
        block = '\n'.join([str(qb.__repr__()) + ',' for qb in self.all])
        indented = '\n '.join(block.split('\n'))
        return '[{}]'.format(indented)

    def __repr__(self):
        return self.__str__()

def prepare_in_X(circuit:cirq.Circuit,
                 ions_to_flip: typing.List[cirq.LineQubit],
                 encoding: int = 0) -> cirq.Circuit:
    """Assume qubits are all in the ground state, will prepare them in the X basis according to provided encoding

    Args:
        circuit: circuit to add X preparation to
        ions_to_flip: list of cirq.LineQubits to perform the encoding on
        encoding: an integer whos binary coding encodes the binary string for the X basis of the qubits.
                    The last qubit in the list is the most significant bit.
            encoding = 0 -> +X+X+X+X
            encoding = 1 -> -X+X+X+X
            encoding = 8 -> +X+X+X-X
    Returns: cirq circuit with added operation

    """
    n_ions = len(ions_to_flip)
    assert 2**n_ions > encoding, "Encoded bit string longer than addressed ions"
    b = bin(encoding)
    b = b[2:]
    state = np.zeros(n_ions)
    for j in range(len(b)):
            state[j] = int(b[-1-j])
    state = np.array(state) * 2 - 1
    gates = [cirq.ry(np.pi/2*(state[i]*-1)).on(ions_to_flip[i]) for i in range(n_ions)]
    circuit.append(gates,_NEWINLINE)
    return circuit

def cnot(i, j):
    """return a sequence of native gates (cirq) that is equivalent to a CNOT (i controlling j)"""
    temp = cirq.Circuit()
    temp.append(cirq.ry(np.pi / 2).on(i))
    temp.append(cirq.ms(np.pi / 4).on(i, j))
    temp.append(cirq.ry(-np.pi / 2).on(i))
    temp.append(cirq.rx(-np.pi / 2).on(j))
    temp.append(cirq.rz(-np.pi / 2).on(i))
    return temp


def hadamard(i):
    """return a sequence of native gates (cirq) that is equivalent to a Hadamard (on i) """
    temp = cirq.Circuit()
    temp.append(cirq.rx(np.pi).on(i))
    temp.append(cirq.ry(-np.pi / 2).on(i))
    return temp

def xx(i: cirq.LineQubit,j: cirq.LineQubit, x: float,precision=0.001):
    """return a trimmed XX gate (optimized)"""
    x=np.fmod(x,np.pi) # xx angle has period of pi
    if x>np.pi/2:
        x=x-np.pi
    elif x<-np.pi/2:
        x=x+np.pi   #pull the xx angle into [-pi/2, pi/2]

    temp = cirq.Circuit()

    if x>np.pi/4:
        temp.append(cirq.rx(np.pi).on(i))
        temp.append(cirq.rx(np.pi).on(j))
        x=x-np.pi/2
    elif x<-np.pi/4:
        temp.append(cirq.rx(-np.pi).on(i))
        temp.append(cirq.rx(-np.pi).on(j))
        x=x+np.pi/2

    if abs(x)>precision:
        temp.append(cirq.ms(x).on(i, j))

    return temp


def prepare_in_Z(circuit:cirq.Circuit,
                 ions_to_flip: typing.List[cirq.LineQubit],
                 encoding: int = 0) -> cirq.Circuit:
    """Assume qubits are all in the ground state, will prepare them in the Z basis according to provided encoding

    Args:
        circuit: circuit to add Z preparation to
        ions_to_flip: list of cirq.LineQubits to perform the encoding on
        encoding: an integer whos binary coding encodes the binary string for the Z basis of the qubits.
                    The last qubit in the list is the most significant bit.
            encoding = 0 -> +Z+Z+Z+Z
            encoding = 1 -> -Z+Z+Z+Z
            encoding = 8 -> +Z+Z+Z-Z
    Returns: cirq circuit with added operation

    """
    n_ions = len(ions_to_flip)
    assert 2**n_ions > encoding, "Encoded bit string longer than addressed ions"
    b = bin(encoding)
    b = b[2:]
    state = np.zeros(n_ions)
    for j in range(len(b)):
            state[j] = int(b[-1-j])
    state = np.array(state)
    gates = [cirq.ry(np.pi*(state[i])).on(ions_to_flip[i]) for i in range(n_ions)]
    circuit.append(gates,_NEWINLINE)
    return circuit


def add_ghz(circuit: cirq.Circuit,
            ions: typing.List[cirq.LineQubit],
            basis: str = "z",
            rotation_correction: list = [0, 0],
            phase_correction: list = [0, 0]
            ):
    assert len(ions) == 3, "Currently only three qubit GHZ is supported"

    circuit.append(cirq.ms(np.pi / 4 + rotation_correction[0]).on(ions[0], ions[1]))
    circuit.append([cirq.rz(np.pi / 2 + phase_correction[0]).on(ions[0])])
    circuit.append(cirq.ms(-np.pi / 4 + rotation_correction[1]).on(ions[0], ions[2]))
    circuit.append([cirq.rz(-np.pi / 2 + phase_correction[1]).on(ions[2])])

    if basis.lower() == "x":
        circuit.append([cirq.ry(np.pi / 2).on(ions[0]),
                        cirq.ry(np.pi / 2).on(ions[1]),
                        cirq.ry(np.pi / 2).on(ions[2])],
                       _NEWINLINE)

    return circuit

def add_parity_scan(circuit: cirq.Circuit,
                    ions: typing.List[cirq.LineQubit],
                    symbol: sympy.Symbol=sympy.Symbol("phase"),
                    basis: str = 'z'):
    if type(ions) != list:
        ions = [ions]

    first_ops = []
    next_ops = []
    for i in ions:
        if basis.lower() == "z":
            first_ops.append(cirq.ry(Pi / 2).on(i))
            next_ops.append(cirq.rz(symbol).on(i))
        else:
            first_ops.append(cirq.rz(symbol).on(i))

    circuit.append(first_ops, _NEWINLINE)
    if len(next_ops) != 0:
        circuit.append(next_ops, _INLINE)
    circuit.append([cirq.ry(-Pi/2).on(i) for i in ions], _INLINE)

    return circuit


def print_circuit_output(circuit, repetitions=10000):
    simulator = cirq.Simulator()
    # Add measurments if they dont exist
    if circuit[-1].operations[-1].gate.__class__ != cirq.ops.common_gates.MeasurementGate:
        qbits = circuit.all_qubits()
        measurement = cirq.measure(*qbits)
        measured_circuit = cirq.Circuit(circuit)
        measured_circuit.append(measurement)
    else:
        measured_circuit = cirq.Circuit(circuit)

    print("Circuit:")
    print(measured_circuit)
    print()
    print("Waveform: \n" + simulator.simulate(circuit).dirac_notation())
    result = simulator.run(measured_circuit, repetitions=repetitions).histogram(key=qbits)
    num_qbit = len(qbits)
    bspec = "|{:0" + str(num_qbit) + "b}> : {:d}"
    print("\nSimulated Counts:")
    for i in result:
        print(bspec.format(i, result[i]))




# THESE ARE MOSTLY DEPRECATED, BUT YOU CAN USE THEM IF YOU WISH
def add_RX(circuit: cirq.Circuit,
           ion: cirq.LineQubit,
           sign: int,
           theta: float = Pi / 2,
           pre_phase_shift: float = 0,
           post_phase_shift: float = 0,
           ):

    if pre_phase_shift != 0:
        circuit.append(cirq.rz(pre_phase_shift).on(ion))

    circuit.append([cirq.rx(sign * theta).on(ion)])

    if post_phase_shift != 0:
        circuit.append(cirq.rz(post_phase_shift).on(ion))

    return circuit


def add_RY(circuit: cirq.Circuit,
           ion: cirq.LineQubit,
           sign: int,
           theta: float = Pi / 2,
           pre_phase_shift: float = 0,
           post_phase_shift: float = 0,
           ):

    if pre_phase_shift != 0:
        circuit.append(cirq.rz(pre_phase_shift).on(ion))

    circuit.append(cirq.ry(sign * theta).on(ion))

    if post_phase_shift != 0:
        circuit.append(cirq.rz(post_phase_shift).on(ion))

    return circuit


def add_RZ(circuit: cirq.Circuit,
           ion: cirq.LineQubit,
           sign: int,
           theta: float = Pi / 2,
           pre_phase_shift: float = 0,
           post_phase_shift: float = 0,
           ):

    if pre_phase_shift != 0:
        circuit.append(cirq.rz(pre_phase_shift).on(ion))

    circuit.append(cirq.rz(sign * theta).on(ion))

    if post_phase_shift != 0:
        circuit.append(cirq.rz(post_phase_shift).on(ion))

    return circuit


def add_XX(circuit: cirq.Circuit,
           ions: typing.List[cirq.LineQubit],
           sign: int,
           pre_phase_shift: float = 0,
           post_phase_shift: float = 0,
           ):

    assert len(ions) == 2, "XX gate is defined on two ions"

    if pre_phase_shift != 0:
        circuit.append([cirq.rz(pre_phase_shift).on(ions[0]),
                        cirq.rz(pre_phase_shift).on(ions[1])])

    circuit.append(cirq.ms(sign * Pi / 4).on(ions[0],ions[1]))

    if post_phase_shift != 0:
        circuit.append([cirq.rz(post_phase_shift).on(ions[0]),
                        cirq.rz(post_phase_shift).on(ions[1])])

    return circuit