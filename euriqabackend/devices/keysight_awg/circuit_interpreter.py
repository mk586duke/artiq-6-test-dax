import logging
import pathlib
import pickle
import typing

import cirq
import numpy as np

import euriqabackend.devices.keysight_awg.common_types as common_types
from euriqabackend.devices.keysight_awg import physical_parameters as pp
from euriqabackend.devices.keysight_awg import RFCompiler as rfc
import euriqabackend.devices.keysight_awg.gate as gate
from euriqafrontend import EURIQA_NAS_DIR


_LOGGER = logging.getLogger(__name__)
_PRINT_GATE_NAMES = True
PATH_TO_SERVER = EURIQA_NAS_DIR / "CompactTrappedIonModule"

class WallClock:
    '''Used to keep track of time in a circuit and linearly feeds forward the angles.
    The native unit here is milliseconds
    '''

    def __init__(self, offset: float, feedforward_1Q: float = 1.0, feedforward_2Q: float = 1.0):
        """

        Args:
            offset: start the clock at a non-zero value everytime
            feedforward_1Q: A linear feedfoward to 1Q gate amplitudes. Should be in units of fractional %/ms
                            Example:
                                feedforward_1Q = 0.01 will increase the angle of the gates by 1% per 1 ms
            feedforward_2Q:
        """
        self.offset = offset
        self.feedforward_1Q = feedforward_1Q
        self.feedforward_2Q = feedforward_2Q

        self.time = self.offset

    def add(self, duration:float):
        self.time += duration

    def reset(self):
        self.time = self.offset

    def feed_forward(self, angle:float, two_qubit: bool=False):
        if two_qubit:
            ff_angle = angle * (1 + self.feedforward_2Q * self.time)
        else:
            ff_angle = angle * (1 + self.feedforward_1Q * self.time)

        return ff_angle


class GateDictionary:
    def __init__(self):
        self.RX_gate = ["X", "XPowGate", "Rx", "rx", "_PauliX"]
        self.RY_gate = ["Y", "YPowGate", "Ry", "ry", "_PauliY"]
        self.RZ_gate = ["Z", "ZPowGate", "Rz", "rz", "S", "T", "_PauliZ"]
        self.MS_gate = ["MS", "XXPowGate", "XX", "ms", "MSGate"]

        self.software_gates = self.RZ_gate + ["DT", "WaitGate"]

class CircuitInterpreter:
    """This class takes in a circuit of native gates expressed in Cirq and adds the
    appropriate gates to the sequence being assembed by the RFCompiler that instantiates
    this class."""

    def __init__(self, RFCompiler: rfc.RFCompiler, physical_params: pp.PhysicalParams):

        self.RFCompiler = RFCompiler
        self.physical_params = physical_params
        self.gate_dictionary = GateDictionary()
        self.circuit = None
        self.circuit_index = 0
        self.print_gate_list = False
        self.use_SK1_AM = False
        self.return_string = ""
        self.circuit_scan = False
        self.max_1Q_angle = 0
        self.max_2Q_angle = 0
        self.one_indexed = True
        self.clocktime_1Q = 0
        self.clocktime_2Q = 0
        self.circuit_slots = []

    ###################################################
    #                  DEFINE GATES                   #
    ###################################################

    def R(self, slot: int, phi: float, theta: float = np.pi / 2, suppress_circuit_scan: bool = False, wait_after=False):

        if phi == 0:
            gate_name = r"RY+" if np.sign(theta) > 0 else r"RY-"
        elif phi == -np.pi / 2:
            gate_name = r"RX+" if np.sign(theta) > 0 else r"RX-"
        else:
            gate_name = (
                r"R({0:.2f}pi)+".format(phi / np.pi)
                if np.sign(theta) > 0
                else r"R({0:.2f}pi)-".format(phi / np.pi)
            )

        if abs(theta) != np.pi / 2:
            gate_name = gate_name + " ({0:.2f}pi)".format(abs(theta / np.pi))

        # SK1s cannot take a negative angle, phase shift by pi instead
        if theta < 0:
            phi += np.pi
            theta = abs(theta)

        if self.use_SK1_AM:
            self.RFCompiler.add_SK1_am(
                slots=[slot],
                phi=phi,
                theta=theta,
                wait_after=int(wait_after),
                gate_name=gate_name,
                circuit_index=self.circuit_index,
                suppress_circuit_scan=suppress_circuit_scan,
            )
        else:
            self.RFCompiler.add_SK1(
                slots=[slot],
                theta=theta,
                phi=phi,
                wait_after=int(wait_after),
                gate_name=gate_name,
                circuit_index=self.circuit_index,
                suppress_circuit_scan=suppress_circuit_scan,
            )


        if self.print_gate_list:
            self.return_string += self.print_gate(gate_name, slot, suppress_circuit_scan)

    def RX(self, slot: int, theta: float = np.pi / 2, suppress_circuit_scan: bool = False, wait_after=False):
        if abs(theta) == np.pi:
            self.R(slot=slot, phi=-np.pi / 2, theta=theta/2, suppress_circuit_scan=suppress_circuit_scan,
                   wait_after=0)
            self.R(slot=slot, phi=-np.pi / 2, theta=theta/2, suppress_circuit_scan=suppress_circuit_scan,
                   wait_after=wait_after)
        else:
            self.R(slot=slot, phi=-np.pi/2, theta=theta, suppress_circuit_scan=suppress_circuit_scan, wait_after=wait_after)

    def RY(self, slot: int, theta: float = np.pi / 2, suppress_circuit_scan: bool = False, wait_after=False):
        if abs(theta) == np.pi:
            self.R(slot=slot, phi=0, theta=theta/2, suppress_circuit_scan=suppress_circuit_scan, wait_after=0)
            self.R(slot=slot, phi=0, theta=theta/2, suppress_circuit_scan=suppress_circuit_scan, wait_after=wait_after)
        else:
            self.R(slot=slot, phi=0, theta=theta, suppress_circuit_scan=suppress_circuit_scan, wait_after=wait_after)


    def RZ(self, slot: int, theta: float = np.pi / 2, suppress_circuit_scan: bool = False, wait_after=False):

        gate_name = r"RZ+" if np.sign(theta) > 0 else r"RZ-"
        if abs(theta) != np.pi / 2:
            gate_name = gate_name + " ({0:.2f}pi)".format(abs(theta / np.pi))

        self.RFCompiler.add_phase(
            slots=[slot],
            phase=theta,
            wait_after=int(wait_after),
            gate_name=gate_name,
            circuit_index=self.circuit_index,
            suppress_circuit_scan=suppress_circuit_scan,
        )

        if self.print_gate_list:
            self.return_string += self.print_gate(gate_name, slot, suppress_circuit_scan)

    def XX(self, slots: typing.List[int], rotation: float, suppress_circuit_scan: bool = False, wait_after=False):
        # Always reference ion pair in ascending order, then convert to tuple
        slots.sort()
        if isinstance(slots, list):
            slots = tuple(slots)
        sign = np.sign(rotation)
        rotation = abs(rotation)
        gate_name = r"XX+" if sign > 0 else r"XX-"
        gate_angle = self.RFCompiler.gate_solution.loc[slots,"XX_angle"] * 2

        if rotation == gate_angle:
            self.RFCompiler.add_XX(
                slots=slots,
                gate_sign=sign,
                wait_after=int(wait_after),
                gate_name=gate_name,
                circuit_index=self.circuit_index,
                suppress_circuit_scan=suppress_circuit_scan,
            )
        else:
            gate_name += " ({0:.2f}pi)".format(rotation / np.pi)

            ind_multiplier = self.RFCompiler.gate_tweaks.loc[slots, :].individual_amplitude_multiplier
            diff_stark = self.RFCompiler.gate_tweaks.loc[slots, :].stark_shift_differential
            scale = np.sqrt(rotation / gate_angle)
            scaled_ind_multiplier = scale * ind_multiplier
            scaled_diff_stark = scale * diff_stark

            self.RFCompiler.add_XX(
                slots=slots,
                gate_sign=sign,
                wait_after=int(wait_after),
                gate_name=gate_name,
                circuit_index=self.circuit_index,
                suppress_circuit_scan=suppress_circuit_scan,
                individual_amplitude_multiplier=scaled_ind_multiplier,
                stark_shift_differential=scaled_diff_stark
            )


        if self.print_gate_list:
            self.return_string += self.print_gate(gate_name,slots,suppress_circuit_scan)

    ###################################################
    #           COMPILE CIRCUIT FROM FILE             #
    ###################################################
    @staticmethod
    def print_gate(gate_name,slots,suppress_circuit_scan):
        gate_string = ""
        gate_string += " {0} ".format(gate_name)
        gate_string += " on slots {0}, ".format(slots)
        gate_string += " Scanned = {0}".format(not suppress_circuit_scan)
        print(gate_string)
        return gate_string


    @staticmethod
    def get_circuit_scan_size(cirq_file: str = ""):
        """ Opens a circuit scan and gets the length of the scan.
        Required for ARTIQ to set up the scan_value correctly

        Args:
            cirq_file: string to circuit scan pickle file

        Returns:
            int - length of circuit scane

        """
        with open(str(PATH_TO_SERVER / cirq_file), "rb") as c_file:
            loaded_circuit = pickle.load(c_file)

        return len(loaded_circuit)

    @staticmethod
    def get_max_angles(circuit_scan: typing.List[cirq.Circuit]):
        """

        Args:
            circuit_scan: A List of circuits meant to be scanned over

        Returns: The maximum angle in all of the circuit scan for both 1Q and 2Q gates

        """
        if type(circuit_scan) != list:
            circuit_scan = [circuit_scan]

        max_1Q_angle = 0
        max_2Q_angle = 0

        for circuit in circuit_scan:
            for moment in circuit:
                for operation in moment:
                    try:

                        angle = abs(operation.gate._exponent * np.pi)
                        if len(operation.qubits) > 1:
                            if angle > max_2Q_angle:
                                max_2Q_angle = angle
                        else:
                            # Don't count phase gates in max 1Q angle since they aren't physical gates
                            name = str(operation.gate)[0:2]
                            phase_gate = (name == "Z" or name == "Z*" or name == "Rz" or name == "S" or name == "S*")

                            if angle > max_1Q_angle and not phase_gate:
                                max_1Q_angle = angle
                    except AttributeError:
                        pass

        return max_1Q_angle, max_2Q_angle

    def get_1Q_clocktime(self, angle: float=np.pi/2, use_SK1_AM: bool=True):
        """ Calculate the wall clock time it takes to run a 1 qubit gate

        Args:
            angle: angle of the gate in radians. Use the max 1Q angle since this sets timing in the switch network
            use_SK1_AM: If using AM pulses the gate time changes.

        Returns:
            duration: in ms of a 1Q gate including extra delays due to switch network
        """

        # For now just use the center slot for calculating timing. In the future one could use the A_array in the aom
        # calibration to actually figure out how long the max angle take to run on the worst slot (see below) This
        # would also not be accurate though if the biggest angle is not run on the worst slot.
        worst_slot = common_types.ion_to_slot(ion=0, N_ions=self.N_qubits, one_indexed=False)
        #worst_slot = np.argmax(self.RFCompiler.calibration.A_array)+1

        if use_SK1_AM:
            gateSK1 = gate.SK1_AM(physical_params=self.physical_params,
                                  slot=worst_slot,
                                  theta=angle,
                                  phi=0,
                                  ind_amp=self.physical_params.SK1_AM.amp_ind,
                                  global_amp=self.physical_params.SK1_AM.amp_global)
            duration = gateSK1.T_rotation + gateSK1.T_correction1 + gateSK1.T_correction2
            duration += gateSK1.global_ramp*2

        else:
            gateSK1 = gate.SK1(physical_params=self.physical_params,
                               slot=worst_slot,
                               theta=angle,
                               phi=0,
                               ind_amp=self.RFCompiler.physical_params.SK1.amp_ind,
                               global_amp=self.RFCompiler.physical_params.SK1.amp_global)
            duration = gateSK1.T_rotation + gateSK1.T_correction1 + gateSK1.T_correction2

        extra_delays = {"enqueueD_delay": 1.000,
                        "HVI_delay" : 3.37,
                        "switching_delay" : 2.00,
                        "t_delay": self.physical_params.t_delay}
        duration += sum(extra_delays.values())
        duration_ms = duration / 1e3
        return duration_ms


    def get_2Q_clocktime(self):
        # For now just use the longest gate, don't check whether it is actually used in the circuit.
        duration = np.max(self.RFCompiler.gate_solution.loc[:, "XX_duration_us"])
        extra_delays = {"enqueueD_delay": 1.000,
                        "HVI_delay" : 3.37,
                        "switching_delay" : 2.00,
                        "twoQ_correction" : 0.01,
                        "global_ramp" : 10*2,
                        "t_delay": self.physical_params.t_delay}
        duration += sum(extra_delays.values())
        duration_ms = duration / 1e3
        return duration_ms

    @staticmethod
    def get_circuit_slots(circuit_scan):
        """ Gets information about slots and qubits used in the circuit scan

        Args:
            circuit_scan: a list of circuits you want to scan

        Returns:
            circuit_slots: all the slots used in that circuit scan disregarding the measurement operation
            one_indexed: whether the ions are one or zero indexed see common_types.ion_to_slot
        """


        for i, circuit in enumerate(circuit_scan):
            if i == 0:
                all_qubits = len(circuit.all_qubits())
                qubit_ind = np.array([q.x for q in circuit.all_qubits()])
                active_qubits = len(qubit_ind)

                if np.all(qubit_ind > 0):
                    one_indexed = True
                else:
                    one_indexed = False
            icircuit_slots = np.sort(np.array([
                common_types.ion_to_slot(ion=iq,N_ions=all_qubits,one_indexed=one_indexed) for iq in qubit_ind
            ]))

            if i == 0:
                circuit_slots = icircuit_slots
            else:
                assert np.all(circuit_slots==icircuit_slots), "Qubit assignment must be identical across circuit scans"

        return all_qubits, active_qubits, circuit_slots, one_indexed


    def load_circuit_from_file(
        self,
        cirq_file: str = "",
        circuit_index: int = 0,
        suppress_circuit_scan: typing.Union[np.ndarray, bool] = False,
        use_SK1_AM: bool = True,
        print_circuit: bool = False,
        print_gate_list: bool = False,
    ):
        """This function reads a Cirq circuit from file and compiles it.

        Args:
            cirq_file: The pickled file containing the cirq circuit to be parsed
            circuit_index: In circuit scan mode, the index of the circuit being written
            suppress_circuit_scan: Flags whether circuit scan will be suppressed
                for the gate we are adding (i.e., only one waveform file will be written)
                Can either be a bool to specify the whole circuit as scanned or static, or
                can be an nd_array of size (num_moments, max(operations/moment)) that specifies
                whether each operation is scanned or static
            use_SK1_AM: Determines whether we use shaped or square SK1 pulses
            print_circuit: Determines whether we print the circuit loaded from file
            print_gate_list: Determines whether we print the generated gate list
        """

        with open(str(PATH_TO_SERVER / cirq_file), "rb") as c_file:
            loaded_circuit = pickle.load(c_file)

        if len(loaded_circuit) > 1:
            self.circuit_scan = True
        else:
            self.circuit_scan = False

        if type(suppress_circuit_scan) == list and self.circuit_scan is True:
            suppress_circuit_scan = np.array(suppress_circuit_scan)
        elif type(suppress_circuit_scan) != bool and self.circuit_scan is False:
            suppress_circuit_scan = suppress_circuit_scan[0]

        # Inspect circuit and gather information before running
        self.max_1Q_angle, self.max_2Q_angle = self.get_max_angles(loaded_circuit)
        self.N_qubits, self.active_qubits, self.circuit_slots, self.one_indexed = self.get_circuit_slots(loaded_circuit)
        self.clocktime_1Q = self.get_1Q_clocktime(self.max_1Q_angle)
        self.clocktime_2Q = self.get_2Q_clocktime()
        print("1Q gate Length = {0} ms".format(self.clocktime_1Q))
        print("2Q gate Length = {0} ms".format(self.clocktime_2Q))

        wall_clock_offset = 0.0
        feedforward_1Q = 0.0
        feedforward_2Q = 0.0093 #calibrated 10/07/2020
        #feedforward_2Q = 0.008  #12/26/20, gate (2,3)
        #feedforward_2Q = 0.014 #12/26/20, gate (10,11)
        feedforward_2Q = 0.011  #12/26/20, average of (2,3) and (10,11)
        print("Here!")
        self.wall_clock = WallClock(offset=wall_clock_offset,
                                    feedforward_1Q=feedforward_1Q,
                                    feedforward_2Q=feedforward_2Q)

        if self.circuit_scan is True:
            _LOGGER.info("Ignoring circuit_index argument since the file specified is a circuit scan")

            for i, icircuit in enumerate(loaded_circuit):
                self.compile_circuit(
                    circuit=icircuit,
                    circuit_index=i,
                    suppress_circuit_scan_array=suppress_circuit_scan,
                    use_SK1_AM=use_SK1_AM,
                    print_circuit=print_circuit,
                    print_gate_list=print_gate_list,
                )

        else:
            if type(loaded_circuit)==list:
                loaded_circuit = loaded_circuit[0]
            self.compile_circuit(
                circuit=loaded_circuit,
                circuit_index=circuit_index,
                suppress_circuit_scan_array=suppress_circuit_scan,
                use_SK1_AM=use_SK1_AM,
                print_circuit=print_circuit,
                print_gate_list=print_gate_list,
            )


    def compile_circuit(
        self,
        circuit: cirq.Circuit,
        circuit_index: int = 0,
        suppress_circuit_scan_array: typing.Union[np.ndarray, bool] = False,
        use_SK1_AM: bool = True,
        print_circuit: bool = False,
        print_gate_list: bool = False,
    ):
        """This function reads through a Cirq circuit and adds its gates to the array
        held by the RFCompiler object that calls this function.  We will use exclusively
        XX and SK1 gates.

        Args:
            circuit: The cirq circuit to be parsed
            circuit_index: In circuit scan mode, the index of the circuit being written
            suppress_circuit_scan_array: Flags whether circuit scan will be suppressed
                for the gate we are adding
                (i.e., only one waveform file will be written)
            use_SK1_AM: Determines whether we use shaped or square SK1 pulses
            print_circuit: Determines whether we print the circuit loaded from file
            print_gate_list: Determines whether we print the generated gate list
        """

        self.circuit = circuit

        num_moments = len(circuit)
        max_ops = max([len(moment) for moment in circuit])

        if type(suppress_circuit_scan_array) == bool:
            suppress_circuit_scan_array = np.full((num_moments, max_ops), suppress_circuit_scan_array)

        assert suppress_circuit_scan_array.shape == (num_moments, max_ops),\
            "Circuits must all be the same shape in circuit scan and must match shape of supress_circuit_scan_array"

        self.use_SK1_AM = use_SK1_AM
        self.circuit_index = circuit_index

        self.print_gate_list = print_gate_list
        self.return_string = ""


        N_ions = self.physical_params.N_ions
        if self.N_qubits != N_ions:
            _LOGGER.error(
                "Circuit with %i qubits loaded for chain of %i ions", self.N_qubits, N_ions
            )
            raise Exception(
                "Circuit with {0} qubits loaded for chain of {1} ions".format(
                    self.N_qubits, N_ions
                )
            )

        if print_circuit:
            print("Circuit loaded: \n", self.circuit, "\n")
            self.return_string += "Circuit loaded: \n" + str(self.circuit) + "\n"

        if self.print_gate_list:
            print("Generated gates:")
            self.return_string += "Generated gates:"

        measured_slots = []
        # Once an RZ gate is scanned, all the following gates must also be scanned as well
        # Add it to this list and then always check array before adding a new gate
        phase_scan_slots = []

        # Right now we can only one length of wait gate and it is global.
        global_wait_duration = None

        # Keep track of time elapsed in circuit, in milliseconds
        self.wall_clock.reset()

        for imoment, moment in enumerate(self.circuit):
            for iop, op in enumerate(moment):

                if op.gate.num_qubits() == 1:
                    oneQ_gate = True
                else:
                    oneQ_gate = False

                name = op.gate.__class__.__name__

                if name in self.gate_dictionary.software_gates:
                    physical_gate = False
                else:
                    physical_gate = True

                slots = []
                for channel in op.qubits:
                    slots.append(
                        common_types.ion_to_slot(
                            ion=int(str(channel)), N_ions=N_ions, one_indexed=self.one_indexed
                        )
                    )

                try:
                    rotation = op.gate._exponent * np.pi
                    if physical_gate:
                        rotation = self.wall_clock.feed_forward(rotation, two_qubit=(not oneQ_gate))

                except:
                    rotation = 0.0

                    if name == "MeasurementGate":
                        name = "DT"
                    elif name == "WaitGate":
                        if self.print_gate_list:
                            self.return_string += self.print_gate("Wait Gate", "Global", True)


                # Check that the gate has not been measured already
                if any([s in measured_slots for s in slots]):
                    _LOGGER.error(
                        "Cannot perform an operation on qubits (%s)"
                        " that have already been measured",
                        slots,
                    )
                    raise Exception(
                        "Cannot perform an operation on qubits ({0})".format(slots)
                        + " that have already been measured"
                    )

                # Check to see if the next gate is a wait gate
                wait_after = False
                # If we are at the end of a moment, go to the next one
                if iop == (len(moment)-1):
                    # unless its the last moment
                    if imoment == (len(self.circuit)-1):
                        next_op = None
                    else:
                        next_op = self.circuit[imoment+1].operations[0]
                # Else go to the next operation in this moment
                else:
                    next_op = self.circuit[imoment].operations[iop+1]

                try:
                    next_name = next_op.gate.__class__.__name__
                except:
                    next_name = None

                if next_name == "WaitGate":
                    wait_duration = next_op.gate.duration.total_micros()
                    if global_wait_duration is None:
                        global_wait_duration = wait_duration
                        self.RFCompiler.wait_after_time = global_wait_duration
                    else:
                        assert wait_duration == global_wait_duration,\
                                "Currently we can only accept WaitGates of the same duration within a circuit"
                    wait_after = True


                # Make gate scanned if there is a scanned Rz before it
                if any([s in phase_scan_slots for s in slots]):
                    suppress_circuit_scan = False
                else:
                    suppress_circuit_scan = suppress_circuit_scan_array[imoment, iop]

                if name in self.gate_dictionary.RX_gate:
                    self.RX(slots[0], rotation, suppress_circuit_scan, wait_after)

                elif name in self.gate_dictionary.RY_gate:
                    self.RY(slots[0], rotation, suppress_circuit_scan, wait_after)

                elif name in self.gate_dictionary.RZ_gate:

                    # If the gate is scanned we need to scan all the following gates
                    if not suppress_circuit_scan:
                        phase_scan_slots.append(slots[0])

                    self.RZ(slots[0], rotation, suppress_circuit_scan, wait_after)

                elif name in self.gate_dictionary.MS_gate:
                    self.XX(slots, rotation, suppress_circuit_scan, wait_after)

                elif name == "DT":
                    measured_slots.append(slots[0])

                elif name == "WaitGate":
                    global_wait_duration_ms = global_wait_duration / 1e3
                    self.wall_clock.add(global_wait_duration_ms)

                else:
                    raise Exception(
                        "Gate not currently supported in compiler {0} {1} {2}".format(name, rotation, slots))

                # Keep time of the time elapsed during the circuit so far.
                if oneQ_gate and physical_gate:
                    self.wall_clock.add(self.clocktime_1Q)
                elif not oneQ_gate and physical_gate:
                    self.wall_clock.add(self.clocktime_2Q)
                else:
                    pass

                print("time = {0}".format(self.wall_clock.time))

        # This is a hack to make sure that all the circuit scans have the same 1Q length.
        # Add a blank 1Q gate at the end of the sequence to coerce the max length to be
        # the same across all circuit indices.
        if self.circuit_scan:
            if self.use_SK1_AM:
                self.RFCompiler.add_SK1_am(
                    slots=[17],
                    theta=self.max_1Q_angle,
                    phi=0,
                    ind_amp=0,
                    global_amp=0,
                    gate_name="blank",
                    circuit_index=self.circuit_index,
                    suppress_circuit_scan=True,
                )

            else:
                self.RFCompiler.add_SK1(
                    slots=[17],
                    theta=self.max_1Q_angle,
                    phi=0,
                    ind_amp=0,
                    global_amp=0,
                    gate_name="blank",
                    circuit_index=self.circuit_index,
                    suppress_circuit_scan=True,
                )

        return self.return_string


class CircuitBuilder:
    """ THIS CLASS IS DEPRECATED. IT IS REMAINING HERE FOR BACKWARDS COMPATIBILITY BUT HAS BEEN MOVED TO
    EURIQFRONTEND.CIRCUITS.CIRCUIT_BUILDER

     This class provides methods that build a circuit of native ion gates in Cirq.

    Its main function is to provide a compact notation for writing out circuits in
    ARTIQ.
    """

    def __init__(self, N_qubits: int, qubit_offset: int = 0):

        self.N_qubits = N_qubits
        self.qubit_offset = qubit_offset

        self.qubit_list = cirq.LineQubit.range(N_qubits)
        # us = 1000 * cirq.Duration(nanos=1)
        # ion_device = cirq.ion.IonDevice(measurement_duration=100 * us,
        #                                 twoq_gates_duration=200 * us,
        #                                 oneq_gates_duration=10 * us,
        #                                 qubits=self.qubit_list)

        self._circuit = cirq.Circuit()

    def get_circuit(self):

        for i in range(self.N_qubits):
            self._circuit.append([cirq.measure(self.qubit_list[i])])

        return self._circuit

    def qubit_to_slot(self, qubit: int):
        common_types.ion_to_slot(
            ion=qubit + self.qubit_offset + 1, N_ions=self.N_qubits
        )

    ###################################################
    #                  DEFINE GATES                   #
    ###################################################

    def RX(
            self,
            qubit: int,
            sign: int,
            theta: float = np.pi / 2,
            pre_phase_shift: float = 0,
            post_phase_shift: float = 0,
    ):

        if pre_phase_shift != 0:
            self._circuit.append(
                [cirq.Rz(pre_phase_shift).on(self.qubit_list[qubit + self.qubit_offset])]
            )
        self._circuit.append(
            [cirq.Rx(sign * theta).on(self.qubit_list[qubit + self.qubit_offset])]
        )
        if post_phase_shift != 0:
            self._circuit.append(
                [cirq.Rz(post_phase_shift).on(self.qubit_list[qubit + self.qubit_offset])]
            )

    def RY(self,
           qubit: int,
           sign: int,
           theta: float = np.pi / 2,
           pre_phase_shift: float = 0,
           post_phase_shift: float = 0,
           ):

        if pre_phase_shift != 0:
            self._circuit.append(
                [cirq.Rz(pre_phase_shift).on(self.qubit_list[qubit + self.qubit_offset])]
            )
        self._circuit.append(
            [cirq.Ry(sign * theta).on(self.qubit_list[qubit + self.qubit_offset])]
        )
        if post_phase_shift != 0:
            self._circuit.append(
                [cirq.Rz(post_phase_shift).on(self.qubit_list[qubit + self.qubit_offset])]
            )

    def RZ(self,
           qubit: int,
           sign: int,
           theta: float = np.pi / 2,
           pre_phase_shift: float = 0,
           post_phase_shift: float = 0,
           ):

        if pre_phase_shift != 0:
            self._circuit.append(
                [cirq.Rz(pre_phase_shift).on(self.qubit_list[qubit + self.qubit_offset])]
            )
        self._circuit.append(
            [cirq.Rz(sign * theta).on(self.qubit_list[qubit + self.qubit_offset])]
        )
        if post_phase_shift != 0:
            self._circuit.append(
                [cirq.Rz(post_phase_shift).on(self.qubit_list[qubit + self.qubit_offset])]
            )

    def XX(self,
           qubits: typing.List[int],
           sign: int,
           pre_phase_shift: float = 0,
           post_phase_shift: float = 0,
           ):

        if pre_phase_shift != 0:
            self._circuit.append(
                [cirq.Rz(pre_phase_shift).on(self.qubit_list[qubits[0] + self.qubit_offset]),
                 cirq.Rz(pre_phase_shift).on(self.qubit_list[qubits[1] + self.qubit_offset])]
            )
        self._circuit.append(
            [
                cirq.MS(sign * np.pi / 4).on(
                    self.qubit_list[qubits[0] + self.qubit_offset],
                    self.qubit_list[qubits[1] + self.qubit_offset],
                )
            ]
        )
        if post_phase_shift != 0:
            self._circuit.append(
                [cirq.Rz(post_phase_shift).on(self.qubit_list[qubits[0] + self.qubit_offset]),
                 cirq.Rz(post_phase_shift).on(self.qubit_list[qubits[1] + self.qubit_offset])]
            )
