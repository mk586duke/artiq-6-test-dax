import functools
import logging
import pathlib
import typing
import logging
import pathlib
from pprint import pprint
from shutil import copyfile

import artiq.language.environment as artiq_env

import pulsecompiler.qiskit.backend as qbe
import pulsecompiler.rfsoc.structures.channel_map as rfsoc_mapping
import pulsecompiler.rfsoc.tones.tonedata as tones
from pulsecompiler.qiskit.configuration import BackendConfigCenterIndex, QuickConfig

# import jaqalpaw.utilities.parameters as OctetParams
# import pulsecompiler.qiskit.pulses as pc_pulse
# import pulsecompiler.qiskit.schedule_converter as octetconverter
# import pulsecompiler.rfsoc.structures.splines as spl
# import pulsecompiler.rfsoc.tones.upload as uploader
# import pulsecompiler.rfsoc.tones.record as record

from qiskit import QuantumCircuit
import qiskit.pulse as qp
import sipyco.pyon as pyon
import numpy as np
#np.set_printoptions(precision=3)
import matplotlib.pyplot as plt

from sipyco.pc_rpc import Client

from qiskit.assembler import disassemble

import euriqabackend.devices.keysight_awg.gate_parameters as gate_params

import euriqafrontend.settings.calibration_box as calibrations
from euriqabackend import _EURIQA_LIB_DIR
from euriqafrontend import EURIQA_NAS_DIR
from euriqafrontend.settings import RF_CALIBRATION_PATH

import cirq
import euriqafrontend.circuits.cirq_converter as cirq_conv
import qiskit.circuit.parameter as qiskit_param
import pulsecompiler.qiskit.transforms.gate_barrier as barrier

import qiskit.compiler as q_compile

import euriqabackend.waveforms.instruction_schedule as eur_inst_sched
import euriqabackend.waveforms.conversions as wf_convert
import euriqabackend.waveforms.single_qubit as single_qb_wf
import euriqabackend.waveforms.multi_qubit as multi_qb_wf

_LOGGER = logging.getLogger(__name__)

ScheduleList = typing.List[qp.Schedule]
ScheduleOrList = typing.Union[qp.Schedule, ScheduleList]
CompiledScheduleList = typing.List[tones.ChannelSequence]
QiskitChannel = typing.Union[qp.DriveChannel, qp.ControlChannel]


class RFSOCoffline():

    _DEFAULT_RFSOC_DESCRIPTION_PATH = (
        _EURIQA_LIB_DIR
        / "euriqabackend"
        / "databases"
        / "rfsoc_system_description.pyon"
    )

    def get_dataset(self, name):
        self.datasets.get(name)

    def build(self):

        master_ip = "192.168.78.152"
        self.schedule, self.exps, self.datasets = [
            Client(master_ip, 3251, "master_" + i) for i in
            "schedule experiment_db dataset_db".split()]

        self.num_shots = 100

        self.upload_in_streaming_mode = False

        self.record_upload_sequence = False

        self.default_sync = True

        self.openpulse_schedule_qobj = artiq_env.PYONValue(default=None)

        self.rfsoc_board_description = pyon.load_file(self._DEFAULT_RFSOC_DESCRIPTION_PATH)

        self._rf_calib_file_path = pathlib.Path(str(RF_CALIBRATION_PATH))

        self.keep_global_on_global_beam_detuning = 5e6

        self._qiskit_backend = None

    @property
    def qiskit_backend(self) -> qbe.MinimalQiskitIonBackend:
        if self._qiskit_backend is None:
            raise RuntimeError("Trying to access qiskit backend before initialization")
        return self._qiskit_backend

    @qiskit_backend.setter
    def qiskit_backend(self, backend: qbe.MinimalQiskitIonBackend):
        self._qiskit_backend = backend

    @staticmethod
    def _load_gate_solutions(
        path: typing.Union[str, pathlib.Path],
        num_ions: int,
        load_calibrations: bool = True,
        tweak_file_name: str = "gate_solution_tweaks2.h5",
    ) -> calibrations.CalibrationBox:
        """Load Gate Solutions & tweaks from file path specified.

        If ``load_calibrations`` and the file does not exist, then it will
        create a new, empty :class:`GateCalibration` object.
        """
        result = calibrations.CalibrationBox({"gate_solutions": {}, "gate_tweaks": {}})
        solutions = pathlib.Path(path)
        if not solutions.is_file():
            raise FileNotFoundError(f"Solution file '{solutions}' does not exist")
        result["gate_solutions.struct"] = gate_params.GateSolution.from_h5(solutions)
        result["gate_solutions.path"] = solutions
        if load_calibrations:
            potential_tweaks_file = solutions.with_name(tweak_file_name)
            result["gate_tweaks.path"] = potential_tweaks_file
            if potential_tweaks_file.is_file():
                result["gate_tweaks.struct"] = gate_params.GateCalibrations.from_h5(
                    potential_tweaks_file
                )
            else:
                _LOGGER.warning(
                    "No pre-existing gate calibrations found. "
                    "Initializing from gate solutions"
                )
                result[
                    "gate_tweaks.struct"
                ] = gate_params.GateCalibrations.from_gate_solution(
                    result["gate_solutions.struct"]
                )
            assert (
                result["gate_solutions.struct"].solutions_hash
                == result["gate_tweaks.struct"].solutions_hash
            )
            assert (
                result["gate_solutions.struct"].num_ions
                == result["gate_tweaks.struct"].num_ions
            )
        assert num_ions == result["gate_solutions.struct"].num_ions
        return result

    @staticmethod
    def plot_schedule(
        schedule: qp.Schedule, save_path: pathlib.Path = None, **kwargs,
    ) -> "Figure":  # noqa: F821
        """Plot a Qiskit Schedule.

        Simple wrapper, but takes care of the proper imports, and allows saving
        automatically if needed.
        If ``save_path`` is not specified, this method will block until the figure is
        closed, then execution will continue. Else, it will attempt to save the figure
        to a specified file.
        Any ``kwargs`` are passed directly to the schedule drawing method.
        """
        fig = schedule.draw(**kwargs)
        if save_path is None:
            # plot immediately
            import matplotlib.pyplot as plt

            plt.show()
        else:
            fig.savefig(str(save_path))

        return fig

    @property
    def _has_schedule_argument(self):
        """Check if a schedule is being passed as an argument."""
        return self.openpulse_schedule_qobj is not None

    def number_of_ions(self):
        return int(self._rf_calib.other.number_of_ions.value)

    def prepare(
        self,
        schedules: ScheduleList = None,
        load_gate_solutions: bool = True,
        compile_schedules: bool = True,
    ):
        """Load the parameters and calibration values that will be used.

        DO NOT SEND THEM TO RF COMPILER. Doing so will override whatever state
        the RF compiler is in, and it could be preparing/running some other experiment.
        """
        # Load RF calibrations
        # Based on settings/rf_calibration.json, adds datasets & calculated values
        self._rf_calib = calibrations.CalibrationBox.from_json(
            filename=self._rf_calib_file_path,
            dataset_dict=self.datasets#self._HasEnvironment__dataset_mgr,
        )

        num_ions = int(self._rf_calib.other.number_of_ions.value)

        self.rfsoc_map = rfsoc_mapping.RFSoCChannelMapping(self.rfsoc_board_description)
        # use the total number of ions in the chain as the # of qubits,
        # NOT the addressable qubits
        # global: bd 0, ch0
        # ion+1: bd 1, ch0
        # ion-1: bd 2, ch0
        # ion0: bd 0, ch4
        # ion2: fake channel -> oscilloscope, bd2 ch7
        # ind ch means the index of the individual channel.
        # bd 0: ch1 = ind ch 0, ch7 = ind ch 6
        # bd 1: ch0 = ind ch 7, ch7 = ind ch 14
        # bd 2: ch0 = ind ch 15, ch7 = ind ch 22

        #original autocalib
        # config = QuickConfig(num_ions, rfsoc_map, {0: 3, -1: 15, 1: 7, 2: 22})
        if(num_ions==1):
            config = QuickConfig(num_ions, self.rfsoc_map, {0: 3})

        if(num_ions==15):
            config = QuickConfig(num_ions, self.rfsoc_map, {
                7: 6, # bd0_01 # PMT1
                6: 18,# bd1_03 # PMT2
                5: 15, # bd1_00 # PMT3
                4: 17, # bd1_02 # PMT4
                3: 16, # bd1_01 # PMT5
                2: 4, # bd0_03 # PMT6
                1: 5, # bd0_02 # PMT7
                0: 3, # bd0_04 # PMT8
                -1: 1, # bd0_06 # PMT9
                -2: 2, # bd0_05 # PMT10
                -3: 8, # bd2_01 # PMT11
                -4: 9, # bd2_02 # PMT12
                -5: 7, # bd2_00 # PMT13
                -6: 10, # bd2_03 # PMT14
                -7: 0 # bd0_07 PMT15
                }
                )
        elif(num_ions == 23):
            config = QuickConfig(num_ions, self.rfsoc_map, {
                -11: 14,
                -10: 13,
                -9: 12,
                -8: 11,
                -7: 0,
                -6: 10,
                -5: 7,
                -4: 9,
                -3: 8,
                -2: 2,
                -1: 1,
                0: 3,
                1: 5,
                2: 4,
                3: 16,
                4: 17,
                5: 15,
                6: 18,
                7: 6,
                8: 19,
                9: 20,
                10: 21,
                11: 22})
        else:
            config = QuickConfig(num_ions, self.rfsoc_map, {
                -15:26,
                -14:25,
                -13:24,
                -12:23,
                -11: 14,
                -10: 13,
                -9: 12,
                -8: 11,
                -7: 0,
                -6: 10,
                -5: 7,
                -4: 9,
                -3: 8,
                -2: 2,
                -1: 1,
                0: 3,
                1: 5,
                2: 4,
                3: 16,
                4: 17,
                5: 15,
                6: 18,
                7: 6,
                8: 19,
                9: 20,
                10: 21,
                11: 22,
                12:27,
                13:28,
                14:29,
                15:30})

        #multi-rfsoc branch
        # config = QuickConfig(num_ions, rfsoc_map, {0: 3, -1: 15, 1: 7, -2:0, 2:1, -3:2, 3:4, -4:5, 4:6, -5:8, 5:9, -6:10, 6:11, -7:12, 7:13 })
        # config = QuickConfig(num_ions, rfsoc_map, {-7:0,-6:1,-5:2,-4:3,-3:4,-2:5,-1:6,0:7,1:8,2:9,3:10,4:11,5:12,6:13,7:22})

        self.zero_qiskit_backend = qbe.MinimalQiskitIonBackend(num_ions-2,self.rfsoc_map,use_zero_index=True)

        self._qiskit_backend = qbe.MinimalQiskitIonBackend(
            num_ions,
            self.rfsoc_map,
            config=config,
            properties=self._rf_calib.to_backend_properties(num_ions),
        )
        # don't load gate solutions if only have one ion, the gate solutions are
        # only for >= 2 qubit gates

        # load_gate_solutions = False
        if load_gate_solutions and num_ions > 1:
            try:
                # first try the raw path. If that doesn't exist, search in the EURIQA
                # NAS, and try checking subfolder for # of ions
                path_test = pathlib.Path(
                    self._rf_calib.gate_solutions.solutions_top_path
                )
                if not path_test.exists():
                    path_test = EURIQA_NAS_DIR / path_test
                if (path_test / self._rf_calib[f"gate_solutions.{num_ions}"]).exists():
                    path_test = path_test / self._rf_calib[f"gate_solutions.{num_ions}"]
                    self._rf_calib["gate_tweak_path"] = path_test.with_name("gate_solution_tweaks2.h5")
                self._rf_calib.merge_update(
                    self._load_gate_solutions(path_test, num_ions,)
                )
            except KeyError:
                _LOGGER.warning(
                    "Could not find gate solutions for %s ions", num_ions, exc_info=True
                )

    @property
    def dt(self) -> float:
        """Length of a schedule "dt" (base time unit) in seconds."""
        return self.qiskit_backend.configuration().dt

    @property
    def rfsoc_hardware_description(self) -> rfsoc_mapping.RFSoCChannelMapping:
        return self.qiskit_backend.configuration().rfsoc_channel_map

cirq_circuit = cirq.Circuit()

qb = [cirq.LineQubit(i) for i in range(19)]

for i in range(19):
    cirq_circuit.append(cirq.rz(np.pi/2).on(qb[i]))

# 4q GHZ preparation
# IDX_OFFSET = 6

# cirq_circuit.append([cirq.ry(np.pi).on(qb[i]) for i in [IDX_OFFSET-6,IDX_OFFSET-4]])
# cirq_circuit.append(cirq.ms(np.pi/4).on(qb[IDX_OFFSET-6],qb[IDX_OFFSET-5]))
# cirq_circuit.append(cirq.rz(np.pi/2).on(qb[IDX_OFFSET-5]))

# cirq_circuit.append(cirq.ms(-np.pi/4).on(qb[IDX_OFFSET-5],qb[IDX_OFFSET-4]))
# cirq_circuit.append(cirq.rz(np.pi/2).on(qb[IDX_OFFSET-4]))

# cirq_circuit.append(cirq.ms(np.pi/4).on(qb[IDX_OFFSET-4],qb[IDX_OFFSET-3]))
# cirq_circuit.append(cirq.rz(np.pi/2).on(qb[IDX_OFFSET-3]))

# cirq_circuit.append([cirq.ry(np.pi/2).on(qb[i]) for i in range(IDX_OFFSET-6,IDX_OFFSET-2)])

data = [ [18,13,14], [17,15,1], [12,2,7] ]
anc_z = [ [11,3], [16,4] ]
anc_x = [ [10,5], [6,0] ]

#0x,1x,2x,3x,4x,5x,6x,7x,8,9,10x,11x,12x,13x,14x,15,16x,17x,18x

from euriqafrontend.circuits.circuit_builder import IonRegister, add_ghz, print_circuit_output

msth = np.pi/4
# State Prep Logical |0>
for i in range(3):
    cirq_circuit.append(cirq.ms(msth).on(qb[data[i][0]], qb[data[i][1]]))
    cirq_circuit.append(cirq.rz(np.pi/2).on(qb[data[i][1]]))
    cirq_circuit.append(cirq.ms(-msth).on(qb[data[i][1]], qb[data[i][2]]))
    cirq_circuit.append(cirq.rz(-np.pi/2).on(qb[data[i][2]]))

# for j in range(3):
#     for i in range(3):
#         cirq_circuit.append(cirq.ry(-np.pi/2).on(qb[data[i][j]]))

for R in range(1):  #two rounds of extractions
    # Z syndrome extraction
    for i in range(3):
        for j in range(3):
            cirq_circuit.append(cirq.ry(np.pi/2).on(qb[data[i][j]]))

    for i in range(2):
        for j in range(3):
            cirq_circuit.append(cirq.ms(msth).on(qb[data[i][j]], qb[anc_z[R][i]]))
            cirq_circuit.append(cirq.ms(-msth).on(qb[data[i+1][j]], qb[anc_z[R][i]]))

    for j in range(3):
        cirq_circuit.append(cirq.rx(-np.pi/2).on(qb[data[0][j]]))
        cirq_circuit.append(cirq.rx(np.pi/2).on(qb[data[2][j]]))

    for i in range(3):
        for j in range(3):
            cirq_circuit.append(cirq.ry(-np.pi/2).on(qb[data[i][j]]))

    # X syndrome extraction
    for j in range(2):
        for i in range(3):
            cirq_circuit.append(cirq.ms(-msth).on(qb[data[i][j]], qb[anc_x[R][j]]))
            cirq_circuit.append(cirq.ms(msth).on(qb[data[i][j+1]],qb[anc_x[R][j]]))

    for i in range(3):
        cirq_circuit.append(cirq.rx(-np.pi/2).on(qb[data[i][0]]))
        cirq_circuit.append(cirq.rx(np.pi/2).on(qb[data[i][2]]))

#print_circuit_output(cirq_circuit)

num_ions = 23
qiskit_circuit = cirq_conv.convert_cirq_to_qiskit(cirq_circuit,num_ions=19)

#############################################################
# qubit mapping

# 18 13 14 17 15 1 12 2 7 11 3 10 5 16 4 6 0

circ = QuantumCircuit(19)

lst = []
def myrxx(angle, i, j):
    circ.rxx(1.0*angle,i,j)
    if i<j:
        lst.append((i-9,j-9))
    else:
        lst.append((j-9,i-9))

do_prep = True
do_y_rot = True
Z_stabs = [0,1]
X_stabs = []
rounds = [0,1]
n_wait = 0

for i in range(n_wait):
    myrxx(np.pi/2, 8,9)

circ.barrier(range(19))

#State Prep Logical |0>
if do_prep:
    for i in range(3):
        myrxx(np.pi/2, data[i][0], data[i][1])
        circ.rz(np.pi/2, data[i][1])
        myrxx(-np.pi/2, data[i][1], data[i][2])
        circ.rz(-np.pi/2, data[i][2])

circ.barrier(range(19))

for R in rounds:  # rounds of extractions
    # Z syndrome extraction
    if do_y_rot:
        for i in range(3):
            for j in range(3):
                circ.ry(np.pi/2, data[i][j])

    for i in Z_stabs:
        for j in range(3):
            myrxx(np.pi/2, data[i][j], anc_z[R][i])
            myrxx(-np.pi/2, data[i+1][j], anc_z[R][i])

    for j in range(3):
        if Z_stabs==[0,1]:
            # both stabs
            circ.rx(-np.pi/2, data[0][j])
            circ.rx(np.pi/2,  data[2][j])
        elif Z_stabs==[0]:
            # top Z only
            circ.rx(-np.pi/2, data[0][j])
            circ.rx(np.pi/2, data[1][j])
        elif Z_stabs==[1]:
            # bottom Z only
            circ.rx(-np.pi/2, data[1][j])
            circ.rx(np.pi/2,  data[2][j])

    if do_y_rot:
        for i in range(3):
            for j in range(3):
                circ.ry(-np.pi/2, data[i][j])

    circ.barrier(range(19))
#    X syndrome extraction
    for j in X_stabs:
        for i in range(3):
            myrxx(-np.pi/2, data[i][j], anc_x[R][j])
            myrxx(np.pi/2, data[i][j+1], anc_x[R][j])

    for i in range(3):
        if X_stabs==[0,1]:
            # both Xs
            circ.rx(-np.pi/2, data[i][0])
            circ.rx(np.pi/2,  data[i][2])
        elif X_stabs==[0]:
            # left X
            circ.rx(-np.pi/2, data[i][0])
            circ.rx(np.pi/2,  data[i][1])
        elif X_stabs==[1]:
            # right X
            circ.rx(-np.pi/2, data[i][1])
            circ.rx(np.pi/2,  data[i][2])

#     circ.barrier(range(19))
# for i in range(19):
#     circ.measure(i, i)
lst.sort()

sstr=""
for x in lst:
    sstr=sstr+("%d,%d;" % (x[0], x[1]))
print(sstr)
#print(lst)
#############################################################
master_ip = "192.168.78.152"
offlineModule = RFSOCoffline()
offlineModule.build()
offlineModule.prepare()

rfsoc_map = offlineModule.rfsoc_map
qiskit_backend = offlineModule._qiskit_backend

# %%

def center_to_23_pair(fnc):
    def _decorated_fnc(ions:typing.Sequence[int], th:float, *args, **kwargs):
        return fnc([i-9 for i in ions], theta=th*0.5, *args, **kwargs)
    return _decorated_fnc

def center_to_23_idx(fnc):
    def _decorated_fnc(ion_index:int, *args, **kwargs):
        return fnc(ion_index-9, *args, **kwargs)
    return _decorated_fnc

def myrz(ion_index:int, angle:float):
    return single_qb_wf.rz(ion_index, -angle, backend=qiskit_backend)

single_qubit_gates = {
    "id": single_qb_wf.id_gate,
    "r": center_to_23_idx(functools.partial(single_qb_wf.sk1_gaussian, backend = qiskit_backend)),
    "rx": center_to_23_idx(functools.partial(single_qb_wf.sk1_gaussian, phi=np.pi / 2, backend = qiskit_backend)),
    "ry": center_to_23_idx(functools.partial(single_qb_wf.sk1_gaussian, phi=0, backend = qiskit_backend)),
    "rz": center_to_23_idx(myrz)#functools.partial(single_qb_wf.rz, backend = qiskit_backend)),
}

multi_qubit_gates = {
    "rxx": center_to_23_pair(functools.partial(multi_qb_wf.xx_am_gate, backend = qiskit_backend)),
}

from pulsecompiler.qiskit.instruction_schedule import IonInstructionScheduleMap

instruction_schedule = IonInstructionScheduleMap(
        offlineModule.zero_qiskit_backend,
        single_qubit_gates,
        multi_qubit_gates
    )
instruction_schedule._backend = qiskit_backend

scheduled_scan_circuits = []
circuit_sequential = barrier.BarrierBetweenGates().run_circuit(circ)

circuit_scheduled = q_compile.schedule(
    circuit_sequential,
    qiskit_backend, inst_map=instruction_schedule
)
scheduled_scan_circuits.append(circuit_scheduled)

# print(qiskit_circuit)

# for theta_val in np.linspace(0.9, 1.1, 16):
#     circuit_copy = circ.copy()

#     circuit_copy.ry(np.pi/2,9)
#     circ.barrier(range(19))
#     circuit_copy.ry(np.pi/2,0)
#     circuit_copy.ry(np.pi/2,0)
#     circuit_copy.ry(np.pi/2,0)
#     circuit_copy.ry(np.pi/2,0)
#     #circuit_copy.r(-np.pi/2,theta_val,9)
#     circ.barrier(range(19))
#     circuit_copy.r(np.pi/2*theta_val,np.pi, 9)
#     #circuit_copy = QuantumCircuit(19)

#     # for i in range(1):
#     #     circuit_copy.rx(theta=np.pi, qubit = i)
#     #
#     circuit_sequential = barrier.BarrierBetweenGates().run_circuit(circuit_copy)

#     circuit_scheduled = q_compile.schedule(
#         circuit_sequential,
#         qiskit_backend, inst_map=instruction_schedule
#     )
#     scheduled_scan_circuits.append(circuit_scheduled)

# %%
# submit converted qiskit circuit
import euriqafrontend.interactive.rfsoc.submit as rfsoc_submit

rfsoc_submit.submit_schedule(
        scheduled_scan_circuits,
        master_ip,
        qiskit_backend,
        experiment_kwargs={
            "xlabel": "red and blue detuning",
            # "x_values": pyon.encode(w_k_vec),
            # "xlabel": "rred blue spin phase",
            # "x_values": pyon.encode(rel_phase_vec),
            "default_sync": False,
            "num_shots": 2000,
            "PMT Input String": "-3:19",
            "lost_ion_monitor": False,
            "schedule_transform_aom_nonlinearity": True,
            "schedule_transform_keep_global_beam_on": True,
            "schedule_transform_pad_schedule": True,
            "do_sbc": True,
            "do_SBC": False,
            "priority": 0,
        })

# from euriqabackend.devices.keysight_awg.gate_parameters import GateCalibrations
# gate_tweaks_struct = GateCalibrations.from_h5("/media/euriqa-nas/CompactTrappedIonModule/Data/gate_solutions/2023_02_16/gate_solution_tweaks2.h5")

# #print(i,j,gate_tweaks_struct.gate_parameters_df.loc[(8,9),"individual_amplitude_multiplier"])

# for i in range(8,27):
#     for j in range (8,27):
#         if i<j:
#             print(i-17,j-17,gate_tweaks_struct.gate_parameters_df.loc[(i,j),"individual_amplitude_multiplier"])
