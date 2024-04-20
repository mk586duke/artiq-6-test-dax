"""Support for RFSoC Sideband Cooling schedules/sequences.

General way that this works on the RFSoC (single-schedule mode):
1. RFSoC SBC module requests a sideband cooling schedule,
    given a certain set of frequencies & cooling increments.
2. Define single Qiskit schedule for entire SBC sequence.
    Define pump timings on a dummy channel, ``MeasurementChannel(100)``.
    Have some buffer for timing misalignment b/w RFSoC & ARTIQ (different clock
    periods, RFSoC triggering, etc)
3. Convert pump on/off timings to differential timings
    (i.e. on for X us, off for X us, etc.)
4. Pass schedule & pump timings to RFSoC SBC module.
5. RFSoC SBC module prepends the SBC schedule before every RFSoC output
    sequence (e.g. circuit).
    These are all compiled during ``prepare()``, and uploaded to the RFSoC
6. For every shot of the experiment/circuit:
    Trigger RFSoC once to start the SBC sequence.
    While SBC sequence is playing, turn pumping on/off at the appropriate times.
    Trigger RFSoC a second time to start the nominal experiment/sequence.
7. Repeat 6 for every shot.
"""
import logging
import typing
from euriqabackend.coredevice.ad9912 import freq_to_mu
import artiq.language.environment as artiq_env
from artiq.language.types import TInt64
import numpy as np
import numpy.matlib as matlib
import pulsecompiler.rfsoc.tones.tonedata as tones
import pulsecompiler.qiskit.backend as pc_backend
import qiskit.pulse as qp
from artiq.language.core import kernel, delay, delay_mu, parallel

import euriqafrontend.modules.cw_lasers as cw_lasers
import euriqabackend.waveforms.sideband_cooling as wf_sbc
import euriqafrontend.modules.rfsoc as rfsoc

_LOGGER = logging.getLogger(__file__)


class RFSoCSidebandCooling(rfsoc.RFSOC):
    DATASET_KEYS = {
        "duration_init": "global.RFSOC.SBC.duration_init",
        "duration_increment": "global.RFSOC.SBC.duration_increment",
        "ind_amp": "global.RFSOC.SBC.ind_amp",
        "global_amp": "global.RFSOC.SBC.global_amp",
        "Ncycles": "global.RFSOC.SBC.Ncycles",
        "SD_cool_duration": "global.RFSOC.SBC.SD_cool_duration",
        "SD_cool_on": "global.RFSOC.SBC.SD_cool_on"
    }

    kernel_invariants = {
        "pump_states",
        "pump_timings_mu",
    }
    def __init__(self, pump_module :cw_lasers.PumpDetect, SD_module: cw_lasers.BaseDoubleDDS, managers_or_parent, *args, **kwargs) -> None:
        self.pump_detect = pump_module
        self.SD = SD_module
        self.__managers_or_parent: artiq_env.HasEnvironment = managers_or_parent
        super().__init__(managers_or_parent, *args, **kwargs)

    def build(self):
        super().build()
        self.equil_loops = 0
        self._name = self.__class__.__name__
        self.do_sbc = self.get_argument(
            "do_sbc", artiq_env.BooleanValue(default=True), group=self._name
        )
        self.setattr_device("eom_935_3ghz")
        # needed for compilation
        self.pump_timings_mu = []
        self.SD_cool_duration_mu = np.int64(0)
        self.pump_states = [False]  # needed for compilation
        self.do_SD_cool = False

    def prepare(self):
        # Initialize variables
        self.pump_timings = None
        self.compensated_sequence_durations = []
        # prepare pump_detect & second-stage cooling modules
        # self.call_child_method("prepare")
        if _LOGGER.getEffectiveLevel() <= logging.DEBUG:
            raise RuntimeError(
                "Experiments w/ single-schedule SBC & logging.DEBUG will hang "
                "after ~5 shots during the @kernel run() section. "
                "Fix this error or raise the logging level of the experiment to >= INFO"
            )

        super().prepare()  # call rfsoc prepare. compiles schedules.

        self.sbc_global_amplitude = self.get_dataset(
            self.DATASET_KEYS["global_amp"], default=1e-6
        )
        self.sbc_individual_amplitude = self.get_dataset(
            self.DATASET_KEYS["ind_amp"], default=1e-6
        )

        self.parallel_sbc_duration_init = self.get_dataset(
            self.DATASET_KEYS["duration_init"], default=1e-6
        )
        self.parallel_sbc_duration_increment = self.get_dataset(
            self.DATASET_KEYS["duration_increment"], default=5e-6
        )

        # save the sequence durations before adding SBC schedules
        self.compensated_sequence_durations = list(
            map(
                lambda sched: self.core.seconds_to_mu(
                    tones.sequence_duration_cycles(sched) * tones.CLKPERIOD + 1e-6
                ),
                self.compiled_sequence_list,
            )
        )

        total_pump_wait = self.pump_detect.pump_duration
        self.do_SD_cool = (self.get_dataset(self.DATASET_KEYS["SD_cool_on"], 0) != 0)
        self.SD_cool_duration = 0.0
        if(self.do_SD_cool):
            self.SD_cool_duration += self.get_dataset(self.DATASET_KEYS["SD_cool_duration"], 2e-6)
            total_pump_wait += self.SD_cool_duration

        # generate SBC cooling schedules, then prepend those to compiled schedules
        offset_freq = -2.8e6 ## to make sure that most of the frequency shift is done on the global
        if self.do_sbc:
            num_ions = int(self._rf_calib.other.number_of_ions.value)
            (
                sbc_schedule,
                (pump_timings, pump_states),
            ) = self.get_parallel_sbc_single_schedule(
                num_ions,
                pump_pulse_duration=total_pump_wait,
                # TODO: fill in with actual value here
                relative_ion_durations={i: 1.0 for i in range(num_ions)},
                duration_start=self.parallel_sbc_duration_init,
                duration_increment=self.parallel_sbc_duration_increment,
                individual_frequency=self._rf_calib.frequencies.individual_carrier_frequency.value+offset_freq,
                global_frequency=self._rf_calib.frequencies.global_carrier_frequency.value+offset_freq,
                individual_amplitude=self.sbc_individual_amplitude,
                global_amplitude=self.sbc_global_amplitude,
                # off_detuning=-20e6,
                backend=self._qiskit_backend,
            )
            new_sequences, _durations = self.prepend_schedules(
                [sbc_schedule], self.compiled_sequence_list
            )
            for _i in range(int(self.equil_loops)):
                new_sequences.insert(0, new_sequences[0])

            self.compiled_sequence_list = new_sequences
            self.pump_timings = pump_timings
            self.pump_states = pump_states
            self.pump_timings_mu = list(
                map(self.core.seconds_to_mu, pump_timings)
            )
            self.SD_cool_duration_mu = self.core.seconds_to_mu(self.SD_cool_duration)

    def get_sbc_mode_sequence(self, num_ions: int, num_repeats: int = 1) -> np.ndarray:
        if num_ions == 1:
            # mode_array = wf_sbc.awg_riffle_to_modes(
            #     "e 17\n" * num_repeats, min_slot=17, max_slot=17
            # )
            modes = np.array(self._rf_calib["chain_radial_modes.1.sideband_cooling_mode_indices.value"])
            mode_array = matlib.repmat(modes, int(num_repeats/(np.shape(modes)[0])), 1)
        elif num_ions == 3:
            modes = np.array(
                self._rf_calib[
                    "chain_radial_modes.3.sideband_cooling_mode_indices.value"
                ]
            )
            mode_array = matlib.repmat(modes, num_repeats, 1)
        elif num_ions <= 15:
            # mode_array = wf_sbc.awg_riffle_to_modes(
            #     """e 10-24
            #     e 17
            #     d 16;e 17;d 18
            #     e 15;d 16;e 17;d 18;e 19
            #     c 14;e 15;d 16;e 17;d 18;e 19;d 20
            #     e 13;c 14;e 15;d 16;e 17;d 18;e 19;d 20;e 21
            #     c 12;e 13;c 14;e 15;d 16;e 17;d 18;e 19;d 20;e 21;c 22
            #     d 11;c 12;e 13;c 14;e 15;d 16;e 17;d 18;e 19;d 20;e 21;c 22;d 23
            #     c 10;d 11;c 12;e 13;c 14;e 15;d 16;e 17;d 18;e 19;d 20;e 21;c 22;d 23;c 24
            #     d 11;c 12;e 13;c 14;e 15;d 16;e 17;d 18;e 19;d 20;e 21;c 22;d 23
            #     c 12;e 13;c 14;e 15;d 16;e 17;d 18;e 19;d 20;e 21;c 22
            #     e 13;c 14;e 15;d 16;e 17;d 18;e 19;d 20;e 21
            #     c 14;e 15;d 16;e 17;d 18;e 19;d 20
            #     e 15;d 16;e 17;d 18;e 19
            #     d 16;e 17;d 18
            #     e 17
            #     """
            #     * num_repeats,
            #     min_slot=10,
            #     max_slot=24,
            # )

            modes = np.array(
                self._rf_calib[
                    "chain_radial_modes.15.sideband_cooling_mode_indices.value"
                ]
            )
            if(np.mod(num_repeats,(np.shape(modes)[0]))>0):
                print("number of SBC cycle shouldb be divisable with number of different cooling configurations")
            mode_array = matlib.repmat(modes, int(num_repeats/(np.shape(modes)[0])), 1)

        elif num_ions == 23:
            QZY = self.get_dataset("global.Voltages.QZY")
            if(QZY>0):
                modes = np.array(self._rf_calib["chain_radial_modes.23.sideband_cooling_mode_indices.value"])
            else:
                modes = np.array(self._rf_calib["chain_radial_modes.-23.sideband_cooling_mode_indices.value"])

            if(np.mod(num_repeats,(np.shape(modes)[0]))>0):
                print("number of SBC cycle shouldb be divisable with number of different cooling configurations")
            mode_array = matlib.repmat(modes, int(num_repeats/(np.shape(modes)[0])), 1)

        else:
            raise NotImplementedError("SBC Sequence for > 15 ions not yet defined.")

        return mode_array

    def get_sbc_detunings(self, num_ions: int, awg_mode: bool = False):
        if num_ions == 1:
            return (
                (np.array(self._rf_calib["chain_radial_modes.1.nominal.value"])-self._rf_calib["chain_radial_modes.1.freq_offset.value"])
                * (-1)
                )
        elif num_ions == 3:
            # invert b/c using the RSB, these are all positive values
            scaling_factor_vec = np.array(
                self._rf_calib["chain_radial_modes.3.measured.value"]
            ) / np.array(self._rf_calib["chain_radial_modes.3.nominal.value"])
            return (
                np.array(self._rf_calib["chain_radial_modes.3.nominal.value"])
                * (-1)
                * scaling_factor_vec[0]
            )
        elif num_ions == 15:
            # invert sign b/c using the RSB, these are all positive values
            QZY = self.get_dataset("global.Voltages.QZY")
            if(QZY>0):
                return (
                    (np.array(self._rf_calib["chain_radial_modes.15.nominal.value"])-self._rf_calib["chain_radial_modes.15.freq_offset.value"])
                    * (-1)
                )
            else:
                return (
                (np.array(self._rf_calib["chain_radial_modes.-15.nominal.value"])-self._rf_calib["chain_radial_modes.-15.freq_offset.value"])
                * (-1)
                )
        elif num_ions == 23:
            # invert sign b/c using the RSB, these are all positive values
            QZY = self.get_dataset("global.Voltages.QZY")
            mode_freq_offset0 = self.get_dataset("monitor.calibration.radial_mode_offset_frequency")
            if(QZY>0):
                return (
                (np.array(self._rf_calib["chain_radial_modes.23.nominal.value"]) + mode_freq_offset0 - self._rf_calib["chain_radial_modes.23.freq_offset.value"])
                * (-1)
                )
            else:
                return (
                (np.array(self._rf_calib["chain_radial_modes.-23.nominal.value"])-self._rf_calib["chain_radial_modes.-23.freq_offset.value"])
                * (-1)
                )
        else:
            raise ValueError(f"Unsupported number of ions specified: {num_ions}")

    def get_sbc_relative_amplitudes(self, num_ions: int) -> np.ndarray:
        if num_ions == 1:
            return np.array(self._rf_calib["chain_radial_modes.1.rel_amp.value"])
        elif num_ions == 3:
            # note that mode3_dt was tuned to 2e-7 and mode3t0 was tuned to 5e-6
            # to acheive optimal parameters in this configuration
            # global is 0.5 and individual is 0.37 for relative amplitude of 1.
            # For this configuration it is roughly 700 nanoseconds pi time on carrier
            return np.array(self._rf_calib["chain_radial_modes.3.rel_amp.value"])
        elif num_ions == 15:
            return np.array(self._rf_calib["chain_radial_modes.15.rel_amp.value"])
        elif num_ions == 23:
            QZY = self.get_dataset("global.Voltages.QZY")
            if(QZY>0):
                return np.array(self._rf_calib["chain_radial_modes.23.rel_amp.value"])
            else:
                return np.array(self._rf_calib["chain_radial_modes.-23.rel_amp.value"])
        else:
            raise ValueError(f"Unsupported number of ions specified: {num_ions}")

    def get_parallel_sbc_single_schedule(
        self,
        num_ions: int,
        relative_ion_durations: typing.Dict[int, float],
        duration_start: float,
        duration_increment: float,
        pump_pulse_duration: float,
        individual_frequency: float,
        global_frequency: float,
        # off_detuning: float,
        individual_amplitude: float,
        global_amplitude: float,
        backend: pc_backend.MinimalQiskitIonBackend,
    ) -> typing.Tuple[
        qp.Schedule, typing.Tuple[typing.Sequence[float], typing.Sequence[bool]]
    ]:
        NUM_SBC_CYCLES = self.get_dataset(self.DATASET_KEYS["Ncycles"],default=24)

        mode_array = self.get_sbc_mode_sequence(num_ions, NUM_SBC_CYCLES)
        total_sbc_cycles = mode_array.shape[0]
        detunings = self.get_sbc_detunings(num_ions)
        fsec = self.get_dataset("global.Ion_Freqs.frf_sec")
        fsec_offset = fsec - 3.148e6
        detunings = [x-fsec_offset for x in detunings]

        mode_timings = wf_sbc.generate_timings_increment(
            (total_sbc_cycles, 1), duration_start, duration_increment
        )
        assert set(relative_ion_durations.keys()) == set(range(num_ions)), (
            "Out-of-range or invalid ion index provided. Should be 0-indexed, "
            f"and all < num_ions: {relative_ion_durations.keys()}"
        )
        relative_timings_arr = np.array(
            [relative_ion_durations[k] for k in sorted(relative_ion_durations.keys())]
        )
        ion_timing_array = wf_sbc.relative_timings_to_absolute(
            mode_timings, relative_timings_arr
        )
        relative_amplitudes = self.get_sbc_relative_amplitudes(num_ions)
        ion_amplitudes_array = matlib.repmat(relative_amplitudes * individual_amplitude,
                                int(total_sbc_cycles/(np.shape(relative_amplitudes)[0])), 1)
        # ion_amplitudes_array = wf_sbc.expand_relative_amplitudes(
        #     relative_amplitudes,
        #     num_repeats=int(total_sbc_cycles/(np.shape(relative_amplitudes)[0])),
        #     scaled_amplitude=individual_amplitude,
        # )
        backend_config = backend.configuration()

        sbc_freqs = {
            ion_index: individual_frequency - detunings[i]
            for i, ion_index in enumerate(backend_config.all_qubit_indices_iter)
        }
        sbc_schedule, pump_timing_states = wf_sbc.single_schedule_parallel_sbc(
            ion_frequencies=sbc_freqs,
            config_ion_mode_map = mode_array,
            pulse_durations=ion_timing_array,
            pulse_amplitudes_individual=ion_amplitudes_array,
            global_frequency=global_frequency,
            pump_duration=pump_pulse_duration,
            backend=backend,
            global_amplitude=global_amplitude,
            # off_detuning=off_detuning,
            pump_buffer_duration=500e-9,
        )
        return sbc_schedule, pump_timing_states

    @kernel
    def kn_init_rfsoc_sbc(self):
        pass
    #   the pump-detect module should already be prepared in the basic_environment
    #   which takes hundreds of us for DDS ramping
    #   (abstraction leakage)
    #   no need to prepare it again here

    @kernel
    def kn_do_rfsoc_sbc(self,):  # noqa: ATQ301 doesn't like artiq_types.*
        if self.do_SD_cool:
            self.SD.dds1.set_mu(
                bus_group=1, frequency_mu=freq_to_mu(196.31e6), amplitude_mu=1000
            )
        # trigger RFSoC SBC pulse, then pump to 0 for each pulse. repeat.
        self.trigger()
        for i in range(len(self.pump_timings_mu)):
            if self.pump_states[i]:
                self.pump_detect.on()
                if self.do_SD_cool:
                    self.SD.on1()
                    self.eom_935_3ghz.on()
                    delay_mu(self.SD_cool_duration_mu)
                    self.SD.off1()
                    self.eom_935_3ghz.off()
                    delay_mu(self.pump_timings_mu[i] - self.SD_cool_duration_mu)
                else:
                    delay_mu(self.pump_timings_mu[i])
            else:
                self.pump_detect.off()
                delay_mu(self.pump_timings_mu[i])

        # set to last value, untimed
        if self.pump_states[-1]:
            self.pump_detect.on()
        else:
            self.pump_detect.off()
