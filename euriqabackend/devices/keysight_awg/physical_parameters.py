import logging

import euriqabackend.devices.keysight_awg.interpolation_functions as intfn

_LOGGER = logging.getLogger(__name__)


class Monitor:
    def __init__(self):

        self.monitor_ind = -1
        self.detuning = -1
        self.amp = -1

    def set_params(self, monitor_ind: float, detuning: float, amp: float):

        self.monitor_ind = monitor_ind
        self.detuning = detuning
        self.amp = amp

    def check_set(self):

        params_loaded = (
            (self.monitor_ind >= 0) and (self.detuning >= 0) and (self.amp >= 0)
        )

        return params_loaded


class Rabi:
    def __init__(self):

        self.amp_ind = -1
        self.amp_global = -1
        self.Stark_shift = -1

    def set_params(self, amp_ind: float, amp_global: float, Stark_shift: float):

        self.amp_ind = amp_ind
        self.amp_global = amp_global
        self.Stark_shift = Stark_shift

    def check_params(self):
        params_loaded = (
            (self.amp_ind >= 0) and (self.amp_global >= 0) and (self.Stark_shift >= 0)
        )

        return params_loaded


class SK1:
    def __init__(self):

        self.amp_ind = -1
        self.amp_global = -1
        self.Tpi_multiplier = -1
        self.Stark_shift = -1

    def set_params(
        self,
        amp_ind: float,
        amp_global: float,
        Tpi_multiplier: float,
        Stark_shift: float,
    ):

        self.amp_ind = amp_ind
        self.amp_global = amp_global
        self.Tpi_multiplier = Tpi_multiplier
        self.Stark_shift = Stark_shift

    def check_params(self):
        params_loaded = (
            (self.amp_ind >= 0)
            and (self.amp_global >= 0)
            and (self.Tpi_multiplier >= 0)
            and (self.Stark_shift >= 0)
        )

        return params_loaded


class SK1_AM:
    def __init__(self):

        self.amp_ind = -1
        self.amp_global = -1
        self.theta = -1
        self.envelope_type = None
        self.envelope_scale = -1
        self.rotation_pulse_length = -1
        self.correction_pulse_1_length = -1
        self.correction_pulse_2_length = -1
        self.Stark_shift = -1
        self.phase_correction = -1

    def set_params(
        self,
        amp_ind: float,
        amp_global: float,
        theta: float,
        envelope_type: intfn.InterpFunction.FunctionType,
        envelope_scale: float,
        rotation_pulse_length: float,
        correction_pulse_1_length: float,
        correction_pulse_2_length: float,
        Stark_shift: float,
        phase_correction: float
    ):

        self.amp_ind = amp_ind
        self.amp_global = amp_global
        self.theta = theta
        self.envelope_type = envelope_type
        self.envelope_scale = envelope_scale
        self.rotation_pulse_length = rotation_pulse_length
        self.correction_pulse_1_length = correction_pulse_1_length
        self.correction_pulse_2_length = correction_pulse_2_length
        self.Stark_shift = Stark_shift
        self.phase_correction = phase_correction

    def check_params(self):
        # We don't require the envelope_scale parameter to be populated because
        # some envelope types don't require it
        params_loaded = (
            (self.amp_ind >= 0)
            and (self.amp_global >= 0)
            and (self.theta >= 0)
            and (self.envelope_type is not None)
            and (self.rotation_pulse_length >= 0)
            and (self.correction_pulse_1_length >= 0)
            and (self.correction_pulse_2_length >= 0)
            and (self.Stark_shift >= 0)
            and (self.phase_correction >= 0)
        )

        return params_loaded

class XX:
    def __init__(self):
        self.phase_offset = None

    def set_params(self, phase_offset: float):
        self.phase_offset = phase_offset

    def check_params(self):
        # We don't require the envelope_scale parameter to be populated because
        # some envelope types don't require it
        params_loaded = (self.phase_offset is not None)

        return params_loaded

class PhysicalParams:
    """This class stores the default physical parameters that are used for all gates.

    We first define gate-specific inner classes that contain parameters that are only
    applicable to a specific type of gate.  Each inner class can set its parameters and
    check that its parameters have been set.
    """

    def __init__(self):

        self.f_carrier = -1
        self.f_ind = -1
        self.N_ions = -1
        self.t_delay = -1
        self.Rabi_max = -1
        self.PI_center_freq_1Q = -1
        self.PI_center_freq_2Q = -1

        self.monitor = Monitor()
        self.Rabi = Rabi()
        self.SK1 = SK1()
        self.SK1_AM = SK1_AM()
        self.XX = XX()


    def set_params(
        self,
        f_carrier: float,
        f_ind: float,
        N_ions: float,
        t_delay: float,
        Rabi_max: float,
        PI_center_freq_1Q: float,
        PI_center_freq_2Q: float,
    ):

        self.f_carrier = f_carrier
        self.f_ind = f_ind
        self.N_ions = N_ions
        self.t_delay = t_delay
        self.Rabi_max = Rabi_max
        self.PI_center_freq_1Q = PI_center_freq_1Q
        self.PI_center_freq_2Q = PI_center_freq_2Q

    def check_params(self, gate_type: "gate.GateType"):
        import euriqabackend.devices.keysight_awg.common_types as common_types

        params_loaded = (
            (self.f_carrier >= 0)
            and (self.f_ind >= 0)
            and (self.N_ions >= 0)
            and (self.t_delay >= 0)
            and (self.Rabi_max >= 0)
            and (self.PI_center_freq_1Q >= 0)
            and (self.PI_center_freq_2Q >= 0)
        )

        if gate_type == common_types.GateType.Rabi:
            params_loaded = params_loaded and self.Rabi.check_params()

        if gate_type == common_types.GateType.SK1:
            params_loaded = params_loaded and self.SK1.check_params()

        if gate_type == common_types.GateType.XX:
            # HACK: hard-code this to pass, we can't really check that XX params are
            # loaded b/c they're all loaded/set in a different datastructure.
            params_loaded = params_loaded and True

        if not params_loaded:
            raise RuntimeError(
                "Not all physical parameters for the {} gate have been loaded".format(
                    gate_type.name
                )
            )
        else:
            return True
