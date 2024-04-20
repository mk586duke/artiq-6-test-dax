import logging
import pickle
import typing

import numpy as np

from .interpolation_functions import InterpFunction


_LOGGER = logging.getLogger(__name__)

_MAX_AMPLITUDE = 1000


def Rabis_to_amplitudes(Rabi_array: typing.List[float], Rabi_max: float):
    """This function rescales an array of Rabi frequencies to an array of individual
    channel amplitudes.  We include this simple function here to gain access to the
    private _MAX_AMPLITUDE constant.

    Args:
        Rabi_array:
        Rabi_max:
    """
    return [r / Rabi_max * _MAX_AMPLITUDE for r in Rabi_array]


def check_amplitudes(amplitude_array: typing.List[float]):
    """This function rescales an array of Rabi frequencies to an array of individual
    channel amplitudes.  We include this simple function here to gain access to the
    private _MAX_AMPLITUDE constant.

    Args:
        amplitude_array:
    """
    return all([abs(a) <= _MAX_AMPLITUDE for a in amplitude_array])


class Calibration:

    _SATURATION_FILE = r"C:\RF System\AOM saturation values.dat"
    _LEVELS_FILE = r"C:\RF System\AOM levels.dat"

    def __init__(self):

        self.level_calibration_active = True
        self.linearity_calibration_active = True
        self.A_array = []
        self.xsat_array = []

    def disable_calibration(
        self, level_active: bool = False, linearity_active: bool = False
    ):
        """This function selectively disables either the linearity calibration, the
        level calibration, or both.

        Args:
            level_active: Determines whether the level calibration is active
            linearity_active: Determines whether the linearity calibration is active
        """
        self.level_calibration_active = level_active
        self.linearity_calibration_active = linearity_active

    def write_saturation_params(self, xsat_array: typing.List[float]):
        """This function writes the saturation values, which are used below in
        calibrate(), to file.

        Args:
            xsat_array:
        """

        self.xsat_array = xsat_array

        # If calibration was successful, then we write the new level values to
        # file and re-enable calibration
        self.linearity_calibration_active = True
        with open(self._SATURATION_FILE, "wb") as sat_file:
            pickle.dump(xsat_array, sat_file)

    def write_levels(
        self,
        nominal_Tpi: float,
        calibrated_Tpi_array: typing.List[float],
        used_shaped: bool = False,
        envelope_type: InterpFunction.FunctionType = InterpFunction.FunctionType.constant,
        envelope_scale: float = 0,
    ):
        """This function writes the AOM levels, used in :meth:`calibrate`, to file.

        Args:
            nominal_Tpi: The nominal Tpi, which sets the Rabi frequency we
                calibrate the channels to produce
            calibrated_Tpi_array: The measured Tpi values
            used_shaped: Specifies whether shaped or square Rabi pulses were
                used to perform the calibration whose results are being passed in
            envelope_type: The functional form of the interpolation function
                used for calibration
            envelope_scale: Width scaling parameter that was used by the
                envelope function during calibration
        """

        if not nominal_Tpi > 0:
            _LOGGER.error(
                "Trying to calibrate the individual levels without having first set the"
                "nominal Tpi value"
            )
            raise Exception(
                "ERROR: We are trying to calibrate the individual levels without "
                "having first set the nominal Tpi value"
            )

        # If we used shaped pulses to perform the calibration, we derate the measured
        # Tpi values appropriately. We assume that the current shaped SK1 global
        # parameters currently have the same values that were used to perform
        # the calibration.
        if used_shaped:
            interpfn = InterpFunction(
                function_type=envelope_type,
                start_value=0,
                stop_value=0,
                scale_params=[envelope_scale],
            )
            calibrated_Tpi_array_to_use = [
                interpfn.Tpi_derating() * c for c in calibrated_Tpi_array
            ]
        else:
            calibrated_Tpi_array_to_use = calibrated_Tpi_array

        # We define the array of A parameters as the measured Tpi over the nominal Tpi.
        # Since we're throttling the power levels down to accommodate the weakest
        # individual channel plus some headroom, this ratio should always
        # be less than 1 (i.e. all channels can drive faster than our nominal max).
        # We set these amplitude scaling parameters to 1 for all individual
        # channels that have not been calibrated and for the global.
        self.A_array = [1.0] + [c / nominal_Tpi for c in calibrated_Tpi_array_to_use]

        if max(self.A_array) > 1:
            _LOGGER.error(
                "Calibration forcing following channels to run >= full amplitude: %s",
                np.where(np.array(self.A_array) > 1),
            )
            raise RuntimeError(
                "Calibration is forcing >= 1 individual channels to full amplitude, %f", max(self.A_array)
            )

        # If calibration was successful, then we write the new level values to
        # file and re-enable calibration
        self.level_calibration_active = True
        with open(self._LEVELS_FILE, "wb") as lev_file:
            pickle.dump(self.A_array, lev_file)

        return self.A_array

    def load_params(self):
        """This function reads the saturation params, which are used below in
        calibrate(), from file."""

        with open(self._LEVELS_FILE, "rb") as lev_file:
            self.A_array = pickle.load(lev_file)

        with open(self._SATURATION_FILE, "rb") as sat_file:
            self.xsat_array = pickle.load(sat_file)

    def calibrate(self, unscaled_amplitude: float, slot: int):
        """This function calibrates the amplitude of an AWG segment so that all AOM
        channels are linear and have the same maximum Rabi frequency.  See Notes/RF
        Compilation/AOM Calibration for details.

        Args:
            unscaled_amplitude:
            slot:
        """
        x_max = 1
        x_unscaled = min(abs(unscaled_amplitude / _MAX_AMPLITUDE), x_max) * np.sign(
            unscaled_amplitude
        )

        # We don't want to calibrate slot -1, which is used for the reference
        # pulse or other non-AOM waveforms
        if slot < 0:
            return x_unscaled

        A = self.A_array[slot]
        x_sat = self.xsat_array[slot]

        # If we want neither calibration active, simply return the unscaled x
        if (not self.level_calibration_active) and (
            not self.linearity_calibration_active
        ):
            return x_unscaled

        # If we want only the level calibration, we scale x from 0-1 to 0-A,
        # where A <= 1 is set by the measured Tpis
        elif self.level_calibration_active and (not self.linearity_calibration_active):
            return A * x_unscaled

        # If we want only the linearity calibration, we set A = 1 but otherwise
        # carry out the arcsin calculation
        elif (not self.level_calibration_active) and self.linearity_calibration_active:
            x_scaled = (
                x_sat
                * np.arcsin(abs(x_unscaled) * np.sin(x_max / x_sat))
                * np.sign(x_unscaled)
            )
            return x_scaled

        # If we want both the level and linearity calibrations, we carry out
        # the standard arcsin calculation
        else:
            x_scaled = (
                x_sat
                * np.arcsin(A * abs(x_unscaled) * np.sin(x_max / x_sat))
                * np.sign(x_unscaled)
            )
            return x_scaled

    def calibrate_array(self, unscaled_amplitudes: typing.List[float], slot: int):
        """This function calibrates the amplitude of an AWG segment so that all AOM
        channels are linear and have the same maximum Rabi frequency.  See Notes/RF
        Compilation/AOM Calibration for details.

        Args:
            unscaled_amplitudes:
            slot:
        """

        x_max = 1
        A = self.A_array[slot]
        x_sat = self.xsat_array[slot]
        # We don't want to calibrate slot -1, which is used for the reference
        # pulse or other non-AOM waveforms
        # In this case, or if we want neither calibration active,
        # simply return the unscaled x
        if slot < 0 or (
            not(self.level_calibration_active) and not(self.linearity_calibration_active)
        ):
            x_unscaled = [
                min(abs(ua / _MAX_AMPLITUDE), x_max) for ua in unscaled_amplitudes
            ]
            return x_unscaled

        # If we want only the level calibration, we scale x from 0-1 to 0-A,
        # where A <= 1 is set by the measured Tpis
        elif self.level_calibration_active and (not self.linearity_calibration_active):
            x_lin_scaled = [
                min(abs(A * ua / _MAX_AMPLITUDE), x_max) for ua in unscaled_amplitudes
            ]
            return x_lin_scaled

        # If we want only the linearity calibration,
        # we set A = 1 but otherwise carry out the arcsin calculation
        elif (not self.level_calibration_active) and self.linearity_calibration_active:
            coeff = np.sin(x_max / x_sat) / _MAX_AMPLITUDE
            x_scaled = x_sat * np.arcsin(unscaled_amplitudes * coeff)
            return x_scaled

        # If we want both the level and linearity calibrations,
        # we carry out the standard arcsin calculation
        else:
            # t1 = time.time()
            coeff = A * np.sin(x_max / x_sat) / _MAX_AMPLITUDE
            # t2 = time.time()
            x_scaled = x_sat * np.arcsin(unscaled_amplitudes * coeff)
            # t3 = time.time()
            # print("{0} {1}".format(t2-t1, t3-t2))
            return x_scaled
