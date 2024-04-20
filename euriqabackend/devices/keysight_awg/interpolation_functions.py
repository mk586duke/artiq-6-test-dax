import logging
import typing
from enum import IntEnum

import numpy as np
from scipy import optimize
from scipy import special

_LOGGER = logging.getLogger(__name__)


class InterpFunction:
    class FunctionType(IntEnum):
        constant = 0
        linear = 1
        half_cosine = 2
        full_cosine = 3
        half_Gaussian = 4
        full_Gaussian = 5
        ramp = 6

    def __init__(
        self,
        function_type: FunctionType = FunctionType.constant,
        start_value: float = 0,
        stop_value: float = 0,
        scale_params: typing.List[float] = None,
    ):
        """This function creates an interpolation function, which interpolates between
        the given start and stop values with the given functional form.  The start and
        stop values as originally given are retained, but we can rescale them. When the
        function is evaluated, it uses the scaled values.

        Args:
            function_type: The functional form of the interpolation function
                (e.g., linear, Gaussian)
            start_value: The initial value of the function
            stop_value: The final value of of the function, or the midpoint
                value in the case of a symmetric pulse
            scale_params: Width scaling parameters (e.g., width of a Gaussian)
                that are used by some functions
        """

        self.function_type = function_type
        self.unscaled_start_value = start_value
        self.scaled_start_value = start_value
        self.unscaled_stop_value = stop_value
        self.scaled_stop_value = stop_value
        self.scale_params = scale_params

    def rescale(self, scale_factor: float):
        """This function rescales the start and stop values of this InterpFunction
        instance.  The rescalings are not cumulative, which is to say a rescaling erases
        all previous rescalings.

        Args:
            scale_factor: The scale factor by which the start and stop values are
                both multiplied
        """

        self.scaled_start_value = scale_factor * self.unscaled_start_value
        self.scaled_stop_value = scale_factor * self.unscaled_stop_value

    def abs_min(self):
        """Returns the minimum absolute value expected once the InterpFunction is
        evaluated."""

        return min(abs(self.scaled_start_value), abs(self.scaled_stop_value))

    def abs_max(self):
        """Returns the maximum absolute value expected once the InterpFunction is
        evaluated."""

        return max(abs(self.scaled_start_value), abs(self.scaled_stop_value))

    def evaluate(self, N_points):
        """This function evaluates the InterpFunction over the specified number of
        points.  The range of x values runs from 0 to 1, inclusive.

        Args:
            N_points: The number of points at which the function is evaluated
        """

        amps = None

        if self.function_type == InterpFunction.FunctionType.constant:
            amps = np.array([self.scaled_start_value] * N_points)
        elif self.function_type == InterpFunction.FunctionType.linear:
            amps = np.linspace(
                self.scaled_start_value,
                self.scaled_stop_value,
                num=N_points,
                endpoint=True,
            )
        elif self.function_type == InterpFunction.FunctionType.half_cosine:
            a = (self.scaled_stop_value - self.scaled_start_value) / 2
            b = self.scaled_start_value
            amps = (
                a * (1 - np.cos(np.linspace(0, np.pi, num=N_points, endpoint=True))) + b
            )
        elif (
            self.function_type == InterpFunction.FunctionType.half_cosine
            or self.function_type == InterpFunction.FunctionType.full_cosine
        ):
            a = (self.scaled_stop_value - self.scaled_start_value) / 2
            b = self.scaled_start_value
            limits = (
                [0, np.pi]
                if self.function_type == InterpFunction.FunctionType.half_cosine
                else [0, 2 * np.pi]
            )
            amps = (
                a
                * (
                    1
                    - np.cos(
                        np.linspace(limits[0], limits[1], num=N_points, endpoint=True)
                    )
                )
                + b
            )
        elif (
            self.function_type == InterpFunction.FunctionType.half_Gaussian
            or self.function_type == InterpFunction.FunctionType.full_Gaussian
        ):
            N_widths = self.scale_params[0]
            o = np.exp(-N_widths ** 2 / 2)
            a = (self.scaled_stop_value - self.scaled_start_value) / (1 - o)
            b = self.scaled_start_value
            limits = (
                [-N_widths, 0]
                if self.function_type == InterpFunction.FunctionType.half_Gaussian
                else [-N_widths, N_widths]
            )
            amps = (
                a
                * (
                    np.exp(
                        -np.linspace(limits[0], limits[1], num=N_points, endpoint=True)
                        ** 2
                        / 2
                    )
                    - o
                )
                + b
            )
        elif self.function_type == InterpFunction.FunctionType.ramp:
            # first scale 0-1 to -2pi to 2 pi
            middle = (self.scaled_stop_value-self.scaled_start_value)/2
            x_values = np.linspace(-np.pi,np.pi,num=N_points,endpoint=True)
            amps = self.scaled_start_value+middle+middle*np.tanh(
                (x_values)
            )
        return amps

    def Tpi_derating(self):
        """This function calculates the derating factor we must apply to the measured
        Tpi value according to the specific envelope of the shaped pulse that was used
        to perform the calibration measurement.  Essentially, a shaped pulse of a given
        total duration and maximum amplitude has less integrated area than a square
        pulse of the same duration and amplitude, so the Rabi oscillations we measure
        while sweeping the duration of the shaped pulse are slower than those we would
        observe performing the same calibration with a square pulse.  We therefore
        derate the measured Tpi in order to find the Tpi corresponding to the Rabi
        frequency at the pulse's maximum amplitude."""

        if self.function_type == InterpFunction.FunctionType.full_Gaussian:
            # See details of this calculation in
            # Notebook/Notes/RF compilation/Calculating area of shaped SK1 pulse
            return (
                np.sqrt(np.pi / 2)
                * special.erf(self.scale_params[0] / np.sqrt(2))
                / self.scale_params[0]
                - np.exp(-self.scale_params[0] ** 2 / 2)
            ) / (1 - np.exp(-self.scale_params[0] ** 2 / 2))
        else:
            _LOGGER.error(
                "ERROR: Tpi derating not calculated for envelope type %s",
                self.function_type.name,
            )
            raise Exception(
                "ERROR: Tpi derating not calculated for envelope type "
                + self.function_type.name
            )

    def calculate_segment_area(self, x_start: float, x_end: float):
        """This function calculates the area of a segment between the two given times,
        expressed as a fraction of the area of the entire shaped pulse.  This function
        is unitless; the variable x runs from 0 at the start of the pulse to 1 at the
        end of the pulse.

        Args:
            x_start: The start time of the pulse segment
            x_end: The end time of the pulse segment

        Returns:
            The desired fractional area of the pulse segment
        """

        x1 = x_start
        x2 = x_end
        Nw = self.scale_params[0]

        if self.function_type == InterpFunction.FunctionType.full_Gaussian:
            # See details of this calculation in
            # Notebook/Notes/RF compilation/Calculating area of shaped SK1 pulse
            return (
                np.sqrt(np.pi / 2.0)
                * (
                    special.erf(Nw * (2 * x2 - 1) / np.sqrt(2))
                    - special.erf(Nw * (2 * x1 - 1) / np.sqrt(2))
                )
                - (2 * Nw * (x2 - x1)) * np.exp(-Nw ** 2 / 2.0)
            ) / (
                np.sqrt(2 * np.pi) * special.erf(Nw / np.sqrt(2))
                - 2 * Nw * np.exp(-Nw ** 2 / 2.0)
            )
        else:
            _LOGGER.error(
                "ERROR: Segment area not calculated for envelope type %s",
                self.function_type.name,
            )
            raise Exception(
                "ERROR: Segment not calculated for envelope type "
                + self.function_type.name
            )

    def find_segment_end_time(self, fractional_area: float, x_start: float):
        """This function finds the end time of a segment, starting at the given time,
        within the shaped pulse that encloses the given fraction of the total pulse
        area.  This function is unitless; the variable x runs from 0 at the start of the
        pulse to 1 at the end of the pulse.

        Args:
            fractional_area: The desired fractional area of the pulse segment
            x_start: The start time of the pulse segment

        Returns:
            x_stop, the end time of the pulse segment
        """

        self_intfn = self

        def f(x):
            return (
                self_intfn.calculate_segment_area(x_start, x_start + x)
                - fractional_area
            )

        try:
            # the optimize.brentq function finds a root of the function f(x) in the window 0 to 1
            x_stop = optimize.brentq(f, 0, 1)
        except ValueError:
            _LOGGER.error(
                "Cannot find end time of shaped pulse segment "
                "- too much fractional area requested"
            )
            raise ValueError(
                "Cannot find end time of shaped pulse segment "
                "- too much fractional area requested"
            )

        return x_stop

    def calculate_SK1_segment_lengths(self, theta: float, Tpi: float):
        """This function calculates the lengths of the three segments that comprise a
        shaped SK1 pulse with a desired rotation angle theta.

        Args:
            theta: The rotation angle of the Tpi pulse
            Tpi: The Tpi value corresponding to the maximum pulse amplitude
        """

        # First, calculate the total duration of the shaped pulse
        t_tot = (4 * np.pi + theta) * (Tpi / self.Tpi_derating()) / np.pi

        dx1 = self.find_segment_end_time(
            fractional_area=theta / (4 * np.pi + theta), x_start=0
        )
        dx2 = self.find_segment_end_time(
            fractional_area=2 * np.pi / (4 * np.pi + theta), x_start=dx1
        )
        dx3 = 1.0 - dx1 - dx2
        return [dx * t_tot for dx in [dx1, dx2, dx3]]


class Constant(InterpFunction):
    def __init__(self, value: float):

        super().__init__(
            function_type=InterpFunction.FunctionType.constant, start_value=value
        )


class Linear(InterpFunction):
    def __init__(self, start_value: float, stop_value: float):

        super().__init__(
            function_type=InterpFunction.FunctionType.linear,
            start_value=start_value,
            stop_value=stop_value,
        )


class Cosine(InterpFunction):
    def __init__(self, start_value: float, stop_value: float, half_pulse: bool = False):

        fn_type = (
            InterpFunction.FunctionType.half_cosine
            if half_pulse
            else InterpFunction.FunctionType.full_cosine
        )

        super().__init__(
            function_type=fn_type, start_value=start_value, stop_value=stop_value
        )


class Gaussian(InterpFunction):
    def __init__(
        self,
        start_value: float,
        stop_value: float,
        N_widths: float,
        half_pulse: bool = False,
    ):

        fn_type = (
            InterpFunction.FunctionType.half_Gaussian
            if half_pulse
            else InterpFunction.FunctionType.full_Gaussian
        )

        super().__init__(
            function_type=fn_type,
            start_value=start_value,
            stop_value=stop_value,
            scale_params=[N_widths],
        )

class Special(InterpFunction):
    def __init__(
            self,
            start_value: float,
            stop_value: float,
            tail_value: float
    ):
        fn_type = (
            InterpFunction.FunctionType.special
            if half_pulse
            else InterpFunction.FunctionType.special
        )

        super().__init__(
            function_type=fn_type,
            start_value=start_value,
            stop_value=stop_value,
        )


# testintfn = InterpFunction(InterpFunction.FunctionType.full_Gaussian,
#                            0,
#                            1,
#                            [1])
#
# print(testintfn.calculate_segment_area(0.1, 0.35))
#
# print(testintfn.find_segment_end_time(0.7, 0.2))
