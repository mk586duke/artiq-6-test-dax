import logging
import typing

import artiq.language.environment as artiq_env
import numdifftools
import numpy as np
import scipy.stats as sp_stats
import uncertainties
from scipy.optimize import curve_fit, minimize

_LOGGER = logging.getLogger(__name__)


# define fitting functions
def func_gaussian(x, a, b, c):
    return a * np.exp(-(((x - b) / c) ** 2))


def func_linear(x, a, b):
    return a * x + b


def func_sine(x, amp, freq, phase):
    return amp * np.sin(freq * x + phase)


def func_XXparity(x, amp, offset, phase):
    # assumes scanning analysis pulse phase from 0 -> pi
    return amp * np.sin(2*x + phase) + offset


class PopulationAnalysis(artiq_env.HasEnvironment):
    def build(self):
        # build the GUI
        self.target_states_input = self.get_argument(
            "Specify the target state\n"
            "(input a binary strings of #active_pmt digits, separated by ',')",
            artiq_env.StringValue(default="000000"),
            group="Population Analysis",
        )

        self.setattr_argument(
            "autofill_target_states",
            artiq_env.BooleanValue(default=True),
            group="Population Analysis",
        )

        self.population_analysis = self.get_argument(
            "Population Analysis\n"
            "(options: crosstalk, target_subspace, individual_flip, "
            "target_subspace_oscillation,average_transfer)",
            artiq_env.StringValue(
                default="target_subspace, crosstalk, individual_flip, "
                "target_subspace_oscillation, average_transfer"
            ),
            group="Population Analysis",
        )

        self.setattr_argument(
            "show_parity",
            artiq_env.BooleanValue(default=True),
            group="Population Analysis",
        )

        self.setattr_argument(
            "show_population_analysis",
            artiq_env.BooleanValue(default=True),
            group="Population Analysis",
        )

        self.setattr_device("ccb")
        _LOGGER.debug("Done building Population Analyzer")

    def population_analysis_init(self):

        temp0 = ["0"] * len(self.active_pmts)
        temp1 = ["0"] * len(self.active_pmts)
        for i in self.XX_slots_pmt_index:
            temp1[i] = "1"

        if self.autofill_target_states:
            self.target_states_input = "".join(temp0) + "," + "".join(temp1)

        # print(self.target_states_input)

        self.target_states = self.target_states_input.split(",")

        for sequence in self.population_analysis:
            if len(self.target_states[0]) != len(self.active_pmts):
                _LOGGER.error("target states must be specified for all active pmts")

        activate_population_plots = list()

        if self.population_analysis.find("crosstalk") > -1:
            self.set_dataset(
                "data.population.crosstalk",
                np.full(self.num_steps, np.nan),
                broadcast=True,
            )
            activate_population_plots.append("data.population.crosstalk")

        if self.population_analysis.find("target_subspace") > -1:
            self.set_dataset(
                "data.population.target_subspace",
                np.full(self.num_steps, np.nan),
                broadcast=True,
            )
            activate_population_plots.append("data.population.target_subspace")

        if self.population_analysis.find("individual_flip") > -1:
            self.set_dataset(
                "data.population.individual_flip",
                np.full(self.num_steps, np.nan),
                broadcast=True,
            )
            activate_population_plots.append("data.population.individual_flip")

        if self.population_analysis.find("average_transfer") > -1:
            self.set_dataset(
                "data.population.average_transfer",
                np.full(self.num_steps, np.nan),
                broadcast=True,
            )
            activate_population_plots.append("data.population.average_transfer")

        if self.population_analysis.find("target_subspace_oscillation") > -1:
            self.set_dataset(
                "data.population.target_subspace_oscillation",
                np.full(self.num_steps, np.nan),
                broadcast=True,
            )
            activate_population_plots.append(
                "data.population.target_subspace_oscillation"
            )

        self.ccb.issue(
            "create_applet",
            name="Population Analysis",
            command="$python -m euriqafrontend.applets.plot_multi "
            "--x data.{}.x_values "
            "--y-names {} "
            "--y-label 'population' "
            "--x-label 'step index'".format(
                self.data_folder, " ".join(activate_population_plots)
            ),
            group="population_analysis",
        )

    def parity_analysis_init(self):
        self.set_dataset(
            "data.population.parity", np.full(self.num_steps, np.nan), broadcast=True
        )
        self.set_dataset(
            "data.population.parity_errorbars_bottom",
            np.full(self.num_steps, np.nan),
            broadcast=True,
        )
        self.set_dataset(
            "data.population.parity_errorbars_top",
            np.full(self.num_steps, np.nan),
            broadcast=True,
        )
        self.set_dataset("data.population.x_fit", np.nan, broadcast=True)
        self.set_dataset(
            "data.population.parity_y_fit",
            np.full(self.num_steps, np.nan),
            broadcast=True,
        )

        self.ccb.issue(
            "create_applet",
            name="Parity",
            command="$python -m euriqafrontend.applets.plot_multi "
            "--x data.{data_folder}.x_values "
            "--y-names data.population.parity "
            "--rid data.{data_folder}.rid "
            "--x-fit data.population.x_fit "
            "--y-fits data.population.parity_y_fit "
            "--y-label 'parity' "
            "--x-label 'phase (radians / pi)' "
            "--error-bars-bottom data.population.parity_errorbars_bottom "
            "--error-bars-top data.population.parity_errorbars_top".format(
                data_folder=self.data_folder
            ),
            group="population_analysis",
        )

    def module_init(self, data_folder, active_pmts, num_steps, detect_thresh, XX_slots):
        # Data_set initialization

        self.data_folder = data_folder
        self.active_pmts = active_pmts
        self.num_steps = num_steps
        self.detect_thresh = detect_thresh
        self.XX_slots = XX_slots
        self.XX_slots_pmt_index = [self.active_pmts.index(x - 9) for x in self.XX_slots]

        if self.show_population_analysis:
            self.population_analysis_init()

        if self.show_parity:
            self.parity_analysis_init()

        _LOGGER.debug("Done initialize Population Analyzer")

    def prepare(self):
        pass

    def population_analysis_update(self, istep):
        # update the plot every experiment steps

        counts = np.array(self.get_dataset("data." + self.data_folder + ".raw_counts"))

        population_other = 1

        if self.population_analysis.find("target_subspace") > -1:
            mean, std = self.Statistics(self.Target_Subspace_State, counts[:, :, istep])
            self.mutate_dataset("data.population.target_subspace", (istep), mean)
            population_other -= mean

        if self.population_analysis.find("individual_flip") > -1:
            mean, std = self.Statistics(self.Individual_Flip_State, counts[:, :, istep])
            self.mutate_dataset("data.population.individual_flip", (istep), mean)
            population_other -= mean

        if self.population_analysis.find("crosstalk") > -1:
            mean, std = self.Statistics(self.Crosstalk_State, counts[:, :, istep])
            self.mutate_dataset("data.population.crosstalk", (istep), mean)
            population_other -= mean

        if self.population_analysis.find("average_transfer") > -1:
            mean, std = self.Statistics(self.Average_Transfer, counts[:, :, istep])
            self.mutate_dataset("data.population.average_transfer", (istep), mean)
            population_other -= mean

        if self.population_analysis.find("target_subspace_oscillation") > -1:
            try:
                denom = self.Statistics(self.Target_Subspace_State, counts[:, :, istep])[0]
                if denom!=0:
                    mean = (
                        self.Statistics(self.Target_State1, counts[:, :, istep])[0] / denom
                    )
                else:
                    mean = 0
            except ZeroDivisionError:
                import pdb

                _LOGGER.error(
                    "Zero division error. Check cmd-line window for artiq_master for pdb debug"
                )
                pdb.set_trace()
            self.mutate_dataset(
                "data.population.target_subspace_oscillation", (istep), mean
            )
            population_other -= mean

        if self.population_analysis.find("others") > -1:
            self.mutate_dataset("data.population.others", (istep), population_other)

    def parity_analysis_update(self, istep):
        # NOTE: import can't be top-level import b/c it imports pandas, and pandas
        # import fails in ARTIQ b/c too many recursion levels.
        import statsmodels.stats.proportion as stats_prop

        counts = np.array(self.get_dataset("data." + self.data_folder + ".raw_counts"))
        num_shots = counts.shape[1]

        mean, _ = self.Statistics(self.Parity, counts[:, :, istep])
        self.mutate_dataset("data.population.parity", (istep), mean)
        norm_parity = mean / 2 + 0.5  # normalize to range [0, 1]
        parity_errorbars_abs = stats_prop.proportion_confint(
            norm_parity * num_shots, num_shots, alpha=0.05, method="wilson"
        )
        # calculate length of error bars below/above the parity value
        parity_errorbars = np.abs(parity_errorbars_abs - norm_parity) * 2
        self.mutate_dataset(
            "data.population.parity_errorbars_bottom", (istep), parity_errorbars[0]
        )
        self.mutate_dataset(
            "data.population.parity_errorbars_top", (istep), parity_errorbars[1]
        )

    def module_update(self, istep):
        """update the module after every experiment step"""

        if self.show_population_analysis:
            self.population_analysis_update(istep)

        if self.show_parity:
            self.parity_analysis_update(istep)

    def function_fit(self, function, data_name):
        x_values = np.array(self.get_dataset("data." + self.data_folder + ".x_values"))
        y_values = np.array(self.get_dataset("data.population." + data_name))

        p_fit, cov_fit = curve_fit(function, x_values, y_values)

        print("fitting result: {}\nfit covariance: {}".format(p_fit, cov_fit))

        if data_name == "parity":
            step_size = np.divide(x_values[-1] - x_values[0], len(x_values) * 10)
            x_values_fit = np.arange(x_values[0], x_values[-1], step_size)
            y_values_fit = [[function(x, *p_fit) for x in x_values_fit]]

            self.set_dataset("data.population.x_fit", x_values_fit, broadcast=True)
            self.set_dataset(
                "data.population.parity_y_fit", y_values_fit, broadcast=True
            )
        return p_fit

    def linear_fit(self, data_name):
        fit_param = self.function_fit(func_linear, data_name)
        return fit_param

    def gaussian_fit(self, data_name):
        fit_param = self.function_fit(func_gaussian, data_name)
        return fit_param

    def sine_fit(self, data_name):
        fit_param = self.function_fit(func_sine, data_name)
        return fit_param

    def XXparity_fit(self, data_name):
        fit_param = self.function_fit(func_XXparity, data_name)
        return fit_param

    def Digitize(self, state):
        return np.asarray([int(1) if x > self.detect_thresh else int(0) for x in state])

    @staticmethod
    def threshold(pmt_data: np.ndarray, threshold: int) -> np.ndarray:
        """
        Threshold PMT data to determine quantum state, return integers.

        Args:
            pmt_data (np.ndarray): Numpy array of PMT counts.
            threshold (int): Threshold that denotes line demarkating between measuring
                in the \ket{0} or \ket{1} state.
                The \ket{1} state is defined as > threshold.

        Returns:
            np.ndarray: Array of same size as input,
            where values are exclusively 0 or 1.
        """
        return np.where(pmt_data > threshold, 1, 0)

    @staticmethod
    def threshold_bool(pmt_data: np.ndarray, threshold: int) -> np.ndarray:
        """
        Threshold PMT data to determine quantum state, return booleans.

        Args:
            pmt_data (np.ndarray): Numpy array of PMT counts
            threshold (int): Threshold that denotes line demarkating between measuring
                in the \ket{0} or \ket{1} state.
                The \ket{1} state is defined as > threshold.

        Returns:
            np.ndarray: Array of same size as input,
            where values are exclusively True or False.
        """
        return np.where(pmt_data > threshold, True, False)

    @staticmethod
    def calculate_parity(
        pmt_counts: np.ndarray, threshold: int, entangled_indices: typing.Tuple[int]
    ) -> np.ndarray:
        """Calculate parity for each shot of the experiment and return separately.

        Calculating the average parity can then be done with
        ``np.mean(result, axis=0)``.

        Args:
            pmt_counts (np.ndarray): Numpy array of raw PMT counts.
            threshold (int): Value to threshold the PMT counts at.
                Passed through to :meth:`threshold`.
            entangled_indices (typing.Tuple[int]): Tuple of indices in pmt_counts
                that are entangled. That is, if the ions on PMTs 1 & 2 are entangled,
                then this value should be (0, 1). These indices will be used to take
                data from the ``pmt_counts`` variable, and incorrect values could cause
                no significant parity to show up.

        Returns:
            np.ndarray: array of per-shot parity values (i.e. ``(+1, -1)``).
            This will be rather large still if you have a lot of scan points.
            The rough array format format is [scan_point_index, parity_of_single_shot].
            Per-scan-point parity can be calculated with ``np.mean(result, axis=0)``.
        """
        thresholded_counts = PopulationAnalysis.threshold(pmt_counts, threshold)
        parity_per_shot = np.prod(
            thresholded_counts[entangled_indices, :] * -2 + 1, axis=0
        )
        return parity_per_shot

    def Statistics(self, Function, raw_data):
        counter = 0
        counter_square = 0
        num_shots = raw_data.shape[1]
        for i in range(num_shots):
            temp = Function(raw_data[:, i])
            counter += temp
            counter_square += temp * temp
        return (
            counter / num_shots,
            np.sqrt(counter_square / num_shots - np.square(counter / num_shots))
            / np.sqrt(num_shots),
        )

    def Parity(self, state):
        state = self.Digitize(state)
        counter = 1
        for i in self.XX_slots_pmt_index:
            counter = counter * (state[i] - 0.5) * 2

        return counter

    def Crosstalk_State(self, state):
        counter = 0
        state = self.Digitize(state)
        for target in self.target_states:
            b = [int(x) for x in target]
            for i in range(len(b)):
                if not (i in self.XX_slots_pmt_index):
                    for j in self.XX_slots_pmt_index:
                        temp = np.copy(b)
                        temp[i] = 1 - temp[i]
                        temp[j] = 1 - temp[j]
                        counter += int(np.array_equal(temp, state))
        return int(counter >= 1)

    def Average_Transfer(self, state):
        counter = 0
        state = self.Digitize(state)
        counter = state[self.XX_slots_pmt_index[0]] + state[self.XX_slots_pmt_index[1]]
        return counter / 2

    def Target_Subspace_State(self, state):
        counter = 0
        state = self.Digitize(state)
        for target in self.target_states:
            b = [int(x) for x in target]
            counter += int(np.array_equal(b, state))
        return int(counter >= 1)

    def Target_State1(self, state):
        counter = 0
        state = self.Digitize(state)
        b = [int(x) for x in self.target_states[1]]
        counter += int(np.array_equal(b, state))
        return int(counter >= 1)

    def Individual_Flip_State(self, state):
        counter = 0
        state = self.Digitize(state)
        for target in self.target_states:
            b = [int(x) for x in target]
            for j in range(len(b)):
                temp = np.copy(b)
                temp[j] = 1 - temp[j]
                counter += int(np.array_equal(temp, state))
        return int(counter >= 1)

    @staticmethod
    def calculate_parity_contrast(
        num_entangled_ions: int,
        num_shots_per_phase: int,
        phase_values: typing.Sequence[float],
        parity_values: typing.Sequence[float],
    ) -> typing.Tuple[uncertainties.UFloat, uncertainties.UFloat, uncertainties.UFloat]:
        """Calculate the parity contrast.

        Uses Norbert's SPAM Screed method, effectively minimizing a Binomial
        distribution to the average parity measurement given a certain number of trials.

        Returns the parity contrast and its ABSOLUTE, SYMMETRIC uncertainty.
        I would only trust this method to ~2.5 9's (i.e. ~5e-3 error). Seems
        to misread test values from perfect sine wave by about that much when at
        the extreme upper bound (i.e. amp = 0.999). It seems to be more reliable
        with parity ~= 99%.

        Args:
            num_entangled_ions (int): number of ions entangled together.
                This determines the oscillation frequency of the parity curve.
            num_shots_per_phase (int): number of trials at each x-value (phase value).
            phase_values (typing.Sequence[float]): The nominal phase values at which
                the parity was measured (i.e. the x-values).
                Expects these values to be ordered, monotonically increasing. Units of radians.
            parity_values (typing.Sequence[float]): The measured average parity after
                a number of shots.
                Calculated using method like :meth:`calculate_parity`.
                Values should be in range [-1, 1].

        Returns:
            Tuple[uncertainties.UFloat, uncertainties.UFloat, uncertainties.UFloat]:
            the (parity contrast, parity phase, parity y_offset) fitted.
            Uncertainty is symmetric & 1-sigma standard deviation,
            i.e. contrast = retval[0].n +- retval[0].s (67% conf. interval)

        Examples (realistic-ish case, then best-case):
            >>> import numpy as np
            >>> x_vals = np.linspace(0, 2 * np.pi)
            >>> PopulationAnalysis.calculate_parity_contrast(
            ...     2, 500, x_vals, -0.921 * np.sin(2 * x_vals + .2))   # doctest: +ELLIPSIS
            (-0.9208...+/-0.0047..., 0.2003...+/-0.008..., -0.0001...+/-0.003...)
            >>> PopulationAnalysis.calculate_parity_contrast(
            ...     2, 200, x_vals, -0.9985 * np.sin(2 * x_vals + .001) + 0.001)   # doctest: +ELLIPSIS
            (-0.9989...+/-0.001..., -0.00025...+/-0.01..., 0.0008...+/-0.001...)
        """
        # shift parity from [-1, 1] to [0, 1]
        _LOGGER.debug(
            "Parity fit input values: x=%s,\ny = %s", phase_values, parity_values
        )
        machine_eps = np.finfo(float).eps
        normalized_parity = parity_values * 0.5 + 0.5
        # Number of hits (successes) needed to produce that parity with given num_shots
        num_hits_k = np.around(normalized_parity * num_shots_per_phase)

        def _ideal_parity_sine_probability(
            i: int, amp: float, phase_offset: float, y_offset: float
        ):
            """Calculates the ideal parity curve as a sine wave in [0, 1].

            Needed for mapping to probability for Binomial distribution: p in [0, 1].
            y_offset should be in range [0, 1].

            To prevent weird results in logpmf function, forces result to be in
            [0 + eps, 1 - eps]. That is, the logpmf seems to struggle with p==0 or p==1.
            """
            # Normalized so sine curve in [0, 1]
            norm_amp, norm_offset = amp / 2, y_offset / 2
            retval = norm_amp * np.sin(
                num_entangled_ions * phase_values[i] + phase_offset
            ) + (norm_offset + 0.5)
            return np.clip(retval, 0.0 + machine_eps, 1.0 - machine_eps)

        AmpPhaseOffset = typing.Tuple[float, float, float]

        def _negLogLikelihoodBinom(x: AmpPhaseOffset):
            """Calculate the negative log-likelihood of the parity binomial.

            Effectively -sum(log[binomial_distribution(parity_trial)]).
            From Norbert Linke's SPAM Screed eqn. 6.
            """
            amplitude, phase, y_offset = x
            retval = -np.sum(
                [
                    sp_stats.binom.logpmf(
                        k,
                        n=num_shots_per_phase,
                        p=_ideal_parity_sine_probability(i, amplitude, phase, y_offset),
                        # NOTE: for proper results, p != {0.0, 1.0}. Returns +/- inf.
                    )
                    for i, k in enumerate(num_hits_k)
                ]
            )
            # _LOGGER.debug("loglikeliresult(%s)=%f", x, retval)
            return retval

        # Initialize Parameters for minimization. Guess amp = -1, phase = 0, offset = 0
        # constrain -1 <= amp <= 0, -pi <= phase <= pi,
        # -1 <= offset <= 1
        init_guess = [-0.99, 0.0, 0.0]
        param_bounds = [
            (-1.0 + machine_eps, 0.0 - machine_eps),
            (-np.pi, np.pi),
            (-1.0 + machine_eps, 1 - machine_eps),
        ]
        optimization_result = minimize(
            _negLogLikelihoodBinom,
            init_guess,
            bounds=param_bounds,
            # options={"disp": 1},
        )
        if not optimization_result.success:
            raise RuntimeError(
                "Parity curve contrast fitting failed.\n"
                "Error from optimizer: {}\n{}".format(
                    optimization_result.message, optimization_result
                )
            )
        else:
            _LOGGER.debug("Parity Fit optimization result:\n%s", optimization_result)

        opt_amplitude, opt_phase, opt_offset = optimization_result.x

        # Uncertainty calculation.
        # Doesn't trust L-BFGS-B hessian b/c depends on guess & the minimization path
        hessian = numdifftools.Hessdiag(_negLogLikelihoodBinom)(optimization_result.x)
        fit_uncertainty = np.sqrt(1 / hessian)  # variable order: amp, phase, y_offset

        _LOGGER.debug(
            "Parity result: Amp = %.3f, Phi = %.3f, offset = %.3f",
            opt_amplitude,
            opt_phase,
            opt_offset,
        )

        return (
            uncertainties.ufloat(opt_amplitude, fit_uncertainty[0]),
            uncertainties.ufloat(opt_phase, fit_uncertainty[1]),
            uncertainties.ufloat(opt_offset, fit_uncertainty[2]),
        )
