import logging

from artiq.coredevice import RTIOOverflow
from artiq.experiment import StringValue
from artiq.language import delay
from artiq.language import delay_mu
from artiq.language import kernel
from artiq.language import now_mu
from artiq.language.environment import HasEnvironment
from artiq.language.types import TBool
from artiq.language.types import TInt32
from artiq.language.types import TInt64
from artiq.language.types import TList
from artiq.language.types import TNone
from artiq.language.units import ns
from artiq.language.units import us
from artiq.master.worker_db import DeviceError
import artiq.language.environment as artiq_env
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import numpy as np

_LOGGER = logging.getLogger(__name__)


def gaussian(x, a, b, c):
    return a * np.exp(-(((x - b)/c) ** 2))

class Spec_Analyzer(HasEnvironment):


    def build(self):
    # build the GUI

        self.prominence = self.get_argument(
            "specify the minimum prominence of peaks",
            artiq_env.NumberValue(default=0.3),
            group="Spec_Analysis")

        self.width = self.get_argument(
            "specify the minimum width of peaks",
            artiq_env.NumberValue(default=2),
            group="Spec_Analysis")

        self.num_peaks = self.get_argument(
            "How many peaks are you expecting?",
            artiq_env.NumberValue(default=0),
            group="Spec_Analysis")

        self.auto_adjustment=self.get_argument("engage auto-adjustment (unstable)", artiq_env.BooleanValue(default=False),group="Spec_Analysis")


        _LOGGER.debug("Done building Spec Analyzer")




        self.setattr_device("ccb")
        self.setattr_device("scheduler")

    def module_init(self, data_folder,num_steps):
    # Data_set initialization

        self.data_folder=data_folder
        self.num_steps=num_steps

        self.set_dataset(
            "data.spec_analysis.heights",
            np.full(self.num_steps,np.nan),
            broadcast=True,
        )

        self.set_dataset(
            "data.spec_analysis.heights_x_fit",
            np.nan,
            broadcast=True,
        )

        self.set_dataset(
            "data.spec_analysis.heights_y_fit",
            np.nan,
            broadcast=True,
        )

        self.ccb.issue(
            "create_applet",
            name="Spec Analysis",
            command="$python -m euriqafrontend.applets.plot_multi" + " "
                    + " --x" + " data." + self.data_folder + ".x_values"
                    + " --y-names " + "data.spec_analysis.heights"
                    + " --x-fit " + " data.spec_analysis.heights_x_fit"
                    + " --y-fits " + " data.spec_analysis.heights_y_fit"
                    + " --y-label 'Height'"
                    + " --rid {0}".format("data." + self.data_folder + ".rid")
                    + " --x-label 'Frequency'",
            group="Spec_analysis"
        )

        _LOGGER.debug("Done initialize Spec Analyzer")



    def prepare(self):
        pass


    def module_update(self,istep):
        """update the module after every experiment step"""

        counts = np.array(self.get_dataset("data." + self.data_folder + ".avg_thresh"))

        height = np.sum(counts[:, istep])

        self.mutate_dataset("data.spec_analysis.heights", istep, height)


    def peaks(self):
        """If the efficiency turns out to be very low, put some extra effort on the auto_adjustment"""
        x_values=np.array(self.get_dataset("data." + self.data_folder + ".x_values"))
        y_values=np.array(self.get_dataset("data.spec_analysis.heights"))
        peak_index,info=find_peaks(y_values,prominence=self.prominence,width=self.width)
        if self.auto_adjustment:
            while abs(len(peak_index)-self.num_peaks)>0.5:

                if len(peak_index)<self.num_peaks-0.5:
                    self.prominence-=0.1
                    peak_index,info=find_peaks(y_values,prominence=self.prominence,width=self.width)


                if len(peak_index)<self.num_peaks-0.5:
                    self.width-=0.5
                    peak_index,info=find_peaks(y_values,prominence=self.prominence,width=self.width)


                if len(peak_index)>self.num_peaks+0.5:
                    self.prominence+=0.1
                    peak_index,info=find_peaks(y_values,prominence=self.prominence,width=self.width)


                if len(peak_index)>self.num_peaks+0.5:
                    self.width+=0.5
                    peak_index,info=find_peaks(y_values,prominence=self.prominence,width=self.width)



        return peak_index

    def gaussian_fit(self,data_x,data_y):
        x_values = data_x
        y_values = data_y
        x_min = min(x_values)
        y_min = min(y_values)
        x_range = max(x_values) - min(x_values)
        y_range = max(y_values) - min(y_values)
        normalized_x = np.divide(np.subtract(x_values, x_min), x_range)
        normalized_y = np.divide(np.subtract(y_values, y_min), y_range)

        center = normalized_x[np.argmax(normalized_y)]

        p_fit, cov_fit = curve_fit(gaussian, normalized_x, normalized_y, [1, center, 0.2])

        step_size = np.divide(normalized_x[-1] - normalized_x[0], len(normalized_x) * 10)

        x_values_fit = np.arange(normalized_x[0], normalized_x[-1], step_size)

        y_values_fit = [gaussian(x, *p_fit) for x in x_values_fit]

        x_values_fit = np.add(np.multiply(x_values_fit, x_range), x_min)

        y_values_fit = np.add(np.multiply(y_values_fit, y_range), y_min)

        return p_fit[1]*x_range+x_min,x_values_fit,y_values_fit


    def fit_single_peak(self):

        x_values = np.array(self.get_dataset("data." + self.data_folder + ".x_values"))
        y_values = np.array(self.get_dataset("data.spec_analysis.heights"))
        center,fit_x,fit_y=self.gaussian_fit(x_values,y_values)

        self.set_dataset(
            "data.spec_analysis.heights_x_fit",
            fit_x,
            broadcast=True,
        )


        self.set_dataset(
            "data.spec_analysis.heights_y_fit",
            [fit_y],
            broadcast=True,
        )

        return center

    def fit_multiple_peaks(self):

        x_values = np.array(self.get_dataset("data." + self.data_folder + ".x_values"))
        y_values = np.array(self.get_dataset("data.spec_analysis.heights"))
        peak_index=self.peaks()

        fit_x=[]
        fit_y=[]
        centers=[]

        for index in peak_index:
            temp_center,temp_fit_x,temp_fit_y=self.gaussian_fit(x_values[index-3:index+3],y_values[index-3:index+3])
            centers.append(temp_center)
            fit_x.extend(temp_fit_x)
            fit_y.extend(temp_fit_y)

        self.set_dataset(
            "data.spec_analysis.heights_x_fit",
            fit_x,
            broadcast=True,
        )

        self.set_dataset(
            "data.spec_analysis.heights_y_fit",
            [fit_y],
            broadcast=True,
        )

        self.set_dataset(
            "data.spec_analysis.peaks",
            centers,
            broadcast=True,
        )




        return centers
