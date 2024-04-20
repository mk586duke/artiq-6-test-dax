import re

import h5py
import numpy as np


class h5wrapper:
    """ H5Reader
    Is passed a filepath and serves as a wrapper for that hdf5 file.
    fit_params_names is a list of strings with names for all fitparams
    fit_params is a dictionary of np.ndarrays where key is the fit param name
    rid is an np array of one value that is the rid
    everything else is an np.ndarray of the corresponding dataset in the hdf5 file

    """

    def __init__(self, filepath: str) -> None:
        self.avg_thresh = None
        self.x_values = None
        self.fit_params_names = []
        self.fit_params = {}
        self.fit_y = None
        self.fit_x = None
        self.rid = None
        self.dataset_names = None
        self.pmt_nums = None
        self.filename = None

        # Open file and make sure it has the datasets group
        self.__file = h5py.File(filepath, "r")
        if (
            self.__file.keys().__contains__("datasets") is False
            or len(list(self.__file["datasets"])) == 0
            or self.__file.keys().__contains__("rid") is False
        ):
            raise Exception("No Artiq datasets found in hdf5 file: {}".format(filepath))

        # Extract rid from rid dataset
        self.rid = np.array(self.__file["rid"])

        # Extract the filename from the path
        regex = re.compile("[\/|\\|\\\\]([\w|-]*\.h5)$")
        match = regex.search(filepath)
        self.filename = match.group(1)

        # Extracts all dataset names under datasets
        self.dataset_names = list(self.__file["datasets"])

        # Initialize Pmt Nums
        self.pmt_nums = self.extract_dataset("active_pmts")

        # Extract fit_y and fit_x from file
        self.fit_x = self.extract_dataset("fit_x")
        self.fit_y = self.extract_dataset("fit_y")

        # Set fits to none if no fits are recorded
        if np.isnan(self.fit_x).any():
            self.fit_x = None
            self.fit_y = None

        # Extract the fit params and names for all the params
        regex = re.compile("^\w+\.\w+\.fitparam_(\w+)$")
        for name in self.dataset_names:
            match = regex.match(name)
            if match is not None:
                self.fit_params_names.append(match.group(1))
                self.fit_params[match.group(1)] = np.array(
                    self.__file["datasets"][match.group(0)]
                )

    # Takes in a name for dataset ("avg_thresh" for example) and extracts it
    # from the hdf5 file 'datasets' group. Raises Exception if no such dataset is found
    def extract_dataset(self, name) -> np.ndarray:
        regex = re.compile("^\w+\.\w+\.{}$".format(name))
        matched_keys = list(filter(regex.match, self.dataset_names))

        if len(matched_keys) > 1:
            print(
                "More than one match for '{}' dataset. Using the first match".format(
                    name
                )
            )
        elif len(matched_keys) == 0:
            raise Exception("No '{}' Dataset Found".format(name))

        return np.array(self.__file["datasets"][matched_keys[0]])
