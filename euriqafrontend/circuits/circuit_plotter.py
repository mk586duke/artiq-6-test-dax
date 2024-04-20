from euriqafrontend.utilities.run_id import find_rid_in_path
import matplotlib.pyplot as plt
from datetime import datetime, date
import h5py
import pathlib
import numpy as np
import re
from typing import List, Union


def get_x_values(
    data_path: pathlib.WindowsPath, rid: int, date: datetime.date = date.today()
):
    """Pull the x axis from an experiment RID + artiq file path

    Args:
        data_path: PathLib path to artiq data base directory
        rid: int that has the RID you want to pull raw counts from
        data: a datetime.date object that specifies the date of data, defaults to today
    """
    date_path = date.strftime("%Y-%m-%d")
    file = find_rid_in_path(rid, data_path / date_path)
    data = h5py.File(str(file), "r")
    key_filter = re.compile("[\w.]*\.x_values")
    data_key = list(filter(key_filter.match, list(data["datasets"].keys())))[0]
    x_values = np.array(data["datasets"][data_key])
    return x_values


def get_raw_counts(
    data_path: pathlib.WindowsPath, rid: int, date: datetime.date = date.today()
):
    """Pull the raw counts from an experiment RID + artiq file path

    Args:
        data_path: PathLib path to artiq data base directory
        rid: int that has the RID you want to pull raw counts from
        data: a datetime.date object that specifies the date of data, defaults to today

    Returns:

    """
    if isinstance(date,str):
        date_path=date
    else:
        date_path = date.strftime("%Y-%m-%d")
    file = find_rid_in_path(rid, data_path / date_path)
    data = h5py.File(str(file), "r")

    key_filter = re.compile("[\w.]*\.interrupted")
    data_key = list(filter(key_filter.match, list(data["datasets"].keys())))[0]
    interrupted = data["datasets"][data_key]

    if interrupted[()] is True:
        raise RuntimeError("Data taking was interrupted, invalid data")

    key_filter = re.compile("[\w.]*\.raw_counts")
    data_key = list(filter(key_filter.match, list(data["datasets"].keys())))[0]
    raw_counts = np.array(data["datasets"][data_key])
    return raw_counts


def get_binary_data(
    data_path: pathlib.WindowsPath, rid: int, date: datetime.date = date.today()
):
    """Pull the raw counts and thresholds them from an experiment RID + artiq file path

    Args:
        data_path: PathLib path to artiq data base directory
        rid: int that has the RID you want to pull raw counts from
        data: a datetime.date object that specifies the date of data, defaults to today

    Returns:
        PMT counts from each scan point, shot, and PMT digitized to be either 0 or 1.
    """

    raw_counts = get_raw_counts(data_path=data_path, rid=rid, date=date)
    thresh_counts = raw_counts.copy()
    thresh_counts[raw_counts > 1] = 1
    thresh_counts[raw_counts <= 1] = 0

    return thresh_counts


def hist_states(thresh_counts: np.ndarray, register_index: List[int]):
    """Gather histogram information of the different state populations

    Args:
        thresh_counts: thresh_counts returned from get_binary_data()
        register_index: zero indexed list of ions/pmts you wish to plot,
            lists of lists to group analysis
    """
    n_qbits = len(register_index)
    n_states = 2 ** n_qbits

    raw_pop = thresh_counts[register_index, :]

    states = np.zeros((n_states, n_qbits))
    counts = np.zeros(n_states)
    labels = list()
    for i in np.arange(n_states):
        b = bin(i)
        b = b[2:]
        label = ("{:0" + str(n_qbits) + "d}").format(int(b))
        state = np.zeros(n_qbits)
        for j in range(len(b)):
            state[-j - 1] = int(b[-j - 1])

        states[i, :] = state
        counts[i] = np.sum(np.all(raw_pop.T == state.T, axis=1))
        labels.append(label)

    labels = tuple(labels)
    return counts, labels


def plot_avg_counts(
    thresh_counts: np.ndarray,
    register_index: List[int] = None,
    scan: np.ndarray = None,
    title: str = None,
):
    """Plot the average population

    Args:
        thresh_counts: thresh_counts returned from get_binary_data()
        register_index: zero indexed list of ions/pmts you wish to plot, lists of lists to group analysis
        scan: 1-D vector of the scan variables.
        title: title of plot
    """

    fig, ax = plt.subplots()

    if scan is None:
        scan = np.arange(thresh_counts.shape[-1])
    if register_index is None:
        register_index = np.arange(thresh_counts.shape[0])

    # if it is a list of lists
    if type(register_index) == list and type(register_index[0]) == list:
        for n, p in enumerate(register_index):
            avg_thresh = np.mean(thresh_counts[p, :, :], axis=1)
            ax.plot(scan, avg_thresh, label="Group {0}".format(n + 1), marker=".")
        leg = ax.legend(loc="best")
    else:
        avg_thresh = np.mean(thresh_counts[register_index, :, :], axis=1)
        ax.plot(scan, avg_thresh.T, marker=".")

    ax.set_xlabel("Scan")
    ax.set_ylabel("Population")
    if title is not None:
        ax.set_title(title)
    fig.set_figheight(4)
    fig.set_figwidth(8)


def plot_parity(
    thresh_counts: np.ndarray,
    register_index: Union[List[List[int]], List[int]] = None,
    scan: np.ndarray = None,
    title: str = None,
):
    """Plot the parity

    Args:
        thresh_counts: thresh_counts returned from get_binary_data()
        register_index: zero indexed list of ions/pmts you wish to plot, lists of lists to group analysis
        scan: 1-D vector of the scan variables.
        title: title of plot
    """
    fig, ax = plt.subplots()
    if scan is None:
        scan = np.arange(thresh_counts.shape[-1])
    if register_index is None:
        register_index = np.arange(thresh_counts.shape[0])
    if type(register_index) == list and type(register_index[0]) == list:
        for n, p in enumerate(register_index):
            parity = np.prod(thresh_counts[p, :, :] * -2 + 1, axis=0)
            avg_parity = np.mean(parity, axis=0)
            ax.plot(scan, avg_parity, label="Group {0}".format(n + 1), marker=".")
        leg = ax.legend(loc="best")

    else:
        parity = np.prod(thresh_counts[register_index, :, :] * -2 + 1, axis=0)
        avg_parity = np.mean(parity, axis=0)
        ax.plot(scan, avg_parity, marker=".")

    ax.set_xlabel("Scan")
    ax.set_ylabel("Parity")
    if title is not None:
        ax.set_title(title)
    fig.set_figheight(4)
    fig.set_figwidth(8)


def plot_correlations(
    thresh_counts: np.ndarray,
    register_index: Union[List[List[int]], List[int]] = None,
    scan: np.ndarray = None,
    title: str = None,
):
    """Plot the correlations, when all correlated, i.e 000/111 map to 1

    Args:
        thresh_counts: thresh_counts returned from get_binary_data()
        register_index: zero indexed list of ions/pmts you wish to plot, lists of lists to group analysis
        scan: 1-D vector of the scan variables.
        title: title of plot
    """

    fig, ax = plt.subplots()
    if scan is None:
        scan = np.arange(thresh_counts.shape[-1])
    if register_index is None:
        register_index = np.arange(thresh_counts.shape[0])

    if type(register_index) == list and type(register_index[0]) == list:
        for n, p in enumerate(register_index):

            bit_count = np.sum(thresh_counts[p, :, :], axis=0)
            avg_correlation = np.mean(bit_count % len(p) == 0, axis=0)
            ax.plot(scan, avg_correlation, label="Group {0}".format(n + 1), marker=".")

        leg = ax.legend(loc="best")

    else:
        parity = np.prod(thresh_counts[register_index, :, :] * 2 - 1, axis=0)
        avg_parity = np.mean(parity, axis=0)
        ax.plot(scan, avg_parity, marker=".")

    if title is not None:
        ax.set_title(title)
    ax.set_xlabel("Phase of Analysis Pulse")
    ax.set_ylabel("Correlation")
    leg = ax.legend(loc="best")

    fig.set_figheight(4)
    fig.set_figwidth(8)


def plot_histogram(
    thresh_counts: np.ndarray,
    register_index: Union[List[List[int]], List[int]] = None,
    title: str = None,
):
    """Plot the correlations, when all correlated, i.e 000/111 map to 1

    Args:
        thresh_counts: thresh_counts returned from get_binary_data()
        register_index: zero indexed list of ions/pmts you wish to plot, lists of lists to group analysis
        title: title of plot
    """
    fig, ax = plt.subplots()

    if register_index is None:
        register_index = np.arange(thresh_counts.shape[0])

    if type(register_index) == list and type(register_index[0]) == list:
        assert (
            len(set([len(i) for i in register_index])) == 1
        ), "All register groups must be the same size"
    elif type(register_index) == list and (register_index[0]) == int:
        register_index = [register_index]
    else:
        raise TypeError(
            "Register index data format not recognized, must be a list (or list of list) of zero"
            "indexed data indices "
        )

    return_counts = np.zeros((len(register_index), 2 ** len(register_index[0])))
    width = 0.75 / len(register_index)
    xshift = np.linspace(-0.5, 0.5, len(register_index) + 2)
    x = np.arange(2 ** len(register_index[0]))

    for ireg, reg in enumerate(register_index):
        counts, labels = hist_states(thresh_counts, reg)
        return_counts[ireg, :] = counts
        ax.bar(
            x + xshift[ireg + 1],
            counts,
            width=width,
            alpha=0.5,
            label="{0}".format([i + 1 for i in reg]),
        )

    ax.set_xlabel("Quantum State")
    ax.set_ylabel("Counts")
    fig.set_figheight(4)
    fig.set_figwidth(14)
    leg = ax.legend(loc="best")
    fig.tight_layout()
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    plt.show()

    return return_counts
