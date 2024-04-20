"""Helper for submitting Qiskit pulse schedule -> ARTIQ experiment."""
import logging
import typing

import pulsecompiler.qiskit.backend as pc_backend
import qiskit.compiler.assemble as assemble
import qiskit.pulse as qp
import sipyco.pyon as pyon

import euriqafrontend.interactive.artiq_clients as artiq_clients


_LOGGER = logging.getLogger(__name__)


def submit_schedule(
    schedule: qp.Schedule,
    master_ip: str,
    backend: pc_backend.MinimalQiskitIonBackend,
    disable_calibrations: bool = False,
    experiment_name: str = "RFSoCSequence",
    experiment_file: str = "rfsoc/gate_tests.py",
    submit_kwargs: typing.Dict = None,
    experiment_kwargs: typing.Dict = None,
) -> int:
    """Submits a Qiskit Pulse Schedule as an ARTIQ experiment."""
    if submit_kwargs is None:
        submit_kwargs = {
            "pipeline_name": "main",
            "priority": 0,
            "due_date": None,
            "flush": False,
        }
    default_experiment_args = {
        "PMT Input String": '-1:17',
        "use_RFSOC": True,
        "do_calib": False,
        "use_AWG": False,
    }
    used_experiment_kwargs = default_experiment_args
    if experiment_kwargs is not None:
        used_experiment_kwargs.update(experiment_kwargs)
    artiq_scheduler = artiq_clients.get_artiq_scheduler(master_ip)

    # TODO: check experiment_name in experiment_db
    # assert experiment_name in experiment_db
    expid = {
        "class_name": experiment_name,
        "file": experiment_file,
        "log_level": 20,
        "repo_rev": "N/A",
        "arguments": used_experiment_kwargs,
    }

    # TODO: set arguments for calibrations
    if disable_calibrations:
        raise NotImplementedError("TODO")
        # arguments = {}

    num_qubits = backend.configuration().n_qubits
    expid["arguments"]["openpulse_schedule_qobj"] = pyon.encode(
        assemble(
            schedule,
            backend,
            qubit_lo_freq=[0.0] * num_qubits,
            meas_lo_freq=[0.0] * num_qubits,
        ).to_dict(validate=False)
    )

    _LOGGER.debug("Submitting experiment with expid: %s", expid)
    return artiq_scheduler.submit(expid=expid, **submit_kwargs)
