import typing

import qiskit.providers.backend as qbe
import qiskit.pulse as qp


def wait_gate_channels(
    duration: float,
    channels: typing.Iterable[typing.Union[qp.DriveChannel, qp.ControlChannel]] = None,
    backend: qbe.Backend = None,
) -> qp.Schedule:
    """
    Add a delay (wait) on specified output channels.

    If no channels are specified, will default to delaying all output channels
    that the backend supports. In other words, it prevents parallel operations.

    Args:
        duration (float): Time that the schedule should be delayed for, in seconds.
        channels (typing.Set[Channel, optional): Set of channels that should be
            delayed by this call. If set to ``None``, then all channels in the
            backend will be delayed. Defaults to None.

    Returns:
        qp.Schedule: Schedule to be inserted at a given time. It includes
        delays on all specified channels for the duration.
    """
    if backend is None:
        backend = qp.active_backend()
    if channels is None:
        channels = backend.configuration().all_channels
    with qp.build(backend) as wait_schedule:
        for chan in channels:
            qp.delay(qp.seconds_to_samples(duration), chan)

    return wait_schedule


def wait_gate_ions(
    duration: float, ions: typing.Sequence[int] = None, backend: qbe.Backend = None
) -> None:
    """
    Add a delay (wait) on specified ion indices.

    Assumes ions are indexed the same as whatever the Qiskit backend is using.

    NOTE: this has a different return type than :func:`wait_gate_channels`, it does not
    return a :class:`Schedule`.

    Args:
        duration (float): Duration of the wait (delay). Units of seconds.
        ions (typing.Sequence[int], optional): Ion indices to delay. If ``None``,
            then will delay all ions. Defaults to None.
    """
    if backend is None:
        backend = qp.active_backend()
    if ions is None:
        ions = list(backend.configuration().all_qubit_indices_iter)
    return qp.delay_qubits(qp.seconds_to_samples(duration), *ions)
