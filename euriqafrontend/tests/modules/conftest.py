import pathlib
import typing

import pytest
import dax.sim.signal as dax_signal
from dax.sim import enable_dax_sim
from dax.util.artiq import get_managers

from euriqabackend.databases.device_db_main_box import device_db as _DEVICE_DB

# ignore unused imports. These are needed so the fixtures are used in tests in this dir
from euriqabackend.tests.waveforms.conftest import (  # noqa: F401
    rfsoc_qiskit_backend,
    qiskit_backend_with_real_cals,
)
from euriqabackend.tests.conftest import pytest_assertrepr_compare  # noqa: F401

_TEST_RESOURCE_PATH = (
    pathlib.Path(__file__).parents[3] / "euriqabackend/tests/resources/"
)

_dataset_db_path = _TEST_RESOURCE_PATH / "dataset_db-rfsoc-tests.pyon"


@pytest.fixture
def dax_sim_device_db(tmp_path) -> typing.Dict:
    """Fixture for a DAX sim-enabled EURIQA Device DB.

    Signal manager is unused, but must be initialized before coredevice.
    """
    yield enable_dax_sim(
        _DEVICE_DB.copy(), enable=True, moninj_service=False, output="peek"
    )
    dax_signal.get_signal_manager().write_vcd(
        str(tmp_path / "test_sequence.vcd"), ref_period=1e-9
    )


@pytest.fixture
def experiment_args() -> typing.Dict:
    """Defaults to nothing, but can be overridden on a per-test basis.

    Example:
    ```
    @pytest.mark.parametrize("experiment_args", [{"my_argument": "my_value"}])
    def test_function():
        do_something()
    """
    return {}


@pytest.fixture
def dataset_db_path() -> str:
    """Gives the path to the frozen dataset DB resource."""
    return str(_dataset_db_path)


@pytest.fixture
def artiq_experiment_managers(
    dax_sim_device_db, dataset_db_path: str, experiment_args: typing.Dict
) -> typing.Tuple:
    """Yield the ARTIQ "managers" needed to initialize an ARTIQ experiment."""
    with get_managers(
        dax_sim_device_db, dataset_db=dataset_db_path, arguments=experiment_args
    ) as managers:
        yield managers


# @pytest.fixture
# def dax_signal_manager(tmp_path) -> dax_signal.PeekSignalManager:
#     """Fixture for a DAX signal manager.

#     Used to record """
#     peek_signal_manager = dax_signal.PeekSignalManager()
#     dax_signal.set_signal_manager(peek_signal_manager)
#     yield peek_signal_manager
#     peek_signal_manager.write_vcd(str(tmp_path / "test_sequence.vcd"), ref_period=1e-9)
#     peek_signal_manager.close()
