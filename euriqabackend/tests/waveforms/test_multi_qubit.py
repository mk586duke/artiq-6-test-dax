"""Test :mod:`euriqabackend.waveforms.multi_qubit`."""
import pytest
import qiskit.pulse as qp

import euriqabackend.waveforms.multi_qubit as multiqb


def test_xx_am_gate(qiskit_backend_with_gate_solutions, tmp_path):
    if qiskit_backend_with_gate_solutions.configuration().num_qubits == 1:
        pytest.xfail("Not enough qubits in backend for 2-qubit gate")
    # The original units were in MHz, so these are all multiplied by 1e6
    # Taken from gate solutions 2019_12_20/15ions_fullset_interpolated_225us
    example_xx_gate_rabis = [
        (0.0e6, 0.025194691167612547e6),
        (0.025194691167612547e6, 0.054822342364947034e6),
        (0.05482234236494704e6, 0.07698756800036738e6),
        (0.07698756800036738e6, 0.07811827277337292e6),
        (0.07811827277337294e6, 0.07770438546378304e6),
        (0.07770438546378307e6, 0.10182748599076295e6),
        (0.10182748599076298e6, 0.14338488793547732e6),
        (0.14338488793547732e6, 0.15881654537076514e6),
        (0.15881654537076517e6, 0.1433775552674004e6),
        (0.1433775552674004e6, 0.1018189808570081e6),
        (0.10181898085700815e6, 0.07770466843382218e6),
        (0.07770466843382219e6, 0.07811933099414312e6),
        (0.07811933099414313e6, 0.07698423772497663e6),
        (0.07698423772497665e6, 0.05481563039059622e6),
        (0.054815630390596226e6, 0.02518890554441995e6),
        (0.025188905544419955e6, 0.0e6),
    ]
    durations = ((225e-6 / len(example_xx_gate_rabis)),) * len(example_xx_gate_rabis)
    detuning = 2.98571e6

    with qp.build(qiskit_backend_with_gate_solutions) as test_sched:
        qp.call(
            multiqb.xx_am_gate(
                (0, -1),
                list(zip(durations, example_xx_gate_rabis)),
                nominal_detuning=detuning,
            )
        )

    test_sched.draw().savefig(tmp_path / "xx_am_gate_waveform.png")
