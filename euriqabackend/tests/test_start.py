"""Basic tests to check that this library will start and recognize ARTIQ."""
import artiq


def test_artiq_load():
    """Tests that ARTIQ has loaded correctly."""
    dummy_experiment = artiq.language.environment.Experiment()
