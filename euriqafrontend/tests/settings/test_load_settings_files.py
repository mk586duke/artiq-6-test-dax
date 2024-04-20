"""Test proper loading of :mod:`euriqafrontend.settings`."""
import pytest


@pytest.mark.parametrize("settings_name", ["auto_calibration", "rf_calibration"])
def test_load_settings_module(settings_name: str):
    import euriqafrontend.settings as settings_mod

    settings = getattr(settings_mod, settings_name)
    assert len(settings) != 0
    assert "comments" in settings.keys()

    # Test equivalent dotted access
    assert settings["comments"] == settings.comments
