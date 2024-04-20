"""
Extra features to add to EURIQA experiments.

Examples include cooling, autoloading, fitting, etc.
"""
from .experiment import EuriqaExperiment


class CoolingMixin(EuriqaExperiment):
    """Mixin to add cooling features to an experiment.

    Mixes with :class:`~euriqabackend.experiments.experiment.EuriqaExperiment`.

    >>> from .experiment import EuriqaExperiment
    >>> class CoolingExperiment(CoolingMixin, EuriqaExperiment):
            pass
    >>> exp = CoolingExperiment(cooling_hardware=["cooling_dds1"])
    >>> exp.has_feature("coo")
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the cooling mixin.

        Kwargs:
            cooling_hardware(Sequence[str]): list of devices used for cooling
        """
        self._FEATURES |= set(["cool"])  # update here to not overwrite other mixins

        cooling_hardware = kwargs.pop("cooling_hardware")
        # todo: Make cooling-hardware a dictionary to group them?
        for device_name in cooling_hardware:
            self.setattr_device(device_name)

        raise NotImplementedError

    def cool(self, *args, **kwargs):
        """
        Cool the ion chain.

        Kwargs:
            cooling_settings(Dict[str, Dict[str, value]]): Dictionary of settings.
                Format: {device_name: {setting1: value, setting2: value}}
        """
        raise NotImplementedError
