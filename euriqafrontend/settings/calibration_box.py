"""A data structure to assist with easily storing ARTIQ calibration data.

It handles a mix of static & ARTIQ datasets, as well as calculated values based on
sums/multiples of other properties.

Values can be accessed as ``CalibrationBox().path.to.property.value``.
"""
import datetime
import functools
import logging
import typing

import box
from box.converters import _from_json
import numpy as np
import qiskit.providers.models.backendproperties as q_be_props


_LOGGER = logging.getLogger(__name__)


def _filter_keys(
    keys: typing.Set[str], suffix: str, return_parent: bool = True
) -> typing.Tuple[str, ...]:
    """Filters a set of dotted Box keys to only ones containing a specific entry key/suffix.

    Example:
        >>> _filter_keys({"box.key1.value", "box.key2.other"}, ".value", return_parent=True)
        ("box.key1")
        >>> _filter_keys({"box.key1.value", "box.key2.other"}, ".value", return_parent=False)
        ("box.key1.value")
    """
    found_keys = filter(lambda k: k.endswith(suffix), keys)
    if return_parent:
        found_keys = map(lambda k: k[: -len(suffix)], found_keys)
    return tuple(found_keys)


class CalibrationBox(box.Box):
    """Wrapper around :class:`box.Box`, allows dotted access to calibration settings.

    Allows special, referencing calibration settings to be used, which either
    lookup their value at creation time in an external dictionary, or generate their
    value based on other calibration values.
    """

    def __init__(self, *args, dataset_dict: typing.Dict = None, **kwargs):
        kwargs["box_dots"] = True
        kwargs["camel_killer_box"] = True
        # prevent recursively initializing dataset & calculated args
        kwargs["box_class"] = box.Box
        super().__init__(*args, **kwargs)
        self._init_dates()
        self._init_dataset_args(dataset_dict)
        self._init_array_args()
        self._init_calculated_args()

    def _init_array_args(self):
        """Convert any entries with ``list_to_array=True`` -> numpy array.

        Just for convenience so that lists don't always need converted -> arrays
        when using values.
        """
        suffix = ".list_to_array"
        for parent_key in filter(
            lambda k: self[k + suffix],
            _filter_keys(self.keys(dotted=True), suffix, return_parent=True),
        ):
            raw_value = self[f"{parent_key}.value"]
            self[f"{parent_key}.value"] = np.array(raw_value)

    def _init_dataset_args(self, dataset_dict: typing.Dict = None):
        """Get values for any keys referring to datasets.

        Example item:
        ```
        "name": {
            "type": "dataset",
            "key": "path.to.global.dataset",
            "scale": 100.0
        }
        ```
        Any keys not explicitly specified above will be unused, allowing
        you to include comments, dates, etc.
        ``scale`` can also be omitted, in which case it will default to ``1.0``.
        ``scale`` multiplies whatever value it receives from the dataset by ``scale``
        before setting it to ``value``.
        If you want to disable ``scale`` multiplication
        (e.g. for lists, which raise TypeError), then set ``scale=False``.
        Note: this function will automatically convert datasets of type ``list``
        -> ``np.ndarray`` if they are being scaled. This change should be transparent,
        but can be disabled as-needed by setting ``list_to_array=False``.
        """
        # TODO: handle populating "time" field, needs dataset support
        type_suffix = ".type"
        for parent_key in filter(
            lambda k: self[k + type_suffix].lower() == "dataset",
            _filter_keys(self.keys(dotted=True), type_suffix, return_parent=True),
        ):
            # Only error if has a "dataset" key
            if dataset_dict is None:
                raise ValueError(
                    "No dataset dictionary provided. Cannot look up dataset values"
                )
            scale_path = f"{parent_key}.scale"
            if scale_path in self.keys(dotted=True):
                val_scale = self[scale_path]
            else:
                val_scale = False
            unscaled_value = dataset_dict.get(self[f"{parent_key}.key"])
            if val_scale:
                if isinstance(unscaled_value, (list, tuple)) and self.get(
                    f"{parent_key}.list_to_array", True
                ):
                    unscaled_value = np.array(unscaled_value)
                scaled_value = unscaled_value * val_scale
            else:
                scaled_value = unscaled_value
            self[f"{parent_key}.value"] = scaled_value
            self[f"{parent_key}.date"] = datetime.datetime.now().astimezone()

    def _init_calculated_args(self):
        """Resolves calculated arguments, which depend on other keys.

        Can be used for calculating values that depend on other values, with all
        relationships maintained in the dictionary.
        Currently, these are only calculated at initialization of the dictionary,
        dynamic updates are a future upgrade.

        For exactness, all keys to "addends" and "products" must be specified
        (unless either "addends" or "products" is not desired, in which case it
        should be an empty list: ``[]``).

        Example item:
        ```
        "name": {
            "type": "calculated",
            "addends": [
                {
                    "key": "internal.path.to.other.calibration.value",
                    "scale": 1.0
                }
            ]
            "products": [
                {
                    "key": "internal.path.to.second.calibration.value",
                    "scale": 100E+6,
                    "coefficient": 0.5
                    "exponent": 2.0
                }
            ],
            "comment": "name.value will be set to sum(addends.key * addends.scale) +
                product(p.coefficient * ((p.scale * p.key) ^ p.exponent))
                for p in products)."
        }
        ```
        """

        def _calculate_value(
            addends: typing.List[typing.Dict] = None,
            products: typing.List[typing.Dict] = None,
        ) -> float:
            out = 0.0
            if addends is not None and len(addends):
                out += sum(map(lambda c: self[c["key"]] * c["scale"], addends))
            if products is not None and len(products):
                out += functools.reduce(
                    lambda x, y: x * y,
                    map(
                        lambda p: p["coefficient"]
                        * np.pow(p["scale"] * self[p["key"]], p["exponent"]),
                        products,
                    ),
                )
            return out

        type_suffix = ".type"

        for parent_key in filter(
            lambda k: self[k + type_suffix].lower() == "calculated",
            _filter_keys(self.keys(dotted=True), type_suffix, return_parent=True),
        ):
            addends = self[f"{parent_key}.addends"]
            products = self[f"{parent_key}.products"]
            self.__setitem__(f"{parent_key}.value", _calculate_value(addends, products))
            try:
                last_update_time = max(
                    map(
                        lambda k: self[k.rstrip("value") + "date"],
                        set(map(lambda c: c["key"], addends))
                        | set(map(lambda c: c["key"], products)),
                    )
                )
            except (TypeError, KeyError):
                last_update_time = datetime.datetime.now().astimezone()
            self.__setitem__(f"{parent_key}.date", last_update_time)

    def _init_dates(self):
        """Converts any ISO-format string dates to timezone-aware dates."""
        for k in _filter_keys(self.keys(dotted=True), ".date", return_parent=False):
            if isinstance(self[k], str):
                try:
                    self.__setitem__(
                        k, datetime.datetime.fromisoformat(self[k]).astimezone()
                    )
                except ValueError as err:
                    raise ValueError(f"Calibration date {self[k]} at key '{k}' is invalid") from err

    @classmethod
    def from_json(cls, *args, dataset_dict: typing.Dict = None, **kwargs):
        """Use kwarg ``dataset_dict`` to auto-populate any datasets.

        Allows setting static values via JSON, and then pull latest
        calibrations from ARTIQ datasets.
        """
        data = _from_json(*args, **kwargs)

        return cls(data, dataset_dict=dataset_dict)

    def to_backend_properties(self, num_qubits: int) -> q_be_props.BackendProperties:
        """Convert a CalibrationBox to Qiskit Backend properties."""
        general = []
        for parent_key in _filter_keys(self.keys(dotted=True), ".value"):
            value = self[parent_key + ".value"]
            try:
                unit = self[parent_key + ".unit"]
            except KeyError:
                unit = ""
            try:
                date = self[parent_key + ".date"]
            except KeyError as err:
                raise KeyError(
                    f"Could not find date in entries for '{parent_key}': {self[parent_key]}"
                ) from err
            general.append(q_be_props.Nduv(date, parent_key, unit, value))

        # has_max_ions = any(map(lambda k: k.endswith("max_number_ions.value"), self.keys(dotted=True)))
        # populate list of qubits with the number of *qubits* (not ions) currently used
        # TODO: maybe populate with more properties?
        if not isinstance(num_qubits, int):
            num_qubits_int = int(num_qubits)
            assert (
                num_qubits == num_qubits_int
            ), f"Passed non-integer number of qubits {num_qubits}"
            num_qubits = num_qubits_int

        qubits = [
            [
                q_be_props.Nduv(
                    datetime.datetime.now().astimezone(), "operational", "", True
                )
            ]
        ] * num_qubits

        return q_be_props.BackendProperties(
            backend_name="",
            backend_version=datetime.date.today(),
            last_update_date=max(map(lambda nduv: nduv.date, general)),
            qubits=qubits,
            gates=[],
            general=general,
            rf_calibration=self,
        )
