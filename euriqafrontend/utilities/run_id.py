"""Utilities to find a file by RID (Run ID #).

Does not work on remote/network paths.
"""
import logging
import pathlib
import typing

_LOGGER = logging.getLogger(__name__)


def find_rid_in_path(rid: int, path: typing.Union[str, pathlib.Path]) -> pathlib.Path:
    """Return the path of an ARTIQ dataset, given the run ID & storage path."""
    path = pathlib.Path(path).resolve()
    assert path.exists()

    rid_glob_string = "{:09}-*.h5".format(rid)
    rid_files = list(path.rglob(rid_glob_string))
    _LOGGER.debug("Found rid_files: %s", rid_files)
    assert len(rid_files) == 1
    return rid_files[0]
