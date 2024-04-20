"""Utilities to hash basic Python types."""
import copy
import hashlib
import logging
import pathlib
import typing

_LOGGER = logging.getLogger(__name__)


def make_hash(
    obj: typing.Union[typing.Dict, typing.Sequence, typing.Tuple, typing.Set]
):
    """
    Hash a dictionary/list/tuple/set, recursively.

    Must contains only other hashable types (i.e. list/set/tuple/dictionary)
    From StackOverflow: https://stackoverflow.com/a/8714242
    """
    import pandas as pd # moved here b/c else fails when importing in ARTIQ experiment

    if isinstance(obj, (set, tuple, list, pd.Series)):
        return hash(tuple([make_hash(element) for element in obj]))
    elif not isinstance(obj, dict):
        return hash(obj)
    new_obj = copy.deepcopy(obj)
    for key, value in new_obj.items():
        new_obj[key] = make_hash(value)
    return hash(frozenset(sorted(new_obj.items())))


def hashdir(
    directory: typing.Union[str, pathlib.Path], hashObject: object = hashlib.sha256
) -> str:
    """Hash a directory into a static hash.

    Args:
        directory (str, pathlib.Path): path for the directory to be hashed.
        hashObject (Object, optional): A hash object that supports :meth:`update` and
            :meth:`hexdigest`, such as any in :mod:`hashlib`.
            Defaults to :class:`hashlib.sha256`.

    Returns:
        str: hashed contents of the directory, effectively freezing their contents
        & metadata.

    """
    hasher = hashObject()
    dir_to_hash = pathlib.Path(directory).resolve()
    assert dir_to_hash.exists(), "Given directory does not exist"
    assert dir_to_hash.is_dir(), "Given path is not a directory"

    # Walk all files in directory & hash them
    for path in sorted(dir_to_hash.iterdir()):  # sort to make sequence deterministic
        assert path.is_file(), "Directory has non-file in it (maybe a directory)"
        _LOGGER.debug("Hashing file `%s`", str(path))
        hasher.update(path.read_bytes())
    return hasher.hexdigest()
