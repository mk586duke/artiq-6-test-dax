"""EURIQA settings files.

These files can be accessed as ``euriqafrontend.settings.file_name`` due to the minor magic done here.
"""
import json
import logging
import pathlib

import box

_LOGGER = logging.getLogger(__name__)


def _load_settings_files():
    """Load settings files from JSON files stored in this directory (recursive).

    Allows access to them as ``this_module.file_name``.
    Expects files to be a top-level dictionary.
    This is a function to avoid cluttering the namespace of this module.
    """
    this_directory = pathlib.Path(__file__).parent
    for json_file in this_directory.rglob("*.json"):
        try:
            settings = box.Box.from_json(filename=json_file)
        except json.JSONDecodeError:
            _LOGGER.error(
                "JSON file %s could not be decoded properly. Skipping",
                json_file,
                exc_info=True,
            )
        else:
            # Allow access to the contents of the JSON file as this_module.file_name
            globals()[json_file.stem] = settings

        # Add the path of the JSON file as this_module.FILE_NAME_PATH = pathlib.Path("path/to/file")
        json_path_name = f"{json_file.stem.upper()}_PATH"
        globals()[json_path_name] = json_file.resolve()


_load_settings_files()
