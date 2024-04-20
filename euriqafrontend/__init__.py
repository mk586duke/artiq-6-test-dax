"""Python module to run trapped ion experiments.

Central repository for UMD EURIQA's experiment code.
Should not include any hardware or device dependencies, mostly just code that
describes experiments.
"""

import distutils.util as _dist_util
import os
import pathlib
import sys
import warnings

_ENVIRONMENT_VARIABLE_NAS_DIR = "EURIQA_NAS_DIR"
_ENVIRONMENT_VARIABLE_CHECK_NAS = "CHECK_NAS_DIR"  # default to ON
_DEFAULT_NAS_DIR = {
    "win32": r"\\EURIQA-NAS\lab",
    "linux": "/media/euriqa-nas/",  # should be auto-mounted via cifs-tools
}

if os.getenv(_ENVIRONMENT_VARIABLE_NAS_DIR) is not None:
    _nas_dir = pathlib.Path(os.getenv(_ENVIRONMENT_VARIABLE_NAS_DIR))
else:
    _nas_dir = _DEFAULT_NAS_DIR[sys.platform.lower()]
    # TODO: remove this, but it's here for transition.
    warnings.warn(
        "No environment variable ({}) defined for where the EURIQA NAS is mounted. "
        "I guessed '{}' based on your platform".format(
            _ENVIRONMENT_VARIABLE_NAS_DIR, _nas_dir
        )
    )

EURIQA_NAS_DIR = pathlib.Path(_nas_dir)
if _dist_util.strtobool(os.getenv(_ENVIRONMENT_VARIABLE_CHECK_NAS, "ON")):
    assert EURIQA_NAS_DIR.exists() and EURIQA_NAS_DIR.is_dir()
