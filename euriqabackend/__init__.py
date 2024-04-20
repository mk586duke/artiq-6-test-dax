"""Package for UMD EURIQA project experiment backend code, built on M-Labs ARTIQ."""
import pathlib
import shlex
import subprocess

try:
    import artiq
    _ARTIQ_MAJOR_VERSION = artiq.__version__[0]
except (ModuleNotFoundError, ImportError):
    _ARTIQ_MAJOR_VERSION = "MISSING"

_EURIQA_LIB_DIR = pathlib.Path(__file__).resolve().parents[1]

_STATIC_BUILD = False  # TODO: make read from environment variable somehow?

if not _STATIC_BUILD:
    # Deduce configuration settings: ARTIQ version, folder, and git status
    _EURIQA_GIT_HASH = (
        subprocess.check_output(
            shlex.split(
                'git -C "{}" rev-parse --verify --short HEAD'.format(_EURIQA_LIB_DIR)
            )
        )
        .strip()
        .decode("ascii")
    )
    _IS_EURIQA_GIT_CLEAN = (
        subprocess.check_output(
            shlex.split('git -C "{}" status --short'.format(_EURIQA_LIB_DIR))
        )
        .strip()
        .decode("ascii")
        == ""
    )
else:
    # Expects "$PACKAGE/euriqabackend/GIT_REV.hash" to be populated by builder/freezer
    _EURIQA_GIT_HASH = (
        (_EURIQA_LIB_DIR / "euriqabackend" / "GIT_REV.hash").open("r").read().strip()
    )
    _IS_EURIQA_GIT_CLEAN = True
__version__ = "artiq_{}_euriqa_{}.{}".format(
    _ARTIQ_MAJOR_VERSION, _EURIQA_GIT_HASH, "clean" if _IS_EURIQA_GIT_CLEAN else "dirty"
)
