r"""Script to build ARTIQ for the EURIQA hardware.

Example uses:
```bash
$ build_artiq clean
$ build_artiq -V euriqasandiadac build flash 192.168.9.2   # build and flash new gateware
# flash pre-built gateware
$ build_artiq -V euriqasandiadac flash --flash-dir /tmp/artiq_4_euriqa_HASH.clean/ 192.168.9.2
$ build_artiq -V euriqa build --extra-args --sandia-dac \
    flash --log-bootup --idle-kernel none 192.168.9.2   # v4+
```
"""
import importlib
import logging
import os
import pathlib
import random
import re
import shlex
import shutil
import subprocess
import threading
import time
import typing
import warnings
from datetime import datetime

import click
from artiq.coredevice.comm_mgmt import CommMgmt
from progress.spinner import Spinner

from euriqabackend import __version__
from euriqabackend import _ARTIQ_MAJOR_VERSION
from euriqabackend import _EURIQA_LIB_DIR

# NOTE: Vars to modify
_GATEWARE_MODULES = {"euriqabackend.gateware.kc705_soc"}
_ALLOWED_ARTIQ_VERSIONS = ["4", "5", "6"]

# NOTE: leave alone
_LOGGER = logging.getLogger(__name__)
_LOGGER.setLevel(logging.DEBUG)
_CONTEXT_SETTINGS = {
    "help_option_names": ["-h", "--help"],
    "auto_envvar_prefix": "ARTIQ_BUILD",
}

# Generalize this script to apply to any gateware variant imported here
# e.g. VARIANTS = {"euriqa_kc705": "euriqabackend.gateware.kc705_soc.EURIQA",
# "euriqa_kasli_main": "euriqabackend.gateware.kasli_soc.main"}
VARIANT_TO_MODULE_DICT = dict()
# NOTE: expects a VARIANTS var in each gateware module
for mod in _GATEWARE_MODULES:
    _gateware_mod = importlib.import_module(mod)
    VARIANT_TO_MODULE_DICT.update({var: mod for var in _gateware_mod.VARIANTS})
variant_to_build = None


@click.group(chain=True, context_settings=_CONTEXT_SETTINGS)
@click.option(
    "--build-variant",
    "-V",
    required=True,
    type=click.Choice(VARIANT_TO_MODULE_DICT.keys()),
    help="Which variant of the gateware you would like to build",
)
@click.option("-v", "--verbose", count=True, help="debug level (default = Info)")
@click.option(
    "--log",
    "--log-file",
    "log_file",
    default="build_artiq.log",
    show_default=True,
    type=str,
    help="log file to write to",
)
def cli(build_variant: str, verbose: int, log_file: str) -> None:
    """Top level command-line interface to run.

    Sets up logging and creates build directory, because it is always run.
    """
    global variant_to_build  # pylint: disable=global-statement
    assert build_variant in VARIANT_TO_MODULE_DICT.keys()
    variant_to_build = build_variant

    log_file = build_dir(log_file)
    build_dir().mkdir(parents=True, exist_ok=True)

    cli_log_handler = logging.StreamHandler()
    cli_log_handler.setFormatter(logging.Formatter("[%(levelname)-8s]: %(message)s"))
    cli_log_handler.setLevel(logging.INFO - 10 * verbose)  # output INFO by default
    file_log_handler = logging.FileHandler(str(log_file))
    file_log_handler.setFormatter(
        logging.Formatter("%(asctime)s | [%(levelname)-8s]: %(message)s")
    )
    file_log_handler.setLevel(logging.DEBUG)  # log all >= DEBUG in file.
    _LOGGER.addHandler(file_log_handler)
    _LOGGER.addHandler(cli_log_handler)
    _LOGGER.debug("Logging to %s", log_file)


@cli.command("clean")
@click.option(
    "--build-files-only/--all-files",
    is_flag=True,
    default=True,
    help="clean temporary build files (NOT files needed to flash ARTIQ). "
    "Useful if you want to save space/release bitfiles, but still be able to flash",
)
def clean_cmd(build_files_only: bool) -> None:
    """Command line command to clean the build directory."""
    if build_files_only:
        clean_dir = build_dir(variant_to_build)
    else:
        clean_dir = build_dir()
    _LOGGER.info("Cleaning dir %s", clean_dir)
    shutil.rmtree(clean_dir, ignore_errors=True)


@cli.command("build")
@click.option(
    "--compile-gateware/--no-compile-gateware",
    default=True,
    show_default=True,
    help="If the gateware/firmware should be compiled. If not set, "
    "it will just attempt to copy existing gateware to a binaries folder.",
)
@click.option(
    "-a",
    "--artiq-version",
    type=click.Choice(_ALLOWED_ARTIQ_VERSIONS),
    default=_ARTIQ_MAJOR_VERSION,
    show_default=True,
    help="Choose which version of ARTIQ you are building",
)
@click.option("--extra-args", type=str, help="Extra build arguments to ARTIQ build")
def compile_cmd(compile_gateware: bool, artiq_version: str, extra_args: str) -> None:
    """Compile ARTIQ for EURIQA.

    Called with `build` command-line argument.
    """
    artiq_version = int(artiq_version)
    output_dir = build_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    if compile_gateware:
        _compile_artiq_gateware(artiq_version, output_dir, extra_args)
    else:
        _LOGGER.info("NOT compiling ARTIQ. Just copying files instead.")

    # Copy files to central folder for flashing
    _archive_flash_binaries()


def _compile_artiq_gateware(
    artiq_version: int, output_dir: str, extra_args: typing.Optional[str]
) -> None:
    if artiq_version >= 4:
        build_args = [
            "--output-dir",
            str(output_dir),
            "--toolchain",
            "vivado",
            "-V",
            variant_to_build,
            "-vv",
        ]
    build_args_string = " ".join(build_args)
    gateware_module = VARIANT_TO_MODULE_DICT[variant_to_build]
    if extra_args:
        build_command = (
            f"python3 -m {gateware_module} {build_args_string} "
            f"{shlex.quote(extra_args)}"
        )
    else:
        build_command = f"python3 -m {gateware_module} {build_args_string}"
    _LOGGER.info("Building ARTIQ. This may take 15-30 mins")
    _LOGGER.debug("Building ARTIQ for %s with args: %s", variant_to_build, build_args)

    start = datetime.now()
    run_shell_command(build_command, f"artiq_{variant_to_build}_builder")
    _LOGGER.info(
        "Compilation completed in %s", str(datetime.now() - start).split(".")[0]
    )

    # Log compilation flags
    with build_dir("compilation_info.txt").open("w") as f:
        f.write(f"compiled on {datetime.now()}\n")
        f.write(f"compiled with command: {build_command}")


def _archive_flash_binaries() -> None:
    binaries_folder = build_dir("flash_binaries")
    binaries_folder.mkdir(exist_ok=True)
    # NOTE: this will copy an undetermined binary if >1 variants in build_dir().
    bootloader_file = sorted(build_dir().rglob("bootloader.bin"))
    runtime_file = sorted(build_dir().rglob("runtime.fbi"))
    gateware_file = sorted(build_dir().rglob("top.bit"))
    shutil.copy(str(bootloader_file[0]), str(binaries_folder))
    shutil.copy(str(runtime_file[0]), str(binaries_folder))
    shutil.copy(str(gateware_file[0]), str(binaries_folder))
    shutil.copy(str(build_dir("compilation_info.txt")), str(binaries_folder))
    _LOGGER.info("Moved compiled binaries to %s", str(binaries_folder))


@cli.command("flash")
@click.argument("ip_address")
@click.option("--mac", type=str, help="FPGA MAC address to set", default=None)
@click.option(
    "--log-bootup/--no-log-bootup",
    "serial_port",
    default=True,
    show_default=True,
    is_flag=True,
    help="should we log dev board bootup after flashing",
)
@click.option(
    "-t",
    "--timeout",
    default=15,
    show_default=True,
    type=click.IntRange(min=5, max=60),
    help="time to monitor FPGA serial output before ending",
)
@click.option(
    "-f",
    "--flash-dir",
    default=None,
    type=click.Path(
        exists=True, file_okay=False, dir_okay=True, readable=True, resolve_path=True
    ),
    help="(OPTIONAL): build directory with ARTIQ binaries/gateware to flash",
)
@click.option(
    "-i",
    "--idle-kernel",
    default="none",
    type=click.Path(exists=False, file_okay=True, dir_okay=False, resolve_path=True),
    help="Path of idle kernel to compile & flash (ARTIQ >=4)",
)
@click.option(
    "-sk",
    "--startup-kernel",
    default=str(
        pathlib.Path(
            _EURIQA_LIB_DIR,
            "euriqafrontend",
            "experiments",
            "bootup",
            "startup_kernel.py",
        )
    ),
    show_default=True,
    type=click.Path(exists=False, file_okay=True, dir_okay=False, resolve_path=True),
    help="Path of startup kernel to compile & flash (ARTIQ >=4)",
)
@click.option(
    "-db",
    "--device-db",
    default=str(
        pathlib.Path(
            _EURIQA_LIB_DIR, "euriqabackend", "databases", "device_db_main_box.py"
        )
    ),
    show_default=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, resolve_path=True),
    help="Path of device database used to compile & flash (ARTIQ >=4)",
)
@click.option(
    "-a",
    "--artiq-version",
    type=click.Choice(_ALLOWED_ARTIQ_VERSIONS),
    default=_ARTIQ_MAJOR_VERSION,
    help="Choose which version of ARTIQ you are using to flash",
)
def flash_cmd(
    ip_address: str,
    mac: typing.Optional[str],
    serial_port: typing.Union[str, bool],
    timeout: int,
    flash_dir: typing.Optional[str],
    artiq_version: str,
    idle_kernel: str,
    startup_kernel: str,
    device_db: str,
) -> None:
    """Flash ARTIQ to the FPGA."""
    artiq_version = int(artiq_version)
    idle_kernel = pathlib.Path(idle_kernel)

    # Check arguments
    if flash_dir is None:
        root_dir = build_dir()
    else:
        root_dir = pathlib.Path(flash_dir)
    binaries_dir = pathlib.Path(root_dir, "flash_binaries")
    if not (
        root_dir.exists()
        and root_dir.is_dir()
        and binaries_dir.exists()
        and binaries_dir.is_dir()
    ):
        raise FileNotFoundError(
            f"Binary directory {root_dir} not found. "
            "Make sure to build ARTIQ before flashing."
        )
    if re.match(r"^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$", ip_address) is None:
        raise ValueError(f"Did not provide a valid IP address: {ip_address}")
    if mac is None:
        mac = _gen_mac_address()

    # Flash
    flash_storage_file = pathlib.Path(binaries_dir, "flash_storage.img")
    storage_command = f"artiq_mkfs {flash_storage_file} -s mac {mac} -s ip {ip_address}"
    run_shell_command(storage_command, "make_storage")

    target_fpga = (
        "kc705"
        if "kc705" in VARIANT_TO_MODULE_DICT[variant_to_build].lower()
        else "kasli"
    )
    if artiq_version == 4:
        flash_command = f"artiq_flash --srcbuild {root_dir} "
    elif artiq_version >= 5:
        flash_command = f"artiq_flash -d {root_dir} --srcbuild "
    flash_command += (
        f"-V {variant_to_build} -t {target_fpga} -f {flash_storage_file} "
        "gateware bootloader firmware storage start"
    )
    run_shell_command(flash_command, "flash_artiq")
    _LOGGER.info("Flashed ARTIQ gateware to FPGA. Check above and in logs for errors.")
    _LOGGER.info("Set IP = %s, MAC = %s", ip_address, mac)

    # Log/Check bootup
    if serial_port is not None:
        if artiq_version >= 4:
            # wait for core to boot and establish network connections
            # confirm that booted
            run_shell_command(
                f"ping {ip_address} -c {timeout}", "ping_fpga", timeout=2 * timeout
            )
            # get log
            with build_dir("monitoring_bootup_stdout.log").open("w") as log:
                try:
                    log.write(CommMgmt(ip_address).get_log())
                except OSError as connection_err:
                    # shouldn't occur because ping checks that the device is there
                    # & connected
                    raise ConnectionError(
                        f"Could not connect to core device at {ip_address}. "
                        "Make sure firmware was built properly and you are "
                        "using the proper IP address"
                    ) from connection_err
            _LOGGER.info("Core device booted and connected.")

    # Flash startup/idle kernels
    if artiq_version >= 4:
        compile_kernel(idle_kernel, binaries_dir, device_db, ip_address, "idle_kernel")
        compile_kernel(
            startup_kernel, binaries_dir, device_db, ip_address, "startup_kernel"
        )


def kernel_is_valid(kernel_experiment_path: pathlib.Path) -> bool:
    """Check if kernel is a valid python file."""
    if kernel_is_none(kernel_experiment_path):
        return False
    return kernel_experiment_path.is_file() and kernel_experiment_path.suffix == ".py"


def kernel_is_none(kernel_experiment_path: pathlib.Path) -> bool:
    """Check if the path is None or ends in None (ie no file passed)."""
    return str(kernel_experiment_path).lower().endswith("none")


def compile_kernel(
    kernel_experiment_path: typing.Union[str, pathlib.Path],
    binaries_dir: pathlib.Path,
    device_db_path: str,
    ip_address: str,
    kernel_type: str,
) -> None:
    """Compile a kernel for the ARTIQ FPGA that is being flashed.

    Does nothing for kernels ending with "none".

    kernel_type: one of ["idle_kernel", "startup_kernel"].
    """
    assert kernel_type in ("startup_kernel", "idle_kernel")
    # ARTIQ 4 only
    if kernel_is_none(kernel_experiment_path):
        # Remove kernel types set as None from the coredevice
        # (i.e. if don't set idle_kernel, make sure it's not recognized/in memory)
        _LOGGER.info("Removing/disabling %s", kernel_type)
        run_shell_command(
            f"artiq_coremgmt -D {ip_address} config remove {kernel_type}",
            f"removing_{kernel_type}",
        )
        return
    kernel_experiment_path = pathlib.Path(kernel_experiment_path)
    assert kernel_experiment_path.is_file()
    try:
        kernel_output_file = pathlib.Path(binaries_dir, f"{kernel_type}.elf")
        kernel_compile_cmd = (
            f"artiq_compile {kernel_experiment_path} "
            f"--device-db={device_db_path} -vv -o {kernel_output_file}"
        )
        run_shell_command(kernel_compile_cmd, f"compiling_{kernel_type}")
        flash_kernel_cmd = (
            f"artiq_coremgmt -D {ip_address} -vv config write "
            f"-f {kernel_type} {kernel_output_file}"
        )
        run_shell_command(flash_kernel_cmd, f"flashing_{kernel_type}")
        run_shell_command(f"artiq_coremgmt -D {ip_address} -vv reboot", "reboot_kernel")
        # TODO: return kernel_experiment_path/kernel_output_file?
    except subprocess.CalledProcessError:
        _LOGGER.error(
            "Error while trying to compile & flash the %s. "
            "Device still flashed properly (will boot), but without %s.",
            kernel_type,
            kernel_type,
            exc_info=True,
        )
        warnings.warn(f"{kernel_type} not flashed")
    else:
        _LOGGER.info(
            "Flashed %s %s to core device @ %s",
            kernel_type,
            kernel_experiment_path,
            ip_address,
        )


def build_dir(*path_suffixes: typing.Sequence[str]) -> pathlib.Path:
    """Determine the build directory, optionally add a path on top of build dir."""
    version = "_".join((__version__, variant_to_build))
    return pathlib.Path("~/artiq_builds", version, *path_suffixes).expanduser()


def run_shell_command(
    command: str, name: str, notes: str = None, **kwargs
) -> subprocess.CompletedProcess:
    """Run a shell command and log output to files."""
    stdout_file = build_dir(f"{name}_stdout.log")
    stderr_file = build_dir(f"{name}_stderr.log")
    with stdout_file.open("wb") as out_log:
        with stderr_file.open("wb") as err_log:
            # TODO: log stdout/stderr to _LOGGER
            _LOGGER.debug("Running command `%s`", command)
            _LOGGER.debug(
                "Logging subcommand stdout to `%s`, stderr to `%s`",
                stdout_file,
                stderr_file,
            )
            if notes:
                spinner_text = f"Running {name} ({notes})"
            else:
                spinner_text = f"Running {name}"
            spinner = ThreadedSpinner(spinner_text + " ")
            # if in a nix shell, modify the subprocess
            # environment to use the shell's python first,
            # not any unwrapped python that might not have all packages.
            nix_shell_path = os.getenv("HOST_PATH", False)
            if nix_shell_path or os.getenv("IN_NIX_SHELL", False):
                env = dict(os.environ.items())
                env["PATH"] = nix_shell_path + env["PATH"]
            else:
                env = None
            try:
                spinner.start()
                subprocess.run(
                    shlex.split(command),
                    check=True,
                    stdout=out_log,
                    stderr=err_log,
                    env=env,
                    **kwargs,
                )
            except subprocess.CalledProcessError as err:
                _LOGGER.error(
                    "Error while running `%s` (logged to `%s` and `%s`)",
                    command,
                    stdout_file,
                    stderr_file,
                )
                raise err
            except subprocess.TimeoutExpired as err:
                _LOGGER.warning(
                    "Timed out while running `%s` (see log in `%s` and `%s`)",
                    command,
                    stdout_file,
                    stderr_file,
                )
                raise err
            else:
                _LOGGER.debug("Successfully completed command `%s`", command)
            finally:
                spinner.stop()


def _gen_mac_address() -> str:
    """Generate a random locally-administered MAC address.

    No collision guarantee.
    """
    random.seed()
    return "02:00:00:{:02X}:{:02X}:{:02X}".format(
        random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
    )


class ThreadedSpinner:
    """Threads spinning so it can happen while doing some other task."""

    def __init__(self, label: str, delay: float = 0.3):
        """Create a spinner."""
        self.spinner_generator = Spinner(label, suffix="%(elapsed)d")
        self.busy = False
        self.delay = delay

    def spinner_task(self):
        """Run main loop for the spinner."""
        while self.busy:
            self.spinner_generator.next()
            time.sleep(self.delay)

    def start(self):
        """Start spinning."""
        self.busy = True
        threading.Thread(target=self.spinner_task).start()

    def stop(self):
        """Close spinner and restore state."""
        self.busy = False
        self.spinner_generator.finish()
        print("")  # new line
        time.sleep(self.delay)


if __name__ == "__main__":
    cli()  # pylint: disable=no-value-for-parameter
