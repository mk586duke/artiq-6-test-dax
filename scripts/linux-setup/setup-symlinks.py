#! /usr/bin/env nix-shell
#! nix-shell -E "with import <nixpkgs> {}; (python3.withPackages(ps: with ps; [ click ])).env" -i python
"""Set up the symlinks used in this repository for the Control PC.

These are primarily the pre-commit & result symlinks
(i.e. results stored to NAS automatically).
"""
import pathlib
import shlex
import subprocess
import textwrap

import click


_this_dir = pathlib.Path(__file__).parent


def _symlink(
    source: pathlib.Path, destination: pathlib.Path, use_sudo: bool = False
) -> None:
    """Wrapper around the Linux symlink utility."""
    cmd = "ln -s {source} {destination}".format(source=source, destination=destination)
    if use_sudo:
        cmd = "sudo " + cmd
    click.echo("Creating symlink '{}' to file '{}'".format(destination, source))
    subprocess.run(shlex.split(cmd), check=True)


def _add_results_nas_symlink():
    """Add symlink from ARTIQ results folder to save directly to the NAS drive."""
    if click.confirm(
        "Add a symlink to store results directly to the NAS?", default=True,
    ):
        click.echo(
            "Please make sure that the EURIQA NAS is mounted "
            "& set to auto-mount on startup (check `/etc/fstab`)"
        )
        nas_path = pathlib.Path(
            click.prompt(
                "Where is the NAS mounted locally?",
                type=str,
                default="/media/euriqa-nas/",
            )
        )
        results_path = (_this_dir / "../../results").resolve()
        nas_folder = nas_path / click.prompt(
            "Where on the NAS would you like to store the results?",
            type=str,
            default="CompactTrappedIonModule/Data/artiq_data/",
            confirmation_prompt=True,
        )
        if not (nas_folder.exists() and nas_folder.is_dir() and nas_path.is_mount()):
            raise click.ClickException(
                "NAS Path '{}' not valid: either does not exist, is not a directory, "
                "or is not an external mounted directory".format(nas_folder)
            )
        _symlink(nas_folder, results_path)


def _add_precommit_author_check():
    """Install the Git pre-commit check to disable generic commit authors."""
    if click.confirm("Install the pre-commit author check?", default=True):
        git_path = (_this_dir / "../../.git").resolve()
        pre_commit_path = git_path / "hooks" / "pre-commit"
        pre_commit_script = (_this_dir / "../pre-commit").resolve()
        assert not pre_commit_path.is_file(), "Destination {} already exists".format(
            pre_commit_path
        )
        assert pre_commit_script.is_file(), "Script '{}' not found".format(
            pre_commit_script
        )
        _symlink(pre_commit_script, pre_commit_path)
        # before committing a merge, run the pre-commit check
        # Uses default pre-merge-commit script
        pre_merge_script = pre_commit_path.with_name("pre-merge-commit.sample")
        pre_merge_new_path = pre_merge_script.with_name("pre-merge-commit")
        _symlink(pre_merge_script, pre_merge_new_path)


def _add_artiq_influx_service():
    """Installs the ARTIQ InfluxDB background process by symlinking to .service file."""
    if click.confirm("Install the ARTIQ InfluxDB connector?", default=True):
        influx_script_path = (_this_dir / "artiq_influxdb.service").resolve()
        assert influx_script_path.is_file(), "Service file {} not found".format(
            influx_script_path
        )
        systemd_install_path = (
            pathlib.Path("/etc/systemd/system/") / influx_script_path.name
        )
        assert (
            not systemd_install_path.exists()
        ), "Install location {} already exists".format(systemd_install_path)
        _symlink(influx_script_path, systemd_install_path, use_sudo=True)
        click.echo("ARTIQ InfluxDB connector installed to systemd")
        click.secho(
            textwrap.dedent(
                """
                To finish installation, you must run
                    $ sudo systemctl daemon-reload
                    $ sudo systemctl enable artiq_influxdb.service
                    $ sudo systemctl restart artiq_influxdb.service
                """
            ),
            fg="yellow",
        )


@click.command()
def cli():
    """Set up the symlinks for the pre-commit check & storing results to the NAS.

    This will prompt you to set up the appropriate symlinks.
    """
    # ARTIQ results symlink
    _add_results_nas_symlink()

    # Pre-commit hook
    _add_precommit_author_check()

    # ARTIQ InfluxDB Service
    _add_artiq_influx_service()


if __name__ == "__main__":
    cli()
