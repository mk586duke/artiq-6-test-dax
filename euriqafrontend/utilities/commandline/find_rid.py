"""Command-line tool to search for an experiment data file in a directory.

Searches by RID (Run ID #).
"""
import logging

import click

import euriqafrontend.utilities.run_id as rid

_LOGGER = logging.getLogger(__name__)


@click.command()
@click.argument(
    "base_path",
    type=click.Path(exists=True, dir_okay=True, readable=True, resolve_path=True),
)
@click.argument("run_id", type=int)
def cli(base_path: str, run_id: int) -> None:
    """Given a directory, find an experiment data file (by run ID)."""
    logging.basicConfig(level=logging.INFO)
    click.echo(rid.find_rid_in_path(run_id, base_path))


if __name__ == "__main__":
    cli()  # pylint: disable=no-value-for-parameter
