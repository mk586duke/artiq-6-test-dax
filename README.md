# euriqa-artiq-backend

The backend-type code used for running experiments on the ARTIQ platform.

## Installation

Download this repository with the command `git clone --recurse-submodules URL`, because it includes submodules.
It can be installed with the command (for Conda, run from within the git directory) `pip install .`.
This project requires Python 3.6+. Or it can be run with Nix (see [Nix README](./nix/README.md)).

### Installing with Conda

Prebuilt conda environments with certain configurations can be found
in the [./conda/](./conda/) folder here. These are sample environments,
and might not meet your exact needs.
The main versions are in the filename. These can be installed using
``conda env create -f FILENAME.yml``

These files can also be created with
``conda env export > FILENAME.yml``

## Development

If developing (i.e. committing), you should run the command `pre-commit install` as soon as you download.
To manually run all pre-commit checks, use `pre-commit run --all-files`.
The first time you use this command, it will take ~10-15 mins to install all the dependencies.
This also requires that you have `pylint` installed in your environment.
NOTE: ``pre-commit`` should be installed from an environment with ``python>=3.6``.

If developing with this, you can/should use the ``nix-shell`` environment, basically mandated by ARTIQ >= 5.
This can be launched with ``nix-shell $EURIQA_DIR/shell.nix -j auto``.
See [Nix README](./nix/README.md) for more details.

### Committing

This repository uses pre-commit hooks, which don't allow committing until certain checks are met, such as formatting and code errors.
To do this seamlessly, you should commit from the command line, using the same environment that your code is running in. i.e. `conda activate artiq` -> `git commit` from this repository's directory.
See documentation on `conda` or `git` if you need help using their command line tools.

If you are unable to commit, it is probably because you failed tests.
If pylint/flake8/pydocstyle failed, you can see the relevant output.
Other tools, like formatters, will fix the files, and then you should retry committing (`git add ... && git commit`).

## Documentation

Documentation is in the `/docs/` folder. To launch it, go to [/docs/_build/html/index.html](/docs/_build/html/index.html).
If it does not exist, then you need to build the documentation yourself.
Make sure `sphinx` is installed (`pip/conda install sphinx`), then navigate to [docs](docs/) and run `make html`.

If you have added new modules, you will need to update the API documentation so they are recognized & built.
Navigate to [docs](docs/) and run `make api`.
