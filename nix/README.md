# Nix Scripts for EURIQA's ARTIQ extension

Nix basically provides a package manager & environment manager in one.
In EURIQA, we use it to provide deterministic environments for our software,
basically allowing everyone to work from the same software environment and stay in sync
via Git.

Here are the basics of what you need to know:

## SETUP

1. Install Nix with ``curl -L https://nixos.org/nix/install | sh``

2. Follow the ARTIQ instructions for installing ARTIQ:
    * Installing ARTIQ via Nix: https://m-labs.hk/artiq/manual/installing.html#installing-nix-users
        * Follow this until it asks you to install ARTIQ in your environment or create a nix file.
        You will be using the EURIQA nix shell file in this directory.
    * (OPTIONAL) Developing gateware (needed for building & flashing gateware):
        https://m-labs.hk/artiq/manual/developing.html
3. Use Drew's Nix package cache for some quantum packages:
    * Install Cachix: https://docs.cachix.org/installation.html
    * Use [Drew's NUR](https://github.com/drewrisinger/nur-packages) Cachix cache: ``cachix use drewrisinger``
        * NOTE: Cachix free version has limited storage space.
            Nix will cache the packages used locally,
            but if those are lost or this is run on a new computer, and Cachix no longer
            stores the version you desired, then it might take a while to rebuild the
            packages you requested from source. This should only need done once though.
4. Allow building Unfree packages:
    ```bash
    mkdir -p ~/.config/nixpkgs/
    echo "{ allowUnfree = true; }" > ~/.config/nixpkgs/config.nix
    ```

## Running

To launch a conda-like shell where the ``euriqa[back/front]end`` packages are available (along with ARTIQ), you can run
``nix-shell ./shell.nix -j auto``.
After first launch, the ``-j auto`` flag is optional, it is used to speed building.

You can then launch ``artiq_master``, ``artiq_dashboard``, etc from there.

The conda environment method should still work as long as M-Labs supports it,
and is the **only** method for Windows-based PCs.

## Updating ARTIQ/Python Packages

We have locked the version of ARTIQ & all dependent python packages using Nix.
The dependent python packages live either on:
* nixpkgs (community-supported)
* M-Lab's ARTIQ Nix buildserver: https://nixbld.m-labs.hk/channel/custom/artiq/full/artiq-full
* Drew Risinger's NUR (Nix User Repository) repo: https://github.com/drewrisinger/nur-packages

To update a given package (say, Qiskit), you must first trace that package's origin
(try looking in [default.nix](../default.nix)), and then find & update the Nix expression
for that package to the latest version. Once that commit has been accepted to the
appropriate repository, then you can change to use that Nix expression in this repository.

i.e. to increase the Qiskit version, which lives in Drew's NUR repo, you would:
1. Commit the Nix expression for a new Qiskit version to Drew's NUR repo. Currently it must make it to the master branch via a Pull Request, but a specific commit can be chosen using Niv.
2. Update [./sources.json](./sources.json) to use the updated NUR repo by running niv:
    ``nix-shell -p niv --run "niv update"``.
    If you want to update across branches (i.e. ARTIQ 5 -> 6), then you need to do
    something like ``nix-shell -p niv --run "niv modify artiq -a ref=release-6 && niv update artiq"``
3. Test the new changes: start a shell, run ARTIQ, run experiments, etc.
4. Commit the new changes.

A similar process can be used for updating the ARTIQ or Nixpkgs repositories.

## Development

### Setup for building Gateware

To get Xilinx to run properly, you must change your Nix config file to allow Xilinx to write to sandboxed paths.

In ``/etc/nix/nix.conf``, insert ``extra-sandbox-paths = /opt/Xilinx``.
NOT ``sandbox-paths = ...``, which can break some builds.

### Building Gateware

On a PC that has Vivado 2019.1, from ``$EURIQA_DIR`` run
``nix-shell ./nix/shell-gateware.nix -j auto``.
The ``build_artiq`` script should work from this environment ONLY.

If you modify the requirements of the EURIQA package, you need to modify the [euriqa nix](../default.nix)
dependencies manually.
It'd be great if it's in the standard Nix Python library (``nix search PACKAGE``),
but otherwise you can try using ``pypi2nix`` on each dependency. It's not ideal, but works
well enough. Drew had to do this for Google's Cirq when setting this up.

## TODO: turn computer into build server with nix-store --serve
