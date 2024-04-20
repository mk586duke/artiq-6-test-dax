{ sources ? import ./sources.nix {}
, nixpkgs ? sources.nixpkgs
}:

# Alternative python environment including jupyter. Used for e.g. submitting circuits
# Can be launched with ``nix-shell ./shell-jupyter.nix --command "jupyter lab"``

let
  euriqa = pkgs.callPackage ../default.nix {};
  drewrisinger-nur = pkgs.callPackage sources.drewrisinger-nur-packages { };
  artiqpkgs = pkgs.callPackage ./artiqpkgs.nix {};

  # Importing overlays from JupyterWith.
  overlays = [
    # Necessary for Jupyter
    (import "${sources.jupyterwith}/nix/python-overlay.nix")
    (import "${sources.jupyterwith}/nix/overlay.nix")
  ];

  # Overlay nixpkgs with the JupyterWith dependencies.
  # Allows completely reproducible jupyter environments
  pkgs = import nixpkgs { inherit overlays; };

  jupyter = pkgs.jupyterWith;

  # Create an "ipython kernel", basically a python virtual environment with
  # EURIQA package, its dependencies, and anything else you want.
  euriqaIPythonKernel = jupyter.kernels.iPythonWith {
    name = "EURIQA_python_environment";
    packages = p: with p; [ euriqa artiqpkgs.artiq-comtools ];
  };

  jupyterEnvironment =
    jupyter.jupyterlabWith {
      kernels = [ euriqaIPythonKernel ];
    };
in
  jupyterEnvironment.env
