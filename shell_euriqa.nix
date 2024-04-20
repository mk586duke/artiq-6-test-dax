{ sources ? import ./nix/sources.nix {}
, pkgs ? import sources.nixpkgs {}
}:

# NOTE: this starts the shell with the python package installed in DEVELOPMENT mode.
# This means that any changes that you make to the package will be immediately reflected
# in the code that is run (maybe have to restart python shell)
# If this is NOT desired, then a separate shell using nix's mkShell (see ARTIQ install instructions)
# should be created
let
  euriqaPackage = pkgs.callPackage ./default.nix {};
  artiqpkgs = pkgs.callPackage ./nix/artiqpkgs.nix {};
  niv = (import sources.niv {}).niv;
in
pkgs.mkShell {
  # This is the key for development mode.
  # We get the inputs we would install otherwise, then install it in development mode w/ setuptoolsBuildHook
  inputsFrom = [ euriqaPackage ];
  buildInputs = [
    (pkgs.python3.withPackages(ps: [
      # euriqaPackage # NOTE: We DON'T want this. Putting this here would NOT install it in development mode.
      # See above inputsFrom
      artiqpkgs.artiq-comtools

      # Development tools
      ps.black
      ps.flake8
      ps.pycodestyle
      ps.pylint
      ps.pytest
      artiqpkgs.flake8-artiq
    ]))
    niv # for version management of e.g. ARTIQ, drewrisinger-nur, pulsecompiler, etc.
    pkgs.git
  ];
  nativeBuildInputs = [ pkgs.python3Packages.setuptoolsBuildHook ];
  inherit (euriqaPackage) preShellHook postShellHook;
}
