{ pkgs ? import <nixpkgs> { config.allowUnfree = true; } }:

let
  artiq-full = import <artiq-full> { inherit pkgs; };
  dax-full = import <dax-full> { inherit pkgs; };
  drew-nur = import <drew-nur> { inherit pkgs; };
  
  #oitg, numdifftools
  nur_packages = import (
    pkgs.fetchFromGitHub {
      owner = "yuyichao";
      repo = "nur-packages";
      rev = "ab62fd72f4781f474203b9a5e91a581fcde67fae";
      sha256 = pkgs.lib.fakeSha256; # Note_1! This will produce a fail, but the error message will contain the functioning hash. Replace.
    } { inherit pkgs; }
  );

in
pkgs.mkShell {
  buildInputs = [
    # Python packages
    (pkgs.python3.withPackages (ps: [

      # DAX
      dax-full.dax # Includes ARTIQ
      dax-full.dax-comtools
      dax-full.dax-comtools-private
      dax-full.dax-applets

      # RFSoC packages
      #nur_packages.python3Packages.symengine.override (args: { ignoreCollisions = true; })
      #(nur_packages.python3Packages.qiskit-terraNoVisual.overridePythonAttrs(oldAttrs: {
      #  patches = [ ./nix/0001-Work-around-artiq-bug.patch ] ++ (oldAttrs.patches or []);
      #  doCheck = false;
      #  })
      #).override (args: { ignoreCollisions = true; })

      drew-nur.pulsecompiler
      drew-nur.pulsecompiler-docs
      drew-nur.dax-pulse
      
      (nur_packages.python3Packages.oitg.overridePythonAttrs(oldAttrs: {
        patches = [(
          # patch a bug in the linear fitting routine w/ 0-indexing vs 1-indexing
          # https://github.com/OxfordIonTrapGroup/oitg/pull/42
          pkgs.fetchpatch {
            name = "pr-42-fix-linear-fit-indexing.patch";
            url = "https://github.com/OxfordIonTrapGroup/oitg/commit/294a22d4366696156c372fb480822b483c5e6d1d.patch";
            sha256 = "sha256-KoJYWzSEqfu8G66kV6TgmOAeTKRUUeiw7tOwfnaZavw=";
          })] ++ (oldAttrs.patches or []);
        })
      )
      (nur_packages.python3Packages.numdifftools.overridePythonAttrs (oldAttrs: { doCheck = false; }))
      pkgs.python3Packages.uncertainties

      # Other packages
      artiq-full.artiq-comtools
      ps.paramiko # needed for flashing boards remotely (artiq_flash -H)
      ps.pandas
      ps.pyqt5

      # Gateware packages
      dax-full.dax-gateware

      # Packages for testing
      ps.pytest
      dax-full.flake8-artiq

    ]))
    # Non-Python packages
    artiq-full.openocd # needed for flashing boards, also provides proxy bitstreams
    # Packages required for testing
    artiq-full.binutils-or1k
  ];
  #shellHook = ''
  #  # Create a link to the current environment in the nix store
  #  if [ -z $SHELL_ENV_PATH ]; then
  #  SHELL_ENV_PATH=~/.brassboard_shell_env
  #  fi
  #  ln -sfTn $(realpath $(dirname $(which python3))/../) $SHELL_ENV_PATH
  #  # Suppresses GTK and QT warnings
  #  export QT_XCB_GL_INTEGRATION=none
  #  export LC_ALL=C
  #'';
  preShellHook = ''
  PYTHONPATH=/home/henryluo/git/local-lib/:$PYTHONPATH
  export PYTHONPATH
  '';
}
