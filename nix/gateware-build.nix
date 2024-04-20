# This needs to be run with --option sandbox false
# e.g. nix-build gateware-build.nix --option sandbox false
{ sources ? import ./sources.nix { }
, pkgs ? import sources.nixpkgs { }
}:

let
  lib = pkgs.lib;
  artiq-board-build-original = import (artiqpkgs.artiq-fast-patched-source + "/artiq-fast/artiq-board.nix") {
    inherit pkgs;
    vivado = pkgs.callPackage ./build-fpga/vivado.nix { vivadoPath = "/opt/Xilinx/Vivado/2019.1"; };
  };
  artiqpkgs = pkgs.callPackage ./artiqpkgs.nix { };
  euriqa = pkgs.callPackage ../default.nix { };
  unwantedPackages = name: value: !(
    lib.hasInfix "vivado" name
  );
  overrideArtiqBuild = deriv: deriv.overrideAttrs(oldAttrs: rec {

  });
  artiqBuild = {target, variant, buildCommand, ...}@args: (artiq-board-build-original { inherit target variant buildCommand args; }).overrideAttrs(oldAttrs: rec {
    buildInputs = [ (pkgs.python3.withPackages(ps:
      (with ps; [ jinja2 numpy])
      ++ (with artiqpkgs; [ artiq jesd204b migen microscope misoc ])
      ++ [ euriqa ]
    ))];
    doCheck = true;
  });
in
{
  mainkc705build= artiqBuild rec {
    target = "kc705";
    variant = "euriqasandiadac";
    buildCommand = "python -m euriqabackend.gateware.kc705_soc -V ${variant}";
  };
}
