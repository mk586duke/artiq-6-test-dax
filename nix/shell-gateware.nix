{ sources ? import ./sources.nix {}
, pkgs ? import sources.nixpkgs {}
, checkVivadoPath ? true
}:
# This Nix shell is designed for building & flashing the gateware for ARTIQ boards.
# It includes everything in a standard EURIQA shell, with ARTIQ gateware dependencies & vivado

# this should be kept somewhat in sync with M-Lab's nix scripts: https://git.m-labs.hk/M-Labs/nix-scripts

# To build & flash gateware, you can run ``nix-shell ./shell-gateware.nix --run "build_artiq build flash"``
let
  vivado = import ./build-fpga/vivado.nix { vivadoPath = "/opt/Xilinx/Vivado/2019.1"; checkPath = checkVivadoPath; };
  euriqa = pkgs.callPackage ../default.nix {};
  artiqpkgs = pkgs.callPackage ./artiqpkgs.nix {};
in
pkgs.mkShell {
  buildInputs = [
    vivado
    pkgs.gnumake
    (
      pkgs.python3.withPackages (ps:
        (with ps; [ jinja2 numpy paramiko pylint ])
        ++ (with artiqpkgs; [ artiq jesd204b microscope migen migen-axi misoc ])
        ++ [ euriqa ]
      )
    )
  ] ++ (with artiqpkgs; [ binutils-arm binutils-or1k llvm-or1k openocd rustc cargo-legacy ]);
  TARGET_AR = "or1k-linux-ar";

  # Environment variables for the build_artiq script. Used for setting defaults for building the gateware
  ARTIQ_BUILD_BUILD_VARIANT = "euriqasandiadac";
}
