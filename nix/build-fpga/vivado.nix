# Install Vivado in /opt and add to /etc/nixos/configuration.nix:
#  nix.sandboxPaths = ["/opt"];
# Or in /etc/nix/nix.conf (non-NixOS):
#   extra-sandbox-paths = /opt/Xilinx/

{ sources ? import ../sources.nix { }
, pkgs ? import sources.nixpkgs {}
, vivadoPath ? "/opt/Xilinx/Vivado/2019.1"  # must be string b/c otherwise tries to hash entire Vivado dir, SLOW/memory-intensive
, checkPath ? true
}:

# I tried to run this with nix-build --argstr vivadoPath "PATH_TO_VIVADO", but that doesn't seem to work. See GitHub https://github.com/NixOS/nix/issues/598 (repl-only, but still issue in nix-build)
# So the obvious way of customizing this, without passing down tons of unnecessary args, doesn't seem to work. Not really sure how to use overrides.
assert (!checkPath || (builtins.pathExists vivadoPath));
(
  pkgs.buildFHSUserEnv {
    name = "vivado";
    targetPkgs = pkgs: (
      with pkgs; [
        ncurses5
        zlib
        libuuid
        glibcLocales
        xorg.libSM
        xorg.libICE
        xorg.libXrender
        xorg.libX11
        xorg.libXext
        xorg.libXtst
        xorg.libXi
      ]
    );
    profile = ''
      source ${vivadoPath}/settings64.sh
      export LOCALE_ARCHIVE_2_27="${pkgs.glibcLocales}/lib/locale/locale-archive"   # Fixes Vivado run issue using https://github.com/NixOS/nixpkgs/issues/38991#issuecomment-496332104
    '';
    runScript = "vivado";
  }
)
# .env  # uncomment to try to launch Vivado as an application
