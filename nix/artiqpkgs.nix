{ sources ? import ./sources.nix {}
, pkgs ? import sources.nixpkgs {}
}:

# See the README for update instructions for ARTIQ.
# You can build your own version of ARTIQ & its accompanying packages w/o M-Labs with this script:
# $ nix-build ./artiqpkgs.nix
let
  lib = pkgs.lib;
  # Generate the ARTIQ version. It's SIMILAR TO the official ARTIQ version, though not 100% matching.
  # Major difference: no beta flag
  artiqVersion = "${lib.strings.fileContents "${sources.artiq}/MAJOR_VERSION"}.${toString sources.artiq.outPath.revCount}.${toString sources.artiq.outPath.shortRev}";
  # Patch files to pin the ARTIQ source directory & version according to niv's sources (see sources above)
  artiq-fast-patched-source = pkgs.applyPatches {
    name = "artiq-fast-build-patched-ARTIQ-${artiqVersion}";
    src = sources.mlabs-nix-scripts;

    patches = [
      (pkgs.substituteAll {
        src = ./pin-artiq-source.patch;
        artiqSrc = sources.artiq;
        inherit artiqVersion;
      })
    ];
  };
  artiq-full-patched = pkgs.applyPatches {
    name = "artiq-full-build-patched-with-artiq-fast-${artiqVersion}";
    src = sources.mlabs-nix-scripts;
    # Don't actually patch, just move artiq-fast (patched) to the directory artiq-full expects.
    postPatch = ''
      cp -r ${artiq-fast-patched-source}/artiq-fast ./artiq-full/fast
    '';
  };

  # Actual ARTIQ build. This corresponds to the <artiq-fast> conda channel on M-Labs's nix buildserver.
  artiq-fast = import "${artiq-fast-patched-source}/artiq-fast/default.nix" { inherit pkgs; };

  # Build tools like artiq-comtools. This gets a few unneeded packages, but it's simple enough
  artiq-extras = import (artiq-full-patched + "/artiq-full/extras.nix") {
    inherit pkgs;
    inherit (artiq-fast) sipyco asyncserial artiq;
  };
  filterUnwantedArtiqPackages = name: value: !(
    # Skips Conda, ARTIQ boards (i.e. the gateware for various labs), and manuals
    lib.hasInfix "conda" name ||
    lib.hasInfix "artiq-board" name ||
    lib.hasInfix "manual-html" name
  );
in
  # Remove unneeded ARTIQ packages that M-Labs builds. We're mostly just interested in vanilla ARTIQ, sipyco, artiq-comtools
  (lib.filterAttrs filterUnwantedArtiqPackages (artiq-fast // artiq-extras)) // {
    inherit artiq-fast artiq-fast-patched-source;
  }
