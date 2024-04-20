{ pkgs }:

rec {
  oitg =
    let
      src = import (fetchGit {
        url = "git@github.com:OxfordIonTrapGroup/oitg.git";
        rev = "a968246ad228c206153a4c5391808517051cf284";
      });
    in
      pkgs.callPackage src { inherit pkgs; };
}
