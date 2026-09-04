{
  description = "Telegram style-model training environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    {
      self,
      nixpkgs,
      rust-overlay,
    }:
    let
      supportedSystems = [ "x86_64-linux" ];
      forAllSystems = nixpkgs.lib.genAttrs supportedSystems;
    in
    {
      devShells = forAllSystems (
        system:
        let
          overlays = [ (import rust-overlay) ];
          pkgs = import nixpkgs {
            inherit system overlays;
            config.allowUnfree = true;
          };
          rustToolchain = pkgs.rust-bin.stable.latest.default.override {
            extensions = [ "rust-src" ];
          };
        in
        {
          default = pkgs.mkShell {
            packages = with pkgs; [
              rustToolchain
              pkg-config
              python312
              uv
            ];

            LD_LIBRARY_PATH =
              pkgs.lib.makeLibraryPath [
                pkgs.cudaPackages.cuda_cudart
                pkgs.stdenv.cc.cc.lib
                pkgs.zlib
              ]
              + ":/run/opengl-driver/lib";
            UV_PYTHON = "${pkgs.python312}/bin/python";
            RUST_SRC_PATH = "${rustToolchain}/lib/rustlib/src/rust/library";

            shellHook = ''
              uv sync
              source .venv/bin/activate
            '';
          };
        }
      );
    };
}
