use sha2::{Digest, Sha256};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

const ALGORITHM_ID: &str = "sha256-framed-source-bundle-v1";
const DOMAIN_TAG: &[u8] = b"stl-rust-source-bundle-v1\0";

fn hash_entry(hasher: &mut Sha256, label: &str, path: &Path) {
    let label = label.as_bytes();
    let contents = fs::read(path).unwrap_or_else(|error| {
        panic!(
            "failed to read source-bundle input {}: {error}",
            path.display()
        )
    });
    let label_len = u64::try_from(label.len()).expect("source-bundle label length exceeds u64");
    let content_len =
        u64::try_from(contents.len()).expect("source-bundle input length exceeds u64");

    hasher.update(label_len.to_be_bytes());
    hasher.update(label);
    hasher.update(content_len.to_be_bytes());
    hasher.update(contents);
}

fn main() {
    let crate_root = PathBuf::from(
        env::var_os("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR is required by Cargo"),
    );
    let workspace_lock = crate_root.join("../../..").join("Cargo.lock");
    let entries = [
        ("Cargo.toml", crate_root.join("Cargo.toml")),
        ("build.rs", crate_root.join("build.rs")),
        ("src/lib.rs", crate_root.join("src/lib.rs")),
        ("Cargo.lock", workspace_lock),
    ];

    // sha256-framed-source-bundle-v1 hashes DOMAIN_TAG (including its trailing
    // NUL), then the entries above in exactly that order. Each entry is framed
    // as u64-BE label byte length, UTF-8 label bytes, u64-BE content byte
    // length, and raw file bytes. No path or newline normalization is applied.
    let mut hasher = Sha256::new();
    hasher.update(DOMAIN_TAG);
    for (label, path) in &entries {
        println!("cargo:rerun-if-changed={}", path.display());
        hash_entry(&mut hasher, label, path);
    }

    let digest = format!("{:x}", hasher.finalize());
    println!("cargo:rustc-env=SOURCE_BUNDLE_DIGEST={digest}");
    println!("cargo:rustc-env=SOURCE_BUNDLE_DIGEST_ALGORITHM={ALGORITHM_ID}");

    // Python extension modules resolve CPython symbols from the host process on
    // macOS.  This crate also emits an rlib so its numerical kernel can be unit
    // tested by `cargo test --workspace`, where Cargo otherwise omits the
    // dynamic-lookup linker mode used by maturin.
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        println!("cargo:rustc-link-arg=-undefined");
        println!("cargo:rustc-link-arg=dynamic_lookup");
    }
}
