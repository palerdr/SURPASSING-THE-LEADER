fn main() {
    // Python extension modules resolve CPython symbols from the host process on
    // macOS.  This crate also emits an rlib so its numerical kernel can be unit
    // tested by `cargo test --workspace`, where Cargo otherwise omits the
    // dynamic-lookup linker mode used by maturin.
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        println!("cargo:rustc-link-arg=-undefined");
        println!("cargo:rustc-link-arg=dynamic_lookup");
    }
}
