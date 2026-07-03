//! Microbenchmark: measure per-iteration cost of alloc+free on the default
//! stream. Tells us whether cuMemAllocAsync pooling makes repeated same-size
//! allocs cheap (pool hit) or not (real driver call each time).
//!
//! Run: cargo run --release --features cuda --example check_pool

#![cfg(feature = "cuda")]

use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let cuda = native_transcribe::engine::CudaState::new(0)?;

    for (label, n) in [
        ("1280*64 (LN out)", 1280 * 64),
        ("5120*64 (FFN mid)", 5120 * 64),
        ("3840*64 (QKV)", 3840 * 64),
    ] {
        const ITERS: usize = 3000;
        // Warm the pool
        for _ in 0..50 {
            let _b = cuda.alloc_zeros_f16(n)?;
        }
        cuda.synchronize()?;
        let t0 = Instant::now();
        for _ in 0..ITERS {
            let _b = cuda.alloc_zeros_f16(n)?;
        }
        cuda.synchronize()?;
        let per = t0.elapsed().as_secs_f64() / ITERS as f64 * 1e6;
        println!("{label:20} n={n:7}: {per:.3} us/alloc (alloc_zeros)");
    }

    // Also time alloc_uninit (no memset) for the typical LN-out size.
    for (label, n) in [("1280*64 (LN out)", 1280 * 64)] {
        const ITERS: usize = 3000;
        let t0 = Instant::now();
        for _ in 0..ITERS {
            let _b = cuda.alloc_uninit_f16(n)?;
        }
        cuda.synchronize()?;
        let per = t0.elapsed().as_secs_f64() / ITERS as f64 * 1e6;
        println!("{label:20} n={n:7}: {per:.3} us/alloc (alloc_uninit)");
    }

    Ok(())
}
