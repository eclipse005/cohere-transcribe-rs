//! Break down the 26ms/step: how much is attention vs layernorm vs allocation?
#![cfg(feature = "cuda")]
use std::time::Instant;
use half::f16;
use native_transcribe::backend::{Backend, CpuBackendF16};

fn ms(t: std::time::Duration) -> f64 { t.as_secs_f64() * 1e3 }

#[test]
fn op_breakdown() -> anyhow::Result<()> {
    let b = CpuBackendF16::new();
    let iters = 200;
    let h = 8usize; let hd = 128; let d = h*hd;
    let enc = 187usize;

    // attention_qk M=1 (self, k_seq grows ~ use 100 as avg)
    let q = b.alloc_uninit(h*hd)?;
    let kcache = b.alloc_uninit(h * 256 * hd)?;
    let t = Instant::now();
    for _ in 0..iters { let _ = b.attention_qk(&q, &kcache, h, 1, 100, hd, 256, 0.088)?; }
    eprintln!("attention_qk M=1 (k_seq=100): {:.4} ms", ms(t.elapsed())/iters);

    // attention_av M=1
    let a = b.alloc_uninit(h*100)?;
    let vcache = b.alloc_uninit(h * 256 * hd)?;
    let t = Instant::now();
    for _ in 0..iters { let _ = b.attention_av(&a, &vcache, h, 1, 100, hd, 256)?; }
    eprintln!("attention_av M=1 (k_seq=100): {:.4} ms", ms(t.elapsed())/iters);

    // cross attention qk (enc_tokens=187)
    let t = Instant::now();
    for _ in 0..iters { let _ = b.attention_qk(&q, &kcache, h, 1, enc, hd, 256, 0.088)?; }
    eprintln!("cross attention_qk M=1 (k_seq=187): {:.4} ms", ms(t.elapsed())/iters);

    // softmax_last_dim
    let s = b.alloc_uninit(h*100)?;
    let t = Instant::now();
    for _ in 0..iters { let _ = b.softmax_last_dim(&s, h, 100)?; }
    eprintln!("softmax_last_dim (h*100): {:.4} ms", ms(t.elapsed())/iters);

    // layer_norm [1, 1024]
    let w = b.alloc_uninit(d)?;
    let bb = b.alloc_uninit(d)?;
    let x = b.alloc_uninit(d)?;
    let t = Instant::now();
    for _ in 0..iters { let _ = b.layer_norm(&x, &w, &bb, 1, d, 1e-5)?; }
    eprintln!("layer_norm [1,1024]: {:.4} ms", ms(t.elapsed())/iters);

    // merge_heads_single + split_qkv_step_cached
    let qkv = b.alloc_uninit(3*d)?;
    let mut kc = b.alloc_uninit(h*256*hd)?;
    let mut vc = b.alloc_uninit(h*256*hd)?;
    let t = Instant::now();
    for _ in 0..iters { let _ = b.split_qkv_step_cached(&qkv, &bb, &mut kc, &mut vc, 10, 256, h, hd)?; }
    eprintln!("split_qkv_step_cached: {:.4} ms", ms(t.elapsed())/iters);

    // bias_residual [1,1024]
    let t = Instant::now();
    for _ in 0..iters { let _ = b.bias_residual(&x, &bb, &x, d, d)?; }
    eprintln!("bias_residual [1024]: {:.4} ms", ms(t.elapsed())/iters);

    Ok(())
}
