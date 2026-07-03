//! Isolated correctness test for `rel_shift_rank3_f16` — the trickiest
//! Conformer kernel (Transformer-XL relative-position shift). Compares the
//! GPU kernel against the exact CPU reference (pad col0 -> reshape -> narrow
//! row0 -> reshape) that candle's `rel_shift_rank3` implements, on a small
//! random [heads, q_len, pos_len] input.

#![cfg(feature = "cuda")]

use half::f16;
use native_transcribe::engine::CudaState;

/// CPU reference: candle `rel_shift_rank3` semantics on [heads, q_len, pos_len]
/// with pos_len = 2*q_len - 1.
fn cpu_rel_shift(bd: &[f32], heads: usize, q_len: usize, pos_len: usize) -> Vec<f32> {
    assert_eq!(pos_len, 2 * q_len - 1, "pos_len must be 2*q_len-1");
    let mut out = vec![0f32; heads * q_len * pos_len];
    for h in 0..heads {
        for i_dst in 0..q_len {
            for j in 0..pos_len {
                // Inverse of candle's pad->reshape->narrow->reshape transform.
                // Step 1: target (i_dst, j) -> flat2 in the narrowed [pos_len, q_len] layout.
                let flat2 = i_dst * pos_len + j;
                // Step 2: decompose into (R', C) in the [pos_len, q_len] view after narrow.
                let rp = flat2 / q_len; // R' in [0, pos_len)
                let c = flat2 % q_len;
                // Step 3: undo narrow — R = R' + 1.
                let r = rp + 1;
                // Step 4: flatten to the padded [q_len, pos_len+1] layout.
                let flat = r * q_len + c;
                // Step 5: recover source (i_src, slot).
                let i_src = flat / (pos_len + 1);
                let slot = flat % (pos_len + 1);
                let val = if slot == 0 {
                    0.0f32 // pad column
                } else {
                    let k = slot - 1;
                    bd[(h * q_len + i_src) * pos_len + k]
                };
                out[(h * q_len + i_dst) * pos_len + j] = val;
            }
        }
    }
    out
}

#[test]
fn rel_shift_matches_cpu_reference() -> anyhow::Result<()> {
    // Small but non-trivial: heads=2, q_len=3 -> pos_len=5.
    let (heads, q_len, pos_len) = (2usize, 3usize, 5usize);
    let total = heads * q_len * pos_len;
    // Distinct values so a wrong index produces an obvious mismatch.
    let bd_f32: Vec<f32> = (0..total).map(|i| (i as f32) * 0.1).collect();
    let expected = cpu_rel_shift(&bd_f32, heads, q_len, pos_len);

    let cuda = CudaState::new(0)?;
    let bd_f16: Vec<f16> = bd_f32.iter().copied().map(f16::from_f32).collect();
    let bd_gpu = cuda.upload_f16(&bd_f16)?;
    let out_gpu = cuda.rel_shift_rank3(&bd_gpu, heads, q_len, pos_len)?;
    cuda.synchronize()?;
    let out_f16 = cuda.download_f16(&out_gpu)?;
    let out_f32: Vec<f32> = out_f16.iter().map(|h| h.to_f32()).collect();

    assert_eq!(out_f32.len(), expected.len());
    let mut max_diff = 0f32;
    for (a, b) in out_f32.iter().zip(&expected) {
        max_diff = max_diff.max((a - b).abs());
    }
    assert!(
        max_diff < 1e-3,
        "rel_shift mismatch: max_diff={max_diff:.5}\n gpu:   {out_f32:?}\n expect:{expected:?}"
    );
    println!("ok: rel_shift matches CPU reference (max_diff={max_diff:.5})");
    Ok(())
}

/// A second, larger random test to catch edge cases the structured small test
/// might miss (e.g. index wraparound at larger dims).
#[test]
fn rel_shift_matches_cpu_reference_large() -> anyhow::Result<()> {
    let (heads, q_len, pos_len) = (8usize, 16usize, 31usize);
    let total = heads * q_len * pos_len;
    // Pseudo-random but deterministic.
    let bd_f32: Vec<f32> = (0..total).map(|i| ((i as i32 % 97) as f32) * 0.03 - 1.5).collect();
    let expected = cpu_rel_shift(&bd_f32, heads, q_len, pos_len);

    let cuda = CudaState::new(0)?;
    let bd_f16: Vec<f16> = bd_f32.iter().copied().map(f16::from_f32).collect();
    let bd_gpu = cuda.upload_f16(&bd_f16)?;
    let out_gpu = cuda.rel_shift_rank3(&bd_gpu, heads, q_len, pos_len)?;
    cuda.synchronize()?;
    let out_f16 = cuda.download_f16(&out_gpu)?;
    let out_f32: Vec<f32> = out_f16.iter().map(|h| h.to_f32()).collect();

    let max_diff = out_f32
        .iter()
        .zip(&expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(
        max_diff < 1e-2,
        "rel_shift (large) mismatch: max_diff={max_diff:.5}"
    );
    println!("ok: rel_shift (large) matches CPU reference (max_diff={max_diff:.5})");
    Ok(())
}
