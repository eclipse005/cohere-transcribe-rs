//! Lightweight GPU/host tensor wrappers.
//!
//! `GpuTensor` owns a `CudaSlice<f16>` plus a shape; no strides (we keep all
//! tensors contiguous and reshape/transpose explicitly). `GpuWeight` is a
//! weight-transposed matrix held resident: `data` is `[rows=N, cols=K]` row-major
//! f16, so `y = x @ W^T` is the cuBLAS call `OP_T(W), OP_N(x)`.

use anyhow::anyhow;

/// Row-major f16 GPU tensor with an explicit shape (no strides).
#[derive(Debug)]
pub struct GpuTensor {
    pub data: cudarc::driver::safe::CudaSlice<half::f16>,
    pub shape: Vec<usize>,
}

impl GpuTensor {
    pub fn new(data: cudarc::driver::safe::CudaSlice<half::f16>, shape: Vec<usize>) -> Self {
        let _ = Self::assert_shape_matches(&data, &shape);
        Self { data, shape }
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    fn assert_shape_matches(
        data: &cudarc::driver::safe::CudaSlice<half::f16>,
        shape: &[usize],
    ) -> anyhow::Result<()> {
        let numel: usize = shape.iter().product();
        let n = data.len();
        if numel != n {
            return Err(anyhow!("GpuTensor shape {shape:?} numel {numel} != data len {n}"));
        }
        Ok(())
    }
}

/// A weight-transposed f16 matrix kept resident on the device: stored as
/// `[rows=N, cols=K]` row-major, so a linear projection `y = x @ W^T` is
/// `cuBLAS OP_T(W) * OP_N(x)` with `lda=K`.
#[derive(Debug, Clone)]
pub struct GpuWeight {
    pub data: cudarc::driver::safe::CudaSlice<half::f16>,
    pub rows: usize, // = N (output features)
    pub cols: usize, // = K (input features)
}

impl GpuWeight {
    pub fn new(data: cudarc::driver::safe::CudaSlice<half::f16>, rows: usize, cols: usize) -> Self {
        assert_eq!(
            data.len(),
            rows * cols,
            "GpuWeight rows*cols {}x{} = {} != data len {}",
            rows,
            cols,
            rows * cols,
            data.len()
        );
        Self { data, rows, cols }
    }
}

/// Row-major f16 host tensor, used for staging before H2D upload.
#[derive(Debug, Clone)]
pub struct CpuTensor {
    pub data: Vec<half::f16>,
    pub shape: Vec<usize>,
}

impl CpuTensor {
    pub fn new(data: Vec<half::f16>, shape: Vec<usize>) -> Self {
        let numel: usize = shape.iter().product();
        assert_eq!(data.len(), numel, "CpuTensor shape mismatch");
        Self { data, shape }
    }
}
