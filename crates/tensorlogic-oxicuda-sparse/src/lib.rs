//! # tensorlogic-oxicuda-sparse
//!
//! GPU-accelerated sparse matrix operations for the TensorLogic project,
//! backed by OxiCUDA on NVIDIA hardware with a pure-Rust CPU fallback.
//!
//! ## Feature flags
//!
//! | Feature | Description |
//! |---------|-------------|
//! | `cpu` (default) | Pure-Rust CSR sparse routines — no CUDA required. |
//! | `gpu` | Routes computation through `oxicuda-sparse`; requires `libcuda.so` at run-time. Falls back to CPU automatically when the driver is unavailable. |
//!
//! ## Core types
//!
//! - [`SparseCsr`] — host-resident CSR matrix with `i32` indices and generic values (`f32` default).
//! - [`SparseCsc`] — host-resident CSC matrix with `i32` indices and generic values (`f32` default).
//!
//! ## Core operations
//!
//! - [`spmv`] — `y = alpha * A * x + beta * y` (f32)
//! - [`spmm`] — `C = alpha * A * B + beta * C`  (B and C are row-major dense, f32)
//! - [`spmv_f64`] — `y = alpha * A * x + beta * y` (f64)
//! - [`spmm_f64`] — `C = alpha * A * B + beta * C` (f64)
//! - [`spmv_batched`] — batched `Y = alpha * A * X + beta * Y` (f32)

#![warn(clippy::all)]
#![deny(clippy::correctness)]
#![deny(clippy::suspicious)]

pub mod csc;
pub mod csr;
pub mod error;

pub use csc::SparseCsc;
pub use csr::SparseCsr;
pub use error::SparseError;

use scirs2_core::numeric::Float;

// ---------------------------------------------------------------------------
// GPU path
// ---------------------------------------------------------------------------

/// GPU-side sparse operations (requires the `gpu` feature and a CUDA driver).
#[cfg(feature = "gpu")]
mod gpu {
    use std::sync::Arc;

    use oxicuda_blas::{Layout, MatrixDesc, MatrixDescMut};
    use oxicuda_driver::{Context, Device};
    use oxicuda_memory::DeviceBuffer;
    use oxicuda_sparse::format::CsrMatrix;
    use oxicuda_sparse::handle::SparseHandle;
    use oxicuda_sparse::ops::{spmm as oxicuda_spmm, spmv as oxicuda_spmv, SpMVAlgo};

    use crate::csr::SparseCsr;
    use crate::error::SparseError;

    /// Attempt to initialise CUDA device 0 and create a [`SparseHandle`].
    ///
    /// Returns `Err` when the driver is absent or there are no devices; the
    /// public API treats this as a cue to fall back to the CPU path.
    fn try_init(device_id: i32) -> Result<(Arc<Context>, SparseHandle), SparseError> {
        let device = Device::get(device_id)
            .map_err(|e| SparseError::GpuError(format!("CUDA device {device_id} init: {e}")))?;
        let ctx = Arc::new(
            Context::new(&device)
                .map_err(|e| SparseError::GpuError(format!("CUDA context create: {e}")))?,
        );
        let handle = SparseHandle::new(&ctx)
            .map_err(|e| SparseError::GpuError(format!("SparseHandle create: {e}")))?;
        Ok((ctx, handle))
    }

    /// Upload a host-resident [`SparseCsr<f32>`] to the GPU.
    fn upload_csr(a: &SparseCsr) -> Result<CsrMatrix<f32>, SparseError> {
        CsrMatrix::from_host(a.rows as u32, a.cols as u32, &a.indptr, &a.indices, &a.data)
            .map_err(|e| SparseError::GpuError(format!("CsrMatrix upload: {e}")))
    }

    /// GPU-accelerated SpMV: `y = alpha * A * x + beta * y`.
    ///
    /// Returns `Err(SparseError::GpuError(_))` when CUDA initialisation fails;
    /// callers should fall back to the CPU path in that case.
    ///
    /// The caller must have validated the `x`/`y` lengths against the matrix
    /// dimensions beforehand — the kernel indexes device memory directly and
    /// performs no bounds checking of its own.
    pub(crate) fn gpu_spmv(
        a: &SparseCsr,
        x: &[f32],
        alpha: f32,
        beta: f32,
        y: &mut [f32],
    ) -> Result<(), SparseError> {
        let (_ctx, handle) = try_init(0)?;
        let d_a = upload_csr(a)?;

        let d_x = DeviceBuffer::from_host(x)
            .map_err(|e| SparseError::GpuError(format!("x upload: {e}")))?;
        let d_y = DeviceBuffer::from_host(y)
            .map_err(|e| SparseError::GpuError(format!("y upload: {e}")))?;

        let x_ptr = d_x.as_device_ptr();
        let y_ptr = d_y.as_device_ptr();

        oxicuda_spmv(&handle, SpMVAlgo::Adaptive, alpha, &d_a, x_ptr, beta, y_ptr)
            .map_err(|e| SparseError::GpuError(format!("spmv kernel: {e}")))?;

        // Download result back to the host output buffer.
        // Both d_x and d_y stay alive until end of scope (CUDA stream is synchronous).
        d_y.copy_to_host(y)
            .map_err(|e| SparseError::GpuError(format!("y download: {e}")))?;

        Ok(())
    }

    /// GPU-accelerated SpMM: `C = alpha * A * B + beta * C`.
    ///
    /// Returns `Err(SparseError::GpuError(_))` when CUDA initialisation fails;
    /// callers should fall back to the CPU path in that case.
    ///
    /// The caller must have validated the `b`/`c` lengths against the matrix
    /// dimensions beforehand — the kernel indexes device memory directly and
    /// performs no bounds checking of its own.
    pub(crate) fn gpu_spmm(
        a: &SparseCsr,
        b: &[f32],
        b_cols: usize,
        alpha: f32,
        beta: f32,
        c: &mut [f32],
    ) -> Result<(), SparseError> {
        let (_ctx, handle) = try_init(0)?;
        let d_a = upload_csr(a)?;

        let d_b = DeviceBuffer::from_host(b)
            .map_err(|e| SparseError::GpuError(format!("B upload: {e}")))?;
        let mut d_c = DeviceBuffer::from_host(c)
            .map_err(|e| SparseError::GpuError(format!("C upload: {e}")))?;

        // Build row-major matrix descriptors from the device buffers.
        let b_desc = MatrixDesc::from_buffer(&d_b, a.cols as u32, b_cols as u32, Layout::RowMajor)
            .map_err(|e| SparseError::GpuError(format!("MatrixDesc(B): {e}")))?;
        let mut c_desc =
            MatrixDescMut::from_buffer(&mut d_c, a.rows as u32, b_cols as u32, Layout::RowMajor)
                .map_err(|e| SparseError::GpuError(format!("MatrixDescMut(C): {e}")))?;

        oxicuda_spmm(&handle, alpha, &d_a, &b_desc, beta, &mut c_desc)
            .map_err(|e| SparseError::GpuError(format!("spmm kernel: {e}")))?;

        d_c.copy_to_host(c)
            .map_err(|e| SparseError::GpuError(format!("C download: {e}")))
    }
}

// ---------------------------------------------------------------------------
// CPU path — generic helper used by both f32 and f64 public functions.
// Always compiled; the GPU path is a wrapper around this for f32.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Shape validation — backend independent.
//
// These live at the API boundary rather than inside the CPU kernels because
// the GPU path hands raw device pointers to the PTX kernels and therefore
// cannot re-derive the host buffer lengths itself.  Validating up front keeps
// the contract identical for both backends and stops an undersized `x` from
// being indexed out of bounds on the device.
// ---------------------------------------------------------------------------

/// Verify that SpMV operand lengths agree with the CSR matrix dimensions.
///
/// # Errors
///
/// Returns [`SparseError::ShapeMismatch`] when `x_len != a.cols` or
/// `y_len != a.rows`.
fn check_spmv_shapes<T>(a: &SparseCsr<T>, x_len: usize, y_len: usize) -> Result<(), SparseError> {
    if x_len != a.cols {
        return Err(SparseError::ShapeMismatch(format!(
            "spmv: x.len()={} but A.cols={}",
            x_len, a.cols
        )));
    }
    if y_len != a.rows {
        return Err(SparseError::ShapeMismatch(format!(
            "spmv: y.len()={} but A.rows={}",
            y_len, a.rows
        )));
    }
    Ok(())
}

/// Verify that SpMM operand lengths agree with the CSR matrix dimensions.
///
/// # Errors
///
/// Returns [`SparseError::ShapeMismatch`] when `b_len != a.cols * b_cols` or
/// `c_len != a.rows * b_cols`.
fn check_spmm_shapes<T>(
    a: &SparseCsr<T>,
    b_len: usize,
    b_cols: usize,
    c_len: usize,
) -> Result<(), SparseError> {
    if b_len != a.cols * b_cols {
        return Err(SparseError::ShapeMismatch(format!(
            "spmm: B.len()={} but expected A.cols*b_cols={}*{}={}",
            b_len,
            a.cols,
            b_cols,
            a.cols * b_cols,
        )));
    }
    if c_len != a.rows * b_cols {
        return Err(SparseError::ShapeMismatch(format!(
            "spmm: C.len()={} but expected A.rows*b_cols={}*{}={}",
            c_len,
            a.rows,
            b_cols,
            a.rows * b_cols,
        )));
    }
    Ok(())
}

/// Pure-Rust O(nnz) CSR sparse matrix-vector multiply: `y = alpha * A * x + beta * y`.
///
/// Generic over any [`Float`] type.
pub(crate) fn cpu_spmv_generic<T: Float>(
    a: &SparseCsr<T>,
    x: &[T],
    alpha: T,
    beta: T,
    y: &mut [T],
) -> Result<(), SparseError> {
    check_spmv_shapes(a, x.len(), y.len())?;

    for (i, y_i) in y.iter_mut().enumerate().take(a.rows) {
        let start = a.indptr[i] as usize;
        let end = a.indptr[i + 1] as usize;
        let mut dot = T::zero();
        for k in start..end {
            dot = dot + a.data[k] * x[a.indices[k] as usize];
        }
        *y_i = alpha * dot + beta * *y_i;
    }

    Ok(())
}

/// Pure-Rust CSR sparse-dense matrix multiply: `C = alpha * A * B + beta * C`.
///
/// Generic over any [`Float`] type.  A is `(m, k)` sparse, B is `(k, n)`
/// row-major dense, C is `(m, n)` row-major dense.
fn cpu_spmm_generic<T: Float>(
    a: &SparseCsr<T>,
    b: &[T],
    b_cols: usize,
    alpha: T,
    beta: T,
    c: &mut [T],
) -> Result<(), SparseError> {
    let m = a.rows;
    let k = a.cols;
    let n = b_cols;

    check_spmm_shapes(a, b.len(), n, c.len())?;

    // Pre-allocate temporary column-vector buffers to avoid repeated heap allocation.
    let mut x_col = vec![T::zero(); k];
    let mut y_col = vec![T::zero(); m];

    for j in 0..n {
        // Extract column j of B (row-major, stride n).
        for row_b in 0..k {
            x_col[row_b] = b[row_b * n + j];
        }

        // Extract column j of C (row-major, stride n).
        for row_c in 0..m {
            y_col[row_c] = c[row_c * n + j];
        }

        // y_col = alpha * A * x_col + beta * y_col
        cpu_spmv_generic(a, &x_col, alpha, beta, &mut y_col)?;

        // Write column j back into C.
        for row_c in 0..m {
            c[row_c * n + j] = y_col[row_c];
        }
    }

    Ok(())
}

// Keep the old monomorphic aliases around so the GPU dispatch (which uses
// `&SparseCsr` = `&SparseCsr<f32>`) and the existing tests still compile.
fn cpu_spmv(
    a: &SparseCsr,
    x: &[f32],
    alpha: f32,
    beta: f32,
    y: &mut [f32],
) -> Result<(), SparseError> {
    cpu_spmv_generic(a, x, alpha, beta, y)
}

fn cpu_spmm(
    a: &SparseCsr,
    b: &[f32],
    b_cols: usize,
    alpha: f32,
    beta: f32,
    c: &mut [f32],
) -> Result<(), SparseError> {
    cpu_spmm_generic(a, b, b_cols, alpha, beta, c)
}

// ---------------------------------------------------------------------------
// Public API — f32
// ---------------------------------------------------------------------------

/// Sparse matrix-vector multiply: `y = alpha * A * x + beta * y`.
///
/// When compiled with the `gpu` feature and a CUDA driver is present this
/// routes through `oxicuda-sparse`'s PTX kernels.  On failure (no GPU,
/// missing driver, etc.) it transparently falls back to the pure-Rust CPU
/// implementation.  Without the `gpu` feature the CPU path is always taken.
///
/// # Arguments
///
/// * `a`     – Sparse CSR matrix (`SparseCsr<f32>`).
/// * `x`     – Input dense vector; must have length `a.cols`.
/// * `alpha` – Scalar multiplier for `A * x`.
/// * `beta`  – Scalar multiplier for the current content of `y`.
/// * `y`     – In-out dense vector; must have length `a.rows`.
///
/// # Errors
///
/// Returns [`SparseError::ShapeMismatch`] when `x.len() != a.cols` or
/// `y.len() != a.rows`.
pub fn spmv(
    a: &SparseCsr,
    x: &[f32],
    alpha: f32,
    beta: f32,
    y: &mut [f32],
) -> Result<(), SparseError> {
    // Validate before dispatch: the GPU kernel receives bare device pointers
    // and would happily read past the end of an undersized `x`.
    check_spmv_shapes(a, x.len(), y.len())?;

    #[cfg(feature = "gpu")]
    {
        if gpu::gpu_spmv(a, x, alpha, beta, y).is_ok() {
            return Ok(());
        }
        // GPU unavailable — fall through to CPU path.
    }

    cpu_spmv(a, x, alpha, beta, y)
}

/// Sparse-dense matrix multiply: `C = alpha * A * B + beta * C`.
///
/// `A` is a sparse CSR matrix of shape `(m, k)`.  `B` is a dense row-major
/// matrix of shape `(k, b_cols)`.  `C` is a dense row-major matrix of shape
/// `(m, b_cols)`.
///
/// When compiled with the `gpu` feature and a CUDA driver is present this
/// routes through `oxicuda-sparse`'s PTX kernels.  On failure it falls back
/// to the pure-Rust CPU implementation.
///
/// # Arguments
///
/// * `a`      – Sparse CSR matrix (`SparseCsr<f32>`).
/// * `b`      – Dense input matrix in row-major order; length `a.cols * b_cols`.
/// * `b_cols` – Number of columns in `B` (and `C`).
/// * `alpha`  – Scalar multiplier for `A * B`.
/// * `beta`   – Scalar multiplier for the current content of `C`.
/// * `c`      – In-out dense output matrix in row-major order; length `a.rows * b_cols`.
///
/// # Errors
///
/// Returns [`SparseError::ShapeMismatch`] when buffer lengths are inconsistent
/// with the declared shape.
pub fn spmm(
    a: &SparseCsr,
    b: &[f32],
    b_cols: usize,
    alpha: f32,
    beta: f32,
    c: &mut [f32],
) -> Result<(), SparseError> {
    // Validate before dispatch — see [`spmv`] for the rationale.
    check_spmm_shapes(a, b.len(), b_cols, c.len())?;

    #[cfg(feature = "gpu")]
    {
        if gpu::gpu_spmm(a, b, b_cols, alpha, beta, c).is_ok() {
            return Ok(());
        }
        // GPU unavailable — fall through to CPU path.
    }

    cpu_spmm(a, b, b_cols, alpha, beta, c)
}

// ---------------------------------------------------------------------------
// Public API — f64
// ---------------------------------------------------------------------------

/// Sparse matrix-vector multiply (f64): `y = alpha * A * x + beta * y`.
///
/// Pure-Rust CPU path (f64 is not supported by the GPU path which is f32-only).
///
/// # Arguments
///
/// * `a`     – Sparse CSR matrix (`SparseCsr<f64>`).
/// * `x`     – Input dense vector; must have length `a.cols`.
/// * `alpha` – Scalar multiplier for `A * x`.
/// * `beta`  – Scalar multiplier for the current content of `y`.
/// * `y`     – In-out dense vector; must have length `a.rows`.
///
/// # Errors
///
/// Returns [`SparseError::ShapeMismatch`] when `x.len() != a.cols` or
/// `y.len() != a.rows`.
pub fn spmv_f64(
    a: &SparseCsr<f64>,
    x: &[f64],
    alpha: f64,
    beta: f64,
    y: &mut [f64],
) -> Result<(), SparseError> {
    cpu_spmv_generic(a, x, alpha, beta, y)
}

/// Sparse-dense matrix multiply (f64): `C = alpha * A * B + beta * C`.
///
/// Pure-Rust CPU path.  `A` is a sparse CSR matrix of shape `(m, k)`.
/// `B` is a dense row-major matrix of shape `(k, b_cols)`.  `C` is a dense
/// row-major matrix of shape `(m, b_cols)`.
///
/// # Arguments
///
/// * `a`      – Sparse CSR matrix (`SparseCsr<f64>`).
/// * `b`      – Dense input matrix in row-major order; length `a.cols * b_cols`.
/// * `b_cols` – Number of columns in `B` (and `C`).
/// * `alpha`  – Scalar multiplier for `A * B`.
/// * `beta`   – Scalar multiplier for the current content of `C`.
/// * `c`      – In-out dense output matrix in row-major order; length `a.rows * b_cols`.
///
/// # Errors
///
/// Returns [`SparseError::ShapeMismatch`] when buffer lengths are inconsistent
/// with the declared shape.
pub fn spmm_f64(
    a: &SparseCsr<f64>,
    b: &[f64],
    b_cols: usize,
    alpha: f64,
    beta: f64,
    c: &mut [f64],
) -> Result<(), SparseError> {
    cpu_spmm_generic(a, b, b_cols, alpha, beta, c)
}

// ---------------------------------------------------------------------------
// Public API — batched f32
// ---------------------------------------------------------------------------

/// Batched sparse matrix-vector multiply: `Y = alpha * A * X + beta * Y`.
///
/// `x_batch` is a row-major matrix of shape `(a.cols, batch_size)` — i.e.
/// each column of `x_batch` is one input vector.  `y_batch` is a row-major
/// matrix of shape `(a.rows, batch_size)` — each column is one output vector.
///
/// This is equivalent to calling [`spmv`] once per column of `X`.
///
/// # Arguments
///
/// * `a`          – Sparse CSR matrix (`SparseCsr<f32>`).
/// * `x_batch`    – Row-major dense matrix of shape `(a.cols, batch_size)`.
/// * `batch_size` – Number of vectors in the batch.
/// * `alpha`      – Scalar multiplier for `A * X`.
/// * `beta`       – Scalar multiplier for the current content of `Y`.
/// * `y_batch`    – In-out row-major dense matrix of shape `(a.rows, batch_size)`.
///
/// # Errors
///
/// Returns [`SparseError::ShapeMismatch`] when buffer lengths are inconsistent.
pub fn spmv_batched(
    a: &SparseCsr,
    x_batch: &[f32],
    batch_size: usize,
    alpha: f32,
    beta: f32,
    y_batch: &mut [f32],
) -> Result<(), SparseError> {
    let k = a.cols;
    let m = a.rows;
    let n = batch_size;

    if x_batch.len() != k * n {
        return Err(SparseError::ShapeMismatch(format!(
            "spmv_batched: x_batch.len()={} but expected A.cols*batch_size={}*{}={}",
            x_batch.len(),
            k,
            n,
            k * n,
        )));
    }
    if y_batch.len() != m * n {
        return Err(SparseError::ShapeMismatch(format!(
            "spmv_batched: y_batch.len()={} but expected A.rows*batch_size={}*{}={}",
            y_batch.len(),
            m,
            n,
            m * n,
        )));
    }

    // Temporary single-column buffers.
    let mut x_col = vec![0.0f32; k];
    let mut y_col = vec![0.0f32; m];

    for j in 0..n {
        // Extract column j of X (row-major, stride n).
        for row_x in 0..k {
            x_col[row_x] = x_batch[row_x * n + j];
        }

        // Extract column j of Y (row-major, stride n).
        for row_y in 0..m {
            y_col[row_y] = y_batch[row_y * n + j];
        }

        // y_col = alpha * A * x_col + beta * y_col
        cpu_spmv(a, &x_col, alpha, beta, &mut y_col)?;

        // Write column j back into Y.
        for row_y in 0..m {
            y_batch[row_y * n + j] = y_col[row_y];
        }
    }

    Ok(())
}
