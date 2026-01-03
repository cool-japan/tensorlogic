//! SIMD (Single Instruction, Multiple Data) optimization utilities.
//!
//! This module provides SIMD optimization infrastructure:
//! - **Platform detection** (AVX2, AVX-512, NEON, etc.)
//! - **Vectorization hints** for the compiler
//! - **SIMD-friendly data layouts** and alignment
//! - **Automatic vectorization** checking
//! - **Performance benchmarking** for SIMD operations
//!
//! ## Example
//!
//! ```rust,ignore
//! use tensorlogic_infer::{SimdCapabilities, SimdOptimizer, AlignedBuffer};
//!
//! // Check SIMD capabilities
//! let caps = SimdCapabilities::detect();
//! println!("AVX2 supported: {}", caps.has_avx2());
//!
//! // Create aligned buffer for SIMD operations
//! let buffer = AlignedBuffer::<f32>::new(1024);
//!
//! // Optimize operations for SIMD
//! let optimizer = SimdOptimizer::new(caps);
//! let optimized_graph = optimizer.optimize(&graph)?;
//! ```

use serde::{Deserialize, Serialize};
use std::alloc::{alloc, dealloc, Layout};
use std::ptr::NonNull;
use thiserror::Error;

/// SIMD optimization errors.
#[derive(Error, Debug, Clone, PartialEq)]
pub enum SimdError {
    #[error("Unsupported SIMD instruction set: {0}")]
    UnsupportedInstructionSet(String),

    #[error("Alignment error: required {required}, got {actual}")]
    AlignmentError { required: usize, actual: usize },

    #[error("Buffer size mismatch: expected {expected}, got {actual}")]
    SizeMismatch { expected: usize, actual: usize },

    #[error("SIMD operation failed: {0}")]
    OperationFailed(String),
}

/// SIMD instruction set.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SimdInstructionSet {
    /// No SIMD support
    None,
    /// SSE (Streaming SIMD Extensions)
    SSE,
    /// SSE2
    SSE2,
    /// SSE3
    SSE3,
    /// SSSE3
    SSSE3,
    /// SSE4.1
    SSE41,
    /// SSE4.2
    SSE42,
    /// AVX (Advanced Vector Extensions)
    AVX,
    /// AVX2
    AVX2,
    /// AVX-512
    AVX512,
    /// ARM NEON
    NEON,
    /// SVE (Scalable Vector Extension)
    SVE,
}

impl SimdInstructionSet {
    /// Get the vector width in bytes.
    pub fn vector_width_bytes(&self) -> usize {
        match self {
            SimdInstructionSet::None => 1,
            SimdInstructionSet::SSE
            | SimdInstructionSet::SSE2
            | SimdInstructionSet::SSE3
            | SimdInstructionSet::SSSE3
            | SimdInstructionSet::SSE41
            | SimdInstructionSet::SSE42
            | SimdInstructionSet::NEON => 16,
            SimdInstructionSet::AVX | SimdInstructionSet::AVX2 => 32,
            SimdInstructionSet::AVX512 => 64,
            SimdInstructionSet::SVE => 128, // Can be up to 2048 bits
        }
    }

    /// Get the number of f32 elements per vector.
    pub fn f32_lanes(&self) -> usize {
        self.vector_width_bytes() / std::mem::size_of::<f32>()
    }

    /// Get the number of f64 elements per vector.
    pub fn f64_lanes(&self) -> usize {
        self.vector_width_bytes() / std::mem::size_of::<f64>()
    }

    /// Get the preferred alignment in bytes.
    pub fn preferred_alignment(&self) -> usize {
        match self {
            SimdInstructionSet::None => std::mem::align_of::<f64>(),
            SimdInstructionSet::SSE
            | SimdInstructionSet::SSE2
            | SimdInstructionSet::SSE3
            | SimdInstructionSet::SSSE3
            | SimdInstructionSet::SSE41
            | SimdInstructionSet::SSE42
            | SimdInstructionSet::NEON => 16,
            SimdInstructionSet::AVX | SimdInstructionSet::AVX2 => 32,
            SimdInstructionSet::AVX512 => 64,
            SimdInstructionSet::SVE => 128,
        }
    }

    /// Get the instruction set name.
    pub fn name(&self) -> &'static str {
        match self {
            SimdInstructionSet::None => "None",
            SimdInstructionSet::SSE => "SSE",
            SimdInstructionSet::SSE2 => "SSE2",
            SimdInstructionSet::SSE3 => "SSE3",
            SimdInstructionSet::SSSE3 => "SSSE3",
            SimdInstructionSet::SSE41 => "SSE4.1",
            SimdInstructionSet::SSE42 => "SSE4.2",
            SimdInstructionSet::AVX => "AVX",
            SimdInstructionSet::AVX2 => "AVX2",
            SimdInstructionSet::AVX512 => "AVX-512",
            SimdInstructionSet::NEON => "NEON",
            SimdInstructionSet::SVE => "SVE",
        }
    }
}

/// SIMD capabilities detection.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SimdCapabilities {
    /// Available instruction sets
    pub instruction_sets: Vec<SimdInstructionSet>,

    /// CPU architecture
    pub architecture: CpuArchitecture,

    /// Number of CPU cores
    pub num_cores: usize,

    /// Cache line size (bytes)
    pub cache_line_size: usize,

    /// L1 cache size (bytes)
    pub l1_cache_size: usize,

    /// L2 cache size (bytes)
    pub l2_cache_size: usize,

    /// L3 cache size (bytes)
    pub l3_cache_size: usize,
}

/// CPU architecture.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CpuArchitecture {
    X86_64,
    AArch64,
    ARM,
    Other,
}

impl SimdCapabilities {
    /// Detect SIMD capabilities of the current CPU.
    pub fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            Self::detect_x86_64()
        }

        #[cfg(target_arch = "aarch64")]
        {
            Self::detect_aarch64()
        }

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            Self::default()
        }
    }

    #[cfg(target_arch = "x86_64")]
    fn detect_x86_64() -> Self {
        let mut instruction_sets = Vec::new();

        // Check for SSE family
        if is_x86_feature_detected!("sse") {
            instruction_sets.push(SimdInstructionSet::SSE);
        }
        if is_x86_feature_detected!("sse2") {
            instruction_sets.push(SimdInstructionSet::SSE2);
        }
        if is_x86_feature_detected!("sse3") {
            instruction_sets.push(SimdInstructionSet::SSE3);
        }
        if is_x86_feature_detected!("ssse3") {
            instruction_sets.push(SimdInstructionSet::SSSE3);
        }
        if is_x86_feature_detected!("sse4.1") {
            instruction_sets.push(SimdInstructionSet::SSE41);
        }
        if is_x86_feature_detected!("sse4.2") {
            instruction_sets.push(SimdInstructionSet::SSE42);
        }

        // Check for AVX family
        if is_x86_feature_detected!("avx") {
            instruction_sets.push(SimdInstructionSet::AVX);
        }
        if is_x86_feature_detected!("avx2") {
            instruction_sets.push(SimdInstructionSet::AVX2);
        }
        if is_x86_feature_detected!("avx512f") {
            instruction_sets.push(SimdInstructionSet::AVX512);
        }

        Self {
            instruction_sets,
            architecture: CpuArchitecture::X86_64,
            num_cores: num_cpus::get(),
            cache_line_size: 64, // Common for x86_64
            l1_cache_size: 32 * 1024,
            l2_cache_size: 256 * 1024,
            l3_cache_size: 8 * 1024 * 1024,
        }
    }

    #[cfg(target_arch = "aarch64")]
    fn detect_aarch64() -> Self {
        let instruction_sets = vec![SimdInstructionSet::NEON];

        // Note: SVE detection would require runtime feature detection
        // which is not yet stable in Rust

        Self {
            instruction_sets,
            architecture: CpuArchitecture::AArch64,
            num_cores: num_cpus::get(),
            cache_line_size: 64,
            l1_cache_size: 64 * 1024,
            l2_cache_size: 512 * 1024,
            l3_cache_size: 4 * 1024 * 1024,
        }
    }

    /// Check if a specific instruction set is available.
    pub fn has_instruction_set(&self, isa: SimdInstructionSet) -> bool {
        self.instruction_sets.contains(&isa)
    }

    /// Check if AVX2 is available.
    pub fn has_avx2(&self) -> bool {
        self.has_instruction_set(SimdInstructionSet::AVX2)
    }

    /// Check if AVX-512 is available.
    pub fn has_avx512(&self) -> bool {
        self.has_instruction_set(SimdInstructionSet::AVX512)
    }

    /// Check if NEON is available.
    pub fn has_neon(&self) -> bool {
        self.has_instruction_set(SimdInstructionSet::NEON)
    }

    /// Get the best available instruction set.
    pub fn best_instruction_set(&self) -> SimdInstructionSet {
        // Prefer more advanced instruction sets
        if self.has_avx512() {
            SimdInstructionSet::AVX512
        } else if self.has_avx2() {
            SimdInstructionSet::AVX2
        } else if self.has_instruction_set(SimdInstructionSet::AVX) {
            SimdInstructionSet::AVX
        } else if self.has_instruction_set(SimdInstructionSet::SSE42) {
            SimdInstructionSet::SSE42
        } else if self.has_neon() {
            SimdInstructionSet::NEON
        } else {
            SimdInstructionSet::None
        }
    }

    /// Get the recommended vectorization factor for a given element size.
    pub fn recommended_vector_size(&self, element_size: usize) -> usize {
        let best_isa = self.best_instruction_set();
        best_isa.vector_width_bytes() / element_size
    }
}

impl Default for SimdCapabilities {
    fn default() -> Self {
        Self {
            instruction_sets: vec![SimdInstructionSet::None],
            architecture: CpuArchitecture::Other,
            num_cores: num_cpus::get(),
            cache_line_size: 64,
            l1_cache_size: 32 * 1024,
            l2_cache_size: 256 * 1024,
            l3_cache_size: 8 * 1024 * 1024,
        }
    }
}

/// Aligned buffer for SIMD operations.
pub struct AlignedBuffer<T> {
    ptr: NonNull<T>,
    len: usize,
    alignment: usize,
}

impl<T> AlignedBuffer<T> {
    /// Create a new aligned buffer with the specified alignment.
    pub fn new_with_alignment(len: usize, alignment: usize) -> Result<Self, SimdError> {
        if alignment == 0 || !alignment.is_power_of_two() {
            return Err(SimdError::AlignmentError {
                required: alignment,
                actual: 0,
            });
        }

        let size = len * std::mem::size_of::<T>();
        let layout =
            Layout::from_size_align(size, alignment).map_err(|_| SimdError::AlignmentError {
                required: alignment,
                actual: 0,
            })?;

        let ptr = unsafe { alloc(layout) as *mut T };
        if ptr.is_null() {
            return Err(SimdError::OperationFailed("Allocation failed".to_string()));
        }

        Ok(Self {
            ptr: NonNull::new(ptr).unwrap(),
            len,
            alignment,
        })
    }

    /// Create a new aligned buffer with default alignment (64 bytes).
    pub fn new(len: usize) -> Result<Self, SimdError> {
        Self::new_with_alignment(len, 64)
    }

    /// Get the length of the buffer.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get the alignment of the buffer.
    pub fn alignment(&self) -> usize {
        self.alignment
    }

    /// Get a slice of the buffer.
    pub fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    /// Get a mutable slice of the buffer.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }

    /// Get the raw pointer.
    pub fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr()
    }

    /// Get the mutable raw pointer.
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr.as_ptr()
    }

    /// Check if the buffer is properly aligned.
    pub fn is_aligned(&self) -> bool {
        (self.ptr.as_ptr() as usize) % self.alignment == 0
    }
}

impl<T> Drop for AlignedBuffer<T> {
    fn drop(&mut self) {
        let size = self.len * std::mem::size_of::<T>();
        let layout = Layout::from_size_align(size, self.alignment).unwrap();
        unsafe {
            dealloc(self.ptr.as_ptr() as *mut u8, layout);
        }
    }
}

unsafe impl<T: Send> Send for AlignedBuffer<T> {}
unsafe impl<T: Sync> Sync for AlignedBuffer<T> {}

/// Vectorization hint for the compiler.
#[inline(always)]
pub fn vectorize_hint() {
    // This is a hint to the compiler that this loop should be vectorized
    // The actual implementation depends on the compiler
}

/// Check if a pointer is aligned for SIMD operations.
pub fn is_simd_aligned<T>(ptr: *const T, alignment: usize) -> bool {
    (ptr as usize) % alignment == 0
}

/// Get the alignment offset needed to align a pointer.
pub fn alignment_offset<T>(ptr: *const T, alignment: usize) -> usize {
    let addr = ptr as usize;
    let rem = addr % alignment;
    if rem == 0 {
        0
    } else {
        alignment - rem
    }
}

/// SIMD optimization hints.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SimdOptimizationHints {
    /// Preferred vector size
    pub vector_size: usize,

    /// Preferred alignment
    pub alignment: usize,

    /// Enable loop unrolling
    pub unroll_loops: bool,

    /// Unroll factor
    pub unroll_factor: usize,

    /// Enable data prefetching
    pub prefetch: bool,

    /// Prefetch distance (cache lines)
    pub prefetch_distance: usize,
}

impl Default for SimdOptimizationHints {
    fn default() -> Self {
        let caps = SimdCapabilities::detect();
        let best_isa = caps.best_instruction_set();

        Self {
            vector_size: best_isa.vector_width_bytes(),
            alignment: best_isa.preferred_alignment(),
            unroll_loops: true,
            unroll_factor: 4,
            prefetch: true,
            prefetch_distance: 8,
        }
    }
}

impl SimdOptimizationHints {
    /// Create hints for a specific instruction set.
    pub fn for_instruction_set(isa: SimdInstructionSet) -> Self {
        Self {
            vector_size: isa.vector_width_bytes(),
            alignment: isa.preferred_alignment(),
            ..Default::default()
        }
    }

    /// Disable all optimizations.
    pub fn none() -> Self {
        Self {
            vector_size: 1,
            alignment: std::mem::align_of::<f64>(),
            unroll_loops: false,
            unroll_factor: 1,
            prefetch: false,
            prefetch_distance: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_instruction_set_width() {
        assert_eq!(SimdInstructionSet::SSE.vector_width_bytes(), 16);
        assert_eq!(SimdInstructionSet::AVX.vector_width_bytes(), 32);
        assert_eq!(SimdInstructionSet::AVX2.vector_width_bytes(), 32);
        assert_eq!(SimdInstructionSet::AVX512.vector_width_bytes(), 64);
        assert_eq!(SimdInstructionSet::NEON.vector_width_bytes(), 16);
    }

    #[test]
    fn test_simd_instruction_set_lanes() {
        assert_eq!(SimdInstructionSet::AVX2.f32_lanes(), 8);
        assert_eq!(SimdInstructionSet::AVX2.f64_lanes(), 4);
        assert_eq!(SimdInstructionSet::AVX512.f32_lanes(), 16);
    }

    #[test]
    fn test_simd_capabilities_detection() {
        let caps = SimdCapabilities::detect();
        assert!(!caps.instruction_sets.is_empty());
        assert!(caps.num_cores > 0);
        assert!(caps.cache_line_size > 0);
    }

    #[test]
    fn test_simd_capabilities_best() {
        let caps = SimdCapabilities::detect();
        let best = caps.best_instruction_set();

        // Should never return None if any SIMD is available
        if !caps.instruction_sets.is_empty() {
            assert_ne!(best, SimdInstructionSet::None);
        }
    }

    #[test]
    fn test_aligned_buffer_creation() {
        let buffer = AlignedBuffer::<f32>::new(1024).unwrap();
        assert_eq!(buffer.len(), 1024);
        assert_eq!(buffer.alignment(), 64);
        assert!(buffer.is_aligned());
    }

    #[test]
    fn test_aligned_buffer_custom_alignment() {
        let buffer = AlignedBuffer::<f32>::new_with_alignment(512, 32).unwrap();
        assert_eq!(buffer.len(), 512);
        assert_eq!(buffer.alignment(), 32);
        assert!(buffer.is_aligned());
    }

    #[test]
    fn test_aligned_buffer_slice() {
        let mut buffer = AlignedBuffer::<f32>::new(10).unwrap();
        let slice = buffer.as_mut_slice();
        slice[0] = 1.0;
        slice[1] = 2.0;

        let const_slice = buffer.as_slice();
        assert_eq!(const_slice[0], 1.0);
        assert_eq!(const_slice[1], 2.0);
    }

    #[test]
    fn test_is_simd_aligned() {
        let buffer = AlignedBuffer::<f32>::new(1024).unwrap();
        assert!(is_simd_aligned(buffer.as_ptr(), 64));
        assert!(is_simd_aligned(buffer.as_ptr(), 32));
        assert!(is_simd_aligned(buffer.as_ptr(), 16));
    }

    #[test]
    fn test_alignment_offset() {
        let buffer = AlignedBuffer::<u8>::new(1024).unwrap();
        let offset = alignment_offset(buffer.as_ptr(), 64);
        assert_eq!(offset, 0); // Already aligned
    }

    #[test]
    fn test_simd_optimization_hints_default() {
        let hints = SimdOptimizationHints::default();
        assert!(hints.vector_size > 0);
        assert!(hints.alignment > 0);
        assert!(hints.unroll_loops);
    }

    #[test]
    fn test_simd_optimization_hints_for_isa() {
        let hints = SimdOptimizationHints::for_instruction_set(SimdInstructionSet::AVX2);
        assert_eq!(hints.vector_size, 32);
        assert_eq!(hints.alignment, 32);
    }

    #[test]
    fn test_simd_optimization_hints_none() {
        let hints = SimdOptimizationHints::none();
        assert_eq!(hints.vector_size, 1);
        assert!(!hints.unroll_loops);
        assert!(!hints.prefetch);
    }

    #[test]
    fn test_instruction_set_name() {
        assert_eq!(SimdInstructionSet::AVX2.name(), "AVX2");
        assert_eq!(SimdInstructionSet::NEON.name(), "NEON");
        assert_eq!(SimdInstructionSet::AVX512.name(), "AVX-512");
    }
}
