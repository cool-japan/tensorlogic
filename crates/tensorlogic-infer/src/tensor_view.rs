//! Zero-copy tensor views and slicing operations.
//!
//! This module provides infrastructure for zero-copy tensor operations,
//! enabling efficient memory access patterns without data duplication.

use std::ops::Range;

/// Tensor view descriptor for zero-copy operations
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorView {
    /// Base tensor identifier
    pub base_tensor_id: usize,
    /// Slice specification for each dimension
    pub slices: Vec<SliceSpec>,
    /// Strides for each dimension (for strided access)
    pub strides: Vec<isize>,
    /// Offset from the base tensor
    pub offset: usize,
}

impl TensorView {
    /// Create a new tensor view
    pub fn new(base_tensor_id: usize, slices: Vec<SliceSpec>) -> Self {
        TensorView {
            base_tensor_id,
            slices,
            strides: vec![],
            offset: 0,
        }
    }

    /// Create a full view of a tensor (no slicing)
    pub fn full(base_tensor_id: usize, rank: usize) -> Self {
        TensorView {
            base_tensor_id,
            slices: vec![SliceSpec::Full; rank],
            strides: vec![],
            offset: 0,
        }
    }

    /// Create a view with specific offset and strides
    pub fn with_strides(mut self, strides: Vec<isize>) -> Self {
        self.strides = strides;
        self
    }

    /// Create a view with specific offset
    pub fn with_offset(mut self, offset: usize) -> Self {
        self.offset = offset;
        self
    }

    /// Check if this view represents a contiguous slice
    pub fn is_contiguous(&self) -> bool {
        self.slices
            .iter()
            .all(|s| matches!(s, SliceSpec::Full | SliceSpec::Range(_)))
            && self.strides.is_empty()
    }

    /// Check if this view is a complete view (no slicing)
    pub fn is_full_view(&self) -> bool {
        self.slices.iter().all(|s| matches!(s, SliceSpec::Full)) && self.offset == 0
    }

    /// Get the rank of the view
    pub fn rank(&self) -> usize {
        self.slices.len()
    }

    /// Compose two views (create a view of a view)
    pub fn compose(&self, other: &TensorView) -> Result<TensorView, String> {
        if self.base_tensor_id != other.base_tensor_id {
            return Err("Cannot compose views from different base tensors".to_string());
        }

        if self.rank() != other.rank() {
            return Err(format!(
                "Rank mismatch: {} vs {}",
                self.rank(),
                other.rank()
            ));
        }

        // Compose slices
        let mut composed_slices = Vec::new();
        for (s1, s2) in self.slices.iter().zip(other.slices.iter()) {
            composed_slices.push(s1.compose(s2)?);
        }

        // Compute composed offset
        let composed_offset = self.offset + other.offset;

        Ok(TensorView {
            base_tensor_id: self.base_tensor_id,
            slices: composed_slices,
            strides: if other.strides.is_empty() {
                self.strides.clone()
            } else {
                other.strides.clone()
            },
            offset: composed_offset,
        })
    }
}

/// Slice specification for a single dimension
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SliceSpec {
    /// Full dimension (no slicing)
    Full,
    /// Range slice [start..end)
    Range(Range<usize>),
    /// Single index (reduces dimension)
    Index(usize),
    /// Strided slice (start, end, stride)
    Strided {
        start: usize,
        end: usize,
        stride: usize,
    },
    /// Reverse slice (full dimension in reverse order)
    Reverse,
}

impl SliceSpec {
    /// Create a range slice
    pub fn range(start: usize, end: usize) -> Self {
        SliceSpec::Range(start..end)
    }

    /// Create a strided slice
    pub fn strided(start: usize, end: usize, stride: usize) -> Self {
        SliceSpec::Strided { start, end, stride }
    }

    /// Get the size of this slice given the dimension size
    pub fn size(&self, dim_size: usize) -> Result<usize, String> {
        match self {
            SliceSpec::Full => Ok(dim_size),
            SliceSpec::Range(r) => {
                if r.end > dim_size {
                    Err(format!(
                        "Range end {} exceeds dimension size {}",
                        r.end, dim_size
                    ))
                } else if r.start >= r.end {
                    Err(format!("Invalid range: {}..{}", r.start, r.end))
                } else {
                    Ok(r.end - r.start)
                }
            }
            SliceSpec::Index(_) => Ok(1), // Single element
            SliceSpec::Strided { start, end, stride } => {
                if *end > dim_size {
                    Err(format!(
                        "Strided end {} exceeds dimension size {}",
                        end, dim_size
                    ))
                } else if start >= end {
                    Err(format!("Invalid strided range: {}..{}", start, end))
                } else if *stride == 0 {
                    Err("Stride cannot be zero".to_string())
                } else {
                    Ok((end - start).div_ceil(*stride))
                }
            }
            SliceSpec::Reverse => Ok(dim_size),
        }
    }

    /// Compose two slice specs
    pub fn compose(&self, other: &SliceSpec) -> Result<SliceSpec, String> {
        match (self, other) {
            (SliceSpec::Full, s) => Ok(s.clone()),
            (s, SliceSpec::Full) => Ok(s.clone()),
            (SliceSpec::Range(r1), SliceSpec::Range(r2)) => {
                let start = r1.start + r2.start;
                let end = r1.start + r2.end;
                if end > r1.end {
                    Err(format!(
                        "Composed range end {} exceeds first range end {}",
                        end, r1.end
                    ))
                } else {
                    Ok(SliceSpec::Range(start..end))
                }
            }
            (SliceSpec::Range(r), SliceSpec::Index(i)) => {
                if *i >= r.len() {
                    Err(format!("Index {} out of range 0..{}", i, r.len()))
                } else {
                    Ok(SliceSpec::Index(r.start + i))
                }
            }
            _ => Err("Cannot compose these slice types".to_string()),
        }
    }
}

/// Trait for tensors that support zero-copy views
pub trait TensorViewable {
    /// Create a view of this tensor
    fn view(&self, slices: Vec<SliceSpec>) -> Result<TensorView, String>;

    /// Create a slice of this tensor
    fn slice(&self, ranges: &[Range<usize>]) -> Result<TensorView, String> {
        let slices = ranges.iter().map(|r| SliceSpec::Range(r.clone())).collect();
        self.view(slices)
    }

    /// Create a strided view
    fn stride(&self, strides: Vec<isize>) -> Result<TensorView, String>;

    /// Get a single element view
    fn at(&self, indices: &[usize]) -> Result<TensorView, String> {
        let slices = indices.iter().map(|&i| SliceSpec::Index(i)).collect();
        self.view(slices)
    }

    /// Reshape view (if possible without copying)
    fn reshape_view(&self, new_shape: Vec<usize>) -> Result<TensorView, String>;
}

/// View builder for ergonomic API
pub struct ViewBuilder {
    base_tensor_id: usize,
    slices: Vec<SliceSpec>,
    strides: Vec<isize>,
    offset: usize,
}

impl ViewBuilder {
    /// Create a new view builder
    pub fn new(base_tensor_id: usize, rank: usize) -> Self {
        ViewBuilder {
            base_tensor_id,
            slices: vec![SliceSpec::Full; rank],
            strides: vec![],
            offset: 0,
        }
    }

    /// Set slice for a dimension
    pub fn slice_dim(mut self, dim: usize, slice: SliceSpec) -> Self {
        if dim < self.slices.len() {
            self.slices[dim] = slice;
        }
        self
    }

    /// Set range for a dimension
    pub fn range_dim(mut self, dim: usize, start: usize, end: usize) -> Self {
        if dim < self.slices.len() {
            self.slices[dim] = SliceSpec::Range(start..end);
        }
        self
    }

    /// Set index for a dimension
    pub fn index_dim(mut self, dim: usize, index: usize) -> Self {
        if dim < self.slices.len() {
            self.slices[dim] = SliceSpec::Index(index);
        }
        self
    }

    /// Set strides
    pub fn with_strides(mut self, strides: Vec<isize>) -> Self {
        self.strides = strides;
        self
    }

    /// Set offset
    pub fn with_offset(mut self, offset: usize) -> Self {
        self.offset = offset;
        self
    }

    /// Build the tensor view
    pub fn build(self) -> TensorView {
        TensorView {
            base_tensor_id: self.base_tensor_id,
            slices: self.slices,
            strides: self.strides,
            offset: self.offset,
        }
    }
}

/// In-place operation marker
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InPlaceMode {
    /// Safe in-place (no aliasing)
    Safe,
    /// Unsafe in-place (potential aliasing, user responsible)
    Unsafe,
    /// No in-place operation
    None,
}

/// Trait for in-place operations
pub trait InPlaceOps {
    type Error;

    /// Check if in-place operation is safe
    fn can_do_inplace(&self, output_view: &TensorView, input_views: &[TensorView]) -> bool;

    /// Execute operation in-place if possible
    fn execute_inplace(
        &mut self,
        output_view: &TensorView,
        input_views: &[TensorView],
        mode: InPlaceMode,
    ) -> Result<(), Self::Error>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_view_creation() {
        let view = TensorView::new(0, vec![SliceSpec::Full, SliceSpec::Range(10..20)]);
        assert_eq!(view.base_tensor_id, 0);
        assert_eq!(view.rank(), 2);
        assert!(!view.is_full_view());
    }

    #[test]
    fn test_full_view() {
        let view = TensorView::full(0, 3);
        assert_eq!(view.rank(), 3);
        assert!(view.is_full_view());
        assert!(view.is_contiguous());
    }

    #[test]
    fn test_slice_spec_size() {
        assert_eq!(SliceSpec::Full.size(100).unwrap(), 100);
        assert_eq!(SliceSpec::Range(10..20).size(100).unwrap(), 10);
        assert_eq!(SliceSpec::Index(5).size(100).unwrap(), 1);
        assert_eq!(
            SliceSpec::Strided {
                start: 0,
                end: 100,
                stride: 10
            }
            .size(100)
            .unwrap(),
            10
        );
    }

    #[test]
    fn test_slice_spec_compose() {
        let s1 = SliceSpec::Range(10..30);
        let s2 = SliceSpec::Range(5..15);
        let composed = s1.compose(&s2).unwrap();
        assert_eq!(composed, SliceSpec::Range(15..25));
    }

    #[test]
    fn test_view_compose() {
        let view1 = TensorView::new(0, vec![SliceSpec::Range(0..100), SliceSpec::Full]);
        let view2 = TensorView::new(0, vec![SliceSpec::Range(10..50), SliceSpec::Range(0..64)]);
        let composed = view1.compose(&view2).unwrap();
        assert_eq!(composed.base_tensor_id, 0);
        assert_eq!(composed.rank(), 2);
    }

    #[test]
    fn test_view_builder() {
        let view = ViewBuilder::new(0, 3)
            .range_dim(0, 10, 20)
            .index_dim(1, 5)
            .with_offset(100)
            .build();

        assert_eq!(view.base_tensor_id, 0);
        assert_eq!(view.offset, 100);
        assert_eq!(view.slices[0], SliceSpec::Range(10..20));
        assert_eq!(view.slices[1], SliceSpec::Index(5));
    }

    #[test]
    fn test_contiguous_check() {
        let view1 = TensorView::new(0, vec![SliceSpec::Full, SliceSpec::Range(0..10)]);
        assert!(view1.is_contiguous());

        // Note: Index slices are considered contiguous only if no explicit strides
        let view2 = TensorView::new(0, vec![SliceSpec::Full, SliceSpec::Range(0..10)]);
        assert!(view2.is_contiguous());

        // View with explicit strides is not contiguous
        let view3 =
            TensorView::new(0, vec![SliceSpec::Full, SliceSpec::Full]).with_strides(vec![128, 1]);
        assert!(!view3.is_contiguous());
    }

    #[test]
    fn test_strided_slice() {
        let spec = SliceSpec::strided(0, 100, 10);
        assert_eq!(spec.size(100).unwrap(), 10);

        let spec2 = SliceSpec::strided(5, 50, 5);
        assert_eq!(spec2.size(100).unwrap(), 9);
    }

    #[test]
    fn test_invalid_slices() {
        // Range exceeds dimension
        assert!(SliceSpec::Range(10..200).size(100).is_err());

        // Invalid range (intentionally reversed to test error handling)
        #[allow(clippy::reversed_empty_ranges)]
        {
            assert!(SliceSpec::Range(20..10).size(100).is_err());
        }

        // Zero stride
        assert!(SliceSpec::Strided {
            start: 0,
            end: 10,
            stride: 0
        }
        .size(100)
        .is_err());
    }

    #[test]
    fn test_view_with_strides() {
        let view = TensorView::new(0, vec![SliceSpec::Full, SliceSpec::Full])
            .with_strides(vec![128, 1])
            .with_offset(0);

        assert_eq!(view.strides, vec![128, 1]);
        assert!(!view.is_contiguous()); // Has explicit strides
    }
}
