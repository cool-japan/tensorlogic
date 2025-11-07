//! Dummy tensor implementation for testing.

/// Minimal tensor implementation for testing and prototyping.
///
/// This is NOT meant for production use - it's a placeholder for
/// validating logic compilation and execution flow.
#[derive(Debug, Clone)]
pub struct DummyTensor {
    pub name: String,
    pub shape: Vec<usize>,
    pub data: Vec<f64>,
}

impl DummyTensor {
    pub fn new(name: impl Into<String>, shape: Vec<usize>) -> Self {
        let size: usize = shape.iter().product();
        DummyTensor {
            name: name.into(),
            shape,
            data: vec![0.0; size],
        }
    }

    pub fn with_data(name: impl Into<String>, shape: Vec<usize>, data: Vec<f64>) -> Self {
        assert_eq!(shape.iter().product::<usize>(), data.len());
        DummyTensor {
            name: name.into(),
            shape,
            data,
        }
    }

    pub fn size(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn zeros(name: impl Into<String>, shape: Vec<usize>) -> Self {
        Self::new(name, shape)
    }

    pub fn ones(name: impl Into<String>, shape: Vec<usize>) -> Self {
        let size: usize = shape.iter().product();
        DummyTensor {
            name: name.into(),
            shape,
            data: vec![1.0; size],
        }
    }
}
