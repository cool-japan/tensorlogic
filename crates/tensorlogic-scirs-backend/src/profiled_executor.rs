//! Performance profiling support for execution monitoring.

use crate::{Scirs2Exec, Scirs2Tensor};
use tensorlogic_infer::{ExecutorError, Profiler, TlAutodiff, TlExecutor, TlProfiledExecutor};
use tensorlogic_ir::EinsumGraph;

/// Profiling-enabled executor wrapper
pub struct ProfiledScirs2Exec {
    /// Underlying executor
    executor: Scirs2Exec,
    /// Profiler for tracking operations
    profiler: Option<Profiler>,
}

impl ProfiledScirs2Exec {
    /// Create a new profiled executor
    pub fn new() -> Self {
        ProfiledScirs2Exec {
            executor: Scirs2Exec::new(),
            profiler: Some(Profiler::new()),
        }
    }

    /// Create with memory pooling enabled
    pub fn with_memory_pool() -> Self {
        ProfiledScirs2Exec {
            executor: Scirs2Exec::with_memory_pool(),
            profiler: Some(Profiler::new()),
        }
    }

    /// Access the underlying executor
    pub fn executor(&self) -> &Scirs2Exec {
        &self.executor
    }

    /// Access the underlying executor mutably
    pub fn executor_mut(&mut self) -> &mut Scirs2Exec {
        &mut self.executor
    }
}

impl Default for ProfiledScirs2Exec {
    fn default() -> Self {
        Self::new()
    }
}

impl TlExecutor for ProfiledScirs2Exec {
    type Tensor = Scirs2Tensor;
    type Error = ExecutorError;

    fn einsum(&mut self, spec: &str, inputs: &[Self::Tensor]) -> Result<Self::Tensor, Self::Error> {
        if let Some(profiler) = &mut self.profiler {
            profiler.time_op(format!("einsum({})", spec), || {
                self.executor.einsum(spec, inputs)
            })
        } else {
            self.executor.einsum(spec, inputs)
        }
    }

    fn elem_op(
        &mut self,
        op: tensorlogic_infer::ElemOp,
        x: &Self::Tensor,
    ) -> Result<Self::Tensor, Self::Error> {
        if let Some(profiler) = &mut self.profiler {
            profiler.time_op(format!("elem_op({:?})", op), || {
                self.executor.elem_op(op, x)
            })
        } else {
            self.executor.elem_op(op, x)
        }
    }

    fn elem_op_binary(
        &mut self,
        op: tensorlogic_infer::ElemOp,
        x: &Self::Tensor,
        y: &Self::Tensor,
    ) -> Result<Self::Tensor, Self::Error> {
        if let Some(profiler) = &mut self.profiler {
            profiler.time_op(format!("elem_op_binary({:?})", op), || {
                self.executor.elem_op_binary(op, x, y)
            })
        } else {
            self.executor.elem_op_binary(op, x, y)
        }
    }

    fn reduce(
        &mut self,
        op: tensorlogic_infer::ReduceOp,
        x: &Self::Tensor,
        axes: &[usize],
    ) -> Result<Self::Tensor, Self::Error> {
        if let Some(profiler) = &mut self.profiler {
            profiler.time_op(format!("reduce({:?})", op), || {
                self.executor.reduce(op, x, axes)
            })
        } else {
            self.executor.reduce(op, x, axes)
        }
    }
}

impl TlAutodiff for ProfiledScirs2Exec {
    type Tape = <Scirs2Exec as TlAutodiff>::Tape;

    fn forward(&mut self, graph: &EinsumGraph) -> Result<Self::Tensor, Self::Error> {
        if let Some(profiler) = &mut self.profiler {
            profiler.time_op("forward_pass", || self.executor.forward(graph))
        } else {
            self.executor.forward(graph)
        }
    }

    fn backward(
        &mut self,
        graph: &EinsumGraph,
        loss_grad: &Self::Tensor,
    ) -> Result<Self::Tape, Self::Error> {
        if let Some(profiler) = &mut self.profiler {
            profiler.time_op("backward_pass", || self.executor.backward(graph, loss_grad))
        } else {
            self.executor.backward(graph, loss_grad)
        }
    }
}

impl TlProfiledExecutor for ProfiledScirs2Exec {
    fn profiler(&self) -> Option<&Profiler> {
        self.profiler.as_ref()
    }

    fn profiler_mut(&mut self) -> Option<&mut Profiler> {
        self.profiler.as_mut()
    }

    fn enable_profiling(&mut self) {
        if self.profiler.is_none() {
            self.profiler = Some(Profiler::new());
        }
    }

    fn disable_profiling(&mut self) {
        self.profiler = None;
    }
}

#[cfg(all(test, feature = "integration-tests"))]
mod tests {
    use super::*;
    use scirs2_core::ndarray::ArrayD;
    use tensorlogic_compiler::compile_to_einsum;
    use tensorlogic_infer::ElemOp;
    use tensorlogic_ir::{TLExpr, Term};

    fn create_test_tensor(shape: &[usize], value: f64) -> ArrayD<f64> {
        ArrayD::from_elem(shape.to_vec(), value)
    }

    #[test]
    fn test_profiled_executor_basic() {
        let mut executor = ProfiledScirs2Exec::new();

        let a = create_test_tensor(&[3, 3], 1.0);
        let b = create_test_tensor(&[3, 3], 2.0);

        // Execute an einsum operation
        let _result = executor
            .einsum("ij,jk->ik", &[a.clone(), b.clone()])
            .unwrap();

        // Check that profiling recorded the operation
        assert!(executor.profiler().is_some());
    }

    #[test]
    fn test_profiled_forward_pass() {
        let x = TLExpr::pred("x", vec![Term::var("i")]);
        let y = TLExpr::pred("y", vec![Term::var("i")]);
        let expr = TLExpr::add(x, y);
        let graph = compile_to_einsum(&expr).unwrap();

        let mut executor = ProfiledScirs2Exec::new();
        executor
            .executor_mut()
            .add_tensor(graph.tensors[0].clone(), create_test_tensor(&[5], 1.0));
        executor
            .executor_mut()
            .add_tensor(graph.tensors[1].clone(), create_test_tensor(&[5], 2.0));

        let _result = executor.forward(&graph).unwrap();

        // Check profiling is active
        assert!(executor.profiler().is_some());
    }

    #[test]
    fn test_enable_disable_profiling() {
        let mut executor = ProfiledScirs2Exec::new();

        let a = create_test_tensor(&[2, 2], 1.0);

        // Execute with profiling enabled
        let _result = executor.elem_op(ElemOp::Relu, &a).unwrap();
        assert!(executor.profiler().is_some());

        // Disable profiling
        executor.disable_profiling();
        assert!(executor.profiler().is_none());

        // Re-enable profiling
        executor.enable_profiling();
        assert!(executor.profiler().is_some());
    }
}
