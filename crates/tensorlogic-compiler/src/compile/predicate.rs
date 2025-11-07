//! Predicate compilation.

use anyhow::Result;
use tensorlogic_ir::{EinsumGraph, Term};

use crate::context::{CompileState, CompilerContext};

pub(crate) fn compile_predicate(
    name: &str,
    args: &[Term],
    ctx: &mut CompilerContext,
    graph: &mut EinsumGraph,
) -> Result<CompileState> {
    for arg in args {
        if let Term::Var(v) = arg {
            ctx.assign_axis(v);
        }
    }

    let axes = ctx.get_axes(args)?;
    let tensor_name = format!("{}[{}]", name, axes);
    let tensor_idx = graph.add_tensor(tensor_name);

    Ok(CompileState { tensor_idx, axes })
}
