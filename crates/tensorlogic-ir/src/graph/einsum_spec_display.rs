//! Display implementation for EinsumSpec.

use std::fmt;

use super::einsum_spec::EinsumSpec;

impl fmt::Display for EinsumSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let input_part = self
            .inputs
            .iter()
            .map(|sub| sub.iter().collect::<String>())
            .collect::<Vec<_>>()
            .join(",");

        let output_part = self.output.iter().collect::<String>();

        write!(f, "{}->{}", input_part, output_part)
    }
}
