//! Domain masks for filtering and constraints.

use serde::{Deserialize, Serialize};
use std::collections::HashSet;

use crate::DomainInfo;

/// Domain mask for filtering and constraints
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DomainMask {
    pub domain: String,
    pub included_elements: HashSet<String>,
    pub excluded_elements: HashSet<String>,
}

impl DomainMask {
    pub fn new(domain: impl Into<String>) -> Self {
        DomainMask {
            domain: domain.into(),
            included_elements: HashSet::new(),
            excluded_elements: HashSet::new(),
        }
    }

    pub fn include(&mut self, element: impl Into<String>) -> &mut Self {
        self.included_elements.insert(element.into());
        self
    }

    pub fn exclude(&mut self, element: impl Into<String>) -> &mut Self {
        self.excluded_elements.insert(element.into());
        self
    }

    pub fn is_allowed(&self, element: &str) -> bool {
        if !self.excluded_elements.is_empty() && self.excluded_elements.contains(element) {
            return false;
        }

        if !self.included_elements.is_empty() {
            return self.included_elements.contains(element);
        }

        true
    }

    pub fn apply_to_indices(&self, domain_info: &DomainInfo) -> Vec<usize> {
        if let Some(elements) = &domain_info.elements {
            elements
                .iter()
                .enumerate()
                .filter(|(_, elem)| self.is_allowed(elem))
                .map(|(idx, _)| idx)
                .collect()
        } else {
            (0..domain_info.cardinality).collect()
        }
    }
}
