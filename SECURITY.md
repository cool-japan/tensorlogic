# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

The COOLJAPAN ecosystem takes security seriously. If you discover a security vulnerability in Tensorlogic, please report it responsibly.

### How to Report

**DO NOT** open a public GitHub issue for security vulnerabilities.

Instead, please report security issues to:

- **Email**: security@cool-japan.org (or open a private security advisory on GitHub)
- **Subject**: [SECURITY] Tensorlogic - Brief description

### What to Include

Please include the following information in your report:

1. **Description**: Clear description of the vulnerability
2. **Impact**: Potential impact and severity assessment
3. **Reproduction**: Step-by-step instructions to reproduce the issue
4. **Environment**: Rust version, OS, and relevant configuration
5. **Proposed Fix**: If you have a suggested solution (optional)

### Response Timeline

- **Initial Response**: Within 48 hours
- **Status Update**: Within 7 days
- **Fix Timeline**: Depends on severity
  - Critical: Within 7 days
  - High: Within 14 days
  - Medium: Within 30 days
  - Low: Next release cycle

### Disclosure Policy

- We follow coordinated disclosure practices
- We will work with you to understand and address the issue
- We will credit reporters in security advisories (unless anonymity is requested)
- Please allow us reasonable time to fix issues before public disclosure

## Security Considerations

### Scope

Tensorlogic is a **planning layer** that compiles logic rules to tensor equations. Security considerations include:

1. **Input Validation**: Malformed logic expressions could cause panics or excessive resource consumption
2. **Resource Limits**: Large graphs may consume excessive memory or CPU
3. **Dependency Security**: We audit dependencies regularly
4. **Type Safety**: Rust's type system provides strong guarantees, but unsafe code requires careful review

### Out of Scope

- Theoretical attacks on logic compilation algorithms (research papers welcome)
- Issues in upstream dependencies (report to those projects)
- Performance issues without security implications

### Security Features

- **No Unsafe Code** in planning layer (IR, compiler, infer traits)
- **Dependency Auditing**: Regular `cargo audit` checks
- **Static Analysis**: Clippy with strict lints enabled
- **Minimal Attack Surface**: Planning layer has no network or filesystem access

### Backend Security

The SciRS2 backend may use:
- SIMD operations (platform-specific)
- GPU operations (future)
- Numerical computations that could overflow or lose precision

We validate inputs and provide safe abstractions over low-level operations.

### Data Governance

The OxiRS bridge handles:
- GraphQL query parsing
- RDF/SPARQL execution
- SHACL constraint validation

These components have their own security policies. See the OxiRS documentation for details.

## Best Practices for Users

### Input Validation

Always validate untrusted input before compiling:

```rust
use tensorlogic_compiler::compile_to_einsum;

// Validate before compilation
if expr_is_too_large(&expr) {
    return Err("Expression exceeds size limit");
}

let graph = compile_to_einsum(&expr)?;
```

### Resource Limits

Set appropriate limits for production use:

```rust
const MAX_GRAPH_NODES: usize = 10000;
const MAX_TENSOR_SIZE: usize = 100_000_000;

if graph.nodes.len() > MAX_GRAPH_NODES {
    return Err("Graph too large");
}
```

### Dependency Management

- Keep dependencies up to date
- Run `cargo audit` regularly
- Review security advisories for the COOLJAPAN ecosystem

### Safe Deployment

- Run with minimal privileges
- Isolate execution environments (containers, VMs)
- Monitor resource usage
- Implement timeout mechanisms for long-running operations

## Security Audits

We perform regular security reviews:

- **Code Review**: All PRs reviewed for security implications
- **Dependency Audits**: Monthly `cargo audit` checks
- **Static Analysis**: Continuous Clippy and rustfmt enforcement
- **Fuzzing**: Planned for Phase 8 (validation & scale)

## Known Limitations

Current known limitations (not vulnerabilities):

1. **No Resource Limits**: Early versions do not enforce graph size limits
2. **No Timeout Mechanisms**: Long compilations may run indefinitely
3. **Limited Input Validation**: Assumes well-formed expressions

These will be addressed in future releases.

## Updates and Advisories

Security advisories will be published:

- As GitHub Security Advisories
- In release notes
- On the COOLJAPAN security mailing list (when available)

## Contact

For security-related questions:
- Email: security@cool-japan.org
- GitHub: Open a private security advisory

For non-security issues:
- Open a public GitHub issue
- See CONTRIBUTING.md for guidelines

Thank you for helping keep Tensorlogic secure!
