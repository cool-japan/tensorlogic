# CI/CD Integration Examples

This directory contains example CI/CD pipeline configurations for integrating TensorLogic CLI into your continuous integration workflows.

## Available Pipelines

### 1. GitHub Actions (`github-actions.yml`)

**Features**:
- Validates all `.tl` rule files
- Compiles rules to JSON
- Generates visualizations (PNG/SVG)
- Tests multiple compilation strategies in parallel
- Quality checks for large graphs
- Artifact uploads

**Setup**:
```bash
mkdir -p .github/workflows
cp ci-examples/github-actions.yml .github/workflows/tensorlogic.yml
```

**Triggers**:
- Push to `main` or `develop` branches
- Pull requests to `main`
- Changes to `rules/**/*.tl` files

---

### 2. GitLab CI (`gitlab-ci.yml`)

**Features**:
- Multi-stage pipeline (validate → compile → visualize → test → deploy)
- Strategy matrix testing
- Caching for faster builds
- Artifact management with expiration
- Environment-specific deployments (staging/production)

**Setup**:
```bash
cp ci-examples/gitlab-ci.yml .gitlab-ci.yml
```

**Stages**:
1. `setup` - Install dependencies
2. `validate` - Validate rules
3. `compile` - Compile to JSON
4. `visualize` - Generate DOT graphs
5. `test` - Test with multiple strategies
6. `deploy` - Deploy to environments

---

### 3. Jenkins Pipeline (`Jenkinsfile`)

**Features**:
- Declarative pipeline syntax
- Parallel execution for statistics and compilation
- Matrix builds for strategy testing
- Conditional visualization (main/develop only)
- Post-build artifact archival

**Setup**:
```bash
cp ci-examples/Jenkinsfile Jenkinsfile
```

**Configuration**:
- Add to Jenkins as a Pipeline job
- Point to Git repository containing Jenkinsfile
- Configure triggers as needed

---

## Common Workflows

### Validation Only

Simple validation workflow for all CI systems:

**GitHub Actions**:
```yaml
- name: Validate rules
  run: tensorlogic batch rules/*.tl --validate
```

**GitLab CI**:
```yaml
validate:
  script:
    - tensorlogic batch rules/*.tl --validate
```

**Jenkins**:
```groovy
sh 'tensorlogic batch rules/*.tl --validate'
```

---

### Compilation and Deployment

**GitHub Actions**:
```yaml
- name: Compile and deploy
  run: |
    for rule in rules/*.tl; do
      name=$(basename "$rule" .tl)
      tensorlogic "$rule" \
        --output-format json \
        --output "compiled/${name}.json"
    done
    # Deploy compiled files
```

**GitLab CI**:
```yaml
deploy:
  script:
    - tensorlogic batch rules/*.tl --output-format json
    - # Upload to artifact registry
  only:
    - main
```

**Jenkins**:
```groovy
stage('Deploy') {
    steps {
        sh '''
            for rule in rules/*.tl; do
                tensorlogic "$rule" --output-format json
            done
        '''
    }
}
```

---

## Best Practices

### 1. Cache Dependencies

**GitHub Actions**:
```yaml
- uses: actions/cache@v4
  with:
    path: |
      ~/.cargo/bin/
      ~/.cargo/registry/
    key: ${{ runner.os }}-cargo-${{ hashFiles('**/Cargo.lock') }}
```

**GitLab CI**:
```yaml
cache:
  paths:
    - .cargo/
    - target/
```

**Jenkins**:
Use workspace caching or Docker volumes

---

### 2. Fail Fast on Validation Errors

All pipelines should fail immediately if validation fails:

```bash
tensorlogic batch rules/*.tl --validate || exit 1
```

---

### 3. Generate Reports

Create human-readable reports:

```bash
for rule in rules/*.tl; do
  name=$(basename "$rule" .tl)
  tensorlogic "$rule" \
    --output-format stats \
    --analyze > "reports/${name}.txt"
done
```

---

### 4. Conditional Deployment

Only deploy on specific branches:

**GitHub Actions**:
```yaml
if: github.ref == 'refs/heads/main'
```

**GitLab CI**:
```yaml
only:
  - main
```

**Jenkins**:
```groovy
when {
    branch 'main'
}
```

---

## Environment Variables

Configure these variables in your CI system:

| Variable | Description | Default |
|----------|-------------|---------|
| `TENSORLOGIC_VERSION` | CLI version to install | `0.1.0-alpha.2` |
| `RUST_TOOLCHAIN` | Rust toolchain version | `stable` |
| `RULE_DIR` | Directory containing rules | `rules/` |
| `OUTPUT_DIR` | Output directory | `compiled/` |

---

## Strategy Testing

Test all compilation strategies in parallel:

**GitHub Actions**:
```yaml
strategy:
  matrix:
    strategy:
      - soft_differentiable
      - hard_boolean
      - fuzzy_godel
      - fuzzy_product
steps:
  - run: tensorlogic "$rule" --strategy ${{ matrix.strategy }}
```

**GitLab CI**:
```yaml
.test_strategy:
  script:
    - tensorlogic "$rule" --strategy ${STRATEGY}

test:soft: { extends: .test_strategy, variables: { STRATEGY: soft_differentiable } }
test:hard: { extends: .test_strategy, variables: { STRATEGY: hard_boolean } }
```

**Jenkins**:
```groovy
matrix {
    axes {
        axis {
            name 'STRATEGY'
            values 'soft_differentiable', 'hard_boolean', 'fuzzy_godel'
        }
    }
}
```

---

## Quality Gates

### Check Graph Size

```bash
stats=$(tensorlogic "$rule" --output-format stats --quiet)
tensors=$(echo "$stats" | grep "Tensors:" | awk '{print $2}')
if [ "$tensors" -gt 100 ]; then
    echo "⚠️  Graph too large!"
    exit 1
fi
```

### Verify Format Conversion

```bash
tensorlogic convert "$rule" --from expr --to json > /dev/null || exit 1
```

### Validate Before Merge

```bash
# Only allow merge if all rules validate
tensorlogic batch rules/*.tl --validate || exit 1
```

---

## Notification Integration

### Slack Notification (GitHub Actions)

```yaml
- name: Notify Slack
  if: failure()
  uses: 8398a7/action-slack@v3
  with:
    status: ${{ job.status }}
    text: 'TensorLogic validation failed!'
    webhook_url: ${{ secrets.SLACK_WEBHOOK }}
```

### Email Notification (GitLab CI)

```yaml
notify:
  script:
    - echo "Validation failed" | mail -s "CI Failure" team@example.com
  only:
    - failure
```

---

## Docker Integration

Build with Docker for consistent environments:

**Dockerfile**:
```dockerfile
FROM rust:latest

RUN cargo install tensorlogic-cli --version 0.1.0-alpha.2

WORKDIR /workspace
COPY rules/ ./rules/

CMD ["tensorlogic", "batch", "rules/*.tl", "--validate"]
```

**Usage**:
```bash
docker build -t tensorlogic-validator .
docker run -v $(pwd)/rules:/workspace/rules tensorlogic-validator
```

---

## Troubleshooting

### Issue: Rust installation timeout

**Solution**: Use pre-built Docker images with Rust installed

### Issue: Large artifact sizes

**Solution**: Set expiration times and compress artifacts

### Issue: Slow compilation

**Solution**:
- Enable caching
- Use smaller domains for CI
- Run expensive tasks only on main branch

---

## Further Reading

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [GitLab CI/CD Documentation](https://docs.gitlab.com/ee/ci/)
- [Jenkins Pipeline Documentation](https://www.jenkins.io/doc/book/pipeline/)
- [TensorLogic CLI Documentation](../README.md)

---

**Happy Integrating!** 🚀
