# Configuration

This document describes the configuration options for the verification approach.

## Verification Parameters

The verification approach has the following parameters:

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `kinf` | Number of candidate responses to generate | 200 |
| `kverif` | Number of verification samples per response | 50 |
| `ktie` | Number of comparison samples per pair | 100 |
| `threshold` | Threshold for considering responses as tied | 0.05 |

## Configuration Example

Here's an example configuration in YAML format:

```yaml
verification:
  enabled: true
  kinf: 200
  kverif: 50
  ktie: 100
  threshold: 0.05
```

## Integration with DAPO Configuration

The verification configuration will be integrated with the DAPO configuration system. The parameters will be added to the configuration file used by the DAPO framework.

Example integration with the DAPO configuration:

```yaml
actor_rollout_ref:
  actor:
    # Existing DAPO configuration...
  rollout:
    # Existing DAPO configuration...
    verification:
      enabled: true
      kinf: 200
      kverif: 50
      ktie: 100
      threshold: 0.05
```

## Scaling Considerations

The paper demonstrates that performance continues to improve as the verification parameters are scaled up. However, there are diminishing returns and computational constraints to consider.

### Recommended Values

Based on the paper, the following values are recommended for different computational budgets:

| Computational Budget | kinf | kverif | ktie |
|----------------------|------|--------|------|
| Low | 10 | 5 | 10 |
| Medium | 50 | 20 | 50 |
| High | 200 | 50 | 100 |

### Performance Impact

The paper shows that:

1. Scaling up `kinf` (number of candidate responses) has the largest impact on performance, especially for difficult problems.
2. Scaling up `kverif` (number of verification samples) improves verification accuracy.
3. Scaling up `ktie` (number of comparison samples) improves tie-breaking accuracy.

The optimal values depend on the specific task and computational budget.
