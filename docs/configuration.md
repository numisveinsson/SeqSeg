# Configuration

[← Back to README](../README.md)

## Configuration Files

SeqSeg uses YAML configuration files located in `seqseg/config/`:

| Config File | Purpose |
|-------------|---------|
| `global.yaml` | Default settings |
| `aorta_tutorial.yaml` | Aortic vessel segmentation |
| `global_coro.yaml` | Coronary arteries |
| `global_cereb.yaml` | Cerebral vessels |
| `global_pulm.yaml` | Pulmonary vessels |

## Key Configuration Parameters

```yaml
# Volume extraction
VOLUME_SIZE_RATIO: 5              # Local volume size vs radius (4.9 for aorta, 5.5 for coronaries)
MAGN_RADIUS: 1                    # Radius magnification factor
ADD_RADIUS: 0.3                   # Additional radius for volume extraction (mm)
MIN_RADIUS: 0.3                   # Minimum vessel radius before stopping (mm)

# Tracing control
NR_CHANCES: 2                     # Retry attempts for failed steps
NR_ALLOW_RETRACE_STEPS: 5         # Steps allowed inside existing vessels before stopping
PREVENT_RETRACE: True             # Avoid tracing already segmented areas
ASSEMBLY_EVERY_N: 20              # Combine predictions into assembly every N steps

# Early stopping
STOP_PRE: True                    # Enable premature stopping
STOP_RADIUS: 0.46                 # Stop tracing if radius drops below this (mm)
MAX_STEPS_BRANCH: 1000            # Max steps per branch

# Centerline extraction
CENTERLINE_EXTRACTION_VMTK: False # Use VMTK (True) or built-in FMM (False) for centerlines
```

## Custom Configuration

1. Copy existing config: `cp seqseg/config/global.yaml seqseg/config/my_config.yaml`
2. Modify parameters for your specific use case
3. Run with: `seqseg -config_name my_config ...`
