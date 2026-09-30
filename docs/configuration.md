# Configuration

[← Back to README](../README.md)

SeqSeg reads a YAML file shipped inside the installed package (`seqseg/config/`). Pass the filename without `.yaml`:

```bash
seqseg run batch -config_name global_aorta ...
```

`seqseg run single` uses `--config-name` for the same files.

Print a config (as JSON) or list keys that differ from another packaged file:

```bash
seqseg config dump --name global_aorta
seqseg config fingerprint --name aorta_tutorial --baseline global
```

## Which file to use

| `-config_name` | When |
|----------------|------|
| `global` | Default. Volume size ratio 5 |
| `global_aorta` | Aorta and femoral vessels (MR or CT weights) |
| `aorta_tutorial` | The sample scan in the [tutorial](../seqseg/tutorial/tutorial.md). Same family as `global_aorta`. `DEBUG` is true, and tracing calls `pdb.set_trace()` if the step index reaches `DEBUG_STEP` (10000), which is past the default step cap |
| `global_coro` | Coronary arteries (volume size ratio 5.5) |
| `global_cereb` | Cerebral vessels |
| `global_pulm` | Pulmonary vessels |

Pair the file with the matching weight folder. The table is in [Usage](usage.md#commands-and-flag-spelling) and [Installation](installation.md#model-weights).

Other YAML files in that folder (`global_default`, `global_debug`, `global_test`, `global_seg`) are baselines and test settings. `global_default` is the baseline `seqseg config fingerprint` compares against.

## Units inside the YAML

Seed coordinates and the seed radius follow `-unit` (default centimeters). Lengths in the YAML stay in **millimeters**, including when `-unit cm`:

- `MIN_RADIUS`, `ADD_RADIUS`, `STOP_RADIUS`
- `CENT_MAX_SPACING`

Comments in the files say `mm (keep in mm)`.

## Key parameters

Values below are typical. The file you select has the exact numbers. Inspect it with `seqseg config dump --name <name>`.

```yaml
# Volume extraction
VOLUME_SIZE_RATIO: 5              # Local volume size vs radius (about 4.9 for aorta, 5.5 for coronaries)
MAGN_RADIUS: 1                    # Radius magnification factor
ADD_RADIUS: 0.3                   # Extra radius for volume extraction (mm). Set in the aorta configs
MIN_RADIUS: 0.3                   # Minimum vessel radius before stopping (mm)

# Tracing control
NR_CHANCES: 2                     # Retry attempts for failed steps
NR_ALLOW_RETRACE_STEPS: 5         # Steps allowed inside existing vessels before stopping
PREVENT_RETRACE: True             # Avoid tracing already segmented areas
ASSEMBLY_EVERY_N: 20              # Combine predictions into the assembly every N steps

# Early stopping
STOP_PRE: True                    # Enable premature stopping
STOP_RADIUS: 0.46                 # Stop tracing if radius drops below this (mm)

# Centerline extraction
CENTERLINE_EXTRACTION_VMTK: False # Use VMTK (True) or the built-in method (False)
```

How far the trace runs is set on the command line: `-max_n_steps` (default 1000), `-max_n_branches` (default 100), and `-max_n_steps_per_branch` (default 100). The YAML key `MAX_STEPS_BRANCH` is not read. The tutorial sets the three flags much lower so the example finishes quickly.

## Your own config

`-config_name` only loads files that sit next to the packaged YAMLs. Find that directory:

```bash
python -c "import seqseg.config, pathlib; print(pathlib.Path(seqseg.config.__file__).resolve().parent)"
```

An editable install (`pip install -e .`) prints `seqseg/config/` inside the clone. A normal `pip install` prints a directory inside the environment.

1. Copy a starting point, for example `global.yaml`, to `my_config.yaml` in that directory. Keep a copy outside the environment as well: upgrading the package can replace the directory.
2. Edit the parameters.
3. Run with `-config_name my_config`.
