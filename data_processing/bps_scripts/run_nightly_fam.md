# Render YAML Template Documentation

## Overview

This tool renders a Jinja2-templated YAML configuration file with a `day_obs` value substituted throughout. This allows you to define `day_obs` once and have it consistently applied everywhere it appears in the configuration, avoiding manual find-and-replace errors.

## Files

| File | Description |
|------|-------------|
| `render_yaml.py` | Python script that performs the template rendering |
| `template.yaml` | Jinja2-templated YAML configuration file |

## Prerequisites

- Python 3.6+
- [Jinja2](https://jinja2.palletsprojects.com/) (`pip install jinja2`)

## Template Format

The template file uses [Jinja2 syntax](https://jinja2.palletsprojects.com/en/3.1.x/templates/). Any occurrence of `{{ day_obs }}` will be replaced with the provided value at render time.

### Example: `template.yaml`

```yaml
day_obs: {{ day_obs }}
computeSite: s3df
pipelineYaml: /sdf/home/b/brycek/u/dev-repos/donut_viz/pipelines/production/lsstcam_usdf/lsstCamScienceSensorUSDF_Danish.yaml#step1a-detectors,step1b-visits
payload:
  butlerConfig: "/repo/embargo"
  payloadName: "aos_fam_danish/wep_v16_9_0/donut_viz_v3_6_1/{{ day_obs }}"
  inCollection: "LSSTCam/defaults,u/gmegias/intrinsic_aberrations_collection_temp"
  dataQuery: >-
    instrument='LSSTCam'
    and exposure.observation_type in ('cwfs')
    and detector.purpose in ('SCIENCE')
    and exposure.day_obs in ({{ day_obs }})
    and band in ('u', 'g', 'r', 'i', 'z', 'y')
clusterAlgorithm: lsst.ctrl.bps.quantum_clustering_funcs.dimension_clustering
cluster:
  cluster1:
    pipetasks: isr, generateDonutDirectDetectTask
    dimensions: detector
    equalDimensions: "exposure:visit"
    partitionDimensions: exposure
    partitionMaxClusters: 10000
```

### Template Variables

| Variable | Type | Description |
|----------|------|-------------|
| `day_obs` | `int` | Observation day identifier (e.g., `20260409`) |

## Script Usage

### Basic Usage

```bash
# Print rendered YAML to stdout
python render_yaml.py 20260409
```

### Command-Line Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `day_obs` | Yes | — | The `day_obs` value to substitute (e.g., `20260409`) |
| `-t`, `--template` | No | `template.yaml` | Path to the Jinja2 template YAML file |
| `-o`, `--output` | No | stdout | Path to write the rendered YAML file |

### Examples

```bash
# Render using default template (template.yaml) and print to stdout
python render_yaml.py 20260409

# Render using a custom template file
python render_yaml.py 20260409 -t path/to/my_template.yaml

# Render and write output to a file
python render_yaml.py 20260409 -o config.yaml

# Render with custom template and output file
python render_yaml.py 20260409 -t path/to/my_template.yaml -o config.yaml
```

### Example Output

Running `python render_yaml.py 20260409` produces:

```yaml
day_obs: 20260409
computeSite: s3df
pipelineYaml: /sdf/home/b/brycek/u/dev-repos/donut_viz/pipelines/production/lsstcam_usdf/lsstCamScienceSensorUSDF_Danish.yaml#step1a-detectors,step1b-visits
payload:
  butlerConfig: "/repo/embargo"
  payloadName: "aos_fam_danish/wep_v16_9_0/donut_viz_v3_6_1/20260409"
  inCollection: "LSSTCam/defaults,u/gmegias/intrinsic_aberrations_collection_temp"
  dataQuery: >-
    instrument='LSSTCam'
    and exposure.observation_type in ('cwfs')
    and detector.purpose in ('SCIENCE')
    and exposure.day_obs in (20260409)
    and band in ('u', 'g', 'r', 'i', 'z', 'y')
clusterAlgorithm: lsst.ctrl.bps.quantum_clustering_funcs.dimension_clustering
cluster:
  cluster1:
    pipetasks: isr, generateDonutDirectDetectTask
    dimensions: detector
    equalDimensions: "exposure:visit"
    partitionDimensions: exposure
    partitionMaxClusters: 10000
```

## Adding New Variables

To add additional template variables:

1. **Update `template.yaml`** — Add `{{ variable_name }}` where you want the value substituted.

2. **Update `render_yaml.py`** — Add a new argument to the parser and pass it to `template.render()`:

    ```python
    parser.add_argument("--my_var", type=str, help="Description of my_var")
    ```

    ```python
    rendered = template.render(day_obs=args.day_obs, my_var=args.my_var)
    ```

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| `FileNotFoundError: Template file not found` | Template path is incorrect or file doesn't exist | Check the `-t` path or ensure `template.yaml` exists in the current directory |
| `ModuleNotFoundError: No module named 'jinja2'` | Jinja2 is not installed | Run `pip install jinja2` |
| `{{ day_obs }}` appears literally in output | File is not using Jinja2 syntax or wrong file was passed | Verify the template file path with `-t` |


# Running BPS file on USDF

`nohup bps submit {$BPS_SUBMIT_FILE} > nohup.out &`

`nohup bash service.sh > service.sh &`

Based upon service file template here: https://developer.lsst.io/usdf/batch.html#allocatenodes-auto

Once this run is finished. Add it to the appropriate collection in https://rubinobs.atlassian.net/wiki/spaces/LTS/pages/761397307/LSSTCam+AOS+Datasets#%2Frepo%2Fembargo-collections by the following command:

`butler collection-chain embargo $PARENT_COLLECTION $NEW_COLLECTION`

so for instance this would look like: `butler collection-chain embargo aos_fam_danish_triplets u/brycek/aos_data_run/wep/donut_viz/20260411`
