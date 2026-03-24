[![Tests](https://github.com/fowler-lab/amygda/actions/workflows/tests.yaml/badge.svg)](https://github.com/fowler-lab/amygda/actions/workflows/tests.yaml)
[![PyPI version](https://badge.fury.io/py/amygda.svg)](https://badge.fury.io/py/amygda)

# Automated Mycobacterial Growth Detection Algorithm (AMyGDA)

AMyGDA analyses photographs of antibiotic-containing 96-well plates, estimates growth in each well, and infers minimum inhibitory concentrations (MICs).

A [paper](https://doi.org/10.1099/mic.0.000733) describing the software and demonstrating its reproducibility and accuracy is available in *Microbiology*.

## Installation

AMyGDA now uses modern Python packaging via `pyproject.toml`.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

## Running the CLI

```bash
analyse-plate-with-amygda --image examples/sample-images/01/image-01-raw.png
```

Useful options:

- `--plate_design UKMYC5`
- `--growth_pixel_threshold 130`
- `--growth_percentage 2`
- `--measured_region 0.5`
- `--sensitivity 4`

The command writes filtered images, detected growth overlays, MIC summaries, and saved arrays next to the input image.

## Library usage

```python
from amygda import PlateMeasurement

plate = PlateMeasurement(
    "examples/sample-images/01",
    categories={"ImageFileName": "image-01"},
    plate_design="UKMYC5",
)
plate.load_image("-raw.png")
plate.mean_shift_filter()
plate.equalise_histograms_locally()
plate.stretch_histogram()
plate.save_image("-filtered.jpg")
```

## Development

Run the tests with:

```bash
pytest
```

Run linting and type checks with:

```bash
ruff check .
mypy
```

## Compatibility notes

The legacy `PlateMeasurement` constructor arguments `new`, `tags`, and
`configuration_path` are still accepted for compatibility, but they now emit
deprecation warnings and no longer affect behavior.

## Plate designs

Plate layouts live in `amygda/config/` and are shipped as package data. Each design is defined by three matrices:

- `*-drug-matrix.txt`
- `*-conc-matrix.txt`
- `*-dilution-matrix.txt`

Adding a new plate design means adding a matching set of those files.

## Licence

The software is available subject to the terms of the attached academic-use licence.
