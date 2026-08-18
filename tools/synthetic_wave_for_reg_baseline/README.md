# Synthetic Wave regularization baseline

This directory contains the phased no-wave-to-Wave baseline experiment described in `R3x1_no_wave_to_wave_BART_regularization_TODO.md`.

## Layout

```text
scripts/       Data inspection, GRAPPA, and theoretical Wave synthesis programs
requirements/  Incremental Python dependency sets for each phase
tests/         Unit and reference-oracle tests
```

Machine-local dataset notes and generated reconstruction artifacts remain at this directory's root. They are ignored by git; paths and scan filenames stay command-line inputs rather than source constants.

## Current baseline

The active result uses the leading 12 columns of the shared 64→24 nested coil-compression basis and joint multicoil R=3 GRAPPA. Full theoretical Wave k-space and first-coil direct-IFFT diagnostics have been generated; sampling-mask application and BART reconstruction remain gated on visual approval. The 24-coil GRAPPA reconstruction is retained locally as a comparison.

## Setup and tests

From the repository root:

```bash
python -m pip install -r tools/synthetic_wave_for_reg_baseline/requirements/phase-d.txt
python -m unittest discover \
    -s tools/synthetic_wave_for_reg_baseline/tests \
    -p 'test_*.py'
```

Run a script with `--help` for its dataset-independent CLI, for example:

```bash
python tools/synthetic_wave_for_reg_baseline/scripts/phase_c_grappa.py --help
```
