# Reconstruction environment setup

This tool uses two independent runtime layers:

1. a Python 3.11--3.13 environment for TWIX/sequence preparation, validation,
   BART CFL I/O, and NIfTI conversion; and
2. a BART executable for CPU reconstruction, optionally built with CUDA for
   reconstructions explicitly requested with `-g`.

Installing CUDA-enabled PyTorch or CuPy in Python does not make BART
GPU-enabled. Conversely, a CUDA-enabled BART build does not install the Python
dependencies. Prepare and validate both layers before using the sample scripts.

## Obtain the source and submodules

From the parent repository root, initialize the pinned Wave-MPRAGE and
Wave-GRE dependencies, including their nested sequence-safety dependency:

```bash
git submodule sync --recursive
git submodule update --init --recursive
cd tools/wave_retro_lr_recon
```

The MPRAGE adapter imports focused helpers from `external/wave-mprage`. The
measured single- or multi-echo GRE adapter imports the independently versioned calibration,
TWIX, coil-compression, trajectory, and NIfTI helpers from the pinned
`external/wave-gre-flow-comp` implementation.
Separately installed packages or unrelated sibling checkouts are not
substitutes for these pinned sources. The tracked submodule URLs use public
HTTPS so read-only users do not need GitHub SSH keys.

## Create the Python environment

### Recommended: standard venv and pip

Pip 25.1 or newer can install the same standardized dependency group directly
from `pyproject.toml`:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade "pip>=25.1"
python -m pip install --group recon
python scripts/prepare_mprage_normal.py --help
```

Python 3.12 and 3.13 are also supported; substitute an explicitly available
supported interpreter when Python 3.11 is unavailable. Keep the environment
inside the tool directory or in another user-controlled location; do not
commit it.

### Optional: uv

Users who already manage Python environments with `uv` may create the same
tool-local `.venv` by running:

```bash
uv sync
source .venv/bin/activate
python scripts/prepare_mprage_normal.py --help
```

The `recon` dependency group in `pyproject.toml` is the default uv group. This
tool currently has no committed `uv.lock`, so `uv sync` resolves versions that
satisfy the recorded bounds rather than reproducing an exact lockfile.

Commands can also be run without activating the environment:

```bash
uv run python scripts/prepare_mprage_normal.py --help
```

## Build BART with CPU support and optional CUDA

Use a reviewed BART revision compatible with this tool; the validated workflow
uses the BART v1.0 command interface. Every BART build retains CPU execution.
Enabling CUDA adds the `-g` backend rather than replacing CPU support.

### CPU-only build

Install a C/C++ compiler, FFTW, BLAS, LAPACK/LAPACKE, GNU Make, and the other
prerequisites listed by the selected BART revision. In the BART source root,
create `Makefile.local`. A typical OpenBLAS configuration is:

```make
CC = gcc
CXX = g++

OPENBLAS = 1
OMP = 1
FFTWTHREADS = 1
```

Do not set `CUDA = 1` for a CPU-only build. `OPENBLAS = 1` selects OpenBLAS as
the CPU BLAS/LAPACK backend; adapt that setting if the host uses another
supported implementation.

### Add CUDA support

To build the same CPU-capable executable with an optional GPU backend, install
a CUDA toolkit compatible with the host compiler and add these settings:

```make
CUDA = 1
CUDA_BASE = /path/to/cuda
CUDA_LIB = lib64
GPUARCH_FLAGS = -gencode arch=compute_SM,code=sm_SM
```

Replace both `SM` tokens with the target GPU compute capability without the
decimal point, for example `89` for compute capability 8.9. Some CUDA layouts
use `lib` rather than `lib64`. BART v1.0 derives `NVCC` from
`CUDA_BASE/bin/nvcc` and defaults `CUDA_CC` to `CC`; set either explicitly only
when those defaults are incorrect. Additional include, linker, rpath, or BLAS
settings may be required by the host; keep them in the host build rather than
in this source repository.

Build BART according to its own revision-specific documentation. A typical
source build is:

```bash
make clean
make -j4
```

Never reuse `Makefile.local` blindly across servers. For CUDA builds, recheck
the target GPU architecture and CUDA/host-compiler pairing.

## Reactivate the environment for continued work

The tool-local `.venv` and compiled BART tree persist after the shell closes.
After a fresh clone, or after pulling a parent-repository change that updates
submodule URLs or pinned revisions, synchronize and initialize all nested
dependencies before reactivating the environment:

```bash
git pull
git submodule sync --recursive
git submodule update --init --recursive
```

These commands are not required every time a new shell is opened when the
recorded submodule revisions are already present.

In every new shell, activate that same venv and then expose the matching BART
source tree. Do not recreate the venv or reinstall packages for routine use:

```bash
cd /path/to/pulseq_wave_calibration/tools/wave_retro_lr_recon
source .venv/bin/activate

export BART_TOOLBOX_PATH=/path/to/bart-source
export TOOLBOX_PATH="$BART_TOOLBOX_PATH"
export PATH="$BART_TOOLBOX_PATH:$PATH"
```

Confirm that both executables resolve from the intended locations:

```bash
command -v python
command -v bart
python --version
bart version
```

`command -v python` should resolve inside `wave_retro_lr_recon/.venv/bin`.
Save the activation and BART exports in `scripts/environment.local.sh` when a
one-command local setup is useful; that machine-specific filename is ignored
by Git. Run `deactivate` when finished. Reinstall the `recon` dependency group
only when `pyproject.toml` changes.

## Validate both layers

Validate the Python environment and either CPU or CUDA-enabled BART with:

```bash
python --version
python - <<'PY'
import nibabel
import numpy
import pymapvbvd
import pypulseq
import scipy
import sigpy
import torch
print("Python reconstruction imports: OK")
PY

command -v bart
bart version
bart wave -h
```

For a CUDA-enabled build, continue on a node or allocation where the intended
GPU is visible:

```bash
nvidia-smi
bart wave -h 2>&1 | grep -- '-g'
```

On Linux, confirm that the CUDA-enabled executable resolves its GPU libraries
and has no missing dependency:

```bash
ldd "$(command -v bart)" | grep -E 'cuda|cufft|cublas|not found'
```

If the BART build includes its CUDA FFT test executable, run it on the GPU
node:

```bash
if [[ -x "$BART_TOOLBOX_PATH/test_cudafft" ]]; then
    "$BART_TOOLBOX_PATH/test_cudafft"
fi
```

The help text and linked libraries establish that CUDA support was compiled;
only a CUDA test or an explicitly requested `-g` operation on a visible GPU
validates runtime execution. `nvidia-smi` failing in a login shell may simply
mean that the GPU is available only inside a scheduled allocation.

## Tool-specific BART behavior

- The sample workflows run `bart wave` on CPU by default. Passing `-g` to a
  sample script selects its explicit `bart wave -g` branch.
- The validated BART v1.0 `ecalib` command has no `-g` option, so this tool runs
  `bart ecalib -m 1 -c ...` on CPU exactly once and reuses its recorded maps.
- Python prepares inputs and converts outputs but never launches BART. Activate
  Python and BART in the same shell before calling a sample Bash script.
- Keep actual environment, CUDA, BART, TWIX, sequence, and output paths only in
  ignored `*.local.sh` or `*.local.json` files.

After setup, review the dataset-independent commands in `README.md` or run:

```bash
scripts/sample_gre_normal_recon.sh --help
scripts/sample_gre_retro_lr_recon.sh --help
scripts/sample_gre_nifti_collection.sh --help
scripts/sample_mprage_normal_recon.sh --help
scripts/sample_mprage_retro_lr_recon.sh --help
scripts/sample_mprage_nifti_collection.sh --help
```

## Macha environment

On host `macha`, use the site environment and compatible host BART build:

```bash
source ~/cluster/miniforge3/etc/profile.d/conda.sh
conda activate cuda133py312-macha
source ~/cluster/bart/bart_startup.sh
command -v python
command -v bart
```

Run tests before entering the production commands in tmux. The agent does not
launch measured reconstruction or choose an output directory; the user must
confirm the exact output root supplied to either sample script.
