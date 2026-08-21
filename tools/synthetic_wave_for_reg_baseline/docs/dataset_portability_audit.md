# Dataset-portability audit

This audit identifies acquisition-specific state that must move behind the
shared dataset manifest before the incoming R1 study can run end to end. It
distinguishes constants that describe the current R3 experiment from algorithm
invariants; those current values remain valid, but must not be defaults for a
new dataset.

| Area | Remaining acquisition-specific state | Planned source |
| --- | --- | --- |
| Dataset inspection | Manifest integration is complete. Measured matrix, source acceleration, coils, readout oversampling, source-grid completeness, optional ACS support, and DICOM image type are checked against the contract. | Dataset manifest plus TWIX/DICOM metadata |
| Coil compression | Manifest integration is complete. TWIX, output prefix, physical/virtual coil expectations, explicit image/refscan covariance source, and passed-inspection provenance come from the contract; chunking remains a runtime option. This supports fully sampled R1 even when no PAT refscan exists. | Dataset manifest and passed inspection report |
| No-Wave source reconstruction | Code support is complete for both roles. The direct R1 path requires a complete centered readout and duplicate-free PE grid and applies no interpolation; existing GRAPPA remains restricted to its validated measured R3x1 stride/residue. The new MID00198 R1 metadata inspection has passed; large direct assembly is pending. | Manifest matrix/settings plus exact measured sampling report |
| Wave synthesis | Manifest integration is complete. The source report, subject, matrix/FOV, virtual-coil count, sequence/TWIX paths, extended readout, trajectory settings, diagnostics, output allocation, and safe complete reuse derive from one hash-matched contract. | Dataset manifest plus passed inspection and validated source report |
| BART input export | Manifest integration is complete. The synthetic target acceleration, residues, and ACS bounds build a separate post-Wave mask; full readback verifies bitwise acquired samples and exact missing zeros without mutating the accepted synthesis. The older exact product-mask CLI remains compatible. | Dataset manifest plus validated full-Wave synthesis |
| ACS export | Manifest integration is complete. Calibration source is explicit: fully sampled R1 copies measured ACS from the validated compressed image source without interpolation or repeated compression, while compatible acquisitions can use compressed TWIX refscan data. Matrix, Ncc, support, target BART tree, provenance, and safe resume are contract-backed. | Dataset manifest plus validated measured source and target export |
| Lambda-zero reconstruction | Manifest integration is complete. Geometry/FOV, subject, Ncc, ESPIRiT settings, lambda-zero convergence settings, measured BART inputs, output location, and exact resume provenance all derive from the contract; `bart wave -g` is mandatory. Execution awaits preparation and full-Wave visual approval. | Dataset manifest plus calibrated target BART inputs |
| Regularized reconstruction | Subject, 256³ output checks, and the R3 maximum-eigenvalue default are fixed. | Manifest subject/geometry; null eigenvalue means measure it for each dataset |
| Evaluation preparation | DICOM matrix and count are fixed at the CLI/validator boundary. | Manifest geometry and selected-series metadata |
| Volume evaluation | Registration, filenames, plots, and metric labels are DICOM-centric. | Configurable GRAPPA/NIfTI/DICOM reference contract, with BET restricted to metrics |

The implementation order follows data flow: run and accept lambda zero next,
then propagate the manifest through regularized BART reconstruction and make
evaluation reference-neutral. The new dataset is not qualified merely because
its metadata inspection passes; direct-source, Wave, and reconstruction
acceptance gates remain mandatory.
