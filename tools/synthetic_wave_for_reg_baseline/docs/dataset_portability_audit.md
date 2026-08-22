# Dataset-portability completion audit

This completion audit records where acquisition-specific state is resolved for
the active R1 route. It distinguishes manifest-backed production behavior from
historical R3, DICOM, and partial-readout branches that remain intentionally
separate rather than silently becoming defaults for a new dataset.

| Area | Completion state | Authoritative source |
| --- | --- | --- |
| Dataset inspection | Complete. Measured matrix, source acceleration, coils, readout oversampling, source-grid completeness, optional ACS support, and DICOM image type are checked against the contract. | Dataset manifest plus TWIX/DICOM metadata |
| Coil compression | Complete. TWIX, output prefix, physical/virtual coil expectations, explicit image/refscan covariance source, and passed-inspection provenance come from the contract; chunking remains a runtime option. This supports fully sampled R1 even when no PAT refscan exists. | Dataset manifest and passed inspection report |
| No-Wave source reconstruction | Complete for both roles. The direct R1 path requires a complete centered readout and duplicate-free PE grid and applies no interpolation; the accepted large R1 assembly is complete. Existing GRAPPA remains restricted to its validated measured R3x1 stride/residue. | Manifest matrix/settings plus exact measured sampling report |
| Wave synthesis | Complete. The source report, subject, matrix/FOV, virtual-coil count, sequence/TWIX paths, extended readout, trajectory settings, diagnostics, output allocation, and safe complete reuse derive from one hash-matched contract. | Dataset manifest plus passed inspection and validated source report |
| Full-sampling operator validation | Complete. The standard preparation wrapper runs all-coil real-data `PSF=1` no-Wave identity and full-sampling Wave inverse gates immediately after synthesis, before visual review, target masking, or BART export. | Dataset manifest plus validated source and synthesis manifests |
| BART input export | Complete. The synthetic target acceleration, residues, and ACS bounds build a separate post-Wave mask; full readback verifies bitwise acquired samples and exact missing zeros without mutating the accepted synthesis. The older exact product-mask CLI remains compatible. | Dataset manifest plus validated full-Wave synthesis |
| ACS export | Complete. Calibration source is explicit: fully sampled R1 copies measured ACS from the validated compressed image source without interpolation or repeated compression, while compatible acquisitions can use compressed TWIX refscan data. Matrix, Ncc, support, target BART tree, provenance, and safe resume are contract-backed. | Dataset manifest plus validated measured source and target export |
| Lambda-zero reconstruction | Complete. Geometry/FOV, subject, Ncc, ESPIRiT settings, convergence settings, measured BART inputs, output location, and exact resume provenance derive from the contract; production reconstruction requires `bart wave -g`. | Dataset manifest plus calibrated target BART inputs |
| Regularized reconstruction | Complete for the R1 route. Subject, geometry, output checks, solver settings, and case-specific maximum-eigenvalue handling are manifest/configuration-backed; production reconstruction is GPU-only. | Dataset manifest, frozen reconstruction configuration, and measured eigenvalue records |
| Evaluation preparation | Complete for R1 through the exact-grid direct-FFT reference route. Historical DICOM conversion remains available for qualitative context but is not an R1 metric dependency. | Dataset geometry plus hash-bound direct-reference and metrics manifests |
| Volume evaluation | Complete for R1 with reference-neutral direct-FFT evaluation, fixed metrics-only BET support, exact-grid validation, and no registration or interpolation. Historical DICOM-oriented consumers remain outside the active metric route. | Direct-reference, mask, reconstruction, and evaluation manifests |

The active R1 path has no remaining dataset-portability implementation item.
Its direct-source, Wave/operator, mask/export, reconstruction, geometry, and
evaluation gates remain mandatory for safe reuse. A future dataset still must
pass those gates independently; metadata inspection alone never qualifies it.
