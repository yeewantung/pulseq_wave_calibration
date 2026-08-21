# Dataset-portability audit

This audit identifies acquisition-specific state that must move behind the
shared dataset manifest before the incoming R1 study can run end to end. It
distinguishes constants that describe the current R3 experiment from algorithm
invariants; those current values remain valid, but must not be defaults for a
new dataset.

| Area | Remaining acquisition-specific state | Planned source |
| --- | --- | --- |
| Dataset inspection | Manifest integration is complete. Measured matrix, source acceleration, coils, readout oversampling, source-grid completeness, optional ACS support, and DICOM image type are checked against the contract. | Dataset manifest plus TWIX/DICOM metadata |
| Coil compression | TWIX, output prefix, and Ncc are separate CLI arguments; the module is otherwise shape-derived. Its product-specific 256×24×256 docstring is stale. | Manifest paths, output root, and virtual-coil count |
| No-Wave GRAPPA | Output volume allocation, PE support, loop bounds, and R=3 offsets assume 256³ and the current product mask. | Manifest matrix and measured sampling report; retain an explicitly R3-only kernel implementation until a general source path is selected |
| Wave synthesis | Subject, 256³×12 input gate, and several output shape records are fixed. | Manifest subject, matrix, FOV, virtual-coil count, and Wave sequence |
| BART input export | The mask filename and description assume the R3x1 product sampling report. | Exact mask derived from the dataset inspection report and synthetic-Wave target acceleration |
| ACS export | Refscan support and BART allocations assume 256³. | Manifest matrix plus measured refscan coordinates |
| Lambda-zero reconstruction | Image/map validation assumes 256³, 12 coils, 256-mm FOV, and a product subject. | Manifest geometry, virtual-coil count, subject, and measured BART inputs |
| Regularized reconstruction | Subject, 256³ output checks, and the R3 maximum-eigenvalue default are fixed. | Manifest subject/geometry; null eigenvalue means measure it for each dataset |
| Evaluation preparation | DICOM matrix and count are fixed at the CLI/validator boundary. | Manifest geometry and selected-series metadata |
| Volume evaluation | Registration, filenames, plots, and metric labels are DICOM-centric. | Configurable GRAPPA/NIfTI/DICOM reference contract, with BET restricted to metrics |

The implementation order follows data flow: propagate the manifest through
coil compression and source reconstruction, then Wave/BART export and
reconstruction, then make evaluation reference-neutral. The incoming dataset
is not qualified merely because its manifest parses; measured inspection checks
must also pass and later reconstruction acceptance gates remain mandatory.
