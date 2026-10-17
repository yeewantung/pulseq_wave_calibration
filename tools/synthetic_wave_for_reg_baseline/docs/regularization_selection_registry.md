# Reviewed regularization selection registry

Updated: 2026-10-12, America/New_York

## Purpose and interpretation

This is the durable, path-independent index of every explicit regularization
choice currently described by this tool's experiment records. It is intended
to survive archival or deletion of large local sweep trees. Generated
manifests, metric tables, figures, and reconstruction payloads remain the
primary provenance and must be retained in the verified external backup.

The values below are reviewed selections, not automatically inferred optima.
They apply only to the stated acquisition, operator, coil-processing, geometry,
and evaluation contract. Identical numerical values in different rows do not
imply parameter transferability.

## Current reviewed selections

| Study contract | Case | Method | Selected lambda | Status and authority |
| --- | --- | --- | ---: | --- |
| Standard-PCA MPRAGE, Ncc=24, no noise prewhitening | native R3x1 | Wavelet | `3e-2` | User-reviewed reassessment on the 2026-08-21 dataset; current measured MPRAGE VCC24 normal default |
| Standard-PCA two-echo GRE, Ncc=24, no noise prewhitening | native R3x1, one shared value across echoes | Wavelet | `1.5e-2` | User-reviewed reassessment on the 2026-08-21 dataset; current measured GRE VCC24 normal default |
| Corrected pure-image-lattice MPRAGE | native R3x1 | Wavelet | `3.5e-2` | Explicit five-case selection |
| Corrected pure-image-lattice MPRAGE | native R3x2 | Wavelet | `3.5e-2` | Explicit five-case selection |
| Corrected pure-image-lattice MPRAGE | LR-X R3x2 | Wavelet | `2.5e-2` | Explicit five-case selection |
| Corrected pure-image-lattice MPRAGE | LR-Y R3x2 | Wavelet | `2.5e-2` | Explicit five-case selection |
| Corrected pure-image-lattice MPRAGE | LR-XY R3x2 | Wavelet | `2.2e-2` | Explicit five-case selection |
| Corrected native-grid MPRAGE | native R3x3 | Wavelet | `4.5e-2` | Independent explicit native-R3x3 selection |
| Native-grid two-echo synthetic-Wave GRE | native R3x3, one shared value across echoes | Wavelet | `1.5e-2` | Explicit shared-lambda selection after magnitude, phase, scaling, and delta-B0 review |
| Fully sampled-source synthetic-Wave MPRAGE, historical Ncc=12 contract | full-resolution R3x2 | Wavelet | `1.5e-2` | Frozen historical study value, superseded for cleaned measured-tool defaults; later low-resolution use was initially a transfer, not a new optimum |
| Synthetic-Wave MPRAGE presentation study | native R3x1 | Wavelet | `2.2e-2` | Explicit user selection; `automatic_selection_performed=false` |
| Retrospective-resolution MPRAGE presentation | lower-X, approximately 1.49 x 1 x 1 mm | Wavelet | `1e-2` | Explicit per-grid presentation choice |
| Retrospective-resolution MPRAGE presentation | lower-Y, approximately 1 x 1.49 x 1 mm | Wavelet | `5e-3` | Explicit per-grid presentation choice |
| Retrospective-resolution MPRAGE presentation | balanced, approximately 1.25 x 1.25 x 1 mm | FISTA | `0` | Explicit unregularized presentation choice |
| No-Wave R3x1 PICS presentation study | native R3x1 | Wavelet | `1e-3` | Explicit user selection for this PICS operator only |

## Hash-bound authorities

- The unwhitened MPRAGE Ncc=24 reassessment has metrics-provenance SHA-256
  `f4f199eae91077b25f742249912da4800a57198b4d742ce5847e01dceb601b78`
  and metric-CSV SHA-256
  `ca5ea4689cd947e0e989d808bd73ae0f4f812b73b4871a6eb69d0c783bcc80f8`.
- The unwhitened GRE Ncc=24 shared-lambda reassessment has evaluation-manifest
  SHA-256
  `fecbac4b66c6ba3b6d8723e4e47cc67bb2e40ead6bf88a9cc509914c111544f6`
  and metric-CSV SHA-256
  `bbcd3dc2245d4a309b4550bb60b099294b7ba350525ecbef66d089c9642ab656`.
- The corrected five-case MPRAGE selection-manifest SHA-256 is
  `07cd8fe9f859ee125e76a338a30fcfc5e79c4c2f46ca9c43d5f454ec32ea90f6`.
- The independent native-R3x3 MPRAGE selection-manifest SHA-256 is
  `07fec1879821dcef6cd177766224f23930a0c556c96a28055a339c6530b6002d`.
- The synthetic-Wave R3x1 and retrospective-resolution choices remain bound
  by their private selection manifests, indexed in `HANDOVER.md` and the
  private output-root indexes. Their machine paths are intentionally absent
  from tracked source.

## Sweeps without a selected winner

The following completed or planned evaluations must not be converted into a
default merely because one metric has a numerical leader:

- full-resolution and retrospective corrected-LLR comparisons: no LLR
  block/lambda was selected;
- retrospective resolution tradeoff: no preferred resolution was selected;
- historical R3 presentation pilots and the ecalib intensity diagnostic: no
  new production regularization default was selected;
- MPRAGE and GRE retrospective VCC24 Wavelet values remain transferred unless
  a geometry-specific review says otherwise; their normal defaults do not
  establish retrospective optima; and
- noise-prewhitened MPRAGE and GRE reassessments are pending. No unwhitened
  value may be relabeled as the prewhitened optimum.

## Deletion and archive gate

Before deleting any local sweep tree, verify its external backup by hash and
retain at least its configuration snapshot, preparation and reconstruction
manifests, selection/evaluation manifests, metric CSVs, review figures, logs,
and the selected candidate's manifest and image products. Record the backup
identity and verification result in the private output index. This tracked
registry records decisions but is not a substitute for generated provenance.
