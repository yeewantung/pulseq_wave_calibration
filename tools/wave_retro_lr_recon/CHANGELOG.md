# Changelog

## 0.2.0

- Integrated the imported history into the parent repository.
- Replaced internal cache discovery and Torch CG with an explicit BART
  manifest/companion-input contract and mandatory GPU `bart wave -g`.
- Added crop-first no-wave synthesis, source PSF and native-operator identity
  gates, target-grid PSF regeneration, PE-only map/calibration handling, norm
  restoration, and canonical-RAS NIfTI export.
- Made target phase-encoding matrices the nearest multiple of four and record
  both the requested and matrix-achieved resolutions.
- Added a config-driven CLI, resumable manifests, and `unittest` coverage.

## 0.1.0

- Added batch retrospective physical-resolution center cropping and undersampling.
- Added in-memory source sampling-mask inference from `kspace_cc`.
- Added acceleration inference and `.seq` validation.
- Added target-grid CSM interpolation and PSF rebuilding.
- Added three intermediate-saving levels with `standard` as the default.
- Added achieved-resolution case folders under the existing output's `retro-LR-us/` directory.
- Added case and batch JSON provenance plus grouped NIfTI outputs.
