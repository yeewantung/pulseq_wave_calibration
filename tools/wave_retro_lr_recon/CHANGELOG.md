# Changelog

## 0.1.0

- Added batch retrospective physical-resolution center cropping and undersampling.
- Added in-memory source sampling-mask inference from `kspace_cc`.
- Added acceleration inference and `.seq` validation.
- Added target-grid CSM interpolation and PSF rebuilding.
- Added three intermediate-saving levels with `standard` as the default.
- Added achieved-resolution case folders under the existing output's `retro-LR-us/` directory.
- Added case and batch JSON provenance plus grouped NIfTI outputs.
