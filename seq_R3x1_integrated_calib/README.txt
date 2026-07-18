# README


## Specifications - MPRAGE
1. Integrated FLASH-calibration + MPRAGE wave R3x1 (SAG), 1x1x1 mm^3, 3:46 min (Adult FOV), 3:01 min (Kid FOV)

Notes:
- Choose "Sagittal" for the slice orientation before any rotation of FOV (Otherwise, the pulseq interpreter won't be able to assign PE to the right direction)
- Align the calibration sequence with the GRE wave sequence so that they have the same FOV center and slice orientation
- Bandwidth = 195 Hz/Px
- Adult FOV = 192x256x256 mm^3
- Kid FOV   = 172x220x256 mm^3

