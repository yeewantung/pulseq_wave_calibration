# README


## Specifications - GRE
1. FLASH for wave calibration (TRA), 8s
2. GRE wave R3x1 (TRA), 0.88x0.88x2.5 mm^3, 3:09 min (Adult FOV), 2:30 min (Kid FOV)

Notes:
- Choose "Transverse" for the slice orientation
- Align the calibration sequence with the GRE wave sequence so that they have the same FOV center and slice orientation
- TE = 20 ms, TR = 30 ms, Bandwidth = 200 Hz/Px
- Adult FOV = 220x220x160 mm^3
- Kid FOV   = 220x172x160 mm^3
- Slice oversampling = 12.5%


## Specifications - MPRAGE
1. FLASH for wave calibration (SAG), 8s
2. MPRAGE wave R3x1 (SAG), 1x1x1 mm^3, 3:33 min (Adult FOV), 3:03 min (Kid FOV)

Notes:
- Choose "Sagittal" for the slice orientation before any rotation of FOV (Otherwise, the pulseq interpreter won't be able to assign PE to the right direction)
- Align the calibration sequence with the GRE wave sequence so that they have the same FOV center and slice orientation
- Bandwidth = 195 Hz/Px
- Adult FOV = 192x256x256 mm^3
- Kid FOV   = 172x220x256 mm^3

