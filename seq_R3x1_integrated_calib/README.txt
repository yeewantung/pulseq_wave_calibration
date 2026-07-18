# README


## Specifications - GRE
1. Integrated FLASH-calibration + GRE/SWI wave R3x1 (TRA), 1x1x1 mm^3, 3:24 min (Adult FOV), 2:45 min (Kid FOV)

Notes:
- Choose "Transverse" for the slice orientation
- Recommend using the adjustment volume + B0 standard shim or using B0 tune up + manual adjustments of B0 3D shim
- TE = 20 ms, TR = 30 ms, Bandwidth = 200 Hz/Px
- Adult FOV = 220x220x160 mm^3
- Kid FOV   = 220x172x160 mm^3
- Slice oversampling = 12.5%



## Specifications - MPRAGE
1. Integrated FLASH-calibration + MPRAGE wave R3x1 (SAG), 1x1x1 mm^3, 3:46 min (Adult FOV), 3:01 min (Kid FOV)

Notes:
- Choose "Sagittal" for the slice orientation before any rotation of FOV (Otherwise, the pulseq interpreter won't be able to assign PE to the right direction)
- Bandwidth = 195 Hz/Px
- Adult FOV = 192x256x256 mm^3
- Kid FOV   = 172x220x256 mm^3

