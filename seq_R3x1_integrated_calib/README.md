# README


## Specifications - GRE
1. Integrated FLASH-calibration + GRE/SWI wave R3x1 (TRA), 1x1x1 mm^3, 3:24 min (Adult FOV)
  - Single echo: TE = 20 ms, TR = 30 ms
  - Dual echo: TE = 10/20 ms, TR = 30 ms

Notes:
- Choose "Transverse" for the slice orientation
- Choose "R->L" for the PE direction
- Avoid aliasing in PE direction
- Recommend using the adjustment volume + B0 standard shim or using B0 tune up + manual adjustments of B0 3D shim
- Bandwidth = 200 Hz/Px
- Adult FOV = 220x220x160 mm^3
- Slice oversampling = 12.5%



## Specifications - MPRAGE
1. Integrated FLASH-calibration + MPRAGE wave R3x1 (SAG), 1x1x1 mm^3
  - 3:48 min (Adult Extended FOV & Adult FOV)
  - 3:03 min (Kid FOV)

Notes:
- Choose "Sagittal" for the slice orientation before any rotation of FOV (Otherwise, the pulseq interpreter won't be able to assign PE to the right direction)
- Recommend to turn neck channels off to avoid signal contamination
- Bandwidth = 195 Hz/Px
- TR = 2500 ms
- Adult Extended FOV = 220x256x256 mm^3 (recommended)
- Adult FOV = 192x256x256 mm^3
- Kid FOV = 172x220x256 mm^3

