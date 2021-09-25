# Notes
In our original submission, we chareacterized the area and power for a 32X32 systolic array (11% area and 36% power overheads).
In the camera-ready version, we will update this analysis on a 128X128 systolic array, with 4M activation, 4M weight, and 16M Embedding SRAM, to be consistent with Table 3 in the paper.

Our plotting script reflects this update (Area overhead: 7.6%; Power overhead: 27.4%). 

The Area and Power numbers are from logic synthesis reports in `/RecPipe-Artifact/RPAccel_characterization/Synthesis_Report/`. Please check `synthesis.py` for details.
