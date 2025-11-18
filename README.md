# Multi LED Solar Simulator Calibration for Tandem-I-V

Please note: This repository is functional, however, it is currently undergoing final cleaning, organization and documentation. 

A jupyter notebook enabling virtual calibration for solar simulators according to IEC 60904-1-1, for tandem I-V measurements. This code was written by Antoine Bourgeois, Stella Hadiwidjaja, and Zoltan Nicot-Senneville. 

Before running the code, the user requires:
- Spectral responses of reference cells and device-under-test.
- Short-circuit of reference cells under reference spectrum.
- Characterization of each LED at control parameter intervals 0.1, 0.2, ...1.

Note for Mac Users: due to the existence of the hidden file '.DS_Store', get_LEDs function may run into an error. Delete the '.DS_Store' in terminal by navigating to the directory and using:

find . -name ".DS_Store" -print -delete
