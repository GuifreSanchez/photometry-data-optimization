# photometry-data-optimization
Filtering and star selection utilities for differential photometry data optimization (as part of my Bachelor's Thesis in Physics).

All programs are (extensively) commented. 

main.py allows for performing comparison star selection through 
  target RSD minimization (main routines are implemented in the starselection.py module) and 
  variability index performance (varindex.py and starselection.py modules are combined in this case).
as well as applying different filtering procedures (sigma-clipping, SNR numerical clipping and binning).

plotter.py contains a variety of plot implementations that allow for generating
graphics from raw, filtered, and filtered + optimized (with results from main.py) target
and comparison stars fluxes (multiple LS periodograms, phase-foldings, index bar charts, etc.).

As stated, details can be found in each of the provided codes. 

examples_TZ_Ari contains some examples using TZ Ari differential photometry
data of graphics that can be obtained combining plotter.py and main.py.

data contains the original (and, in some cases, cropped) photometry files
generated with AstroImageJ from .FITS images taken at Montsec Astronomical Observatory (Catalonia),
corresponding to the 4 studied objects in the thesis: Wolf 1069, TOI-1266, TZ Ari and GJ 555. 
