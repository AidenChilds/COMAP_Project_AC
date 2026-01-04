# COMAP_Project_AC
Work completed on the COMAP project with Stuart Harper. Aiming to quantify various features in the M31 maps and find methods to reduce noise levels of the maps.

In the code directory I have included all the code used to reproduce my results.
## Code
### ReadMaps
This file is used in the other modules to read in the maps from .fits file.

### Source_Comparison
This file is used to calculate the flux density of the source 5C 3.50. When initialising the class the map info can be provided in a range of ways. As an array, although a wcs will need to be passed into self.wcs after initialisation, or as a path to a .fits file. Finally a path can be given to a directory containing a set of folders that contain the fits files. The contained folders should be used to separate the files by band and each file in these folders should correspond to the different feeds.

When initialised the code will cut out the source from the given maps and fit a Gaussian to it and save the results as a text file.

Next the function self.calculate() can be run to calculate various features of the source such as the flux density and the shift from the expected position. The function will then plot graphs of these reults comparing the different bands and feeds. 
