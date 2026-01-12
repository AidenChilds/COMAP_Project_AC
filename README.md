# COMAP_Project_AC
Work completed on the COMAP project with Stuart Harper. Aiming to quantify various features in the M31 maps and find methods to reduce noise levels of the maps.

In the code directory I have included all the code used to reproduce my results.
## Code
### ReadMaps
This file is used in the other modules to read in the maps from .fits file.

### Source_Comparison
This file is used to calculate the flux density of the source 5C 3.50. When initialising the class the map info can be provided in a range of ways. As an array, although a wcs will need to be passed into self.wcs after initialisation, or as a path to a .fits file. Finally a path can be given to a directory containing a set of folders that contain the fits files. The contained folders should be used to separate the files by band and each file in these folders should correspond to the different feeds.

When initialised the code will cut out the source from the given maps and fit a Gaussian to it and save the results as a text file.

Next the function self.calculate() can be run to calculate various features of the source such as the flux density and the shift from the expected position. The function will then plot graphs of these results comparing the different bands and feeds. 

### Map_Noise_Level
This file is used to estimate the red and white noise levels present in each of the maps. When initialising the class the map info can be provided in the same ways as Source_Comparison.

When initialised the code will cut out the centre of the given maps and Fourier Transform this section into a 2D power spectrum. The 2D power spectrum is then binned by radial frequency to produce a 1D power spectrum. The equation $$\sigma_r^2\left(\frac{\nu_{knee}}{\nu}\right)^\alpha + \sigma_w^2$$ is then fitted to the 1D power spectrum, where $$\sigma_r$$ is the red noise, $$\nu_{knee}$$ is the knee frequency, $$\nu$$ is the radial frequency, $$\alpha$$ is the spectral index and $$\sigma_w$$ is the white noise.

The graphs of the power spectrums are then saved along with the results of the fits.
