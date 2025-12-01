# Imports
import numpy as np
import os, re
import matplotlib.pyplot as plt

from astropy import units as u
from astropy import constants as consts
from astropy.modeling.functional_models import Gaussian2D
from astropy.wcs import WCS
from astropy.coordinates import cartesian_to_spherical, spherical_to_cartesian

from scipy.optimize import curve_fit

import ReadMaps

def Elliptical_Gaussian(x, y, amp, centre_x, centre_y, sigma_x, sigma_y, rot, offset, grad_x, grad_y):
    """
    Function used for fitting a 2D elliptical Gaussian

    Parameters
    ----------
    - x:    float or array of floats
            x-coordinates of the Gaussian
    - y:    float or array of floats
            y-coordinates of the Gaussian
    - amp:  float
            Amplitude of the Gaussian
    - centre_x: float
                x-coordinate of the centre of the Gaussian
    - centre_y: float
                y-coordinate of the centre of the Gaussian
    - sigma_x:  float
                Sigma value of the Gaussian in the x direction
    - sigma_y:  float
                Sigma value of the Gaussian in the y direction
    - rot:  float
            Rotation of the Gaussian
    - offset:   float
                Background offset
    - grad_x:   float
                Background gradient in the x direction
    - grad_y:   float
                Background gradient in the y direction
    """

    return Gaussian2D(amplitude=amp, x_mean=centre_x, y_mean=centre_y, x_stddev=sigma_x, y_stddev=sigma_y, theta=rot)(x,y) + offset + grad_x*(x) + grad_y*(y)

def Circular_Gaussian(x, y, amp, centre_x, centre_y, sigma, offset, grad_x, grad_y):
    """
    Function used for fitting a 2D circular Gaussian

    Parameters
    ----------
    - x:    float or array of floats
            x-coordinates of the Gaussian
    - y:    float or array of floats
            y-coordinates of the Gaussian
    - amp:  float
            Amplitude of the Gaussian
    - centre_x: float
                x-coordinate of the centre of the Gaussian
    - centre_y: float
                y-coordinate of the centre of the Gaussian
    - sigma:  float
                Sigma value of the Gaussian
    - offset:   float
                Background offset
    - grad_x:   float
                Background gradient in the x direction
    - grad_y:   float
                Background gradient in the y direction
    """

    return Gaussian2D(amplitude=amp, x_mean=centre_x, y_mean=centre_y, x_stddev=sigma, y_stddev=sigma, theta=0)(x,y) + offset + grad_x*(x) + grad_y*(y)

class SourceComparison:
    """
    Class for analysing a single separated source on a map from a .fits file. Maps should be in units of Kelvin (K) for use in this function

    Parameters
    ----------
    - map_info: None, str or 2D array of floats
                Used to determine how the class will run.
                If map_info remains as None, the class will attempt to read results from already saved data for further analysis.
                If map_info is an array of floats it will be treated as the map for analysis
                If map_info is a string that ends in .fits it will be treated as a path to a singular map to be read and analysed, else it will be treated as a path to folders containing maps from different bands.
    - equation: function
                function that will be used to fit the sources. Can pass in either Circular_Gaussian (default) or Elliptical_Gaussian
    - pixel_size:   float
                    Size of the pixels on the map in arcminutes
    - map_frequency:    float
                        Frequency of the map in GHz
    - freq_shift_per_band:  float
                            Change in frequency between bands when using data from multiple bands in GHz
    - layer:    int
                Determines which layer of the fits files should be used when reading in data
    - data_file:    str
                    Path to a file where data should either be saved or read from
    - delimiter:    str
                    String used to separate values in the file when saving or reading
    - guesses:  1D array of floats
                Initial guesses used for fitting equations must have a length of 7 when using Circular_Gaussian and a length of 9 when using Elliptical Gaussian
    - bounds:   2D array of floats with shape (2, N)
                Defines limits of the fit, where N is the number of coefficients (7 for Circular_Gaussian, 9 for Elliptical_Gaussian)
    """

    def __init__(self, map_info=None, equation=Circular_Gaussian, pixel_size=1/60, map_frequency=26.5, freq_shift_per_band=1, layer=0, data_file='Results.txt', delimiter='\t', guesses=None, bounds=None, cmap=plt.cm.get_cmap('jet')):

        # Map settings
        self.pixel_size = pixel_size
        self.map_frequency = map_frequency
        self.freq_shift_per_band = freq_shift_per_band

        self.data_file = data_file
        self.cmap = cmap
        self.source_location = [120.32184996324, -21.18740142905]

        # Equation to fit to the source
        self.equation = equation
        self.num_coeffs = self.equation.__code__.co_argcount - 2
        self.param_names = self.equation.__code__.co_varnames[2:]

        # Pixels to cut out the source from the map
        self.ypix = [310, 335]
        self.xpix = [335, 360]

        # Lists for storing results from fits
        self.coefficients = []
        self.parameter_errors = []

        # If no map is given attempt to load data from previous runs of the code
        if type(map_info) == type(None):
            self.load_data(data_file, delimiter)
            with open(data_file[:-4]+'_wcs.txt', 'r') as file:
                self.wcs = WCS(file.read())

        # If a map is given convert the map to Jy/pixel and then fit the source
        elif type(map_info) == np.ndarray:
            self.map = map_info
            self.convert_map_units(self.map_frequency)
            coeff, p_err = self.fitting_source(guesses=guesses, bounds=bounds)
            self.coefficients = np.reshape(coeff, (1,1,len(coeff)))
            self.parameter_errors = np.reshape(p_err, (1,1,len(p_err)))
            with open(data_file, 'w') as file:
                string = delimiter.join(self.param_names) + delimiter + delimiter.join([name+'_err' for name in self.param_names]) + '\n'
                file.write(string)
                string = delimiter.join(np.array(self.coefficients[0,0], dtype=str)) + delimiter + delimiter.join(np.array(self.parameter_errors[0,0], dtype=str)) + '\n'
                file.write(string)
            
            with open(data_file[:-4]+'_wcs.txt', 'r') as file:
                self.wcs = WCS(file.read())

        # If path to singular map is given read in the map, convert to Jy/pixel and fit to the source
        elif map_info.endswith(".fits"):
            dataread = ReadMaps.DataRead_Fits(map_info)
            self.map, self.wcs = dataread.extract_data(layer=layer)
            self.convert_map_units(self.map_frequency)
            coeff, p_err = self.fitting_source(guesses=guesses, bounds=bounds)
            self.coefficients = np.reshape(coeff, (1,1,len(coeff)))
            self.parameter_errors = np.reshape(p_err, (1,1,len(p_err)))
            with open(data_file, 'w') as file:
                string = delimiter.join(self.param_names) + delimiter + delimiter.join([name+'_err' for name in self.param_names]) + '\n'
                file.write(string)
                file.write(map_info + '\n')
                string = delimiter.join(np.array(self.coefficients[0,0], dtype=str)) + delimiter + delimiter.join(np.array(self.parameter_errors[0,0], dtype=str)) + '\n'
                file.write(string)
            
            with open(data_file[:-4]+'_wcs.txt', 'w') as file:
                file.write(self.wcs.to_header_string())

        else:
            bands = os.listdir(map_info) # List of band folders
            if map_info[-1] != '/': # Add / to end of path if not already there
                map_info += "/"
            for i, band in enumerate(bands):
                maps = os.listdir(map_info + band) # List of maps in the band folder
                feed=1
                for map in maps:
                    if map[0] == ".": # Skip maps that are saved with . at the start of file name
                        continue
                    else:
                        # Read in each map, convert to Jy/pixel and fit to the sources and save results to a list
                        dataread = ReadMaps.DataRead_Fits(map_info+band+"/"+map, output_info=False)
                        self.map, self.wcs = dataread.extract_data(layer=layer)
                        self.convert_map_units(self.map_frequency + i*self.freq_shift_per_band)
                        coeff, p_err = self.fitting_source(guesses=guesses, bounds=bounds, band_feed=[i, feed])
                        self.coefficients.append(coeff), self.parameter_errors.append(p_err)
                        feed += 1

            # Reshape lists into an array with shape (num bands, num feeds, num coefficients)
            self.coefficients = np.reshape(self.coefficients, (len(bands), len(self.coefficients)//len(bands), self.num_coeffs))
            self.parameter_errors = np.reshape(self.parameter_errors, (len(bands), len(self.parameter_errors)//len(bands), self.num_coeffs))

            with open(data_file, 'w') as file:
                string = delimiter + delimiter.join(self.param_names) + delimiter + delimiter.join([name+'_err' for name in self.param_names]) + '\n'
                file.write(string)
                file.write(map_info)
                shape = np.shape(self.coefficients)
                for i in range(shape[0]):
                    file.write("\nBand {}\n".format(i))
                    for j in range(shape[1]):
                        string = 'Feed {}\t'.format(j+1) + delimiter.join(np.array(self.coefficients[i,j], dtype=str)) + delimiter + delimiter.join(np.array(self.parameter_errors[i,j], dtype=str)) + '\n'
                        file.write(string)
            
            with open(data_file[:-4]+'_wcs.txt', 'w') as file:
                file.write(self.wcs.to_header_string())
        
        self.num_bands, self.num_feeds = np.shape(self.coefficients)[0:2]

    def try_float(self, element):
        """
        Turn given value into a float and return the float. If the given value cannot be converted to a float returns NaN instead
        
        Parameters
        ----------
        - element:  str
                    Value that will be converted to a float or NaN
        
        Returns
        -------
        - element:  float or NaN
                    Value given originally converted to a float or NaN
        """
        try:
            return float(element)
        except:
            return np.nan

    def load_data(self, data_file, delimiter):
        """
        Reads in results from Source Fitting data created by previous runs of the class to be used for further data analysis without having to rerun the fitting

        Parameters
        ----------
        - data_file:    str
                        Path to the data file that will be read in to extract the values of the fits
        - delimiter:    str
                        Delimiter used in the data file to separate the values
        
        Returns
        -------
        - self.coefficients:    array of floats
                                Coeffients found from previous runs of the class.
        - self.parameter_errors:    array of floats
                                    Errors found on the values in self.coefficients
        """
        with open(data_file, 'r') as file:
            data = file.readlines()[1:] # Read lines of the file to create list of strings
            remove_index = []
            band_count = 0
            for i, string in enumerate(data):
                if data[i].startswith('Band'): # Counts number of bands in the data
                    band_count += 1
                data[i] = re.split(delimiter, string) # Splts the data to separate the values
                if len(data[i]) < 2*self.num_coeffs: # Finds lines in the file that do not contain values and saves them to be removed later
                    remove_index.append(i)
                else: # Otherwise each element is converted to a float and if it can't be the element is replaced with NaN
                    for j in range(len(data[i])):
                        data[i][j] = self.try_float(data[i][j])
            for index in remove_index[::-1]: # Removes lines marked above
                del(data[index])
            data = np.array(data) # Converts the list to a numpy array
            if np.isnan(data[:, 0]).all(): # Removes column that marks the different feeds in the file
                data = data[:, 1:]
            
            if not band_count:
                band_count += 1

            self.coefficients = np.reshape(data[:, :self.num_coeffs], (band_count, np.shape(data)[0]//band_count, self.num_coeffs))
            self.parameter_errors = np.reshape(data[:, self.num_coeffs:], (band_count, np.shape(data)[0]//band_count, self.num_coeffs))

            return self.coefficients, self.parameter_errors

    def convert_map_units(self, map_frequency):
        """
        Converts maps from kelvin (K) to Jansky/pixel

        Parameters
        ----------
        - map_frequency:    float
                            Frequency of the map being convert in GHz
        
        Returns
        -------
        - self.map: 2D array of floats
                    Converted map in units of Jy/pixel
        """
        self.map = ((self.pixel_size*u.deg.to(u.rad))**2*2*consts.k_B*((map_frequency)*u.GHz)**2
                      *self.map*u.K*10**30 / (consts.c)**2).to(u.Jy) / u.pixel /10**30
        self.map = np.array(self.map * u.pixel/u.Jy, dtype=float)

        return self.map

    def source_plots(self, x, y, coeff, band_feed):
        """
        Plots and saves figures of the source from the map as well as the model from the fits and the residuals

        Parameters
        ----------
        - x:    2D array of ints
                x-coordinates of each pixel in the cut-out map
        - y:    2D array of ints
                y-coordinates of each pixel in the cut-out map
        - coeff:    1D array of floats
                    Fitted coefficients to be used to model the source
        - band_feed:    1D list of ints with shape (2)
                        Used to name the figures when saving with the first index giving the band number and the second the feed number
        """
        figure = plt.figure(figsize=(10,8))
        vmin, vmax = np.nanmin(self.map), np.nanmax(self.map)
        extent =  [np.nanmin(x)-0.5, np.nanmax(x)+0.5, np.nanmin(y)-0.5, np.nanmax(y)+0.5]
        model_values = self.equation(x, y, *coeff)
        residual = self.map - model_values

        true_plot = figure.add_subplot(131)
        true_img = true_plot.imshow(self.map, vmin=vmin, vmax=vmax, extent=extent, cmap=self.cmap)
        plt.ylabel("Galactic Latitude Shift (arcmin)", fontsize=14)
        plt.xlabel("(a) Data Plot", fontsize=14)
        plt.tick_params(axis='both', labelsize=13, which='both')

        model_plot = figure.add_subplot(132)
        model_img = model_plot.imshow(model_values, vmin=vmin, vmax=vmax, extent=extent, cmap=self.cmap)
        plt.xlabel("(b) Model Plot", fontsize=14)
        plt.tick_params(axis='y', labelleft=False)
        plt.tick_params(axis='both', labelsize=13, which='both')

        resid_plot = figure.add_subplot(133)
        resid_img = resid_plot.imshow(residual, extent=extent, cmap=self.cmap)
        plt.xlabel("(c) Residual Plot", fontsize=14)
        plt.tick_params(axis='y', labelleft=False)
        plt.tick_params(axis='both', labelsize=13, which='both')

        cbar_ax = figure.add_axes([0.1, 0.75, 0.53, 0.05])
        cbar = figure.colorbar(model_img, orientation='horizontal', cax=cbar_ax)
        cbar.set_label("Surface Brightness (mJy pixel$^{-1}$)", fontsize=14)
        cbar_ax.tick_params(axis='x', labelsize=13)
        cbar_ax2 = figure.add_axes([0.67, 0.75, 0.23, 0.05])
        cbar2 = figure.colorbar(resid_img, orientation='horizontal', cax=cbar_ax2)
        cbar2.set_label("Residuals", fontsize=14)
        cbar_ax2.tick_params(axis='x', labelsize=13)

        figure.add_subplot(514, frameon=False)
        # hide tick and tick label of the big axis
        plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
        plt.xlabel("Galactic Longitude Shift (arcmin)", fontsize=14)

        #Save plots
        if type(band_feed) == type(None):
            count = 1
            while os.path.exists(self.data_file + '/../Source_Fit_{}.png'.format(count)):
                count += 1
            plt.savefig(self.data_file + '/../Source_Fit_{}.png'.format(count))
        else:
            count = 1
            while os.path.exists(self.data_file + '/../Band{}_Feed{}_Fit_{}.png'.format(*band_feed, count)):
                count += 1
            plt.savefig(self.data_file + '/../Band{}_Feed{}_Fit_{}.png'.format(*band_feed, count))





    def fitting_source(self, guesses=None, err_data=None, bounds=None, band_feed=None):
        """
        Used to take a full map and cut out a single isolated source. Then fitting that source with a Gaussian function and returning the coefficients found.

        Parameters
        ----------
        - guesses:  1D array of floats
                    Initial guesses used for fitting equations must have a length of 7 when using Circular_Gaussian and a length of 9 when using Elliptical Gaussian
        - err_data: 2D array of floats
                    Error data for each pixel of the map given to provide a weighting to each value when fitting. Otherwise each pixel is given the same weighting
        - bounds:   2D array of floats with shape (2, N)
                    Defines limits of the fit, where N is the number of coefficients (7 for Circular_Gaussian, 9 for Elliptical_Gaussian)
        - band_feed:    1D list of ints with shape (2)
                        Used to name figures when saving them
        """

        self.map = self.map[self.ypix[0]:self.ypix[1], self.xpix[0]:self.xpix[1]] * 1000 #Convert to mJy
        y,x = np.mgrid[:self.map.shape[0], :self.map.shape[1]]
        y -= self.map.shape[0]//2
        x -= self.map.shape[1]//2

        if type(bounds) == type(None):
            bounds = np.tile([[-np.inf], [np.inf]], self.num_coeffs)

        #--------------------------------------------------------------
        # Define new equation with one input using original equation because curve_fit only accepts one variable input
        xy = [x.flatten(),y.flatten()]
        self.param_names = self.equation.__code__.co_varnames[2:] # Names of the coefficients to be fit in the function
        if self.equation == Elliptical_Gaussian:
            self.curve_fit = lambda xy, amp, centre_x, centre_y, sigma_x, sigma_y, rot, offset, grad_x, grad_y: self.equation(xy[0], xy[1], amp, centre_x, centre_y, sigma_x, sigma_y, rot, offset, grad_x, grad_y)
        else:
            self.curve_fit = lambda xy, amp, centre_x, centre_y, sigma, offset, grad_x, grad_y: self.equation(xy[0], xy[1], amp, centre_x, centre_y, sigma, offset, grad_x, grad_y)
        #---------------------------------------------------------------
        # Perform the fit and calculate the error on each coefficient
        coeff, pcov = curve_fit(self.curve_fit, xy, self.map.flatten(), p0=guesses, sigma=err_data, bounds=bounds)
        parameter_err = np.sqrt(np.diag(pcov))
        #--------------------------------------------------------------
        # Output the name of each parameter and the value along with an error value
        for i in range(len(self.param_names)):
            string = f' = ${coeff[i]} \pm {parameter_err[i]}$'
            print(self.param_names[i]+'{}'.format(string))
        
        self.source_plots(x, y, coeff, band_feed)

        return coeff, parameter_err

    def calculate_ellipticity_error(self, maj_ax, maj_ax_err, min_ax, min_ax_err):
        """
        Used to calculate the ellipticity and the error on the ellipticity

        Parameters
        ----------
        - maj_ax:    float
                Major axis of the ellipse (must be larger than min_ax)
        - maj_ax_err:    float
                    Error on the major axis
        - min_ax:    float
                Minor axis of the ellipse (must be smaller than maj_ax)
        - min_ax_err:    float
                    Error on the minor axis

        Returns
        -------
        - ellipticity:  float
                        Calculated ellipticity of the ellipse. Will take a value between 0 and 1
        - ellipticity_error:    float
                                Error on the ellipticity
        """

        ellipticity = (maj_ax-min_ax)/maj_ax
        ellipticity_error = np.sqrt(min_ax**2*(maj_ax**-4)*maj_ax_err**2 + (maj_ax**-2)*min_ax_err**2)
        return ellipticity, ellipticity_error
    
    def Spherical_Rotation(self, coord, source_loc):
        """
        Rotates a point on a sphere by a longitude and latitude angle

        Parameters
        ----------
        - coord:    1D array or list of floats
                    Given as the point that will be rotated in the form [Longitude, Latitude]
        - source_loc:   1D array or list of floats
                        The angle the point will be rotated by [Longitude, Latitude]

        """
        source_loc = np.array(source_loc) * np.pi/180 # Convert from degrees to radians
        coord = np.array(coord) * np.pi/180
        cart = spherical_to_cartesian(1, *coord[::-1]) # Convert to cartesian
        long_rot = [
            [np.cos(-source_loc[0]), -np.sin(-source_loc[0]), 0],
            [np.sin(-source_loc[0]), np.cos(-source_loc[0]), 0],
            [0,0,1]
        ] # Rotation around z axis
        lat_rot = [
            [np.cos(source_loc[1]), 0, np.sin(source_loc[1])],
            [0,1,0],
            [-np.sin(source_loc[1]), 0, np.cos(source_loc[1])]
        ] # Rotation around y axis
        cart = np.dot(lat_rot, np.dot(long_rot, cart)) # Perform both rotations on the coordinates
        spherical = cartesian_to_spherical(*cart) # Convert back to spherical
        spherical = np.array([spherical[2]/u.rad  *180/np.pi, spherical[1]/u.rad  *180/np.pi]) # Convert back to degrees
        spherical = np.where(spherical>180, spherical-360, spherical)

        return spherical
    
    def average_values(self, values, errors):
        """
        Calculate the band average values of a set of values along with the total average value. Uses the mean average.

        Parameters
        ----------
        - values:   2D array of floats
                    Values to be averaged with a shape (N, M), where N is the number of bands and M is the number of feeds
        - errors:   2D array of floats
                    Errors on each of the values given. Must be the same shape as values

        Returns
        -------
        - band_averaged_values: 1D array of floats
                                Band averaged values with a shape (N)
        - band_averaged_errors: 1D array of floats
                                Errors on the band_averaged_values
        - average_value:    float
                            Average value
        - average_error:    float
                            Error on the average_value
        """
        band_averaged_values = np.nansum(values/errors**2, axis=1)/np.nansum(1/errors**2, axis=1)
        band_averaged_errors = np.sqrt(1/np.nansum(1/errors**2, axis=1))

        average_value = np.nansum(values/errors**2)/np.nansum(1/errors**2)
        average_error = np.sqrt(1/np.nansum(1/errors**2))

        return band_averaged_values, band_averaged_errors, average_value, average_error


    def calculate(self):
        """
        Used to calculate all extra values that can be found from the fits of the sources
        """

        if self.equation == Elliptical_Gaussian:
            # Calculate the Ellipticity
            hwhm_x = self.coefficients[:,:,3] * np.sqrt(8*np.log(2)) / 2
            hwhm_x_err = self.parameter_errors[:,:,3] * np.sqrt(8*np.log(2)) / 2
            hwhm_y = self.coefficients[:,:,4] * np.sqrt(8*np.log(2)) / 2
            hwhm_y_err = self.parameter_errors[:,:,4] * np.sqrt(8*np.log(2)) / 2

            self.ellipticity, self.ellipticity_error = np.where(hwhm_x >= hwhm_y, self.calculate_ellipticity_error(hwhm_x, hwhm_x_err, hwhm_y, hwhm_y_err), self.calculate_ellipticity_error(hwhm_y, hwhm_y_err, hwhm_x, hwhm_x_err))

            # Calculate flux density
            amplitude, sigma_x, sigma_y = np.sqeeze(np.split(self.coefficients[:,:,[0,3,4]], [1,2], axis=2), axis=-1)
            amplitude_error, sigma_x_error, sigma_y_error = np.squeeze(np.split(self.parameter_errors[:,:,[0,3,4]], [1,2], axis=2), axis=-1)
            self.flux_density = amplitude * 2 * np.pi * sigma_x * sigma_y
            self.flux_density_error = self.flux_density * np.sqrt((amplitude_error/amplitude)**2 + (sigma_x_error/sigma_x)**2 + (sigma_y_error/sigma_y)**2)
        
        else:
            # Define ellipticity
            self.ellipticity, self.ellipticity_error = [np.zeros(np.shape(self.coefficients[:,:,0])), np.zeros(np.shape(self.parameter_errors[:,:,0]))]

            # Calculate flux density
            amplitude, sigma = np.squeeze(np.split(self.coefficients[:,:,[0,3]], [1], axis=2), axis=-1)
            amplitude_error, sigma_error = np.squeeze(np.split(self.parameter_errors[:,:,[0,3]], [1], axis=2), axis=-1)
            self.flux_density = amplitude * 2 * np.pi * sigma**2
            self.flux_density_error = self.flux_density * np.sqrt((amplitude_error/amplitude)**2 + (sigma_error/sigma)**2 + (sigma_error/sigma)**2)

        # Find centre positions if wcs available
        if hasattr(self, 'wcs'):
            centre_x, centre_y = np.squeeze(np.split(self.coefficients[:,:,[1,2]], [1], axis=2), axis=-1)
            coord_shape = np.shape(centre_x)
            centre_x, centre_y = centre_x.flatten(), centre_y.flatten()
            self.centre_x_err, self.centre_y_err = np.squeeze(np.split(self.parameter_errors[:,:,[1,2]], [1], axis=2), axis=-1)
            self.centre_coords = []
            self.shifted_coords = []
            for x, y in zip(centre_x.flatten(), centre_y.flatten()):
                centre_pos = np.array(self.wcs.pixel_to_world_values(self.xpix[0]+np.diff(self.xpix)//2+x, self.ypix[0]+np.diff(self.ypix)//2+y))
                self.centre_coords.append(centre_pos)
                self.shifted_coords.append(self.Spherical_Rotation(centre_pos, self.source_location))
            
            self.centre_coords = np.reshape(self.centre_coords, (*coord_shape, 2))
            self.shifted_coords = np.reshape(self.shifted_coords, (*coord_shape, 2)) * 60 # arcminutes

            self.pythagorean_shift = np.sqrt(self.shifted_coords[:,:,0]**2 + self.shifted_coords[:,:,1]**2)
            self.pythagorean_shift_error = np.sqrt(((self.shifted_coords[:,:,0]*self.centre_x_err)**2 + (self.shifted_coords[:,:,1]*self.centre_y_err)**2)/(self.shifted_coords[:,:,0]**2 + self.shifted_coords[:,:,1]**2))

        #Average values
        self.band_averaged_ellipticity, self.band_averaged_ellipticity_error, self.average_ellipticity, self.average_ellipticity_error = self.average_values(self.ellipticity, self.ellipticity_error)
        self.band_averaged_flux_density, self.band_averaged_flux_density_error, self.average_flux_density, self.average_flux_density_error = self.average_values(self.flux_density, self.flux_density_error)
        self.band_averaged_pythagorean_shift, self.band_averaged_pythagorean_shift_error, self.average_pythagorean_shift, self.average_pythagorean_shift_error = self.average_values(self.pythagorean_shift, self.pythagorean_shift_error)

        band_averaged_x_shift, band_averaged_x_shift_error, average_x_shift, average_x_shift_error = self.average_values(self.shifted_coords[:,:,0], self.centre_x_err)
        band_averaged_y_shift, band_averaged_y_shift_error, average_y_shift, average_y_shift_error = self.average_values(self.shifted_coords[:,:,1], self.centre_y_err)
        self.average_shift = np.array([average_x_shift, average_y_shift])
        self.average_shift_error = np.array([average_x_shift_error, average_y_shift_error])
    
    def graph_plots(self):
        #Figure for shift in lat and long relative to the source position
        plt.figure()
        plt.scatter(0,0, label="True Location")
        plt.errorbar(self.shifted_coords[:,:,0].flatten(), self.shifted_coords[:,:,1].flatten(), yerr=self.centre_y_err.flatten(), xerr=self.centre_x_err.flatten(), fmt='x', color='k', ecolor='dimgray', capsize=2)
        plt.errorbar(*self.average_shift, yerr=self.average_shift_error[1], xerr=self.average_shift_error[0], fmt='x', color='r', ecolor='red', capsize=2, label="Average Location", zorder=9)
        plt.legend()
        plt.xlabel("Galactic Longitude Shift (arcmins)")
        plt.ylabel("Galactic Latitude Shift (arcmins)")
        plt.gca().invert_xaxis()


        #Figure for pythagorean shift against feed
        if hasattr(self, 'pythagorean_shift'):
            count = np.arange(self.num_bands)
            plt.figure()
            for i, band, band_err in zip(count, self.pythagorean_shift, self.pythagorean_shift_error):
                plt.errorbar(np.arange(1,len(band)+1), band, fmt='x', capsize=2, yerr=band_err, label="Band 0{}".format(i))

            plt.xticks(np.arange(1,self.num_feeds+1))
            plt.xlabel("Feeds")
            plt.ylabel("Shift (arcmins)")
            plt.legend(framealpha=0, loc='upper right', bbox_to_anchor=(0.75, 0.55, 0.5,0.5))
        

        #Figure for flux density against feed
        plt.figure()
        for i in range(self.num_bands):
            plt.errorbar(np.arange(1,self.num_feeds+1), self.flux_density[i, :], label="Band 0{}".format(i), yerr=self.flux_density_error[i,:], fmt='x', capsize=2)
        plt.legend(framealpha=0, loc='upper right', bbox_to_anchor=(0.75, 0.55, 0.5,0.5))
        plt.xticks(np.arange(1,20))
        plt.xlabel("Feed")
        plt.ylabel("Flux Density (mJy)")

        #Figure for band averaged flux density against band
        plt.figure()
        plt.errorbar(np.arange(self.num_bands), self.band_averaged_flux_density, yerr=self.band_averaged_flux_density_error, fmt='kx', capsize=2, ecolor='dimgray')
        plt.xlabel("Band")
        plt.ylabel("Flux Density (mJy)")

