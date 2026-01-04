import numpy as np
import matplotlib.pyplot as plt
import os

from scipy.optimize import curve_fit

import ReadMaps


class NoiseLevel:
    """
    Class for analysing a single separated source on a map from a .fits file. Maps should be in units of Kelvin (K) for use in this function

    Parameters
    ----------
    - map_info: None, str or 2D array of floats
                Used to determine how the class will run.
                If map_info remains as None, the class will attempt to read results from already saved data for further analysis.
                If map_info is an array of floats it will be treated as the map for analysis
                If map_info is a string that ends in .fits it will be treated as a path to a singular map to be read and analysed, else it will be treated as a path to folders containing maps from different bands.
    - pixel_size:   float
                    Size of the pixels on the map in arcminutes
    - layer:    int
                Determines which layer of the fits files should be used when reading in data
    - data_file:    str
                    Path to a file where data should either be saved or read from
    - delimiter:    str
                    String used to separate values in the file when saving or reading
    - cmap: colormap
            Colormap used when plotting the 2D power spectrum
    - fontsize: int
                Fontsize used for labels on the plots and fontsize - 1 used for the ticks
    - guesses:  array of floats
                Initial guesses used for the fits
    - knee_freq:    float
                    Assumed value for the knee frequency of the 1D power spectrum in arcmin$^{-1}$
    - remove_low_bins:  int
                        Number of low frequency bins removed from the start of the array for fitting and plotting
    - error_type:   string
                    Determines the type of error used when fitting or plotting. If 'bin_errors' is passed, the errors are calculated from the standard deviation of values within the bin, if 'num_modes' is passed the errors are calculate using the reciprocal of the number of values in each bin. Otherwise no errors will be applied and all data will have standard weights when fitting.
    - log_value_fitting:    bool
                            Determines if values are logged and fitted with a logged equation before fitting. This is used to reduce weighting of high power low frequency bins that pulls the fit away from the high frequency bins
    - sigma_residual:   bool
                        Determines if the residual shows the number of sigma the value is away from the model (True) or a standard residual (False)
    - normalise_red_chi_2:  bool
                            Determines if errors are normalised by multiplying them by the square root of the reduced chi squared value to bring the reduced chi squared to 1

    """

    def __init__(self, map_info=None, num_bins=100, pixel_size=1, layer=0, data_file='Results.txt', delimiter='\t', cmap=plt.cm.get_cmap('jet'), fontsize=14, guesses=None, knee_freq=0.1, remove_low_bins=1, error_type="bin_errors", log_value_fitting=False, sigma_residual=True, normalise_red_chi_2=False):

        # Map settings
        self.pixel_size = pixel_size

        self.data_file = data_file
        self.cmap = cmap

        self.fontsize = fontsize

        # Pixels to cut out the centre of the map
        self.ypix = [260, 340]
        self.xpix = [200, 375]

        self.num_bins = num_bins

        # Lists for storing results from fits
        self.coefficients = []
        self.parameter_errors = []

        # If a map is given FT the map and bin the data in radial frequency to produce 1D power spec and fit the results
        if type(map_info) == np.ndarray:
            self.map = map_info
            self.powerspec()
            coeff, p_err = self.fitting_power_spec(guesses, knee_freq, remove_low_bins, error_type, log_value_fitting, sigma_residual, normalise_red_chi_2)
            self.coefficients = np.reshape(coeff, (1,1,len(coeff)))
            self.parameter_errors = np.reshape(p_err, (1,1,len(p_err)))
            with open(data_file, 'w') as file:
                string = delimiter.join(self.param_names) + delimiter + delimiter.join([name+'_err' for name in self.param_names]) + '\n'
                file.write(string)
                string = delimiter.join(np.array(self.coefficients[0,0], dtype=str)) + delimiter + delimiter.join(np.array(self.parameter_errors[0,0], dtype=str)) + '\n'
                file.write(string)

        # If path to singular map is given read in the map, FT the map and bin the data in radial frequency to produce 1D power spec and fit the results
        elif map_info.endswith(".fits"):
            dataread = ReadMaps.DataRead_Fits(map_info)
            self.map, self.wcs = dataread.extract_data(layer=layer)
            self.powerspec()
            coeff, p_err = self.fitting_power_spec(guesses, knee_freq, remove_low_bins, error_type, log_value_fitting, sigma_residual, normalise_red_chi_2)
            self.coefficients = np.reshape(coeff, (1,1,len(coeff)))
            self.parameter_errors = np.reshape(p_err, (1,1,len(p_err)))
            with open(data_file, 'w') as file:
                string = delimiter.join(self.param_names) + delimiter + delimiter.join([name+'_err' for name in self.param_names]) + '\n'
                file.write(string)
                file.write(map_info + '\n')
                string = delimiter.join(np.array(self.coefficients[0,0], dtype=str)) + delimiter + delimiter.join(np.array(self.parameter_errors[0,0], dtype=str)) + '\n'
                file.write(string)
            

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
                        # Read in each map, FT the map and bin the data in radial frequency to produce 1D power spec and fit the results, then save results to a list
                        dataread = ReadMaps.DataRead_Fits(map_info+band+"/"+map, output_info=False)
                        self.map, self.wcs = dataread.extract_data(layer=layer)
                        self.powerspec(band_feed=[i, feed])
                        coeff, p_err = self.fitting_power_spec(guesses, knee_freq, remove_low_bins, error_type, log_value_fitting, sigma_residual, normalise_red_chi_2, band_feed=[i, feed])
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

    def powerspec(self, band_feed=None):
        """
        Cuts out a section of the map to be used in the power spectrum and then Fourier transforms the map to create the 2D power spectrum. Each pixel is then given a radial frequency mode calculated by the x and y direction frequency modes. These radial frequencies are then used to bin the 2D power spectrum to produce a 1D power spectrum.

        Parameters
        ----------
        - band_feed:    list of ints
                        Used to name the images when saving the plots
        """

        cut_map = self.map[int(self.ypix[0]):int(self.ypix[1]),int(self.xpix[0]):int(self.xpix[1])] # Cut out required section of the map
        self.fft_cut_map = np.fft.fft2(cut_map) # FT the map
        ky, kx = np.meshgrid(np.fft.fftfreq(cut_map.shape[1], d=self.pixel_size), np.fft.fftfreq(cut_map.shape[0],d=self.pixel_size)) # Calculate the frequency modes for each pixel
        self.kr = np.sqrt(kx**2 + ky**2) # Calculate radial frequency mode for each pixel
        kr_bins = np.linspace(0, self.kr.max(), self.num_bins+1) # Create array of bin edge values

        self.kr_bin_centers = 0.5*(kr_bins[1:] + kr_bins[:-1]) # Calculate the centre value of each bin
        bin_powers = np.histogram(self.kr.flatten(), bins=kr_bins, weights=np.abs(self.fft_cut_map.flatten())**2)[0] # Sum powers in each bin
        self.num_modes = np.histogram(self.kr.flatten(), bins=kr_bins)[0] # Number of modes in each bin
        self.radial_ps = bin_powers / self.num_modes # Average power in each bin
        self.radial_ps_errs = []
        for i in range(len(kr_bins[:-1])):
            self.radial_ps_errs.append(np.nanstd(self.fft_cut_map.flatten()[np.where(np.logical_and(self.kr.flatten()>=kr_bins[i], self.kr.flatten()<=kr_bins[i+1]))]))
        self.radial_ps_errs /= self.num_modes
        
        #1D Power Spectrum
        fig = plt.figure(figsize=(10, 8))
        ax = plt.subplot(111)
        plt.plot(self.kr.flatten(),np.abs(self.fft_cut_map.flatten())**2,'.',alpha=0.1)
        plt.plot(self.kr_bin_centers, self.radial_ps, 'k-')
        plt.yscale('log')
        plt.xscale('log')
        plt.ylabel(r'Power [K$^2$]', fontsize=self.fontsize)
        plt.xlabel(r'$k_r$ [arcmin$^{-1}$]', fontsize=self.fontsize)
        plt.xlim([3e-3,0.5/self.pixel_size])
        plt.tick_params('both', labelsize=self.fontsize-1)
        if type(band_feed) == type(None):
            count = 1
            while os.path.exists(self.data_file + '/../1D_Power_Spectrum_{}.png'.format(count)):
                count += 1
            plt.savefig(self.data_file + '/../1D_Power_Spectrum_{}.png'.format(count))
        else:
            count = 1
            while os.path.exists(self.data_file + '/../Band{}_Feed{}_1D_Power_Spectrum_{}.png'.format(*band_feed, count)):
                count += 1
            plt.savefig(self.data_file + '/../Band{}_Feed{}_1D_Power_Spectrum_{}.png'.format(*band_feed, count))

        #2D Power Spectrum
        fig = plt.figure(figsize=(10, 8))
        ax = plt.subplot(111)
        xextent = 0.5/ self.pixel_size
        yextent = 0.5/ self.pixel_size
        ax.imshow(np.log10(np.fft.fftshift(np.abs(self.fft_cut_map)**2)), cmap=self.cmap, origin='lower', extent=[-xextent,xextent,-yextent,yextent])
        ax.set_xlim([-0.2,0.2])
        ax.set_ylim([-0.2,0.2])
        ax.set_xlabel(r'$k_x$ [arcmin$^{-1}$]', fontsize=self.fontsize)
        ax.set_ylabel(r'$k_y$ [arcmin$^{-1}$]', fontsize=self.fontsize)
        ax.tick_params('both', labelsize=self.fontsize-1)
        cbar = fig.colorbar(ax.images[0], orientation='vertical', pad=0.05)
        cbar.set_label(r'K$^2$', fontsize=self.fontsize)
        cbar.ax.tick_params(labelsize=self.fontsize-1)

        #Save Plots
        if type(band_feed) == type(None):
            count = 1
            while os.path.exists(self.data_file + '/../2D_Power_Spectrum_{}.png'.format(count)):
                count += 1
            plt.savefig(self.data_file + '/../2D_Power_Spectrum_{}.png'.format(count))
        else:
            count = 1
            while os.path.exists(self.data_file + '/../Band{}_Feed{}_2D_Power_Spectrum_{}.png'.format(*band_feed, count)):
                count += 1
            plt.savefig(self.data_file + '/../Band{}_Feed{}_2D_Power_Spectrum_{}.png'.format(*band_feed, count))

        return self.kr_bin_centers, self.radial_ps
    
    def fitting_power_spec(self, guesses=None, knee_freq=0.1, remove_low_bins=1, error_type="bin_errors", log_value_fitting=True, sigma_residual=True, normalise_red_chi_2=False, band_feed=None):
        """
        Take results from binning 1D power spectrum and fit an equation to the results to obtain a value for the red and white noise levels of the map and the spectral index

        Parameters
        ----------
        - guesses:  array of floats
                    Initial guesses used for the fits
        - knee_freq:    float
                        Assumed value for the knee frequency of the 1D power spectrum in arcmin$^{-1}$
        - remove_low_bins:  int
                            Number of low frequency bins removed from the start of the array for fitting and plotting
        - error_type:   string
                        Determines the type of error used when fitting or plotting. If 'bin_errors' is passed, the errors are calculated from the standard deviation of values within the bin, if 'num_modes' is passed the errors are calculate using the reciprocal of the number of values in each bin. Otherwise no errors will be applied and all data will have standard weights when fitting.
        - log_value_fitting:    bool
                                Determines if values are logged and fitted with a logged equation before fitting. This is used to reduce weighting of high power low frequency bins that pulls the fit away from the high frequency bins
        - sigma_residual:   bool
                            Determines if the residual shows the number of sigma the value is away from the model (True) or a standard residual (False)
        - normalise_red_chi_2:  bool
                                Determines if errors are normalised by multiplying them by the square root of the reduced chi squared value to bring the reduced chi squared to 1

        """

        if error_type == "num_modes":
            err_data = 1 / self.num_modes
        elif error_type == "bin_errors":
            err_data = self.radial_ps_errs
        else:
            err_data = np.ones(np.shape(self.radial_ps))
        
        self.equation = lambda x, white_noise, alpha, red_noise: red_noise**2 * (knee_freq/x)**alpha + white_noise**2
        self.param_names = self.equation.__code__.co_varnames[1:]
        self.num_coeffs = self.equation.__code__.co_argcount - 1

        #---------------------------------------------------------------
        # Perform the fit and calculate the error on each coefficient
        if log_value_fitting:
            equation = lambda x, white_noise, alpha, red_noise: np.log10(red_noise**2 * (knee_freq/x)**alpha + white_noise**2)

            coeff, pcov = curve_fit(equation, self.kr_bin_centers[remove_low_bins:], np.log10(self.radial_ps[remove_low_bins:]), p0=guesses, sigma=np.log10((err_data/self.radial_ps)+1)[remove_low_bins:], bounds=[[0, -np.inf, 0], [np.inf, np.inf, np.inf]])
        else:
            coeff, pcov = curve_fit(self.equation, self.kr_bin_centers[remove_low_bins:], self.radial_ps[remove_low_bins:], p0=guesses, sigma=err_data[remove_low_bins:], bounds=[[0, -np.inf, 0], [np.inf, np.inf, np.inf]])
        
        parameter_err = np.sqrt(np.diag(pcov))
        #--------------------------------------------------------------
        # Output the name of each parameter and the value along with an error value
        for i in range(len(self.param_names)):
            string = f' = ${coeff[i]} \pm {parameter_err[i]}$'
            print(self.param_names[i]+'{}'.format(string))

        #--------------------------------------------------------------
        #Plot a graph of the results
        self.residual = self.radial_ps - self.equation(self.kr_bin_centers, *coeff)
        self.reduced_chi_squared = np.sum(self.residual[remove_low_bins:]**2 / err_data[remove_low_bins:]**2) / (self.radial_ps[remove_low_bins:].size - len(self.param_names))

        if normalise_red_chi_2:
            err_data *= np.sqrt(self.reduced_chi_squared)
            self.reduced_chi_squared = np.sum(self.residual[remove_low_bins:]**2 / err_data[remove_low_bins:]**2) / (self.radial_ps[remove_low_bins:].size - len(self.param_names))

        if sigma_residual:
            self.residual /= err_data
            residual_errs = np.ones(np.shape(self.residual))
            resid_ax_name = "Residual ($\sigma$)"
        else:
            residual_errs = err_data
            resid_ax_name = "Residual"

        bin_range = np.diff([np.nanmin(self.kr_bin_centers), np.nanmax(self.kr_bin_centers)])
        x = np.linspace(np.nanmin(self.kr_bin_centers)-bin_range/2, np.nanmax(self.kr_bin_centers)+bin_range/2, 2000) # x-values for plotting a curve of the fitted parameters
        model_values = self.equation(x, *coeff) # y-values calculated using the fitted parameters

        figure = plt.figure(figsize=(8, 6))
        plot = figure.add_subplot(211)
        figure.subplots_adjust(hspace=0)
        plot.errorbar(self.kr_bin_centers[remove_low_bins:], self.radial_ps[remove_low_bins:], yerr=err_data[remove_low_bins:], fmt='x', color='k', ecolor='dimgrey', capsize=2) #Plot data points 

        plt.loglog()
        xlim = plt.xlim()
        ylim = plt.ylim()
        plot.plot(x, model_values, 'r-', label='fit') #Plot a curve of the model using the fitted parameters

        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.yticks(fontsize=self.fontsize-1)
        plt.tick_params(axis='x', labelbottom=False)
        plot.legend()
        plot.set_ylabel(r'Power [K$^2$]', fontsize=self.fontsize)
        plot.grid()

        #Plot the residuals underneath the main plot
        residuals_plot = figure.add_subplot(413)
        residuals_plot.errorbar(self.kr_bin_centers[remove_low_bins:], self.residual[remove_low_bins:], yerr=residual_errs[remove_low_bins:], fmt='x', color='k', ecolor='dimgrey', capsize=2)
        residuals_plot.plot(x, 0*x, color='r')

        plt.semilogx()
        plt.xlim(xlim)
        plt.tick_params(axis='both', labelsize=self.fontsize-1)
        residuals_plot.grid()
        residuals_plot.set_ylabel(resid_ax_name, fontsize=self.fontsize)
        residuals_plot.set_xlabel(r'$k_\perp$ [arcmin$^{-1}$]', fontsize=self.fontsize)
        plt.xticks(fontsize=self.fontsize-1)
        plt.yticks(fontsize=self.fontsize-1)

        #Save plots
        if type(band_feed) == type(None):
            count = 1
            while os.path.exists(self.data_file + '/../Power_Spec_Fit_{}.png'.format(count)):
                count += 1
            plt.savefig(self.data_file + '/../Power_Spec_Fit_{}.png'.format(count))
        else:
            count = 1
            while os.path.exists(self.data_file + '/../Band{}_Feed{}_PS_Fit_{}.png'.format(*band_feed, count)):
                count += 1
            plt.savefig(self.data_file + '/../Band{}_Feed{}_PS_Fit_{}.png'.format(*band_feed, count))
        #--------------------------------------------------------

        return coeff, parameter_err