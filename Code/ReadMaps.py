import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
import matplotlib.pyplot as plt

class DataRead_Fits:
    """
    Class to read in data from fits files

    Parameters
    ----------
    - file_name:    str
                    The name of the fits file to be read (must include the extension .fits)
    - file_location:    str
                        Relative or full path to the file to be read in
    - output_info:  bool
                    Determines if the program will output the information about each of the layers in the .fits file.
    """
    def __init__(self, file_name, file_location='', output_info=True):
        #----------------------------------------------------
        # Sets up where to find file
        self.file_location = file_location
        self.file_name = file_name

        if self.file_location != '' and self.file_location[-1] != '/':
            self.file_location += '/'
        #-------------------------------------------------------
        with fits.open(self.file_location + self.file_name) as hdul:
            if output_info:
                hdul.info()
    
    def preview(self, layer=0, cmap=plt.cm.get_cmap('jet'), replace_with_nan=None, cmap_lims_val=None, cmap_lims_percentage=None, xlim_pix=None, ylim_pix=None, x_axis_name=None, y_axis_name=None, z_axis_name='Z Axis', colour_bar_direction='Auto', fontsize=14):
        """
        Opens a layer from the map and outputs the map as an image to give a easy way to preview the layer before extracting it to an array

        Parameters
        ----------
        - layer:    int
                    Selects which layer from the .fits file you want to preview
        - cmap: colormap
                Colourmap used to display the map
        - replace_with_nan: float or array of floats
                            Determines which values in the map array are replaced with NaN
        - cmap_lims_value:  array of floats
                            Minimum and maximum values represented by the colour map. Must have a shape of (2)
        - cmap_lims_percentage: array of floats
                                Gives the minimum and maximum percentile of the map data that will be represented by the colour map. Must contain values between 0 and 100 with a shape of (2)
        - xlim_pix: array of ints
                    Gives the limits in the x direction of the map shown in pixels
        - ylim_pix: array of ints
                    Gives the limits in the y direction of the map shown in pixels
        - x_axis_name:  str
                        Label given to the x axis of the plot
        - y_axis_name:  str
                        Label given to the y axis of the plot
        - z_axis_name:  str
                        Label given to the colour bar of the plot
        - colour_bar_direction: str
                                Determines whether the colour bar is horizontal or vertical. Defaults to "Auto" where the direction is determined by the shape of the map to try and stop it overlapping with the map plot. "horizontal" can be passed to force the colourbar to be horizontal, else the colour bar will be vertical.
        - fontsize: int
                    Determines the fontsize of the axis labels and the ticks will have a fontsize of fontsize-1
        """
        cmap.set_bad([0.7,0.7,0.7],1.) #Sets NaN values to be gray

        with fits.open(self.file_location + self.file_name) as hdul:
            print(repr(hdul[layer].header)) # Outputs the header of the map
            data = hdul[layer].data # Saves data
            wcs = WCS(hdul[layer].header).celestial # Saves wcs

        #Defines the limits of the colour map if given percentage values
        if type(cmap_lims_percentage) != type(None) and type(cmap_lims_val) == type(None):
            cmap_lims_val = np.nanpercentile(data, cmap_lims_percentage)

        # Replaces specified values with NaN in the data array
        if type(replace_with_nan) != type(None):
            if type(replace_with_nan) == int or type(replace_with_nan) == float:
                replace_with_nan = [replace_with_nan]
            for i in replace_with_nan:
                data = np.where(data == i, np.nan, data)

        #Plots the figure
        fig = plt.figure(figsize=(13,8))
        ax1 = plt.subplot(111, projection=wcs)
        if type(cmap_lims_val) == type(None):
            img = ax1.imshow(data, cmap=cmap)
        else:
            img = ax1.imshow(data, cmap=cmap, vmin=cmap_lims_val[0], vmax=cmap_lims_val[1])

        # Adds labels to the axes
        if type(x_axis_name) != type(None):
            ax1.set_xlabel(x_axis_name, fontsize=fontsize)
        if type(y_axis_name) != type(None):
            ax1.set_ylabel(y_axis_name, fontsize=fontsize)
        
        # Adds secondary axes on the top and right of the plot that display the corresponding pixel values
        if type(wcs) != type(None):
            sec_x_ax = ax1.secondary_xaxis('top', transform=ax1.get_transform('world'))
            sec_y_ax = ax1.secondary_yaxis('right', transform=ax1.get_transform('world'))

            sec_x_ax.set_xlabel("X Pixels", fontsize=fontsize)
            sec_y_ax.set_ylabel("Y Pixels", fontsize=fontsize)

            sec_x_ax.tick_params('both', labelsize=fontsize-1)
            sec_y_ax.tick_params('both', labelsize=fontsize-1)

        # Sets the limits of the plot
        if type(xlim_pix) != type(None):
            xlim = list(plt.xlim())
            for x in range(len(xlim_pix)):
                if xlim_pix[x] != None:
                    xlim[x] = xlim_pix[x]
            ax1.set_xlim(xlim)
            
        if type(ylim_pix) != type(None):
            ylim = list(plt.ylim())
            for x in range(len(ylim_pix)):
                if ylim_pix[x] != None:
                    ylim[x] = ylim_pix[x]
            ax1.set_ylim(ylim)
        
        ax1.tick_params('both', labelsize=fontsize-1)
        

        # Determines the direction and plots the colour bar
        if (np.abs(2*np.diff(ax1.get_ylim())) < np.abs(np.diff(ax1.get_xlim())) or colour_bar_direction.lower() == "horizontal") and colour_bar_direction.lower() != "vertical":
            cbar_ax = fig.add_axes([0.11, 0.25, 0.8, 0.03])
            cbar = fig.colorbar(img, orientation='horizontal', cax=cbar_ax)
            cbar.set_label(z_axis_name, fontsize=fontsize)
            cbar_ax.tick_params(axis='x', labelsize=fontsize-1)
        else:
            cbar_ax = fig.add_axes([0.92, 0.1, 0.03, 0.8])
            cbar = fig.colorbar(img, orientation='vertical', cax=cbar_ax)
            cbar.set_label(z_axis_name, fontsize=fontsize)
            cbar_ax.tick_params(axis='y', labelsize=fontsize-1)

        

    
    def histogram_plot(self, layer=0, replace_with_nan=None):
        """
        Plots a histogram of the map data to see how the values are distributed. This can be useful to help determine where to set the limits of the colour map when plotting as an image

        Parameters
        ----------
        - layer:    int
                    The layer from the .fits file that is being analysed
        - replace_with_nan: float or array of floats
                            Determines which values in the map array are replaced with NaN
        """
        # Gets the data to be analysed
        with fits.open(self.file_location + self.file_name) as hdul:
            image_data = hdul[layer].data
        
        # Replaces specified values in the data with NaN
        if type(replace_with_nan) != type(None):
                    if type(replace_with_nan) == int or type(replace_with_nan) == float:
                        replace_with_nan = [replace_with_nan]
                    for i in replace_with_nan:
                        image_data = np.where(image_data == i, np.nan, image_data)
        
        # Plots the histogram
        plt.figure()
        plt.hist(image_data.flatten()[::1],bins=200)
        plt.xlabel("Value")
        plt.ylabel('Count')
        plt.show()

    def extract_data(self, layer=0, replace_with_nan=None):
        """
        Extract data from fits file to work on

        Parameters
        ----------
        - layer:    int
                    Layer of the fits file to extract
        - replace_with_nan: float or array of floats
                            Determines which values in the map array are replaced with NaN
        """

        # Loads the map data and the wcs
        with fits.open(self.file_location + self.file_name) as hdul:
            image_data = hdul[layer].data
            self.wcs = WCS(hdul[layer].header).celestial
        
        # Replaces specified values in the data with NaN
        if type(replace_with_nan) != type(None):
                    if type(replace_with_nan) == int or type(replace_with_nan) == float:
                        replace_with_nan = [replace_with_nan]
                    for i in replace_with_nan:
                        image_data = np.where(image_data == i, np.nan, image_data)
        

        self.data_array = image_data*1
        return self.data_array, self.wcs