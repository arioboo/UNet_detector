    #-<general_modules>-#
from astropy.io import fits
from astropy.visualization import (MinMaxInterval,PercentileInterval,ZScaleInterval,SqrtStretch,AsinhStretch,LogStretch,ImageNormalize)
from astropy.convolution import Tophat2DKernel, Gaussian2DKernel, Box2DKernel, convolve
from astropy.stats import gaussian_fwhm_to_sigma

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
from scipy import fftpack
import os
import sep

import warnings
from copy import deepcopy
from sklearn.feature_extraction import image

import photutils.utils as pht_utl
    #-<custom modules>-#
import radialProfile

###--------------------------<START>-----------------------------###
''' ###########THANKS TO ALL THE DEVELOPERS WHO MADE THE ABOVE MODULES POSSIBLE############ '''

def identify_objects(image_data,nsigma,min_area,deb_n_thresh,deb_cont,param_dict):
    '''
    This function performs source identification and generates a segmentation map,
    which is then used for masking the sources.
    :param image_data: provide the image data, which is a mxn numpy nd array. e.g., fits.getdata('image_file_name')
    :param nsigma: source detection significance.
    :param min_area: minimum area to be considered as a source
    :param deb_n_thresh: number of threshold values for deblending routine. e.g., 32, 64 etc.
    :param deb_cont: deblend minimum contrast ratio (see source extraction or SEP python page).
    :param param_dict: a dictionary containing the
    'sep_filter_kwarg' = filter keyword argument, which can be a 'tophat', 'gauss', or 'boxcar'
    'sep_filter_size' = the 'size' of the filter. In case of gaussian, it is the FWHM of the gaussian. For tophat, it
    is the radius of the tophat filter. For boxcar, it is the side length of the 2D Box.
    :return: objects: a numpy array of the objects, ordered as per their segmentation values in the segmap.
    segmap: a segmentation map, where each source is marked with unique source identification number.
    '''

    # Note, this whole routine uses a Python-based source identification module named SEP (Barbary et al., 2016)

    # Unpack the filter keyword and its size from the parameter dictionary.
    filter_kwarg = param_dict['sep_filter_kwarg']
    filter_size  = float(param_dict['sep_filter_size'])

    # Look at the SEP webpage, this is suggested for working of SEP.
    byte_swaped_data = image_data.byteswap().newbyteorder()

    # SEP estimates a global background.
    global_bkg = sep.Background(byte_swaped_data)

    # background subtracted data = original data - estimated global background.
    bkg_subtracted = byte_swaped_data - global_bkg

    # In the following block, we check for the user's choice of filter and its size.
    # We define a kernel based on their choice.
    if filter_kwarg.lower() not in ['tophat','gauss','boxcar']:
        warnings.warn('The filter %s is not supported as of yet, defaulting to tophat of radius 5')
        source_kernel = Tophat2DKernel(5)
    elif filter_kwarg.lower() == 'tophat':
        source_kernel = Tophat2DKernel(filter_size)
    elif filter_kwarg.lower() == 'gauss':
        _gauss_sigma = gaussian_fwhm_to_sigma(filter_size)
        source_kernel = Gaussian2DKernel(_gauss_sigma)
    elif filter_kwarg.lower() == 'boxcar':
        source_kernel = Box2DKernel(filter_size)

    # Object detection and Segmentation map generataion.
    objects, segmap = sep.extract(bkg_subtracted, nsigma   , err = global_bkg.globalrms, 
                                  minarea       = min_area , deblend_nthresh = deb_n_thresh, 
                                  deblend_cont  = deb_cont , segmentation_map = True, 
                                  filter_kernel = source_kernel.array)

    return objects, segmap


def get_segmap_value(segm,centroid):
    '''
    This code performs a simple query call on the segmentation map and finds what
    segmentation value does the centroid pixel have. This function plays an important role
    in finding the respective segmentation region and using it for further manipulation (e.g., masking)
    :param segm: Segmentation map from the source identification routine.
    :param centroid: centroid of the object to which one wishes to find its segmentation value.
    :return: segmentation value corresponding to the centroid.
    NOTE: This function also works for GALFIT based centroid values. This function also plays
    a role in mapping outputs of GALFIT cubes (header of residual images) to identified objects via
    source extraction.
    '''
    y,x = centroid
    return segm[int(x)][int(y)]

def make_axis_labels_off(axes):
    '''
    A small wrapper to avoid pain while visualizing images and switches of axes labels.
    :param axes: a numpy array of all the axes. In case of using subplots simply pass the full axes item, which
    is already a numpy nd array.
    :return: Nothing.
    '''
    for each in axes.flat:
        each.xaxis.set_visible(False)
        each.yaxis.set_visible(False)
    return


def make_noise_patch(image_to_be_sampled,shape_of_noise_image):
    '''
    This is a key function that 'stitches' together an arbitrary sized (NxN) noise stamp by randomly sampling
    a 10x10 box region of the source-masked image provided by the user.
    :param image_to_be_sampled: User provided postage stamp in which the sources are masked (i.e., source regions = 0).
    :param shape_of_noise_image: Shape of the output noise mosaic required. Note that this function constructs a square
    output noise stamp. E.g., if i provide 200x500, it will generate 500x500 stamp. I did this as it was simplistic and
    served my purpose.
    :return: stitched noise stamp
    '''

    # Conceptually, here is how i planned my stitching. Lets say i want 500x500 output image. I wanted to randomly
    # sample 10x10 box, i was going to do so by making stitching in this fashion (x x x x x ...) and repeating this step
    # for a number of times and stacking them horizontally such that the dimensions meet user requirement. Think of small
    # pieces of wood to make a 500x500 wooden wall.
    # In case the number is not perfectly divisible by 10, then, i do the above process for a number that is perfectly
    # divisible, say 40 for 450x450, in another loop, i separately stitch the remainder.

    # figure out the x and y shapes of the required noise stamp.
    y_shape, x_shape = shape_of_noise_image
    # How may stacks do i need based on the dimensions of required noise stamp?
    how_many_vstacks = int(y_shape/10)                                                             #<ME> #(OCODE) y_shape/10
    last_attach_vstact = y_shape%10 # remainder in case if y_shape is not perfectly divisible by 10.

    how_many_hstacks = int(x_shape/10)                                                         #<ME> #(OCODE) x_shape/10
    last_attach_hstack = y_shape%10

    # define an empty sequence in which chunks of [x x x x x x...] will be appended to.
    vstacks = []
    # for each number of required horizontal sequence... loop through...
    for iter_hstack in range(how_many_hstacks):
        # define an empty sequence, that will become [x x x x ...] by appending on each internal loop.
        patches_vstack = []
        # Initial set of patches sample from the source-masked image.
        # The reason for max_patches = how_many_vstacks * 5 is to avoid not enough patches that have unmasked regions
        # In such case, the code crashes.
        # The choice of 10x10 is to make sure that the pixel-to-pixel noise-correlations are preserved.
        # Hardcoded for now, please feel free to change, but make sure to change 10's above for how_many_vstacks etc.
        initial_vpatches = image.extract_patches_2d(image_to_be_sampled, (10, 10), max_patches=how_many_vstacks*5)
        added_vstack = 0 # A simple counter to make sure i dont add too many horizontal blocks..
        for each_vpatch in initial_vpatches:
            # For each patch in the randomly sample patch pool..
            # Check if there is ANY zero assigned to any pixel.. suggesting masked region...
            # If no masked regions and added blocks are less than required number of blocks...
            # Append to patches_vstack to make [x x x x x ....]
            if np.any(each_vpatch == 0) == False and added_vstack<how_many_vstacks:
                patches_vstack.append(each_vpatch)
                added_vstack= added_vstack+1

        # In case there need some extra blocks to complete the stitching. Usually this loop is ignored if the
        # user provides a image dimension that is perfectly divisible by 10.
        if last_attach_vstact !=0:
            some_sampled_extras = image.extract_patches_2d(image_to_be_sampled, (last_attach_vstact, 5), max_patches=10)
            added_extra=False
            for each_extra in some_sampled_extras:
                if np.any(each_extra == 0) == False and added_extra==False:
                    patches_vstack.append(each_extra)
                    added_extra=True
                    break
        vstacks.append(np.vstack(tuple(patches_vstack)))
        # stitch [[x] [x] [x]] ... =  [[x x x x x]]

    image_stamp = np.hstack(tuple(vstacks))                #<ME> (ERROR!)
    # Now make a vertical stitch [[x x x x x], [y y y y y]] = [[x x x x x]; [y y y y y]...]
    return image_stamp # stitched final image.

def query_random_exp_stamp(large_stamp,size_of_required_stamp):
    '''
    This function queries a smaller portion (user defined size) of a larger noise mosaic.
    :param large_stamp: a 'master' large noise mosaic.
    :param size_of_required_stamp: size of the output cutout
    :return: a randomly sampled portion of the large mosaic that matches the user's stamp size.
    '''
    patches = image.extract_patches_2d(large_stamp, size_of_required_stamp, max_patches=20)
    for each_patch in patches:
        if np.any(each_patch == 0) == False:
            required_stamp = each_patch
            break
    return required_stamp

def do_autocorr_power_spectrum(image_for_fft):
    '''
    This function performs an auto-correlation of an input image and
    outputs a normalized 1D azimuthally averaged power as a function of spatial frequency (PSD).
    :param image_for_fft: input image for which the user wishes a 1D PSD
    :return: a 1D PSD.
    '''
    Fnoise = fftpack.fft2(image_for_fft) # Fast fourier transform
    Fnoise_shifted = fftpack.fftshift(Fnoise) # Shifting such that m=0 is at the center of the image.
    psd2D_noise = np.abs(Fnoise_shifted) ** 2 # 2D Power map..
    psd1D_noise = radialProfile.azimuthalAverage(psd2D_noise) # azimuthally averaged 1D power profile. 
    #NOTE_TO_SELF: There is an alternative function for 1D PSD, which i should explore...
    psd1d_noise_normed = psd1D_noise / np.max(np.cumsum(psd1D_noise)) # Normalizing by the cumulative area under 1D PSD.
    return psd1d_noise_normed # return normalized 1D PSD.


def extract_noise_and_exp_patches(candels_image,vela_noise_less_image,make_plots,is_image_cube,plots_loc,fits_save_loc):
    '''
    This is the main function that creates a noised image of the input simulated image based on
    the real (HST, candels image). The user has the freedom to store plots or not.
    :param candels_image: Input real postage stamp from which the noise properties will be extracted.
    Note units should be in electrons/sec
    :param vela_noise_less_image: An un-noised image of a simulated observation (note, need to be PSF convolved).
    The units of this image should be in micro-janskies per square arcseconds. G Snyder et al.,
    :param make_plots: To make and store the diagnostic plots on the run, please set it True, otherwise to False.
    :param is_image_cube: Provide True if the real image is a image cube (as in my case), elif it is a simple postage
    stamp of a galaxy then provide False.
    :param plots_loc: save location of the generated plots
    :param fits_save_loc: save location of the generated noise-added fits files.
    :return: Nothing.
    '''
    if is_image_cube == False:
        original_data = fits.getdata(candels_image)  # get the data from the real postage stamp.
    else:
        can_image_cube = fits.open(candels_image) # open the image
        original_data = can_image_cube[1].data # load the data from the second HDU, which has the original image.  #<ME> #OCODE [1]

    vela_noiseless_data = fits.getdata(vela_noise_less_image) # get data of the noise-less simulated image
    vela_header = fits.getheader(vela_noise_less_image)       # load the header information from noise-less image.
    # Identify objects and generate the segmentation map of the sources in the real image.
    # the set of parameters i used are discussed in Mantha+19 (submitted), which i converged based on the
    # official numbers used in Guo et al., 2013 and Galametz et al., 2013. However, some values are modified
    # to aid the detection of small and faint sources in the images and make sure that the source segmentation
    # maps extend out to tails of the light distribution. Otherwise, the source light will bleed into the noise stamps.
    obj, segmap = identify_objects(original_data,0.75, 7, 64, 0.001,
                                   {'sep_filter_kwarg':'tophat','sep_filter_size':'5'})         #(I!!)'tophat','5'


    sky_mask = deepcopy(segmap) # make a deep copy of the segmap for further manipulation.
    sky_mask = sky_mask + 1     # add 1 to all seg values.
    sky_mask[sky_mask > 1] = 0  # make everything that is larger than 1 to zeros.
    # This step makes all the sources to 0 and the 'sky'

    noise_stamp = make_noise_patch(sky_mask * original_data, (700, 700))
    # Generate a large noise mosaic of size 700x700

    # based on the size of the simulated image, this step will query a noise stamp matching its size.
    noise_stamp_size_matched = image.extract_patches_2d(noise_stamp, (vela_header['NAXIS1'],vela_header['NAXIS2']), max_patches=1)

    if make_plots: # If the make plots is True, then do the following steps...
        fig, axs = plt.subplots(2, 2, figsize=[20, 16]) # Create a four panel layout 2 rows x 2 cols..
        [[ax1, ax2], [ax3, ax4]] = axs                  # Axes

        norm = ImageNormalize(original_data, interval=PercentileInterval(98.),
                              stretch=AsinhStretch())
        # find the image normalization for visualization matching DS9 settings. 98 percentile, Asinh stretch.

        ax1.imshow(original_data, origin='lower', cmap='gray', alpha=1, norm=norm) # visualize the real image
        for i in range(len(obj)): # this routine uses the source identified objects to put elliptical regions.
            e = Ellipse(xy=(obj['x'][i], obj['y'][i]),
                        width=6 * obj['a'][i],
                        height=6 * obj['b'][i],
                        angle=obj['theta'][i] * 180. / np.pi)
            e.set_facecolor('none')
            e.set_edgecolor('red')
            ax1.add_artist(e)
            ax1.text(obj['x'][i], obj['y'][i], '%s' % (get_segmap_value(segmap, (obj['x'][i], obj['y'][i]))),
                     fontsize=10, color='black', ha='center', va='center') # also put seg value at the center.

        # visualize the source masked image.
        ax2.imshow(sky_mask * original_data, origin='lower', interpolation='nearest', norm=norm, cmap='gray')

        # visualizing the histogram of pixel values in the source masked image.
        ax4.hist([i for i in (sky_mask * original_data).flat if i != 0], bins=100, normed=True, color='red',histtype='step') #<ME> # normed -> density

        # visualizing the large noise mosaic
        ax3.imshow(noise_stamp_size_matched[0], origin='lower', interpolation='nearest', norm=norm, cmap='gray')

        # Sanity check, overplot another histogram of the large noise stamp.. to compare the two histograms.
        ax4.hist(noise_stamp.flat, bins=100, normed=True, color='blue', histtype='step') #<ME> # normed ->

        # draw the mean of the histogram.
        ax4.axvline(x=np.mean(noise_stamp.flat),linestyle='--',color='green',label='sky mean [e/s] = %s\n sky mean [muj/arcsec^2] = %s'%(round(np.mean(noise_stamp.flat),4), round((1E-7 * np.mean(noise_stamp.flat) * (10**6))/(0.06*0.06),4)))
        # ax4.imshow(noise_stamp,origin='lower', interpolation='nearest',norm=norm,cmap='gray')

        make_axis_labels_off(np.array([ax1, ax2, ax3])) # switch of the axes that show images.
        plt.subplots_adjust(wspace=0.01, hspace=0.01) # Adjust the spacing between subplots to avoid whitespace.
        stamp_gen_image_name = os.path.basename(candels_image).strip('.fits') # figuring out the name for later use.
        ax4.legend(loc=2) # show legend for axis4 (histogram).
        # save figure... CHANGE THE PATHS...
        plt.savefig('%s/%s.png'%(plots_loc,stamp_gen_image_name), bbox_inches='tight')
        plt.close(fig)

    exp_time = 3300.0 # Assuming an exposure time of 3300 seconds based on the table in Koekemoer+11 for ~2-orbit depth.  #<ME> whaat

    # Note that the real image from CANDELS are in units of e/s. It is important to convert into electrons.
    real_noise_stamp_counts = noise_stamp_size_matched[0] * exp_time # converting to counts.

    pht_nu = float(vela_header['PHOTFNU']) * 1E6 # PHOTFNU is in units of J * sec/electron; 1E6 is to convert to uJy.

    #Converting the noise-less vela image into units of electrons.
    # uJy/arcsec^2 * (arcsec^2)/ (uJy * seconds/electrons) * seconds = electrons
    noise_less_vela_electrons = (vela_noiseless_data * (0.06 ** 2) / pht_nu) * exp_time
    if np.any(noise_less_vela_electrons<0): # at some very early redshift snapshots.. there are some negative values.
        # to avoid the poisson routine from crashing, i made them zeros.
        noise_less_vela_electrons[np.where(noise_less_vela_electrons<0)] = 0.0

    # Poisson realization of the noise-less simulated image.
    poisson_noised_vela = np.random.poisson(noise_less_vela_electrons, size=None)

    # Poisson residual. We want to make sure that the source Poisson noise is correlated spatially
    # as the background sky (owing to drizzling etc..)
    poisson_resid = poisson_noised_vela - noise_less_vela_electrons

    # Defining a smoothing kernel to smooth the poisson residual image such that the
    # 1D PSD of the resultant smoothed poisson residual matches the noise PSD.
    # Upon some extensive experimentation, i found that a Gaussian kernel of sigma=0.6
    # works the best (by eye). I feel that there might be better automated (sort of regression of MCMC) way
    # to figure this out on an image-by-image basis, but for now i hardcoded it.
    smth_kernel = Gaussian2DKernel(0.6)
    smoothed_poisson_resid = convolve(poisson_resid, kernel=smth_kernel) # Smooth the poisson to induce correlation.

    correlated_poisson = smoothed_poisson_resid + noise_less_vela_electrons # add this to noise-less image (still in counts)
    # Full noise-added image (correlated poisson realization + real noise stamp)
    full_noise_added = correlated_poisson + real_noise_stamp_counts

    # unit conversion back to uJy/arcsec^2
    full_noise_added_ujy_arcsec2 = (full_noise_added / exp_time) * pht_nu / (0.06 ** 2)

    # Get the VELA ID -- this is a specific thing, whoever is using this code may change it as needed.
    what_vela_id = os.path.basename(vela_noise_less_image).strip('_SB00.fits').split('_')[0]

    # writing additional information to the existing header. Preserving original header info by Snyder et al.,
    # with extra information for back tracking purposes.
    vela_header['expsr_t'] = '%s'%exp_time

    # creating a filename to store the output images.
    output_noise_filename = os.path.basename(vela_noise_less_image).strip('_SB00.fits') + '_real_noise_stamp'
    vela_header['REAL_N'] = '%s'%output_noise_filename # the noise stamp name used in this run, printed to header
    fits.writeto('%s/%s/%s.fits' % (fits_save_loc,
        what_vela_id, output_noise_filename),
                 noise_stamp_size_matched[0] * pht_nu / (0.06**2),
                 overwrite=True) # saving the noise postage stamp in units of ujy/arcsec^2 (same as simulated images).

    output_poiss_filename = os.path.basename(vela_noise_less_image).strip('_SB00.fits') + '_smoothed_poiss'
    vela_header['POISS'] = '%s'%output_poiss_filename
    fits.writeto('%s/%s/%s.fits' % (fits_save_loc,
        what_vela_id, output_poiss_filename),
                 (correlated_poisson / exp_time) * pht_nu / (0.06 ** 2),
                 overwrite=True) # saving the correlated poisson realization in units of uJy/arcsec^2.

    output_noise_added_filename = os.path.basename(vela_noise_less_image).strip('_SB00.fits') + '_noise_added'
    fits.writeto('%s/%s/%s.fits' %(fits_save_loc,what_vela_id,output_noise_added_filename),
                 full_noise_added_ujy_arcsec2,
                 overwrite=True,header=vela_header) # saving the final noise-added image in units of uJy/arcsec^2
    if make_plots: # if the make plots is set, then generate this large figure that showcases all steps.
        fig, axs = plt.subplots(2, 3, figsize=[18, 8]) # 2 rows x 3 cols figure.
        [[ax1, ax2, ax3], [ax4, ax5, ax6]] = axs
        '''Normalizing all the image visualizations to the same setting...'''
        norm_nl = ImageNormalize(noise_less_vela_electrons, interval=PercentileInterval(98),
                                 stretch=AsinhStretch())
        norm_poiss = ImageNormalize(poisson_noised_vela, interval=PercentileInterval(98),
                                    stretch=AsinhStretch())
        norm_resid = ImageNormalize(poisson_resid, interval=PercentileInterval(98),
                                    stretch=AsinhStretch())

        norm_smth_resid = ImageNormalize(smoothed_poisson_resid, interval=PercentileInterval(98),
                                         stretch=AsinhStretch())
        norm_corr_poiss = ImageNormalize(correlated_poisson, interval=PercentileInterval(98),
                                         stretch=AsinhStretch())

        norm_full_noised = ImageNormalize(full_noise_added_ujy_arcsec2, interval=PercentileInterval(98.5),
                                          stretch=AsinhStretch())
        ''' Showing all the images in their respective axes...'''
        ax1.imshow(noise_less_vela_electrons, origin='lower', cmap='gray', alpha=1, norm=norm_nl)
        ax2.imshow(poisson_noised_vela, origin='lower', cmap='gray', alpha=1, norm=norm_poiss)
        ax3.imshow(poisson_resid, origin='lower', cmap='gray', alpha=1, norm=norm_resid)
        ax4.imshow(smoothed_poisson_resid, origin='lower', cmap='gray', alpha=1, norm=norm_smth_resid)
        ax5.imshow(correlated_poisson, origin='lower', cmap='gray', alpha=1, norm=norm_corr_poiss)
        ax6.imshow(full_noise_added_ujy_arcsec2, origin='lower', cmap='gray', alpha=1, norm=norm_full_noised)

        make_axis_labels_off(axs) # switch of all axes labels.. as all are images.
        plt.tight_layout() # tight layout... for effective (somewhat) professional visualization.
        plt.subplots_adjust(wspace=0.0, hspace=0.01) # adjust the white space between rows and cols.

        fig_name = os.path.basename(vela_noise_less_image).strip('_SB00.fits') + '_noise_addition'
        plt.savefig('%s/%s.png'%(plots_loc,fig_name), bbox_inches='tight') #save as png.
        plt.close(fig) # close figure object to avoid clutter in memory.
    if make_plots:
        '''If make plots is set... then this block visualizes the 1D PSD of real noise stamp and
        the correlated poisson image.'''
        fig2 = plt.figure(figsize=[10, 8])
        ax3 = fig2.gca()
        psd1d_real_noise = do_autocorr_power_spectrum(real_noise_stamp_counts) #1D PSD of real noise stamp
        psd1d_un_correlated_poiss = do_autocorr_power_spectrum(poisson_resid) # 1D PSD of uncorrelated poisson (should be flat)
        psd1d_smoothed_poiss = do_autocorr_power_spectrum(smoothed_poisson_resid) # 1D PSD of correlated poisson image.

        '''The following three lines show the 1D PSDs'''
        ax3.semilogy(psd1d_real_noise, color='red', linestyle='-', label='Real noise')
        ax3.semilogy(psd1d_un_correlated_poiss, color='red', linestyle='--', label='Poisson Residual')
        ax3.semilogy(psd1d_smoothed_poiss, color='blue', linestyle='-.', label='Smoothed Poisson Residual')

        ax3.set_xlabel('Spatial Frequency', fontsize=18) # xaxis labels
        ax3.set_ylabel('Normalized AutoCorr Power Spectrum', fontsize=18) # yaxis label.

        ax3.legend(loc=1, fontsize=16) #legend..
        fig_name = os.path.basename(vela_noise_less_image).strip('_SB00.fits') + '_autocorr'
        plt.savefig(
            '%s/%s.png'%(plots_loc,fig_name), bbox_inches='tight') # save the plot as a png.
        plt.close(fig2)
    return


###----------------------------<END>-----------------------------###

'''
This code has all the necessary functions for adding realistic HST-like noise to the
simulated (idealized) images from sunrise.

Author: Kameswara Mantha
Collaborators: Yicheng Guo, Harry Ferguson, Elizabeth McGrath, Haowen Zhang, Joel Primack, Gregory Snyder, Marc Huertas-Company
email: km4n6@mail.umkc.edu
'''
