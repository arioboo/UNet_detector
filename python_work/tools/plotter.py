#import tools.plotter as plotter


#-<plot_fits>-#  Plot a fits image from it's data_stored variable in Python
import matplotlib.pylab as plt  
plt.ion()

def plot_fits(image,cmap_choose='Blues',new_figure=0):	
	fig = plt.figure(1)            
	if new_figure : 
		fig=plt.figure()	
	fig.clf()
	ax = fig.add_subplot(1,1,1)
	ax.set_title('2D map')
	ax.set_xlabel('x_bin') ; ax.set_ylabel('y_bin')
	img = ax.imshow(image , cmap=cmap_choose)
	fig.colorbar(img)
	return fig.show() 


#̣-<str_plot_fits>-#   Plot a fits image from it's string path
from astropy.io import fits

def str_plot_fits(str_image , cmap_choose='Blues' , new_figure=False):
    with fits.open(str_image) as hdul: image=hdul[0].data
    return plot_fits(image,cmap_choose,new_figure)	


#-<pattern_plot>-# : Very useful to plot in terms of a0, cam, ... Be in the folder for glob, (input the folder if glob1)
import glob

def pattern_plot(pattern , cmap_choose='Blues' , new_figure=True):
	for im in glob.glob('*'+pattern+'*.fits'):
		str_plot_fits(im,cmap_choose,new_figure)
	return 




#-<group_images>-# : After recognition with the model Unet, it shows in the same figure the "real image", "binary mask", and the "output"


def group_images(i):
    fig=pl.figure(); 
    ax1=fig.add_subplot(1,3,1); ax1.imshow(imgs[i])         # REAL IMAGE
    ax2=fig.add_subplot(1,3,2); ax2.imshow(imgs_mask[i])    # BINARY MASK    
    ax3=fig.add_subplot(1,3,3); ax3.imshow(imgs_test[i])    # OUTPUT OF UNET
    return




'''
#patterns: 
a0.<xxx>  (redshift)
cam<xx>   (camera)
<ACS>,<WFC3> (instrument)
<F160W>,<F775W>,<F606W>,<F435W>  (filter)
'''
