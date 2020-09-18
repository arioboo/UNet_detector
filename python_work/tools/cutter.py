#import tools.cutter as cutter

#-<cut_image>-# : Cut centered images into (128,128)
from astropy.io import fits
from math import *

def cut_image(str_image,side=128):
	#image=image.reshape(side,side)
    with fits.open(str_image) as hdul:
        image=hdul[0].data	
    xbins,ybins=image_data.shape
    if (xbins,ybins)>=(side,side):    #1. cut image to (128,128)
        xcenter = floor(xbins/2.) ; ycenter = floor(ybins/2.)                        # center pixel
        edge = round(side/2.)    			                                         # edge pixels from center
        
        image_cut = image[xcenter-edge : xcenter+edge , ycenter-edge : ycenter+edge] # image_cut	
    else: 
        print('Error size in <image.shape>:',image.shape)   
    
    return image_cut

'''
print('xbins:%i ybins:%i ; ; xcenter:%i ycenter:%i ; ; xedge:%i yedge:%i ' %(xbins,ybins,xcenter,ycenter,xedge,yedge) )
'''

#̣-< >-#
