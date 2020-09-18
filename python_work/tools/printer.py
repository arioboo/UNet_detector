#import tools.printer as printer

#-<print_dimensions>-#:  Print dimensions of fits HDU data selected (as absolute path)
from astropy.io import fits
import glob

def print_dimensions(str_image):
    
    for file in glob.glob(str_image):
       
        
        with fits.open(file) as hdul:
            image=hdul[0].data
        xbins,ybins=image.shape  
        print('(%i,%i)'%(xbins,ybins))    
        
    return  