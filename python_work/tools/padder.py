#import tools.padder as padder

#-<padding>-# : Padding zeros to images with dims<(128,128)

def padding(str_image,side=128):  
    
    with fits.open(str_image) as hdul:
            image=hdul[0].data	
    xbins,ybins=image.shape
    
    if (xbins,ybins)<=(side,side):    #1. pad zeros to fill (128,128)
        rest_side = floor((side-xbins)/2.)  # convenio: floor      #i.e. xbins (xbins==ybins)
        
        image_padded = np.zeros((side, side), dtype=np.float32)    # all fits in type: "np.float32"
        image_padded[rest_side:rest_side + xbins , rest_side:rest_side + ybins] = image        
    else: 
        print('Error size in <image.shape>:',image.shape)
    return image_padded


#-<>-#





























