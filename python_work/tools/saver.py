#import tools.saver as saver

#-<save_fits>-#
from astropy.io import fits

def save_fits(image,filename,folder):    				#=absprepVELA_dirs['02']
	hdu=fits.PrimaryHDU(image)
	hdu.writeto(folder+'/'+filename,overwrite=True)
	return	

#-<write_output_fits>-#   Write fits with the output of UNET in output_folder
import glob
import os
def write_output_fits(img_folder,output_folder,img_pred,predict_extension):
    real_filenames = glob.glob(img_folder + "/*.fits")                                      
    filename_str   = [ i.split("/")[-1].split(".fits")[0] for i in real_filenames]                 
    for num_file in range(0,len(filename_str)):    
        fits.writeto(os.path.join(output_folder, filename_str[num_file] + predict_extension + ".fits"),
                     img_pred[num_file,:,:,0],overwrite=True)   #(I!!)  #predict_extension!! = "-pred"
        #no he considerado meter las cabeceras "header" de los real_files
    return