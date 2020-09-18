# coding: utf-8
############-----------author:arioboo------------###############
'''Prepare a (128x128) grid on original VELA data, changes zero_point of images, and adds real_HSTlike_noise or stamp_noise. Save results on DATA/ directory.''' 
    #-<main modules>-#
from astropy.io import fits		
import numpy as np
import time
from math import *              
from scipy.ndimage import zoom  
import subprocess
import sys
    #-<custom modules>-#   
import tools.changer as changer
import tools.cutter as cutter 
import tools.noiser as noiser
import tools.padder as padder
import tools.plotter as plotter
import tools.selecter as sel 
import tools.saver as saver
from params.data_params import * 

###---------------------------------------START-----------------------------------------------###
print_parameters(real_HSTlike_noise)                               #print_parameters of this RUN (data_params.py) #(I!!)
print("###---<START of prepare.py>---###")
try:    os.stat(absprepVELA_dirs[VELA_id])  ; print("Statting folder:",absprepVELA_dirs[VELA_id]) 
except: os.mkdir(absprepVELA_dirs[VELA_id]) ; print("Making folder:  ",absprepVELA_dirs[VELA_id]) 

###---<RUIDO_MIXED_FLATS>---###
#-<prepare_image>-#
def prepare_image(str_image,filtro=filtro,side=128,real_HSTlike_noise=real_HSTlike_noise):
    with fits.open(str_image) as hdul:
        image_data = hdul[0].data
    if filtro != 'f160':            #0. Rescale ACS images
        image_data = zoom(image_data, [1./2,1./2], order=3)
    
    xbins,ybins=image_data.shape    
    if (xbins,ybins)>(side,side):   #1. cut image to (128,128)   
        xcenter = floor(xbins/2.) ; ycenter = floor(ybins/2.)                         # center pixel
        edge = round(side/2.)    			                                          # edge pixels from center	
        image = image_data[xcenter - edge : xcenter + edge , ycenter - edge : ycenter + edge]  # image_cut		
    elif (xbins,ybins)<(side,side): #1. pad zeros to fulfill (128,128)
        rest_side = floor((side - xbins)/2.)       # convenio: floor                  # i.e. xbins (xbins==ybins)
        
        image = np.zeros((side, side), dtype=np.float32)                              # all fits in type: "np.float32"
        image[rest_side:rest_side + xbins , rest_side:rest_side + ybins] = image_data # image_pad 
    else :                          #1. equal case
        image = image_data
    
    
    if not real_HSTlike_noise: image = changer.change_zp(image,filtro)  #2. changes the zero_point of the image (if not running noise_gen routines)
    return image

#<CORE!>
if not real_HSTlike_noise:              
    #-select_filter_noise-#
    filter_str={ entry:"/bk_"+entry+"_"     for entry in coded_filters.keys()}	  	
    
    str_noiseim1_flat,str_noiseim2_flat = sel.select_byfilter_noise(filter_str[filtro],noise_dir)     
    print("str_noiseim1_flat:",str_noiseim1_flat)
    print("str_noiseim2_flat:",str_noiseim2_flat)
    
    #-select_(flat_)images-#
    noiseim1_flat = sel.select_image(str_noiseim1_flat)  					# already (128,128)
    noiseim2_flat = sel.select_image(str_noiseim2_flat)
    ##--randomize_noise--##
    noiseim = noiser.randomize_noise(noiseim1_flat, noiseim2_flat,linear_comb=True)  	
    
    tiempo_inicial=time.time()
    #--------RUN------#
    PREPARE = 1        
    if PREPARE:	
        for data_filename in VELA_data:
            str_image = os.path.join(absVELA_dirs[VELA_id] , data_filename)  
            prep_image = prepare_image(str_image , filtro)           +  noiseim      #3. add the noise
            saver.save_fits(prep_image, data_filename , absprepVELA_dirs[VELA_id])
    #------END_RUN----#
    tiempo_final=time.time()
    print('tiempo_stamps_flats:',tiempo_final-tiempo_inicial)
###---</RUIDO_MIXED_FLATS>---###

###---<RUIDO_REAL-HSTlike>---###

#<CORE!>
if real_HSTlike_noise:  
    # list_of_candel_image    (data_params.py)           # 2 images
    ##--NOISE GENERATION routine--##
    cmd_noise = "python %s/noise_addition_frontend.py %s %s"
    sub_noise = subprocess.Popen( cmd_noise %(noise_dir,VELA_id,filtro) , shell=True,stdout=subprocess.PIPE , stderr=subprocess.PIPE)
    exit_code = sub_noise.wait()               #-tiempo_generated_noise-#
    stdout, stderr = sub_noise.communicate()    #stdout,stderr
    
     ##--/NOISE GENERATION routine--##
    if (exit_code != 0):                                  #<ME> Error print message
        print(cmd_noise % (noise_dir,VELA_id,filtro))
        print("Non zero exit code... a problem has occur!")
    else:
        #noiseadd_ext
        #--------RUN------#
        PREPARE = 1                # DEFAULT: 1     
        if PREPARE:	
            for data_filename in VELA_data:
                data_filename_noise = data_filename.split("/")[-1].split("_SB00.fits")[0]+noiseadd_ext+".fits"
                str_image  = os.path.join(destin_dir,"VELA%s"%VELA_id,data_filename_noise)  
                prep_image = prepare_image(str_image,filtro)
                saver.save_fits(prep_image , data_filename , absprepVELA_dirs[VELA_id])   #guardar en python/DATA
        #------END_RUN----#

###---</RUIDO_REAL-HSTlike>---###

print("###---<END of prepare.py>---###")
###-----------------------------------------END----------------------------------------------###
