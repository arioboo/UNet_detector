# coding: utf-8
############-----------author:arioboo------------###############
'''
This code is a frontend to a pipeline which creates noise-added images of idealized HST images of simulated galaxies.

Author: Kameswara Mantha; email: km4n6@mail.umkc.edu
Collaborators: Yicheng Guo, Harry Ferguson, Elizabeth McGrath, Haowen Zhang, Joel Primack, 
Raymond Simons, Gregory Snyder, Marc Huertas-Company + CANDELS et al.,

Load the necessary modules. The main functions are located in noise_addition_pipeline.py
'''
    #-<general_modules>
import glob
import numpy as np
import os
import time
    #-<custom modules>-#
import noise_addition_pipeline 
from params.data_params import *



###--------------------<START noise_addition_frontend.py>---------------------------###

print("###---<START of noise_addition_frontend.py>---###")
#noise_gen_dir = '/notebooks/CLUMPS_VELA/noise_stamps'                                      
#list_of_candels_image = glob.glob( os.path.join(noise_gen_dir,'bk_i-*^.fits') )                      

################<SB00 (noise-less) images location>###################

#sb00_loc = '/home/arl94/TFM/CLUMPS_VELA/'                #VELA folder are inside "parent_dir"   

############<Fits save file destination directory for noise-added images>#############

#destin_dir = os.path.join(noise_dir , "noise_added")                 

#################<Plot save location>#################

#plot_save_loc = os.path.join(noise_dir , "plots")      # folder in which plots are stored for each VELA run
try:            os.stat(plot_save_loc)
except OSError: os.mkdir(plot_save_loc)

#VELA_number = ['01']                                         # VELA_number = [VELA_id]        # data_params.py

time_inicial = time.time()
for each_vela_run in VELA_number:
    if not os.path.exists('%s/VELA%s'%(destin_dir,each_vela_run)):
        os.mkdir('%s/VELA%s'%(destin_dir,each_vela_run))

    # Change filter_name and the total_path as needed.
    list_of_noise_less_vela_image = glob.glob("%s/VELA%s/*_*_*_cam*%s-%s_SB00.fits"%(sb00_loc,each_vela_run,instrument,coded_filters[filtro]))  #por vela_id y filtro 

    # For each vela image in the list,
    for vela_image in list_of_noise_less_vela_image:
        
        can_image = np.random.choice(list_of_candels_image) # choose a real CANDELS image.
        # This step calls the pipeline that takes in the real image, the noise-less simulated image and produces as noise-added image.
        #One can choose to make plots during this process, or turn it to False. Make sure the destination directories are appropriate.
        noise_addition_pipeline.extract_noise_and_exp_patches(can_image,vela_image,
                                                      make_plots=False,is_image_cube=False,
                                                    plots_loc=plot_save_loc,fits_save_loc=destin_dir)
        #print('Added noise to %s\n--------\n'%os.path.basename(vela_image))
time_final = time.time()
print('tiempo_generate_noise:',time_final-time_inicial)                      #157.25 s in '02''b'           

print("###---<END of noise_addition_frontend.py>---###")
###--------------------------<END noise_addition_frontend.py>----------------------------###