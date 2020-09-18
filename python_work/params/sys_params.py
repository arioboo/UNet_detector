# -*- coding:utf-8 -*-
######-----author:arioboo-----######    #Defines :parent_dir, python_dir, sex_dir  , noise_dir(noisegen_dir)
#-<this_file modules>-#
import os,sys						

#-----------------------------USER-SPACE--------------------------------------#
USERNAME=os.environ.get('USER'); USERNAME_raspi='pi'         #arioboo,arl94

# The 1st entry refers to the path of this file, the 2nd entry refers to the path of this file in a given tree directory (NOT RECOMMENDED)
#-------------------------------PATHS-----------------------------------------#
#-parent_directory-#
if_relative_path = 1                                    # Always 1!! (DEFAULT)  
if if_relative_path:
        parent_dir = os.path.dirname(os.path.realpath(__file__+'/../..'))   # "/notebooks/CLUMPS_VELA" (in notebook)
        sys.path.insert(0,parent_dir)
            

#-python_directory-#		
python_dir = os.path.join(parent_dir,"python_work")     ; sys.path.insert(0,python_dir) 	 						               
#-sextractor_directory-#
sex_dir    = os.path.join(parent_dir,"sextractor_work") ; sys.path.insert(0,sex_dir) 	  	        
#-models_directory-#
models_dir   = os.path.join(parent_dir,"models")        ; sys.path.insert(0,models_dir)

###---OTHER_DIRECTORIES---###

#----------------------------GENERAL_PARAMETERS--------------------------------#

#-------------------------------EXTENSIONS------------------------------------#
#-files
fits_ext = '.fits'  #images fits
im_ext	 = '.png'   #images png

#-predictions
predict_extension = "-pred"  # suffix for predicted imgs

#-CANDELS_noise
noiseadd_ext = "_noise_added"
rnoise_ext   = "_real_noise_stamp"
spoiss_ext   = "_smoothed_poiss"
#-----------------------------------------------------------------------------#

'''
NOTES: Here are general variables and paths. 
Execute data_params.py module to test all variables if "if_test_params" is active
'''
#----------------------------------<END>--------------------------------------#



