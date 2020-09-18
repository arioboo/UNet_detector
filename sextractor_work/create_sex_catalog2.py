# coding: utf-8
############-----------author:arioboo------------###############
'''Generates SExtractor catalog using fits images (from model UNET output).''' 
    #-<main modules>#
import sewpy                     
import glob 
import pandas as pd              
import numpy as np
import sys,os
import subprocess
from astropy.io import fits
    #-<added modules>-#
import pdb
    #-<custom modules>-#
from params.data_params import *


print("###---<START of create_sex_catalog2.py>---###")

try:    print("Statting folder:",run_name_abs) ; os.stat(run_name_abs)
except: print("Making folder:  ",run_name_abs) ; os.mkdir(run_name_abs)
    
###---<PREPARATION>---###
'''Juntar las imágenes en una carpeta, hecho con enlaces simbólicos a las imagenes dentro de la carpeta para no sobrescribir la información. '''
#1. Mkdir if not exists run_name
try:    os.stat(run_name_abs)
except: os.mkdir(run_name_abs)       

#2. Inventary of real and predicted imgs
files_real      = sorted(glob.glob(os.path.join(img_folder,'*.fits')))         
files_real_only = [ f.split("/")[-1] for f in files_real]                      

files_pred      = sorted(glob.glob(os.path.join(output_folder,'*-pred.fits'))) 
files_pred_only = [ f.split("/")[-1] for f in files_pred]                      

#3. Symbolic links to the imgs of real and predicted imgs_folders 
message = 1
for lnk in files_real+files_pred:
    try:
        os.symlink(lnk , os.path.join(sex_output_folder, run_name , lnk.split("/")[-1] ) )    #<CORE> #(I!!) path relative to 
        if message:
            print("Creating links...Done")      ; message = 0
    except OSError: 
        if message:
            print("Links were already created") ; message = 0
        
###---</PREPARATION>---###
    
#-<files>-# : correr en imagenes .fits dadas por unet outputs ("-pred.fits" extension)
files      = sorted(glob.glob(os.path.join(run_name_abs, "*-pred.fits")))  
files_only = [ f.split("/")[-1] for f in files ]                    

print("Running SExtractor on dir: " + run_name)

df      = pd.DataFrame()
seg_arr = np.empty((len(files),128,128))
dot_freq = int(len(files)/100.)

###---<MAIN LOOP>---### : Adapted for Python3.4
i = 0
for f in files:   
    if (i % dot_freq == 0):                       
        sys.stdout.write(str(i/dot_freq)+"%")
        sys.stdout.flush()
           
    seg_name = f.split('.fits')[0]+'_seg.tmp.fits'         
    cat_name = f.split('.fits')[0]+'_cat.tmp.csv'          

    #<ME> cmd: sextractor -c /notebooks/CLUMPS_VELA/sextractor_work/clumps.sex %s -CHECKIMAGE_NAME %s -CATALOG_NAME %s
    sub = subprocess.Popen(cmd % (f,seg_name,cat_name), shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)  
    exit_code = sub.wait()
    stdout, stderr = sub.communicate()
    
    if (exit_code != 0):                                  #<ME> Error print message
        print(cmd % (f,seg_name,cat_name))
        print("Non zero exit code... that's a problem!")
        continue                                          #<ME> Continue to next file
        
    seg        = fits.open(seg_name)[0].data
    
    seg_arr[i] = seg
    cat = pd.read_csv(cat_name,delim_whitespace=True,comment='#',names=cols)
    #---------------------------DATAFRAME_COLUMNS----------------------------#
    gal_id = f.split("/")[-1].split("_")[0]    
    cat['gal_id']    = gal_id  
    cat['seg_index'] = i          
    #--<ME>--#
    a0_str = f.split("/")[-1].split("_")[1].split(".")[1]   
    a0 = "%.3f"%(float(a0_str)/1000)                   #<ME> some parsing...
    cat['a0'] = a0                                     #<ME> str (3decim)
    
    z = 1./float(a0) - 1                               #a(t) = 1/(1+z)
    cat['z'] = "%.3f"%z                                #<ME> str (3decim)
    
    cam = f.split("/")[-1].split("_")[3].split("cam")[-1]  
    cat['cam'] = cam
    
    instr = f.split("/")[-1].split("_")[4].split("-")[0]  
    cat['instrument'] = instr
    
    filt = f.split("/")[-1].split("-")[1].split("_")[0]   
    cat['filter'] = filt
    
    filename = f.split("/")[-1].split(".fits")[0] 
    cat['filename'] = filename                                  
    #--</ME>--#
    
    # Initialize variables for clumps detected in catalog csv file
    prob_mean = np.empty(len(cat))       
    prob_int  = np.empty(len(cat))
    flux      = np.empty(len(cat))
    
    # Open data of real_imgs and pred_imgs
    pred = fits.open(f)[0].data                                 # real_images            
    img  = fits.open(f.split('-pred.fits')[0]+'.fits')[0].data  # prediction_images

    for j in range(len(cat)):
        pixels = np.where(seg == (j+1))
        if (len(pixels[0]) == 0):      
            continue
        #print("No matching pixels in seg image for i=%d, clump=%d (tot: %d)"%(i,j,len(cat)))    
        #print(np.max(seg))
        prob_mean[j] = np.mean(pred[pixels])                 #calculates mean from pixels around "detected clump" in the "prediction image"
        prob_int[j]  = np.sum(pred[pixels])                  #calculates sum from pixels around "detected clump" in the "prediction image"
        flux[j]      = np.sum(img[pixels])                   #calculates sum from pixels around "detected clump" in the "real image"
        
    cat['prob_mean'] = prob_mean
    cat['prob_int']  = prob_int
    cat['flux']      = flux
    
    df = df.append(cat)                                      #appends the new builded "cat"(of 1 file) to "df"
    
    # remove SExtractor output files we just read in
    try:                                      
        os.remove(seg_name)                              
        os.remove(cat_name)                   
    except:
        print("Error in removing 'seg_name' and/or 'cat_name'")    
    
    i += 1
###---</MAIN LOOP>---### 

    
###---<Saving results>---###
print("Done!")
df = df.reset_index(drop=True)                                 #Eliminate index for saving

np.save     (catalogs_folder + "/sex_seg_" + run_name + ".npy", seg_arr)      #Save in numpy format 
df.to_pickle(catalogs_folder + "/sex_cat_" + run_name + ".pkl")               #Save in pickle format 
df.to_csv   (csv_folder      + "/sex_cat_" + run_name + ".csv")               #Save in csv (ascii) format, (can be loaded in topcat)

#3 above commands overwrites by default
###---</Saving results>---###
print("###---<END of create_sex_catalog2.py>---###")
###-------------------------------------END-------------------------------------------------###

