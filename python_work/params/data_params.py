# -*- coding:utf-8 -*-
############-----------author:arioboo------------###############  #Defines : <PARAMETERS>,<sys_params imports>,... 
    #-<this_file modules>-#
import glob
import sys   
import datetime
    #-<custom modules>-#
from params.sys_params import *        #parent_dir,python_dir
###------------------------<PARAMETERS>----------------------------### (on data_params) 
real_HSTlike_noise = 0                  #DEFAULT: 0                          # type_of_noise
try:
    VELA_id = sys.argv[1]                             
    filtro  = sys.argv[2]
    msg='(VELA_id,filtro) taken from command input.'  #;print(msg)
except :
    VELA_id = '01'                                   #(id for VELA galaxy)  #'02','04','05','06'
    filtro  = 'b'	                                 #(filter for images)   #'f160'('F160W'),'i'('F775W'),'v'('F606W'),'b'('F435W')    
    msg="(VELA_id,filtro) taken from data_params.py manual setup."  #;print(msg)
    



coded_filters = {'f160':'F160W','i':'F775W','v':'F606W','b':'F435W' }   # BOSS _dict (in nm ,but 'f160')	   
GAL_folder    = "VELA" + VELA_id + "_" + coded_filters[filtro]
instrument = "ACS" if filtro != 'f160' else "WFC3" 

###--<DATE>---###
old_date = '_12_21_18'            #old_date   #'_09_18_18' or '_12_21_18'(D!)
now = datetime.datetime.now() 
now_date ='_%s_%s_%s'%(now.strftime("%m"),now.day,now.strftime('%y'))  #now_date   #actualized each day

###------------------------<DATA_DIRECTORIES>---------------------------###
#-<SYS_PARAMS> imports-#

#-data_directories-#       (absVELA_dirs, absprepVELA_dirs)
VELA_list    = sorted(glob.glob1(parent_dir,"VELA*"))                                    # relative path list, search for pattern "VELA" #ordered
absVELA_list = [os.path.abspath( os.path.join(parent_dir,gal)) for gal in VELA_list]     # absolute path list                            #ordered
ID_list      = sorted([gal.split("VELA")[-1] for gal in VELA_list])                      # VELA galaxy number ordenation                 #ordered

VELA_dirs    = { ID_list[i] : "/"+VELA_list[i]        for i in range(len(ID_list))}      #                #unordered dictionary (doesn't matter)
absVELA_dirs = { entry : parent_dir+VELA_dirs[entry]  for entry in VELA_dirs.keys()}     # absolute path  #unordered dictionary (doesn't matter) 


#-noise(noisegen)_directory-#           (!!!TOKEN!!!)
if real_HSTlike_noise :                         # choose one option or another
    #-noise_addition_frontend.py-#
    noise_dir	  = os.path.join(parent_dir,"noise_gen")        # Images to add noise   
    noise_gen_dir = os.path.join(noise_dir,"noise_stamps")      # CANDELS images to real noise  #GOOD
    list_of_candels_image = glob.glob(noise_gen_dir+"/bk_"+filtro+"*.fits" )    #<ME> (CAMBIAR!)
    
    sb00_loc = parent_dir   #(I!!) modifiable folder SB00(noise-less) data  # directory where VELA noise-less folders are
    VELA_number = [VELA_id]
    
    destin_dir    = os.path.join(noise_dir,'noise_added')    
    plot_save_loc = os.path.join(noise_dir,'plots')                                           # special "/" 
else : 
    noise_dir	  = os.path.join(parent_dir,"noise_stamps")  ; sys.path.insert(0,noise_dir)       # Images to add noise   


#-prepare.py-#                                         
prepVELA_dirs    = {number:"/VELA"+number+"_"+coded_filters[filtro]         for number in ID_list}   #         #ID_list i.e. ['02','04','05','06']
absprepVELA_dirs = {entry : python_dir+"/DATA"+prepVELA_dirs[entry] 		for entry in prepVELA_dirs.keys()} #absolute path	
                                     
VELA_data	  = glob.glob1(absVELA_dirs[VELA_id],    '*'+coded_filters[filtro]+'*.fits')      	
prepVELA_data = glob.glob1(absprepVELA_dirs[VELA_id],'*'+coded_filters[filtro]+'*.fits') 

zp={'f160':25.9400,'i':25.6540,'v':26.4800,'b':25.6700}  #zeropoint correction of each filter
#-unet_me.py-#   
img_folder    = os.path.join(python_dir,"DATA"        ,GAL_folder)    #PREPARED_IMAGEs directory
output_folder = os.path.join(python_dir,"DATA_outputs",GAL_folder)    #UNET_OUTPUT_IMAGEs directory

date = old_date                                                                         #current working date
model_name_dict = {f : 'unet' + date + '-' + f        for f in coded_filters.keys()}    #'f160' missing
model_name      =  model_name_dict[filtro]         

batch_size = 4                  # 4
epochs     = 100                # 100
verbose    = 1                  # 1 (TRUE)
img_size   = 128                # 128
max_n      = 48000              # size of training set (remainder to be used for test set) #48000
ntrain     = int(max_n*9/10)    # training / validation split  #max_n*9/10
nval       = int(max_n/10)      # max_n/10

#-create_sex_catalog2.py-#
sex_output_folder = os.path.join(sex_dir        ,"fits_output" )              #SEXTRACTOR_FITS directory
catalogs_folder   = os.path.join(sex_dir        ,"sex_catalogs")
csv_folder        = os.path.join(catalogs_folder,"csv")

run_name     = GAL_folder
run_name_abs = os.path.join(sex_output_folder, run_name)   

cols = ["NUMBER", "X_IMAGE", "Y_IMAGE", 
        "A_IMAGE", "B_IMAGE", 
        "THETA_IMAGE", "XPEAK_IMAGE", "YPEAK_IMAGE"]                #clumps.param  

path_sexparams = os.path.join(sex_dir,"clumps.sex")                                                 
cmd = "sextractor -c " + path_sexparams + " %s -CHECKIMAGE_NAME %s -CATALOG_NAME %s"                #<ME> #command for ..



'''
NOTES: Here are variables and paths depending on <PARAMETERS> and VELA directories existing.
Below print function is useful to check all out.
'''
#----------------------------------<END>----------------------------#

## to_add to print_params():


def print_parameters(real_HSTlike_noise):
    ###---<PARAMETERS>---###
    str_dash="--------------------------------------------------------"
    str_blanks="\n"
    print("real_HSTlike_noise:",bool(real_HSTlike_noise))
    print(msg)
    print(str_dash)
    print("VELA_id: %s   || filtro: %s" %(VELA_id,filtro))
    try:      print("VELA_folder:", VELA_folder)
    except    NameError: pass
    print(str_dash)
    ###---<SYS_PARAMS> imports---###
    #-parent_directory-#
    print("parent_dir:  ", parent_dir)
    #-python_directory-#
    print("python_dir:  ", python_dir) 
    #-sextractor_directory-#
    print("sex_dir:     ", sex_dir) 
    #-noise(noise_gen)_directory-#
    if real_HSTlike_noise:
        print("noise_gen_dir:%s                           || noise_dir:%s" %(noise_gen_dir,noise_dir))  
        #-noise_addition_frontend.py-#
        #noise_dir
        print("sb00_loc:     ",sb00_loc)           #; print("list_of_candels_image:",list_of_candels_image)        
        print("destin_dir:   ",destin_dir)
        print("plot_save_loc:",plot_save_loc)
    else:
        print("noise_dir:",noise_dir)
        
    
    #-models_directory-#
    print("models_dir:  ",models_dir)
    print(str_blanks)
    ###---<DATA_PARAMS>---###
    #-data_directories-#       (absVELA_dirs, absprepVELA_dirs)
    print("VELA_list:   ", VELA_list)
    print("absVELA_list:", absVELA_list)
    print("ID_list:     ", ID_list)

    #print("VELA_dirs:   ", VELA_dirs)           #(CHANGE!)
    #print("absVELA_dirs:", absVELA_dirs)        #(CHANGE!)
    print(str_blanks)

    
    #-prepare.py-#
    print("coded_filters:   ", coded_filters) 

    #print("prepVELA_dirs:   ", prepVELA_dirs)     #(CHANGE!)
    #print("absprepVELA_dirs:", absprepVELA_dirs)  #(CHANGE!)

    if_print_VELA_data = 0
    if if_print_VELA_data:
        print("VELA_data:    ", VELA_data)
        print("prepVELA_data:", prepVELA_data)

    print(str_blanks)    
    #-unet.py-#
    print("img_folder:   ", img_folder)          #img_dir
    print("output_folder:", output_folder)       #saved_outputs
    print("model_name_dict:", model_name_dict)
    print("model_name:     ", model_name)          #filename of heights (no .hd5)
    print(str_dash[0:4])
    print("batch_size:", batch_size)
    print("epochs:    ", epochs)
    print("verbose:   ", verbose)
    print("img_size:  ", img_size)
    print("max_n:     ", max_n)
    print("ntrain:    ", ntrain)
    print("nval:      ", nval)
    print(str_dash[0:4])
    #-create_sex_catalog2.py-#
    print("sex_output_folder:",sex_output_folder)
    print("catalogs_folder:  ",catalogs_folder)
    print("csv_folder:       ",csv_folder)
    print("path_sexparams:",path_sexparams) 
    
    print("run_name_abs:%s         ||run_name:%s" %(run_name_abs,run_name))
    print("cols:",cols)
    print("cmd:",cmd)
    print("###---<End_of_PRINT-PARAMERS>---###")
    print(str_blanks)
    ##--<END>--##    
    return 


#running print: only if this is the main program (not an imported module)    
if __name__ == "__main__" :
    
    outfile = "output.txt"
    if os.path.isfile(outfile): 
        os.remove(outfile)
    
    old_stdout=sys.stdout                      #save default stdout
    with open(outfile,"w") as outputfile:
        
        sys.stdout = outputfile                    #changes stdout to print to a file
        print_parameters(real_HSTlike_noise)       #now make the function prints
    sys.stdout = old_stdout                    #backs to stdout
    
