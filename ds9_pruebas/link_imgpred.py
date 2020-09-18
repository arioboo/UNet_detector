import sys,os
from glob import glob1
import random


VELA_id = "VELA"+sys.argv[1]  ;print("VELA_id:",VELA_id)  #;print(type(VELA_id))
filtro  = sys.argv[2]         ;print("filtro:",filtro)    #;print(type(filtro))

coded_filters = {'f160':'F160W','i':'F775W','v':'F606W','b':'F435W' }   # BOSS _dict (in nm ,but 'f160')	

try:            
    folder_name=VELA_id+"_"+coded_filters[filtro]         #folder_name containing fits

    parent_dir = os.path.dirname(os.path.realpath(__file__+'/..'))
    cwd = os.getcwd()
    #make if not exists
    try:    os.stat( os.path.join(cwd , folder_name )   )  ; print("Folder name already exists")
    except: os.mkdir( os.path.join(cwd , folder_name )  )  ; print("Folder name created")
        
    try:    os.stat(os.path.join(cwd,folder_name,'pngs')) 
    except: os.mkdir(os.path.join(cwd,folder_name,'pngs'))
        
    #folders of real and predicted imgs
    real_folder = os.path.join(parent_dir,"python_work/DATA"        ,folder_name)    ;print(real_folder)
    pred_folder = os.path.join(parent_dir,"python_work/DATA_outputs",folder_name)    ;print(pred_folder)

    #list files of each folder
    lista_real = sorted(glob1(real_folder,"*.fits"))
    lista_pred = sorted(glob1(pred_folder,"*.fits"))

    #randomize number and take one
    if_random = 1 
    if if_random:
        num = random.randint(0,len(lista_real)) 
    else:
        num = 100
    
    #link files    
    print("Linking fits ...")
    os.symlink( os.path.join(real_folder , lista_real[num])   , os.path.join(cwd,folder_name,lista_real[num]))  ;print(lista_real[num])
    os.symlink( os.path.join(pred_folder , lista_pred[num])   , os.path.join(cwd,folder_name,lista_pred[num]))  ;print(lista_pred[num])
    print("done.")
except:         
    print('Filter not available. Aborting ..')



