from params.data_params import *
import sys,os
import pdb
import subprocess
###---<LARGE RUN!>---###
import time 



def make_command(string_command,ID,fil,if_ierr=False,if_debug=False):
    subp = subprocess.Popen(string_command % (ID,fil), shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)  #Parse strings
    exit_code = subp.wait()       #Wait command to complete
    if if_ierr:                stdout, stderr = subp.communicate()  #stdout,stderr of the subprocess
    if if_debug:               pdb.set_trace()
    return 


if_prepare,if_unet_me,if_create_sex_catalog2 = (0,0,1)   #1 or 0  #<MAY CHANGE> (careful!)
fil_list=['b','v']                                       #'b','v' #<MAY CHANGE> (careful!)


#Python strings to run code from path: 'python_work'(cwd),'sextractor_work'
cmd_prepare = "python prepare.py %s %s"
cmd_unet    = "python unet_me.py %s %s"
cmd_sexcat  = "python ../sextractor_work/create_sex_catalog2.py %s %s"
              #<BORRAR>

time_start = time.time()
for ID in ID_list:                         #ID_list
    for fil in fil_list:                  #['b','v'] #coded_filters.keys()
        print(ID,fil,'       ',VELA_id,filtro)  
        if if_prepare:                                   # 25' mixed ,#
            make_command(cmd_prepare,ID,fil)
        if if_unet_me:                                   # 288' mixed ,#
            make_command(cmd_unet,ID,fil)
        if if_create_sex_catalog2: 
            make_command(cmd_sexcat,ID,fil)              # 33' mixed ,#
time_end = time.time()
print("\n\nTOTAL_TIME:",time_end-time_start)


##NOTES##
'''
[In:]
k1=25 ; k2=288 ; k3=33 ; k=k1+k2+k3
print("{0:.2%}".format(k1/k),"{0:.2%}".format(k2/k),"{0:.2%}".format(k3/k) )

[Out:]
7.25% 83.19% 9.57%
'''