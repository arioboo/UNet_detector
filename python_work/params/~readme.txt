###---<README.txt>---###

##-<python_work/params>-##  MAIN FOLDER

Python configuration files for the run:

"sys_params.py" contains general and fixed directories as well as variables.
"data_params.py" contains variables and directories depending on the run and used explicitly by the programs (.py).



##--<sextractor_work/params>--## -->linked to: <python_work/params>
This folder is a link to the parameters folder of "python_work". To be used by the python scripts of "sextractor_work".



##--<noise_gen_dir/params>--##  -->linked to: <python_work/params>
This folder is a link to the parameters folder of "python_work". To be used by the python scripts of "noise_gen_dir".










#-----------------------------------------------------------------------------------------------------
##-<NOTES>-##:
When doing symbolic links, we can refer to general(and filesystem_adaptable)variable_python_paths as "python_dir","sex_dir","parent_dir",etc. Or just going through "../../" referenced by the current folder.

