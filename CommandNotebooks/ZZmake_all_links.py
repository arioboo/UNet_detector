import glob
import os, errno

#-<function: symlink_force>-#
def symlink_force(target, link_name):
    try:
        os.symlink(target, link_name)
    except OSError as e:
        if e.errno == errno.EEXIST:
            os.remove(link_name)
            os.symlink(target, link_name)
        else:
            raise e
#-</function>

ipynb_list = glob.glob("*.ipynb")    #; print(ipynb_list)

cwd  = os.getcwd()                          ;print("cwd:",cwd)
pdir = os.path.abspath(__file__+"/../../")  ;print("pdir:",pdir)

symlink_force(cwd+"/CommandNotebook_pdir.ipynb",pdir+"/CommandNotebook_pdir.ipynb")          #DEFAULT : comment this

python_dir      = pdir + "/python_work"
sex_dir         = pdir + "/sextractor_work"  ; catalogs_dir = sex_dir + "/sex_catalogs"
noise_gen_dir   = pdir + "/noise_gen"        ; noise_added_dir = noise_gen_dir + "/noise_added"
ds9_pruebas_dir = pdir + "/ds9_pruebas"      ; topcat_ds9_dir = ds9_pruebas_dir + "/topcat_ds9"

try:
    ###---<PRINCIPALES>---###
    #-python_work-#
    symlink_force(cwd+"/CommandNotebook_python_work.ipynb",python_dir+"/CommandNotebook_python_work.ipynb")
    symlink_force(cwd+"/prepare.ipynb"                    ,python_dir+"/prepare.ipynb")               #(CORE!!)
    symlink_force(cwd+"/unet_me.ipynb"                    ,python_dir+"/unet_me.ipynb")               #(CORE!!)
    symlink_force(cwd+"/lookon_data.ipynb"                ,python_dir+"/lookon_data.ipynb")
    symlink_force(cwd+"/MAKE_ALL.ipynb"                   ,python_dir+"/MAKE_ALL.ipynb")           
    #-sextractor_work-#
    symlink_force(cwd+"/CommandNotebook_sextractor_work.ipynb",sex_dir+"/CommandNotebook_sextractor_work.ipynb")
    symlink_force(cwd+"/create_sex_catalog2.ipynb"            ,sex_dir+"/create_sex_catalog2.ipynb")  #(CORE!!)          
    symlink_force(cwd+"/lookon_catalogs.ipynb"                ,sex_dir+"/lookon_catalogs.ipynb")
    #(sex-catalogs)
    symlink_force(cwd+"/CommandNotebook_sex_catalogs.ipynb"   ,catalogs_dir+"/CommandNotebook_sex_catalogs.ipynb")
    #-noise_gen-#
    symlink_force(cwd+"/noise_addition_frontend.ipynb"    ,noise_gen_dir+"/noise_addition_frontend.ipynb")
    symlink_force(cwd+"/CommandNotebook_noise_gen.ipynb"  ,noise_gen_dir+"/CommandNotebook_noise_gen.ipynb")
    #(noise_added)
    symlink_force(cwd+"/CommandNotebook_noise_added.ipynb",noise_added_dir+"/CommandNotebook_noise_added.ipynb")
    ###---<OTROS>---###
    #-ds9_pruebas-#
    symlink_force(cwd+"/CommandNotebook_ds9_pruebas.ipynb",ds9_pruebas_dir+"/CommandNotebook_ds9_pruebas.ipynb")
    #(topcat_ds9_pruebas)
    symlink_force(cwd+"/CommandNotebook_topcat_ds9.ipynb",topcat_ds9_dir+"/CommandNotebook_topcat_ds9.ipynb")
except:
    pass



###-------------------------------<END>-----------------------------###