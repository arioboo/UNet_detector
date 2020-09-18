#import tools.selecter as selecter

from params.data_params import *

#-<select_filter_noise>-#
def select_byfilter_noise(filtro_str,directorio):
	str_noiseim1_flat = directorio+filtro_str+'1'+fits_ext     #;print('im1:',str_noiseim1_flat)  #;os.path.isfile(str_noiseim1_flat)
	str_noiseim2_flat = directorio+filtro_str+'2'+fits_ext     #;print('im2:',str_noiseim2_flat)
	return str_noiseim1_flat,str_noiseim2_flat

#-<select_image>-#
from astropy.io import fits

def select_image(str_image):
	with fits.open(str_image) as hdul:
		data=hdul[0].data
	return data

#̣-< >-#
