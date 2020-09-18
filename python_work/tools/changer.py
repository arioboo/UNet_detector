#import tools.changer as changer

#-<change_zp>-#
zp={'f160':25.9400,'i':25.6540,'v':26.4800,'b':25.6700}  #zeropoint correction of each filter   # duplicado en data_params.py

def change_zp(image,filtro):		#from coded_filters
	zp_im = image/10**((30-zp[filtro])/2.5) # Divide the image # in some cases, the images are highly affected by the noise so that we cannot treat with those images
	return zp_im             


#-< >-#
