#import tools.noiser as noiser

#-<randomize_noise>-#
import numpy.random

def randomize_noise(noiseim1_flat, noiseim2_flat ,linear_comb=True):
	if not linear_comb:   	# shuffle only 1 noise profile
		numpy.random.shuffle(noiseim1_flat)

		noiseim = noiseim1_flat.reshape(128,128)		
	else: 				# do linear comb of noises from 2 noise profiles
		numpy.random.shuffle(noiseim1_flat)
		numpy.random.shuffle(noiseim2_flat)
		noise_rn = numpy.random.random()

		noiseim = noiseim1_flat.reshape(128,128)*noise_rn + noiseim2_flat.reshape(128,128)*(1-noise_rn)	

	print('linear_comb:',linear_comb)
	
	return noiseim

'''
NOTES: use random linear combination of two noise profiles 
note: the noise profile noiseim2 for some reason seems to have a point source type bright spot in the upper right quadrant of the image
use a random shuffling of pixels for noiseim1 profile.  Hopefully this is acceptable.
'''

#̣-< >-#
