Author: Kameswara Mantha; 
email: km4n6@mail.umkc.edu

Collaborators: Yicheng Guo, Harry Ferguson, Elizabeth McGrath, Haowen Zhang, Joel Primack, 
Raymond Simons, Gregory Snyder, Marc Huertas-Company + CANDELS et al.,

This is a brief README file for the realistic noise-addition pipeline.

A breif working of the pipeline: 
1. Consider a real image given by the user, perform source identification and mask all of the sources.
2. Generate a sky-only region and then randomly sample 10x10 regions of the un-masked regions to genrate a large noise-only mosaic
3. Take the user-provided noise-less image and generate a poisson realization of it. Take the residual poisson (poisson realization - noise less image).
4. Smooth the poisson residual image with a Gaussian filter, such that the 1D azimuthally averaged power spectral density (PSD) matches the 1D PSD
of the real noise stamp (noise properties of the user given real image).
5. Add the correlated (smoothed) poisson residual to the noise-less image to induce correlations in the source noise, 
that match the pixel-to-pixel correlations of the sky noise.
6. Add the sky noise mosaic to the correlated source image to generate a realistic noised image.


There are three python files enclosed along with this 
readme file:
1. noise_addition_pipeline.py -- The main functions that carryout the noise-additon pipeline is enclosed.
2. noise_addition_frontend.py -- The frontend that wraps the pipeline for a user-specified set of noise-less images.
3. radialProfile.py -- a supplementary function that is used during the computation of 1D azimulthally average power spectral density.

There is an example SB00 folder in which i placed some F160W no noise added images by G. Snyder et al.,

NOTE TO USER:
1. Please open the noise_addition_frontend.py
2. Please change the paths in which the real CANDELS images are stored
3. Please specify the location in which SB00 (noise-less) images are stored.
4. Please specify the destination directory in which the noise-added images will be stored.
5. Please specify the location in which the plots that are made on the run should be stored.


HOW TO RUN:
1. Make the above necessary changes in the frontend file.
2. Choose the VELA run, for e.g., for VELA01, change the number in the for loop to '01'.
3. Thats it, do 'python noise_addition_frontend.py' in the terminal or open your code and hit run in your favorite IDE. 

Recommendations:
1. I recommend running this pipeline for one VELA run at a time.
2. Make sure that the real images given do not have any artifacts (e.g., image edges.. too noisy)
3. I recommend running this pipeline when in an astroconda environment. However, not strictly required.

Required modules:
glob, numpy, os, astropy, matplotlib, sep, sklearn, warnings, copy, photutils, 
scipy, radialProfile (provided with this code.)

Kindly install necessary modules as needed before running my pipeline.

Known issues:
As pointed by Liz McGrath, some very high-redshift (z>3) snapshots have repeated pattern of noisy patches (currently working on this) 

#####################################################################################################3
--------------------------------#Alvaro Rioboo de Larriva#---------------------------------
ficheros:
@@<noise_addition_pipeline.py>@@
@@<noise_addition_frontend>@@
@@radialProfile>@@


@@requirements_noise_gen.txt@@:
Para la versión Python 3.4 , están todos los módulos disponibles.
En el notebook, instalar "sep" y "photutils" con la opción "--no-deps" con "pip".




urls:
https://photutils.readthedocs.io/en/stable/install.html
https://sep.readthedocs.io/en/v1.0.x/


-------------------------------------------#END#----------------------------------------------






