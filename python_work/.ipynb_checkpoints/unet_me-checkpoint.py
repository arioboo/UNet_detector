# coding: utf-8
############-----------author:arioboo------------###############
'''Generates output recognition images from DeepLearning model "UNET" in the PREDICTION task.Save results on DATA_outputs/ directory'''
    #-<main modules>-#
from keras.models import Model
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import MaxPooling2D
from keras.layers import concatenate
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from glob import glob
import numpy as np
from astropy.io import fits
import random
import sys, os
    #-<custom modules>-#
from params.data_params import *


print("###---<START of unet_me.py>---###")

__all__ = ['UNet']
try:    print("Statting folder:",output_folder) ; os.stat(output_folder)
except: print("Making folder:  ",output_folder) ; os.mkdir(output_folder)

###---<FUNCTIONS>---###
#-<Read CANDELS images>-#        
def read_files_real (input_path, img_size, n = None):
    files = sorted(glob.glob(input_path+'/*.fits'))    #<ME> sorted
    GalID = np.array([f.split('/')[-1][6:-5] for f in files])      #GalID = np.array([f.split('/')[-1].split('_')[1] for f in files])     #(OCODE!)   
    
    if (n is None):
        n = len(files) ; n=int(n)
    imgs = np.empty((n,img_size,img_size))
    print ("Reading in " + str(n) + " fits files...")
    dot_freq = int(np.round(n/100.+0.5))
    for i in range (0,n):
        if (i % dot_freq == 0):
            sys.stdout.write(str(i/dot_freq)+".")
            sys.stdout.flush()
        try:
            hdul = fits.open(files[i])
            imgs[i] = hdul[0].data
        except Exception as excep:
            continue
    return imgs, GalID

#-<Define UNet model>-#
def UNet(input_shape, init_filt_size=64):
   
    img_input = Input(shape=input_shape)

    nfilt1 = init_filt_size
    nfilt2 = nfilt1 * 2
    nfilt3 = nfilt2 * 2

        #-<Block 1>-#
    x = Conv2D(nfilt1, (3, 3), activation='relu', padding='same',
               name='block1_conv1')(img_input)
    x_1a = Conv2D(nfilt1, (3, 3), activation='relu', padding='same',
                  name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2),
                     name='block1_pool')(x_1a)

        #-<Block 2>-#
    x = Conv2D(nfilt2, (3, 3), activation='relu', padding='same',
               name='block2_conv1')(x)
    x_2a = Conv2D(nfilt2, (3, 3), activation='relu', padding='same',
                  name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2),
                     name='block2_pool')(x_2a)

        #-<Block 3>-#
    x = Conv2D(nfilt3, (3, 3), activation='relu', padding='same',
               name='block3_conv1')(x)
    x = Conv2D(nfilt3, (3, 3), activation='relu', padding='same',
               name='block3_conv2')(x)
    x_2b = Conv2DTranspose(nfilt2, (2, 2), strides=(2, 2),
                           input_shape=(None, 23, 23, 1),
                           name='block3_deconv1')(x)

        #-<Deconv Block 1>-#
    x = concatenate([x_2a, x_2b])
    x = Conv2D(nfilt2, kernel_size=(3, 3), activation='relu',
               padding='same', name='dblock1_conv1')(x)
    x = Conv2D(nfilt2, (3, 3), activation='relu', padding='same',
               name='dblock1_conv2')(x)
    x_1b = Conv2DTranspose(nfilt1, kernel_size=(2, 2), strides=(2, 2),
                           name='dblock1_deconv')(x)

        #-<Deconv Block 2>-#
    x = concatenate([x_1a, x_1b], input_shape=(None, 92, 92, None),
                    name='dbock2_concat')
    x = Conv2D(nfilt1, (3, 3), activation='relu', padding='same',
               name='dblock2_conv1')(x)
    x = Conv2D(nfilt1, (3, 3), activation='relu', padding='same',
               name='dblock2_conv2')(x)

    # NOTE: this line hardcodes the number of output channels (currently == 1).
        #-<Output convolution>-#
    x = Conv2D(1, (1, 1), activation=None, padding='same',
               name='dblock2_conv3')(x)

        #-<Create model>-#
    model = Model(img_input, x, name='UNet')

    return model
###---</FUNCTIONS>---###
###---<TASKS>---###
model = UNet ((img_size,img_size,1))
model.compile(optimizer = Adam(lr = 1e-3),
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])

READ_DATA    = 1                # 1
PREDICT_REAL = 1                # 1

#-<READ_DATA>-#
if READ_DATA:
    print ("Reading input data...")
    imgs, GalID = read_files_real (img_folder, img_size)                                       
    np.save(output_folder + '/imgs_' + str(imgs.shape[0]) + date + '.npy', imgs)## no me gusta el "GalID", no está bien definido con mis datos.
    
    print ("Loading images..")   ; print('imgs.shape[0]=',imgs.shape[0])
    imgs = np.load(output_folder + '/imgs_' + str(imgs.shape[0]) + date + '.npy')
    print ("done.")
    
    imgs = np.expand_dims(imgs,3)
    
    
#-<PREDICT_REAL>-#
if (PREDICT_REAL):
    print ("Checking model predictions...")
    model.load_weights(   os.path.join(models_dir,model_name+'.hd5')   )

    img_test = imgs[:]                   
    img_pred = model.predict(img_test)   
    
    (n, m , _, _) = img_test.shape 
    
    test_output = np.zeros((n,m,m,3))               
    test_output[:,:,:,0] = img_test[:,:,:,0]       
    test_output[:,:,:,1] = img_pred[:,:,:,0]          
    print ("done.")
    
    print ("Saving test results...")    
    np.save(output_folder + '/img_test-' + str(img_test.shape[0]) + date + '.npy', test_output)
    print("done.")
###---</TASKS>---###
    
    
###---<Write fits from predicted images>---###
real_filenames = sorted(glob.glob(img_folder + "/*.fits"))                     #1. Takes original filenames strings     #<ME> sorted                                            
filename_str   = [ i.split("/")[-1].split(".fits")[0] for i in real_filenames] #2. Parse them (quit ".fits")               
for num_file in range(0,len(filename_str)):                                    
    fits.writeto(os.path.join(output_folder, filename_str[num_file] + predict_extension + ".fits"),
                 img_pred[num_file,:,:,0],overwrite=True)                      #3. Write predict images
###---</Write>---###
    
print("###---<END of unet_me.py>---###")
####--------------------------------THE_END----------------------------------------###