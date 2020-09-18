# create_sex_catalog2.py
# generate Sextractor catalog using fits images (model output only) using system installation of SExtractor
import sewpy, glob
import pandas as pd
import numpy as np
import sys
import subprocess
from astropy.io import fits
import os

#sew = sewpy.SEW(
#        params=["NUMBER", "X_IMAGE", "Y_IMAGE", "A_IMAGE", "B_IMAGE", "THETA_IMAGE", "XPEAK_IMAGE", "YPEAK_IMAGE"],
#        configfilepath="fits_outputs/clumps.sex",#config={"DETECT_MINAREA":4, "PHOT_APERTURES":"10, 20, 30, 40"},
#        sexpath="sextractor"
#    )

# Get run name from command line if possible. E.g. python create_sex_catalog.py run_name_here
if len(sys.argv)>1:
    run_name = sys.argv[1]
else:
    run_name = "img_test-1999_07_05_18"
    #run_name = "imgs_real_test-1270_05_24_18"
    #run_name = "imgs_real_pred_b_1270_08_15_18"

#files = glob.glob("fits_outputs/*.fits")
#files = glob.glob("fits_outputs/imgs_real_test-1270_05_24_18/*.fits")
files = glob.glob("fits_outputs/"+run_name+"/*-pred.fits")

print "Running SExtractor on dir: "+run_name

cols = ["NUMBER", "X_IMAGE", "Y_IMAGE", "A_IMAGE", "B_IMAGE", "THETA_IMAGE", "XPEAK_IMAGE", "YPEAK_IMAGE"]

df = pd.DataFrame()
seg_arr = np.empty((len(files),128,128))

dot_freq = int(len(files)/100.)

cmd = "sextractor -c fits_outputs/clumps.sex %s -CHECKIMAGE_NAME %s -CATALOG_NAME %s"

i = 0
for f in files:
    #if (i==10):
    #    break
    if (i % dot_freq == 0):
        sys.stdout.write(str(i/dot_freq)+".")
        sys.stdout.flush()
    gal_id = f.split("/")[-1].split(".fits")[0].split("-")[-2]   #i.e. VELA06
    #print gal_id
    seg_name = f.split('.fits')[0]+'_seg.tmp.fits'
    cat_name = f.split('.fits')[0]+'_cat.tmp.csv'
    try:
        os.remove(seg_name)
        os.remove(cat_name)
    except OSError:
        pass
    sub = subprocess.Popen(cmd % (f,seg_name,cat_name), shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    exit_code = sub.wait()
    #stdout, stderr = sub.communicate()
    if (exit_code != 0):
        print cmd % (f,seg_name,cat_name)
        print "Non zero exit code... that's a problem!"
        continue
    seg = fits.open(seg_name)[0].data
    seg_arr[i] = seg
    cat = pd.read_csv(cat_name,delim_whitespace=True,comment='#',names=cols)      
    #<ME> pd.read_csv() Read a comma-separated values (csv) file into DataFrame.
    cat['gal_id'] = gal_id
    cat['seg_index'] = i
    prob_mean = np.empty(len(cat))
    prob_int = np.empty(len(cat))
    flux = np.empty(len(cat))
    pred = fits.open(f)[0].data
    img = fits.open(f.split('pred.fits')[0]+'img.fits')[0].data
    for j in range(len(cat)):
        pixels = np.where(seg==(j+1))
        if (len(pixels[0]) == 0):
            continue
        #    print "No matching pixels in seg image for i=%d, clump=%d (tot: %d)"%(i,j,len(cat))
        #    print np.max(seg)
        prob_mean[j] = np.mean(pred[pixels])
        prob_int[j] = np.sum(pred[pixels])
        flux[j] = np.sum(img[pixels])
    cat['prob_mean'] = prob_mean
    cat['prob_int'] = prob_int
    cat['flux'] = flux
    df = df.append(cat)
    # remove SExtractor output files we just read in
    os.remove(seg_name)
    os.remove(cat_name)
    #out = sew(f)
    #_df = out["table"].to_pandas()
    #_df['gal_id'] = gal_id
    #df = df.append(_df)
    i += 1

print " done!"

df = df.reset_index(drop=True)

np.save("sex_catalogs/sex_seg_"+run_name+".npy", seg_arr)
#df.to_pickle("sex_cat_imgs_real_test-7_4_18.pkl")
#df.to_pickle("sex_catalog_7_4_18.pkl")
df.to_pickle("sex_catalogs/sex_cat_"+run_name+".pkl")
