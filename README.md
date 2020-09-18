# UNET_DETECTOR

**NOTE: This project is actually abandoned. I mantain it public, so feel free to explore it and make me know if you want to collaborate/use code/give ideas on the project. If so, mail me to "riodela.alvaro@gmail.com" to get further support/instructions.**

    #-<DESCRIPTION>-#
The goal is to find regions of high SFR called "clumps" in these simulated VELA galaxies. 
For that purpose, we use Deep Learning (U-Net CNN architecture to perform a feature extraction).
Training has been done in Lee at al. 2019 for filters F435W and F606W of ACS instrument (CANDELS project in the HST)
There has been observations of clumps in real data from other papers from Huertas-Company et al.
This project makes 2D catalogues of clumps with SExtractor software. Catalogues are in .csv format and are exportable to other formats (json,etc.)
A good tool to visualize catalogues is TOPCAT.

    #-<VELA data folders>-#
2 basic sets:
  - VELA_archive_2filters.tar.gz  : (~14Gb) This is the selected 30 VELA folders for F435W and F606W. Redshift ranges may vary as well as sizes and epochs from one gal. to another.
  - VELA_archive.tar.gz : (~30Gb)  This is the whole sample of VELA simulation.
 
Although scripts work with the general dataset, we recommend the  "2filters" dataset. Please email to "alvaroinator7@gmail.com" to 
ask for one of the above datasets.

    #-<INITIAL INSTRUCTIONS>-#
The initial steps are pretty standard ones:

DATASET DECOMPRESSION
```
export clumpsvela_pdir="<repository_path_here>" 
cd $clumpsvela_pdir 
tar -xvzf <dataset> 
```
PYTHON PATH INCLUSSION: (set in .bashrc in analogous way)
```
export PYTHONPATH=$PYTHONPATH:$clumpsvela_pdir ;
```

    #-<GENERAL USAGE>-#
To use the scripts:

1. python_work/ folder
  Use the "make_all.py" script with the required options given. This will make DATA/ and DATA_outputs/ in this folder.
2. sextractor_work/ folder
  Use "sex_catalog_create.py". This will link original maps from DATA/ and predicted maps from DATA_outputs/ in this folder. Then, it
  will run SExtractor and create a pandas.Dataframe to store catalogs in. Finally, catalogues are exported to .csv format to catalogues/
  
