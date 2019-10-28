# pySODA

Adapts the SODA colocalisation method from https://www.nature.com/articles/s41467-018-03053-x#Sec26 from Icy (Java) to Python.

The SODA algorithm calculates the probability that particles in super-resolution images interact for discrete intervals
of distances separating them. 

**Related paper**: *Activity-dependent changes of synaptic protein clusters revealed by multidimensional analysis with multicolor STED nanoscopy*

### Requirements:

  - Python 3.X with the following libraries:
    - numpy
    - scikit-image
    - matplotlib
    - scipy
    - xlsxwriter

The Anaconda distribution of Python covers all of these.

  - Two- or three-channel super-resolution microscopy images. The analysis will run on all combinations of two channels.
    

### How to run:

  1 - Set parameters in run_soda.py:
  
    DIRECTORY : Path containing TIF files
    OUTPUT_DIRECTORY : Path in which to save output excel files and images

    SCALE_LIST : Scales to be used for the multiscale product segmentation of spots.
                 Takes the for of a list containing a list of integers for each channel.
                 ex: [[1,2], [2,3]]
    SCALE_THRESHOLD : Percent modifier of wavelet transform threshold.
                      Higher value means more pixels detected. List of one value per channel.
                      ex: [100, 200]
                      
    MIN_SIZE : Minimum area of spots to be considered in the analysis. List of one value per channel.
    MIN_AXIS_LENGTH : Minimum length of both ellipse axes of spots to analyse. List of one value per channel.

    ROI_THRESHOLD : Percent modifier of ROI threshold. Higher value means more pixels taken.
    N_RINGS : Number of rings around spots (int)
    RING_WIDTH : Width of rings in pixels (int)
    SELF_SODA : Whether to compute SODA for couples of spots in the same channel as well (bool)
    
    SAVE_ROI : Whether to save TIF images of spots detection and masks in OUTPUT_DIRECTORY (bool)
    WRITE_HIST : Whether to create a .png of the coupling probabilities by distance histogram (bool)

2 - Execute run_soda.py

### Output:

**For each TIF file**: An excel file containing information on each individual spot and each couple.

**For the entire dataset**: An excel file containing global information about the analysis: coupling indices for each image, mean coupling distances, etc.

**If SAVE_ROI is True**: For each image, four .tif files are created.
 - *_all_spots.tif: The wavelet transform multiscale product segmentation of the original image.
 - *_filtered_spots.tif: Spots segmentation image with spots that don't correspond to the MIN_SIZE and MIN_AXIS_LENGTH
 parameters filtered out.
 - *_spots_in_mask.tif: The filtered spots image with only the spots within the dendritic mask.
 - *_mask.tif: Image of the dendritic mask.
 
 All of these outputs ares saved in the specified OUTPUT_DIRECTORY.
