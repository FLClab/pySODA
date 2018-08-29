# PySODA

Adapts the SODA colocalisation method from https://www.nature.com/articles/s41467-018-03053-x#Sec26 from Icy to Python.


### How to run:

  1 - Set parameters in run_soda.py:
  
    DIRECTORY : String containing the path to a folder containing 2 colors TIF files

    For spot detection:
    SCALE_LIST : Scales to be used for wavelet transform for spot detection. [2] usually gives decent results, but [1, 2] sometimes
    works better. Takes the form of a list containing a list of scales for each channel.
    SCALE_THRESHOLD : List containing the percent modifier of wavelet transform threshold for each channel.
    MIN_SIZE: Minimum area of spots to analyze. Takes the form of a list containing the min size for each channel in order.
    MIN_AXIS_LENGTH: Minimum length of ellipse axes for all spots. Takes the form of a list containing the minimal axis lengthfor each 
    channel in order.

    For SODA analysis:
    ROI_THRESHOLD: # Percent modifier of ROI threshold. Higher value = more pixels taken.
    N_RINGS : (int) Number of rings around spots in which to detect colocalisations
    RING_WIDTH : Width of rings in pixels
    SELF_SODA : Set to True to compute SODA for couples in the same channel as well
    
    Other
    SHOW_ROI : Set to True to display the ROI mask contour and detected spots
    SAVE_HIST : Set to True to save the binary images of spots (all spots vs filtered spots)
    WRITE_HIST : Set to True to create a .png of the coupling probabilities by distance histogram
    
2 - Execute run_soda.py

### Output:

For each TIF file: An excel file containing information on each individual spot and each couple
For the entire dataset: An excel file containing global information about the analysis: coupling indices for each image, mean coupling distances, etc.
