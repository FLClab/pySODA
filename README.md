# PySODA

Adapts the SODA colocalisation method from https://www.nature.com/articles/s41467-018-03053-x#Sec26 from Icy to Python.


### How to run:

  1 - Set parameters in run_soda.py:
  
    DIRECTORY : String containing the path to a folder containing 2 colors TIF files

    For spot detection:
    SCALE_LIST : (list) Scales to be used for wavelet transform for spot detection. [2] usually gives decent results, but [1, 2] sometimes     works better.
    SCALE_THRESHOLD : Percent modifier of wavelet transform threshold

    For SODA analysis:
    N_RINGS : (int) Number of rings around spots in which to detect colocalisations
    STEP : Width of rings in pixels
    MIN_SIZE: Minimum area of spots to analyze
    SELF_SODA : Set to True to compute SODA for couples in the same channel as well
    
    Other
    SHOW_ROI : Set to True to display the ROI mask contour and detected spots
    WRITE_HIST : Set to True to create a .png of the coupling probabilities by distance histogram
   

2 - Execute run_soda.py

### Output:

For each TIF file: An excel file containing information on each individual spot and each couple
For the entire dataset: An excel file containing global information about the analysis: coupling indices for each image, mean coupling distances, etc.
