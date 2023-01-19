import steps_SODA
import time

""" Change parameters here """
DIRECTORY = r"./example_image"  # Path containing TIF files
OUTPUT_DIRECTORY = r"./example_output"  # Path in which to save outputs

# For ROI mask generation
ROI_THRESHOLD = 2.0  # Multiplier of ROI threshold. Higher value = more pixels taken.
CHANNEL_MASK = None # Channel to use as mask. This channel won't be used for SODA analysis. Set to None to generate mask from all channels.
REMOVE_CHANNEL = None # Channel to remove from SODA analysis (for example, channel used for mask generation). Set to None to remove no channel.

# For spot detection
# Channel 2 is not used for a 2 color image
SCALE_LIST = [[4],  # Channel 0  # Scales to be used for wavelet transform for spot detection
              [4],  # Channel 1  # Higher values mean less details.
              [4]]  # Channel 2  # Multiple scales can be used (e.g. [1,2]). Scales must be integers.
SCALE_THRESHOLD = [4.0,  # Channel 0  # Multiplier of wavelet transform threshold.
                   4.0,  # Channel 1  # Higher value = more pixels detected.
                   4.0]  # Channel 2

# For SODA analysis
MIN_SIZE = [30,  # Channel 0 # Minimum area (pixels) of spots to analyse
            30,  # Channel 1
            5]  # Channel 2
MIN_AXIS_LENGTH = [3,  # Channel 0  # Minimum length of both ellipse axes of spots to analyse
                   3,  # Channel 1
                   3]  # Channel 2
N_RINGS = 16  # Number of rings around spots (int)
RING_WIDTH = 1  # Width of rings in pixels
SELF_SODA = False  # Set to True to compute SODA for couples of spots in the same channel as well
CONTOUR = False  # Set to False to avoid considering the cell contour

# Display and graphs
SAVE_ROI = True  # Set to True to save TIF images of spots detection and masks in OUTPUT_DIRECTORY
WRITE_HIST = True  # Set to True to create a .pdf of the coupling probabilities by distance histogram

if __name__ == '__main__':
    start_time = time.time()

    directory = DIRECTORY
    output_dir = OUTPUT_DIRECTORY
    params = {'scale_list': SCALE_LIST,
              'scale_threshold': SCALE_THRESHOLD,
              'min_size': MIN_SIZE,
              'min_axis': MIN_AXIS_LENGTH,
              'roi_thresh': ROI_THRESHOLD,
              'channel_mask': CHANNEL_MASK,
              'remove_channel': REMOVE_CHANNEL,
              'n_rings': N_RINGS,
              'ring_width': RING_WIDTH,
              'self_soda': SELF_SODA,
              'save_roi': SAVE_ROI,
              'write_hist': WRITE_HIST}

    steps_SODA.main(directory, output_dir, params)
    print("--- Running time: %s seconds ---" % (time.time() - start_time))