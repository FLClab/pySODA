
import matplotlib.pyplot as plt
from utils import *
from skimage import io, filters, morphology
from skimage.measure import label
from skimage.segmentation import clear_border
from skimage.measure import regionprops
import ot
import tifffile
import os
import numpy as np
import time
import sys
from split_crops import img_to_crops
sys.path.insert(1, '../pySODA')
from wavelet_SODA import DetectionWavelets
from steps_SODA import SodaImageAnalysis
plt.style.use('dark_background')

# For ROI mask generation
# Multiplier of ROI threshold. Higher value = more pixels takenP
ROI_THRESHOLD = 2.0
# Channel to use as mask. This channel won't be used for SODA analysis. Set to None to generate mask from all channels.
CHANNEL_MASK = 1
# Channel to remove from SODA analysis (for example, channel used for mask generation). Set to None to remove no channel.
REMOVE_CHANNEL = 1

# For spot detection
# Channel 2 is not used for a 2 color image
SCALE_LIST = [[3, 4],  # Channel 0  # Scales to be used for wavelet transform for spot detection
              [3, 4],  # Channel 1  # Higher values mean less details.
              [3, 4]]  # Channel 2  # Multiple scales can be used (e.g. [1,2]). Scales must be integers.
SCALE_THRESHOLD = [2.0,  # Channel 0  # Multiplier of wavelet transform threshold.
                   2.0,  # Channel 1  # Higher value = more pixels detected.
                   2.0]  # Channel 2

# For SODA analysis
MIN_SIZE = [30,  # Channel 0 # Minimum area (pixels) of spots to analyse
            30,  # Channel 1
            5]  # Channel 2
MIN_AXIS_LENGTH = [3,  # Channel 0  # Minimum length of both ellipse axes of spots to analyse
                   3,  # Channel 1
                   3]  # Channel 2
N_RINGS = 16  # Number of rings around spots (int)
RING_WIDTH = 1  # Width of rings in pixels
# Set to True to compute SODA for couples of spots in the same channel as well
SELF_SODA = False
CONTOUR = False  # Set to False to avoid considering the cell contour

# Display and graphs
# Set to True to save TIF images of spots detection and masks in OUTPUT_DIRECTORY
SAVE_ROI = True
# Set to True to create a .pdf of the coupling probabilities by distance histogram
WRITE_HIST = True


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


def filter_spots(mask):
    """
    Removes the spots that are too small or too linear (using parameters min_size and min_axis)
    :param mask: 2D binary mask of spots to filter
    :return: Filtered 2D binary mask
    """
    out_mask = np.copy(mask).astype(bool)
    morphology.remove_small_objects(
        out_mask, min_size=params['min_size'][0], in_place=True)
    mask_lab, num = label(out_mask, connectivity=1, return_num=True)
    mask_props = regionprops(mask_lab)
    new_props = []
    for p in mask_props:
        if p.minor_axis_length < params['min_axis'][0]:
            mask_lab[mask_lab == p.label] = 0
    out_mask = mask_lab > 0
    return out_mask


def find_ROI(image, sigma=10, threshold=1.0, channel_mask=False):
    """
    Find the region of interest for the analysis using a gaussian blur and thresholding.
    :param sigma: Sigma of the gaussian blur
    :param threshold: Threshold multiplier
    :return roi_mask: mask of the ROI as 2D numpy array.
    """
    # if channel_mask is not None:
    #     stack = image[channel_mask]
    # else:
    #     stack = np.sum(image, axis=0)

    filt = filters.gaussian(image, sigma=sigma)
    threshold = np.mean(filt) * (1.0 / threshold)
    filt[filt < threshold] = 0
    filt[filt >= threshold] = 1
    filt = morphology.binary_closing(filt)

    # Keep only areas with a significant area (> mean)
    labels = label(filt, connectivity=2)
    label_props = regionprops(labels)
    roi_mask = np.copy(labels)
    arealist = []
    for i in range(len(label_props)):
        arealist.append(label_props[i].area)
    roi_mask = morphology.remove_small_objects(
        roi_mask.astype(bool), min_size=np.mean(arealist))
    roi_mask[roi_mask > 0] = 1

    return roi_mask


def get_soda_mask(img):
    """
    Use pySODA to get the regions of interest as a binary mask
    """
    spots_img = DetectionWavelets(
        img, params['scale_list'][0], params['scale_threshold'][0]).computeDetection()
    filtered_spots = filter_spots(spots_img)
    roi_mask = find_ROI(
        img, threshold=params['roi_thresh'], channel_mask=params['channel_mask'])
    out_image = filtered_spots * roi_mask
    out_lab, num = label(out_image, connectivity=1, return_num=True)
    out_props = regionprops(out_lab)
    return out_image, out_lab, out_props, num


def tplans_pixel_by_pixel(imgs):
    # img_a = tifffile.imread(imgs)[0] # Bassoon
    # img_b = tifffile.imread(imgs)[1] # PSD-95 or FUS
    img_a, img_b = imgs[0], imgs[1]

    assert img_a.shape == img_b.shape
    img_a_pos = np.zeros((img_a.shape[0] * img_a.shape[1], 2))
    img_a_prod = np.zeros((img_a.shape[0] * img_a.shape[1], ))
    img_b_prod = np.zeros((img_b.shape[0] * img_b.shape[1], ))

    # Initializing positions and masses
    index = 0
    for i in range(img_a.shape[0]):
        for j in range(img_a.shape[1]):
            img_a_pos[index][0] = i
            img_a_pos[index][1] = j
            img_a_prod[index] = img_a[i, j]
            img_b_prod[index] = img_b[i, j]
            index += 1
    img_b_pos = img_a_pos

    # Normalize mass to be transported so that it sums to 1 in both images
    img_a_prod = img_a_prod / img_a_prod.sum()
    img_b_prod = img_b_prod / img_b_prod.sum()

    # Cost matrix
    cost_matrix = ot.dist(img_a_pos, img_b_pos, metric='euclidean')
    # Transport plan
    transport_plan = ot.emd(img_a_prod, img_b_prod, cost_matrix, 500000)
    return transport_plan, cost_matrix


def soda_tplans(imgs, use_crops=False):
    if use_crops:
        img_a, img_b = imgs[0], imgs[1]
        # print(img_a.shape, img_b.shape)
    else:
        img_a = tifffile.imread(imgs)[0]  # Bassoon
        img_b = tifffile.imread(imgs)[1]  # PSD95 or FUS

    # Binary masks, clusterized into ROIs
    soda_a, labels_a, props_a, components_a = get_soda_mask(img_a)
    soda_b, labels_b, props_b, components_b = get_soda_mask(img_b)

    img_a_pos = np.zeros((components_a, 2))
    img_a_prod = np.zeros((components_a,))
    img_b_pos = np.zeros((components_b, 2))
    img_b_prod = np.zeros((components_b,))
    # Compute mass to be transported in each image, looping through its region props
    for i, r in enumerate(props_a):
        img_a_pos[i][0] = r.centroid[0]
        img_a_pos[i][1] = r.centroid[1]
        min_row, min_col, max_row, max_col = r.bbox
        roi = img_a[min_row:max_row, min_col:max_col]
        mass = np.mean(roi)
        img_a_prod[i] = mass

    for i, r in enumerate(props_b):
        img_b_pos[i][0] = r.centroid[0]
        img_b_pos[i][1] = r.centroid[1]
        min_row, min_col, max_row, max_col = r.bbox
        roi = img_b[min_row:max_row, min_col:max_col]
        mass = np.mean(roi)
        img_b_prod[i] = mass

    # Normalize the mass to be transported so that it sums up to 1 in each image
    img_a_prod = img_a_prod / img_a_prod.sum()
    img_b_prod = img_b_prod / img_b_prod.sum()

    # Computing the cost matrix
    cost_matrix = ot.dist(img_a_pos, img_b_pos, metric='euclidean')
    fig, axs = plt.subplots(1, 2, figsize=(10, 10))
    axs[0].imshow(labels_a > 0, cmap='gray')
    axs[1].imshow(labels_b > 0, cmap="gray")
    plt.tight_layout()
    # fig.savefig('./figures/CaMKII_Actin_FLAVIE/synprot_channels_camkii_actin.png',
    #             bbox_inches='tight')
    # fig.savefig('./figures/Baselines_test/synprot_channels_simulated_data.png',
    #             bbox_inches='tight')
    plt.close()

    # Compute transport plan
    start = time.time()
    transport_plan = ot.emd(img_a_prod, img_b_prod, cost_matrix)
    transport_plan = normalize_mass(transport_plan)
    time_emd = time.time() - start
    print("Compute time for transport plan: {}".format(time_emd))

    fig, axs = plt.subplots(1, 2, figsize=(10, 10))
    im1 = axs[0].imshow(cost_matrix, cmap='coolwarm')
    cbar = plt.colorbar(im1, ax=axs[0], shrink=0.3)
    cbar.ax.set_xlabel('cost')
    axs[0].set_xlabel('PSD', fontsize=14)
    axs[0].set_ylabel('Bassoon', fontsize=14)
    # axs[0].set_yticklabels(axs[0].get_yticklabels(), fontsize=12)
    # axs[0].set_xticklabels(axs[0].get_xticklabels(), fontsize=12)
    im2 = axs[1].imshow(transport_plan, cmap='coolwarm')
    cbar = plt.colorbar(im2, ax=axs[1], shrink=0.3)
    cbar.ax.set_xlabel('transport plan')
    axs[1].set_xlabel('PSD', fontsize=14)
    axs[1].set_ylabel('Bassoon', fontsize=14)
    # axs[1].set_yticklabels(axs[1].get_yticklabels(), fontsize=12)
    # axs[1].set_xticklabels(axs[1].get_xticklabels(), fontsize=12)
    plt.tight_layout()
    # fig.savefig('./figures/Baselines_test/synprot_transport_plan_simulated_data.png',
    #             bbox_inches='tight')
    plt.close()

    return transport_plan, cost_matrix
