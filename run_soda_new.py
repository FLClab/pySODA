import wavelet_SODA_change as wv
from skimage import io, filters
from scipy.ndimage import morphology
from matplotlib import pyplot
from skimage.measure import find_contours, regionprops, label, subdivide_polygon
from itertools import combinations, combinations_with_replacement
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import xlsxwriter

"""
This file contains all the code related to running the SODA method using the classes from wavelet_SODA.py
"""

""" Change parameters here! """
DIRECTORY = r"figure_image\cropped"  # Path containing TIF files

# For spot detection
SCALE_LIST = [[3,4],  # Channel 0  # Scales to be used for wavelet transform for spot detection
              [3,4],  # Channel 1  # Higher values mean less details.
              [3,4]]  # Channel 2  # Multiple scales can be used (e.g. [1,2]). Scales must be integers.
SCALE_THRESHOLD = [200,  # Channel 0  # Percent modifier of wavelet transform threshold.
                   200,  # Channel 1  # Higher value = more pixels detected.
                   200]  # Channel 2
MIN_SIZE = [5,  # Channel 0 # Minimum area (pixels) of spots to analyse
            5,  # Channel 1
            5]  # Channel 2
MIN_AXIS_LENGTH = [3,  # Channel 0  # Minimum length of both ellipse axes of spots to analyse
                   3,  # Channel 1
                   1]  # Channel 2

# For SODA analysis
ROI_THRESHOLD = 200  # Percent modifier of ROI threshold. Higher value = more pixels taken.
N_RINGS = 16  # Number of rings around spots (int)
RING_WIDTH = 1  # Width of rings in pixels
SELF_SODA = False  # Set to True to compute SODA for couples of spots in the same channel as well

# Display and graphs
SHOW_ROI = False  # Set to True to display the ROI mask contour and detected spots for every image
SAVE_ROI = False  # Set to True to save TIF images of all spots and filtered spots for each channel
WRITE_HIST = False  # Set to True to create a .png of the coupling probabilities by distance histogram


class ImageError(ValueError):
    """
    Error to raise when an image has invalid dimensions or can't be read
    """


class SpotsError(ValueError):
    """
    Error to raise when an image doesn't have enough spots to be viable
    """


class SodaImageAnalysis:
    """
    This class contains the steps to run the SODA analysis on a multichannel image.
    """
    def __init__(self, file, image, directory, params):
        self.file = file
        self.image = image
        self.directory = directory
        self.params = params

        print("Detecting spots using wavelet transform...")
        self.roi_mask = self.find_ROI(threshold=self.params['roi_thresh'])
        t0 = time.time()
        self.spots_mask = self.detect_spots()
        print(time.time()-t0)
        io.imsave(os.path.join(directory, "y.tif"), self.spots_mask.astype('uint8'))
        #self.roi_mask = io.imread(r"figure_image/mask.tif")
        #self.spots_mask = io.imread(r"figure_image/all_spots_composite.tif")*self.roi_mask
        self.marked_point_process = self.spatial_distribution(self.spots_mask)

    def find_ROI(self, sigma=10, threshold=100):
        """
        Find the region of interest for the analysis using a gaussian blur and thresholding.
        :param sigma: Sigma of the gaussian blur
        :param threshold: Threshold multiplier
        :return roi_mask: mask of the ROI
        """
        stack = np.sum(self.image, axis=0)
        filt = filters.gaussian(stack, sigma=sigma)
        threshold = np.mean(filt) * (100 / threshold)
        filt[filt < threshold] = 0
        filt[filt >= threshold] = 1
        filt = morphology.binary_closing(filt)

        # Keep only areas with a significant volume
        labels = label(filt, connectivity=2)
        label_props = regionprops(labels)
        roi_mask = np.copy(labels)
        arealist = []
        for i in range(len(label_props)):
            arealist.append(label_props[i].area)
        for i in range(len(label_props)):
            if label_props[i].area < np.mean(arealist):
                roi_mask[roi_mask == i + 1] = 0
        roi_mask[roi_mask > 0] = 1

        return roi_mask

    def find_roi_contour(self):
        """
        Find the points creating the outline of the mask to measure distance to boundary
        :return: Numpy array of points making the contour
        """
        # Find contours around the mask. Padding is used to properly find contours near maximum edges
        pad_mask = np.pad(self.roi_mask, ((0, 2), (0, 2)), mode='constant', constant_values=0)
        roi_contour = find_contours(pad_mask, level=0, fully_connected='high', positive_orientation='high')

        # Subdivide polygon for smoother contours and more precise points for distance to boundary
        for c in range(len(roi_contour)):
            roi_contour[c] = subdivide_polygon(roi_contour[c], 7)

        return roi_contour

    def detect_spots(self):
        """
        Detects spots inside the ROI from multi-channel image using the wavelet transform
        :return: 3D numpy array of spots masks. Z dimension is for different channels.
        """
        z, y, x = self.image.shape
        spots_image = np.ndarray((z, y, x))

        for ch in range(z):
            spots_image[ch] = wv.DetectionWavelets(self.image[ch],
                                                   self.params['scale_list'][ch],
                                                   self.params['scale_threshold'][ch]).computeDetection()

        return spots_image*self.roi_mask

    def spatial_distribution(self, mask):
        """
        Gets the spatial distribution (marked point process) of spots in all channels
        :param mask: Image of spots mask
        :return: List of Marked point process of spots for each channel
        """
        marks_list = []

        for i, m in enumerate(mask):
            marks = wv.SpatialDistribution(m, self.image[i], self.params['min_size'][i], self.params['min_axis'][i]).mark()
            print('Spots in channel {}: {}'.format(i, len(marks)))
            marks_list.append(marks)
        return marks_list

    def spatial_relations(self, ch0, ch1, index):
        """
        Measure the Spatial Relations of spots. This is the actual SODA analysis.
        :param ch0: first channel index to use in analysis
        :param ch1: second channel index to use in analysis
        :param index: something?
        :return SR: Spatial relations object
        :return prob_write: Information on all couples and spots
        :return results_dict: Global results
        """
        roi_contour = self.find_roi_contour()
        roi_volume = np.sum(self.roi_mask)
        window_size = self.image[0].shape

        SR = wv.SpatialRelations(self.marked_point_process[ch0],
                                 self.marked_point_process[ch1],
                                 window_size,
                                 roi_volume,
                                 roi_contour,
                                 self.image,
                                 self.params['n_rings'],
                                 self.params['ring_width'],
                                 self.file,
                                 index)

        print("Computing G...")
        G = SR.correlation_new()

        print("Computing variance...")
        var = SR.variance_theo_delta_new()

        print("Computing A...")
        A = SR.intersection2D_new()

        print("Computing G0...")
        arg_dict = {'G': G, 'var': var, 'A': A}
        G0 = SR.reduced_Ripley_vector(**arg_dict)

        print("Computing statistics...")
        prob_write, results_dict = SR.coupling_prob(**arg_dict, G0=G0)
        print("================================"
              "\nCoupling Index 1:", results_dict['coupling_index'][0],
              "\nCoupling Index 2:", results_dict['coupling_index'][1],
              "\nMean Coupling Distance:", results_dict['mean_coupling_distance'], "pixels")

        probs = np.ndarray.tolist(results_dict['coupling_probabilities'])
        print('Probabilities', probs, '\n\n')

        # Coupling prob. by distance histogram
        if WRITE_HIST:
            dists = [i * RING_WIDTH for i in range(N_RINGS)]
            plt.ylim(0, 1.0)
            plt.bar(dists, probs, align='edge', width=RING_WIDTH, edgecolor='black', linewidth=0.75)
            plt.locator_params(axis='x', nbins=N_RINGS)
            plt.xlabel('Distance (pixels)')
            plt.ylabel('Coupling probability')
            plt.savefig(os.path.join(DIRECTORY, 'hist_{}_ch{}{}.png'.format(os.path.basename(self.file), ch0, ch1)))
            plt.close()

        return SR, prob_write, results_dict

    def soda_analysis(self, index):
        """
        The main analysis function; runs SODA on all chosen combinations of channels on the current image
        """
        channel_list = [n for n in range(self.image.shape[0])]
        if SELF_SODA:
            channel_pairs = combinations_with_replacement(channel_list, 2)
        else:
            channel_pairs = combinations(channel_list, 2)

        out_results = {}
        for ch0, ch1 in channel_pairs:
            print('- Channels {} and {} -'.format(ch0, ch1))
            SR, prob_write, results_dict = self.spatial_relations(ch0, ch1, index)
            #spots0, spots1, couples_data = SR.get_spots_data(prob_write)
            SR.write_spots_and_probs(prob_write, self.directory, self.file + '_{}_{}.xlsx'.format(ch0, ch1))
            out_results['ch{}-ch{}'.format(ch0, ch1)] = results_dict

        return out_results


def main(directory, params):
    """
    Loop through all files in directory to run SODA analysis using chosen parameters
    :param directory: Directory
    :param params: Parameters dictionary
    """

    # Writing Excel file
    results_workbook = xlsxwriter.Workbook(os.path.join(directory, "PySODA_results_{}.xlsx".format(os.path.basename(directory))),
                                           {'nan_inf_to_errors': True})
    worksheets = []
    worksheet_names = []
    titles = ['File', 'Spots in channel 0', 'Spots in channel 1', 'Number of couples',
              'Coupling index 0', 'Coupling index 1', 'Weighted mean coupling distance']
    row = 1
    for sheet in worksheets:
        for t in range(len(titles)):
            sheet.write(0, t, titles[t])

    for file in os.listdir(directory):
        if file.lower().endswith('.tif') or file.lower().endswith('.tiff'):
            print('--- Running SODA on image {} ---'.format(file))

            # Open image and make sure it has valid dimensions
            try:
                img = io.imread(os.path.join(directory, file))
                if len(img.shape) != 3:
                    raise ImageError
                # Turns y,x,z images into z,y,x images. Assumes the z axis has the smallest value; works fine
                # since in this case the z axis is for channels, and SODA won't be used on more than 3 channels.
                if img.shape.index(min(img.shape)) == 2:
                    img = np.moveaxis(img, 2, 0)
            except ImageError:
                continue

            ### Analysis happens here! ###
            out_dict = SodaImageAnalysis(file, img, directory, params).soda_analysis(0)

            # Write results in excel file for each image and channels combination
            for sheet_name, results in out_dict.items():
                elem_list = [file,
                             results['n_spots_0'],
                             results['n_spots_1'],
                             results['n_couples'],
                             results['coupling_index'][0],
                             results['coupling_index'][1],
                             results['mean_coupling_distance']]
                if sheet_name not in worksheet_names:
                    worksheet_names.append(sheet_name)
                    worksheets.append(results_workbook.add_worksheet(name=sheet_name))
                    sheet_index = worksheet_names.index(sheet_name)
                    for t in range(len(titles)):
                        worksheets[sheet_index].write(0, t, titles[t])
                else:
                    sheet_index = worksheet_names.index(sheet_name)
                for index in range(len(elem_list)):
                    worksheets[sheet_index].write(row, index, elem_list[index])
            row += 1
    results_workbook.close()


if __name__ == '__main__':
    start_time = time.time()

    directory = DIRECTORY
    params = {'scale_list': SCALE_LIST,
              'scale_threshold': SCALE_THRESHOLD,
              'min_size': MIN_SIZE,
              'min_axis': MIN_AXIS_LENGTH,
              'roi_thresh': ROI_THRESHOLD,
              'n_rings': N_RINGS,
              'ring_width': RING_WIDTH,
              'self_soda': SELF_SODA}

    main(directory, params)
    print("--- Running time: %s seconds ---" % (time.time() - start_time))
