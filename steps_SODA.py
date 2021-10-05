import wavelet_SODA as wv
from skimage import io, filters, morphology
from skimage.measure import find_contours, regionprops, label, subdivide_polygon
from itertools import combinations, combinations_with_replacement
import os
import matplotlib.pyplot as plt
import numpy as np
import xlsxwriter

"""
This file contains all the code related to running the SODA method using the classes from wavelet_SODA.py
"""

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
    def __init__(self, file, image, directory, output_dir, params):
        self.file = file
        self.image = image
        self.directory = directory
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.params = params

        print("Detecting spots using multiscale product...")
        self.roi_mask = self.find_ROI(threshold=self.params['roi_thresh'], channel_mask=self.params['channel_mask'])

        if self.params['remove_channel'] is not None:
            self.image = np.delete(self.image, self.params['remove_channel'], 0)

        self.spots_mask = self.detect_spots(self.params['save_roi'])
        self.marked_point_process = self.spatial_distribution(self.spots_mask)

    def find_ROI(self, sigma=10, threshold=1.0, channel_mask=False):
        """
        Find the region of interest for the analysis using a gaussian blur and thresholding.
        :param sigma: Sigma of the gaussian blur
        :param threshold: Threshold multiplier
        :return roi_mask: mask of the ROI as 2D numpy array.
        """
        if channel_mask is not None:
            stack = self.image[channel_mask]
        else:
            stack = np.sum(self.image, axis=0)

        filt = filters.gaussian(stack, sigma=sigma)
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
        roi_mask = morphology.remove_small_objects(roi_mask.astype(bool), min_size=np.mean(arealist))
        roi_mask[roi_mask > 0] = 1

        return roi_mask

    def find_roi_contour(self):
        """
        Find the points creating the outline of the mask to measure distance to boundary
        :return: Numpy array of points defining the contour
        """
        # Find contours around the mask. Padding is used to properly find contours near maximum edges
        pad_mask = np.pad(self.roi_mask, ((0, 2), (0, 2)), mode='constant', constant_values=0)
        roi_contour = find_contours(pad_mask, level=0, fully_connected='high', positive_orientation='high')

        # Subdivide polygon for smoother contours and more precise points for distance to boundary
        for c in range(len(roi_contour)):
            roi_contour[c] = subdivide_polygon(roi_contour[c], 7)

        return roi_contour

    def detect_spots(self, save=False):
        """
        Detects spots inside the ROI from multi-channel image using wavelet transform multiscale product as described
        in Olivo-Marin, 2002.
        :param save: If True, saves .tif images of spots and mask in the OUTPUT_DIRECTORY.
        :return: 3D numpy array of spots masks. Z dimension is for different channels.
        """
        spots_image = np.ndarray(self.image.shape)

        for ch in range(self.image.shape[0]):
            spots_image[ch] = wv.DetectionWavelets(self.image[ch],
                                                   self.params['scale_list'][ch],
                                                   self.params['scale_threshold'][ch]).computeDetection()

        spots_image_filtered = self.filter_spots(spots_image)
        out_image = spots_image_filtered*self.roi_mask
        if save:
            io.imsave(os.path.join(self.output_dir, "{}_all_spots.tif".format(os.path.basename(self.file))), spots_image.astype('uint8'))
            io.imsave(os.path.join(self.output_dir, "{}_filtered_spots.tif".format(os.path.basename(self.file))), spots_image_filtered.astype('uint8')*255)
            io.imsave(os.path.join(self.output_dir, "{}_mask.tif".format(os.path.basename(self.file))), self.roi_mask.astype('uint8')*255)
            io.imsave(os.path.join(self.output_dir, "{}_spots_in_mask.tif".format(os.path.basename(self.file))), out_image.astype('uint8')*255)

        return out_image

    def filter_spots(self, mask):
        """
        Removes the spots that are too small or too linear (using parameters min_size and min_axis)
        :param mask: 3D binary mask of spots to filter
        :return: Filtered 3D binary mask
        """
        out_mask = np.copy(mask).astype(bool)
        for i, img in enumerate(out_mask):
            morphology.remove_small_objects(img, min_size=self.params['min_size'][i], in_place=True)
            mask_lab, num = label(img, connectivity=1, return_num=True)
            mask_props = regionprops(mask_lab)
            for p in mask_props:
                if p.minor_axis_length < self.params['min_axis'][i]:
                    mask_lab[mask_lab == p.label] = 0
            out_mask[i] = mask_lab > 0
        return out_mask

    def spatial_distribution(self, mask):
        """
        Gets the spatial distribution (marked point process) of spots in all channels.
        :param mask: Image of spots mask
        :return: List of Marked point process of spots for each channel
        """
        marks_list = []

        for i, m in enumerate(mask):
            marks = wv.SpatialDistribution(m, self.image[i], self.params['min_size'][i], self.params['min_axis'][i]).mark()
            print('Spots in channel {}: {}'.format(i, len(marks)))
            marks_list.append(marks)
            if len(marks) < 50:
                raise SpotsError
        print('\n')
        return marks_list

    def spatial_relations(self, ch0, ch1):
        """
        Measure the Spatial Relations of spots. This is the actual SODA analysis.
        :param ch0: index of first channel to use in analysis
        :param ch1: index of second channel to use in analysis
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
                                 self.file)

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
              "\nCoupling Index 0:", results_dict['coupling_index'][0],
              "\nCoupling Index 1:", results_dict['coupling_index'][1],
              "\nMean Coupling Distance:", results_dict['mean_coupling_distance'], "pixels")

        probs = np.ndarray.tolist(results_dict['coupling_probabilities'])
        print('Coupling probabilities: ', probs, '\n\n')

        # Coupling prob. by distance histogram
        if self.params['write_hist']:
            self.write_prob_histogram(probs, ch0, ch1)

        return SR, prob_write, results_dict

    def write_prob_histogram(self, probs, ch0, ch1):
        dists = [i * self.params['ring_width'] for i in range(self.params['n_rings'])]
        plt.ylim(0, 1.0)
        plt.bar(dists, probs, align='edge', width=self.params['ring_width'], edgecolor='black', linewidth=0.75)
        plt.locator_params(axis='x', nbins=self.params['n_rings'])
        plt.xlabel('Distance (pixels)')
        plt.ylabel('Coupling probability')
        plt.savefig(os.path.join(self.output_dir, 'hist_{}_ch{}{}.pdf'.format(os.path.basename(self.file), ch0, ch1)),
                    bbox_inches='tight', transparent=True, dpi=600)
        plt.close()

    def soda_analysis(self):
        """
        The main analysis function; runs SODA on all chosen combinations of channels on the current image
        :return out_results: Dictionary with input channels as keys and results dictionaries as values
        """
        channel_list = [n for n in range(self.image.shape[0])]
        channel_list = [0,1]
        if self.params['self_soda']:
            channel_pairs = combinations_with_replacement(channel_list, 2)
        else:
            channel_pairs = combinations(channel_list, 2)

        out_results = {}
        for ch0, ch1 in channel_pairs:
            print('- Channels {} and {} -'.format(ch0, ch1))
            SR, prob_write, results_dict = self.spatial_relations(ch0, ch1)
            SR.write_spots_and_probs(prob_write, self.output_dir, 'pySODA_' + self.file + '_ch{}{}.xlsx'.format(ch0, ch1))
            out_results['ch{}-ch{}'.format(ch0, ch1)] = results_dict

        return out_results


def main(directory, output_dir, params):
    """
    Loop through all files in directory to run SODA analysis using chosen parameters
    :param directory: Directory
    :param params: Parameters dictionary
    """

    # Writing Excel file
    results_workbook = xlsxwriter.Workbook(os.path.join(output_dir, "pySODA_results_{}.xlsx".format(os.path.basename(directory))),
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
                print('Image has invalid dimensions. Skipping...\n')
                continue

            ### Analysis happens here! ###
            try:
                out_dict = SodaImageAnalysis(file, img, directory, output_dir, params).soda_analysis()
            except SpotsError:
                print("Not enough spots in image! Skipping...\n")
                continue

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