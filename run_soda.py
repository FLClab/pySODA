import wavelet_SODA as wv
from skimage import io, filters
from scipy.ndimage import morphology
from matplotlib import patches
from skimage.measure import find_contours, regionprops, label, subdivide_polygon
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import xlsxwriter

"""
This file contains all the code related to running the SODA method using the classes from wavelet_SODA.py
"""

""" Change parameters here! """
DIRECTORY = r"C:\Users\Renaud\Pictures\WAVELETS_IMAGES\Stimulation\output"  # Path containing TIF files

# For spot detection
SCALE_LIST = [[2],  # Channel 0  # Scales to be used for wavelet transform for spot detection
              [2],  # Channel 1  # Higher values mean less details.
              [2]]  # Channel 2  # Multiple scales can be used (e.g. [1,2]). Scales must be integers.
SCALE_THRESHOLD = [100,  # Channel 0  # Percent modifier of wavelet transform threshold.
                   100,  # Channel 1  # Higher value = more pixels detected.
                   100]  # Channel 2
MIN_SIZE = [5,  # Channel 0 # Minimum area of spots to analyse
            5,  # Channel 1
            5]  # Channel 2

# For SODA analysis
ROI_THRESHOLD = 200  # Percent modifier of ROI threshold. Higher value = more pixels taken.
N_RINGS = 10  # Number of rings around spots (int)
RING_WIDTH = 1  # Width of rings in pixels
SELF_SODA = False  # Set to True to compute SODA for couples of spots in the same channel as well

# Display and graphs
SHOW_ROI = False  # Set to True to display the ROI mask contour and detected spots for every image
WRITE_HIST = True  # Set to True to create a .png of the coupling probabilities by distance histogram


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
    This class contains the steps to run the SODA analysis on an image.
    """
    def __init__(self, file, image, directory, min_size=(5, 5), scale_list=((2,), (2,)), scale_threshold=(100, 100)):
        """
        Constructor function.
        :param file: Path of the image file to analyse
        :param scale_list: array-like of ints; scales to use for wavelet transform
        :param scale_threshold: Percentage of the wavelet threshold to use.
        """
        self.file = file
        self.scale_list = scale_list
        self.scale_threshold = scale_threshold
        self.directory = directory
        self.min_size = min_size

        self.image = image

        z, y, x = self.image.shape
        print('X:', x, '\nY:', y)

        # Create the binary spots image using wavelet transform
        print("Computing wavelet spot detection...\nScales: {}".format(self.scale_list))

        self.detections = [(wv.DetectionWavelets((self.image[i]), J_list=self.scale_list[i],
                                                 scale_threshold=self.scale_threshold[i]).computeDetection(),
                            self.image[i]) for i in range(z)]

    def analysis(self, ch0, ch1, n_rings, step, img_index):
        """
        Runs the analysis for chosen channels.
        :param ch0: int; first channel to use in analysis
        :param ch1: int; second channel to use in analysis
        :param n_rings: int; number of rings around spots in which to detect coupling
        :param step: float; width of rings in pixels
        :param img_index: int; index of image to be stored in results
        """

        # Find ROI for volume and boundary correction
        contours, roivolume, masked_detections = self.find_roi()

        # Compute Marked Point Process for both channels
        marks = self.spatial_distribution(masked_detections, ch0, ch1)
        mean_ecc = []
        for m in marks:
            ecc_list = []
            for m_spots in m:
                if m_spots[0].minor_axis_length > 0:
                    ecc_list.append(m_spots[0].eccentricity)
            mean_ecc.append(np.mean(ecc_list))

        # Initialize spatial relations object for analysis
        SR = wv.SpatialRelations(*marks, self.image[0].shape,
                                  roivolume, contours, self.image,
                                  n_rings, step, os.path.basename(self.file), img_index)

        # Calls SODA spatial relations analysis
        prob_write, CIndex, mean_dist, raw_mean_dist, coupling, n_couples = self.spatial_relations(SR, ch0, ch1)
        spots0, spots1, couples_data = SR.get_spots_data(prob_write)

        # Display spot detection, ROI and couples
        if SHOW_ROI:
            self.show_roi(contours, masked_detections, couples_data, ch0, ch1)

        # Write Excel file for spots and couples
        SR.write_spots_and_probs(prob_write, self.directory,
                                 'PySODA_spots_{}_ch{}{}.xlsx'.format(os.path.basename(self.file), ch0, ch1))

        return prob_write, len(marks[0]), len(marks[1]), CIndex, mean_dist, raw_mean_dist, coupling, n_couples, mean_ecc, spots0, spots1, couples_data

    def spatial_distribution(self, detections, ch0, ch1):
        """
        Computes the spatial distribution for the chosen list of binary images
        :param detections: list of tuples containing (binary image, base image)
        :param ch0: First channel to use in analysis
        :param ch1: Second channel to use in analysis
        :return: Marked Point Process of each binary image in detections
        """
        print("Computing spatial distribution...")
        marks = [wv.SpatialDistribution(d, i, cs=self.min_size[ch]).mark() for d, i, ch in (detections[ch0], detections[ch1])]
        for m in range(len(marks)):
            print("\tSpots in channel {}:".format(m), len(marks[m]))
            if range(len(marks[m]) < 100):
                raise SpotsError
        return marks

    def spatial_relations(self, SR, ch0, ch1):
        """
        Computes every step of the spatial relations analysis for the chosen SpatialRelations object
        (G, variance, A, G0, coupling index and probabilities)
        :param SR: Spatial Relations object
        :param ch0: First channel to use in analysis
        :param ch1: Second channel to use in analysis
        :return prob write: List of lists containing information on each couple (spots coordinates, coupling prob.)
        :return CIndex: Coupling indices for both channels
        :return mean_dist: Mean coupling distance, weighted by coupling probability
        :return raw_mean_dist: Unweighted mean coupling distance, such as exported in Icy results excel files
        :return coupling: List of coupling probability for each ring
        :return n_couples: Total number of couples
        """
        print("Computing G...")
        G = SR.correlation_new()
        print("G ------------------------------\n", G[0, :], '\n')

        print("Computing variance...")
        var = SR.variance_theo_delta_new()
        print("VAR ----------------------------\n", var, '\n')

        print("Computing A...")
        A = SR.intersection2D_new()
        print("A ------------------------------\n", A, '\n')

        print("Computing G0...")
        arg_dict = {'G': G, 'var': var, 'A': A}
        G0 = SR.reduced_Ripley_vector(**arg_dict)
        print("G0 -----------------------------\n", G0, '\n')
        # SR.draw_G0(G0)

        print("Computing statistics...")
        prob_write, CIndex, mean_dist, raw_mean_dist, coupling, n_couples = SR.coupling_prob(**arg_dict, G0=G0)
        print("================================"
              "\nCoupling Index 1:", CIndex[0],
              "\nCoupling Index 2:", CIndex[1],
              "\nMean Coupling Distance:", mean_dist, "pixels",
              "\nUnweighted Mean Coupling Distance:", raw_mean_dist, "pixels")

        # SR.data_scatter(mean_dist, prob_write)
        # SR.data_boxplot(prob_write)
        # print('Probabilities (Javascript version):', SR.main2D_corr(G, var, A))
        probs = np.ndarray.tolist(coupling)
        print('Probabilities', probs, '\n\n')

        # Coupling prob. by distance histogram
        if WRITE_HIST:
            dists = [i for i in range(N_RINGS)]
            plt.ylim(0, 1.0)
            plt.bar(dists, probs, align='edge', width=1, edgecolor='black', linewidth=0.75)
            plt.locator_params(axis='x', nbins=N_RINGS)
            plt.xlabel('Distance (pixels)')
            plt.ylabel('Coupling probability')
            plt.savefig(os.path.join(DIRECTORY, 'hist_{}_ch{}{}.png'.format(os.path.basename(self.file), ch0, ch1)))
            plt.close()

        return prob_write, CIndex, mean_dist, raw_mean_dist, coupling, n_couples

    def find_roi(self):
        """
        Finds the ROI based on a gaussian filter and threshold.
        Using this mask gives similar results to the Icy mask
        :return: cont: Numpy array of all the points determining the shape of the ROI
        :return: roivolume: Area of the detected ROI
        :return: masked_detections: list of tuples (binary image, base image) with only the spots inside the ROI
        """
        # Filtered image for masking: gaussian filter + threshold + fill holes
        filt = np.copy(self.image[0])
        for ch in range(1, len(self.image)):
            print(ch)
            filt += self.image[ch]
        print('MEAN:', np.mean(filt))
        filt = (filters.gaussian(filt, 10))  # Higher sigma for smoother edges
        print('MEAN:', np.mean(filt))

        threshold = np.mean(filt) * (100/ROI_THRESHOLD)
        print(threshold)

        filt[filt < threshold] = 0
        filt[filt >= threshold] = 1
        filt = morphology.binary_closing(filt)
        # filt = morphology.binary_fill_holes(filt)


        # Keep only areas with a significant volume
        labels = label(filt, connectivity=2)
        label_props = regionprops(labels)
        mask = np.copy(labels)
        arealist = []
        for i in range(len(label_props)):
            arealist.append(label_props[i].area)
        for i in range(len(label_props)):
            if label_props[i].area < np.mean(arealist):
                mask[mask == i + 1] = 0
        mask[mask > 0] = 1

        roivolume = len(mask[np.nonzero(mask)])

        # Find contours around the mask. Padding is used to properly find contours near maximum edges
        pad_mask = np.pad(mask, ((0, 2), (0, 2)), mode='constant', constant_values=0)
        cont = find_contours(pad_mask, level=0, fully_connected='high', positive_orientation='high')

        print('Number of regions in ROI:', len(cont))
        print('Volume of ROI:', roivolume)

        # Subdivide polygon for smoother contours and more precise points for distance to boundary
        for c in range(len(cont)):
            cont[c] = subdivide_polygon(cont[c], 7)

        # Binary image of spots inside the ROI
        masked_detections = [(self.detections[i][0]*mask, self.image[i], i) for i in range(len(self.detections))]

        for c in range(len(cont)):
            cont[c] = np.array(cont[c])

        return cont, roivolume, masked_detections
    
    def show_roi(self, contour, masked_detections, couples, ch0, ch1):
        """
        Displays 3 figures in matplotlib, from left to right:
        Spot detection images, ROI and filtered spots (min size and shape), Marked couples
        First channel is yellow, second channel is cyan, overlap is white
        :param contour: List of polygons delimiting the ROI
        :param masked_detections: spot detection images with ROI mask
        :param couples: numpy array of information on couples
        :param ch0: int; first channel used in analysis
        :param ch1: int; second channel used in analysis
        """
        fig, axs = plt.subplots(1, 3, sharex='all', sharey='all')

        # Original spot detection binary image
        axs[0].imshow(((self.detections[ch0][0] * 2) + (self.detections[ch1][0])), cmap='nipy_spectral')

        # Add patches in second in third images for ROI contour
        for ax in (axs[1], axs[2]):
            patch = [patches.Polygon(np.fliplr(c), fill=False, color='white', linewidth=1) for c in contour]
            for p in patch:
                ax.add_patch(p)

        # Filtered spots image; checks if spots are too small or too linear for proper analysis and removes rejected
        # spots from image. Image is masked using detected ROI mask.
        filtered_spots = []
        for d, i, ch in masked_detections:
            labels = label(d, connectivity=2)
            label_props = regionprops(labels)
            for i in range(len(label_props)):
                if label_props[i].area < MIN_SIZE[ch] \
                        or label_props[i].minor_axis_length <= 1 \
                        or label_props[i].major_axis_length <= 1:
                    labels[labels == i + 1] = 0
            labels[labels > 0] = 1
            filtered_spots.append(labels)
        axs[1].imshow(((filtered_spots[ch0] * 2) + (filtered_spots[ch1])), cmap='nipy_spectral')

        # Same as second image with marker on coupled spots.
        if couples.any():
            axs[2].plot(couples[:, 0], couples[:, 1], 'g.')
            axs[2].plot(couples[:, 2], couples[:, 3], 'm.')
            axs[2].imshow(((filtered_spots[ch0] * 2) + (filtered_spots[ch1])), cmap='nipy_spectral')
        else:
            axs[2].imshow(((filtered_spots[ch0] * 2) + (filtered_spots[ch1])), cmap='nipy_spectral')

        # plt.savefig('ROI_{}.tif'.format(os.path.basename(self.file)))
        plt.show()
        plt.close()


def main(directory, scale_list, scale_threshold, n_rings, step):
    """
    Runs the SODA analysis on all images in the chosen directory. Also writes excel files with results.
    :param directory: string; folder from which to analyse every .tif image
    :param scale_list: array-like of ints; list of scales to use for wavelet transform
    :param scale_threshold: int; percent modifier of wavelet threshold
    :param n_rings: int; number of rings around spots in which to detect coupling
    :param step: float; width of rings around spots
    """

    # Stops program if number of rings isn't an integer
    if type(n_rings) != int:
        raise TypeError("Number of rings must be an integer.")

    # Writing Excel file
    results_workbook = xlsxwriter.Workbook(os.path.join(directory,
                                                        "PySODA_results_{}.xlsx".format(os.path.basename(directory))),
                                           {'nan_inf_to_errors': True})
    worksheets = []
    worksheet_names = []
    titles = ['File', 'Spots in channel 0', 'Spots in channel 1', 'Number of couples',
              'Coupling index 0', 'Coupling index 1', 'Weighted mean coupling distance',
              'Unweighted mean coupling distance']
    row = 1
    for sheet in worksheets:
        for t in range(len(titles)):
            sheet.write(0, t, titles[t])

    # Data arrays to use for graphs.
    # spots_results_list contains two lists: one for each channel. Each channel list contains three numpy arrays, one
    # for each condition. Every numpy arrays contains data on a spot in each row:
    """
    spots_results_list
    Column 
    index       Data
    0           X coordinate
    1           Y coordinate
    2           Area
    3           Distance to nearest neighbor in other channel
    4           Distance to nearest neighbor in same channel
    5           Eccentricity
    6           Index of nearest neighbor in other channel (index in same image's MPP)
    7           Index of nearest neighbor in same channel (index in same image's MPP)
    8           Max intensity          
    9           Min intensity
    10          Mean intensity
    11          Major axis length
    12          Minor axis length
    13          Orientation
    14          Perimeter
    15          Diameter of a circle with the same area as spot
    16          Index of image containing spot
    17          Coupling (0 if uncoupled, 1 if chance of coupling)
    """
    spots_results_list = [[np.ndarray((0, 18)), np.ndarray((0, 18)), np.ndarray((0, 18))],
                          [np.ndarray((0, 18)), np.ndarray((0, 18)), np.ndarray((0, 18))]]

    # couples_results_list contains three numpy arrays, one for each condition. Every numpy array contains data on a
    # couple in each row:
    """
    couples_results_list
    Column 
    index       Data
    0           Spot 1 X coordinate
    1           Spot 1 Y coordinate
    2           Spot 2 X coordinate
    3           spot 2 Y coordinate
    4           Coupling distance
    5           Coupling probability
    6           Index of image containing spots
    """
    couples_results_list = [np.ndarray((0, 7)), np.ndarray((0, 7)), np.ndarray((0, 7))]

    # soda_results_list contains three numpy arrays, one for each condition. Every numpy array contains data on the
    # SODA analysis of an image in each row:
    """
    soda_results_list 
    Column 
    index       Data
    0           Number of spots in first channel
    1           Number of spots in second channel
    2           Number of couples detected
    3           Coupling index of first channel
    4           Coupling index of second channel
    5           Mean coupling distance weighted by coupling probability
    6           Unweighted mean coupling distance
    7           Image index
    """
    soda_results_list = [np.ndarray((0, 8)), np.ndarray((0, 8)), np.ndarray((0, 8))]

    img_index = 0
    # Loops through all files in directory and run analysis only on TIF files
    for file in os.listdir(directory):
        if file.lower().endswith('.tif') or file.lower().endswith('.tiff'):
            print("*****************Computing SODA*****************")
            print(file)
            file_path = os.path.join(directory, file)
            write_row = True

            try:
                img = io.imread(file_path)
                if len(img.shape) != 3:
                    raise ImageError
                # Turns y,x,z images into z,y,x images. Assumes the z axis has the smallest value; works fine
                # since in this case the z axis is for channels, and SODA won't be used on more than 3 channels.
                if img.shape.index(min(img.shape)) == 2:
                    img = np.moveaxis(img, 2, 0)
            except ValueError:
                raise ImageError

            # Progress list prevents redundancy in channel combination (e.g. 1-2 and 2-1)
            progress_list = []
            soda = SodaImageAnalysis(file_path, img, directory, MIN_SIZE, scale_list, scale_threshold)
            # Loop through channel combinations; if SELF_SODA is true then run SODA on same-channel combinations as well
            for ch0 in range(img.shape[0]):
                for ch1 in range(img.shape[0]):
                    if ((SELF_SODA is False and ch0 != ch1) or SELF_SODA is True) and {ch0, ch1} not in progress_list:
                        try:
                            print("*****Computing SODA for channels {} and {}*****".format(ch0, ch1))
                            # Analysis happens here.
                            prob_write, spots0, spots1, CIndex, mean_dist, \
                                raw_mean_dist, coupling, n_couples, mean_ecc, \
                                spots_data0, spots_data1, couples_data = soda.analysis(ch0, ch1, n_rings, step, img_index)

                            # Writing excel file
                            # List of all elements in a line
                            elem_list = [file, spots0, spots1, n_couples, CIndex[0], CIndex[1], mean_dist, raw_mean_dist, img_index]
                            sheet_name = "SODA {}-{}".format(ch0, ch1)  # Add a sheet in excel for each ch. combination
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

                            # Add data in the correct numpy array based on condition in filename for graphs
                            elem_array = np.asarray([elem_list[1:]])
                            if 'groundCherry' in file:
                                spots_results_list[0][0] = np.append(spots_results_list[0][0], spots_data0, axis=0)
                                spots_results_list[1][0] = np.append(spots_results_list[1][0], spots_data1, axis=0)
                                if couples_data.shape != (0,):
                                    couples_results_list[0] = np.append(couples_results_list[0], couples_data, axis=0)
                                soda_results_list[0] = np.append(soda_results_list[0], elem_array, axis=0)
                            elif 'Melon' in file:
                                spots_results_list[0][1] = np.append(spots_results_list[0][1], spots_data0, axis=0)
                                spots_results_list[1][1] = np.append(spots_results_list[1][1], spots_data1, axis=0)
                                if couples_data.shape != (0,):
                                    couples_results_list[1] = np.append(couples_results_list[1], couples_data, axis=0)
                                soda_results_list[1] = np.append(soda_results_list[1], elem_array, axis=0)
                            elif 'pickeledBears' in file:
                                spots_results_list[0][2] = np.append(spots_results_list[0][2], spots_data0, axis=0)
                                spots_results_list[1][2] = np.append(spots_results_list[1][2], spots_data1, axis=0)
                                if couples_data.shape != (0,):
                                    couples_results_list[2] = np.append(couples_results_list[2], couples_data, axis=0)
                                soda_results_list[2] = np.append(soda_results_list[2], elem_array, axis=0)

                        except ImageError:  # Skips image if it doesn't have the right shape (multiple channels)
                            write_row = False
                            print("Skipping invalid TIF file...")
                        except SpotsError:  # Skips image if the amount of spots is too few for a gooda analysis
                            write_row = False
                            print("Not enough spots; skipping image...")
                        progress_list.append({ch0, ch1})
            if write_row:
                row += 1
            img_index += 1

    # Write graphs
    # display = DataDisplay(spots_results_list, couples_results_list, soda_results_list)
    # display.index_ecc_boxplots()
    # display.ecc_boxplots()
    # display.draw_density_ecc()
    # display.coupling_dist_boxplots()
    # display.coupling_prob_boxplots()
    # display.coupling_index_boxplots()
    # # display.draw_iso_density()
    # display.coupling_prob_scatter()
    # display.draw_eq_diameter_density()
    try:
        results_workbook.close()
    except PermissionError:
        print("PermissionError: Couldn't write xlsx file. It might be open already!")


class DataDisplay:
    """
    Class containing the functions related to the display of data
    """
    def __init__(self, spots_data, couples_data, soda_data):
        self.spots_data = spots_data
        self.couples_data = couples_data
        self.soda_data = soda_data
        self.conditions = ['0Mg + Bicc + Gly', 'KCl', 'Block']

    def ecc_boxplots(self):
        """
        Draws spots' eccentricity boxplots for different eccentricity intervals, conditions and coupling
        """
        ch = 0
        bin_list = [0.2, 0.4, 0.6, 0.8, 1.0]
        for channel in self.spots_data:
            for b in range(1, len(bin_list)):
                ax_i = 0
                fig, axs = plt.subplots(1, 3)
                plt.subplots_adjust(wspace=0.4)
                axs[0].set_title('0Mg + Bicc + Gly')
                axs[1].set_title('KCl')
                axs[2].set_title('Block')
                for c in channel:
                    filt_data = c[c[:, 5] > bin_list[b - 1]]
                    filt_data = filt_data[filt_data[:, 5] <= bin_list[b]]
                    self.draw_boxplot_axis(filt_data, axs[ax_i])
                    ax_i += 1
                plt.savefig('ecc_max_{}_global{}.png'.format(bin_list[b], ch))
                plt.close()

            for b in ['less', 'more']:
                ax_i = 0
                fig, axs = plt.subplots(1, 3)
                plt.subplots_adjust(wspace=0.4)
                axs[0].set_title('0Mg + Bicc + Gly')
                axs[1].set_title('KCl')
                axs[2].set_title('Block')
                for c in channel:
                    if b == 'more':
                        filt_data = c[c[:, 5] > 0.5]
                    else:
                        filt_data = c[c[:, 5] <= 0.5]
                    self.draw_boxplot_axis(filt_data, axs[ax_i])
                    ax_i += 1
                plt.savefig('ecc_{}_0.5_global{}.png'.format(b, ch))
                plt.close()

            fig, axs = plt.subplots(1, 3)
            plt.subplots_adjust(wspace=0.4)
            axs[0].set_title('0Mg + Bicc + Gly')
            axs[1].set_title('KCl')
            axs[2].set_title('Block')
            ax_i = 0
            for c in channel:
                self.draw_boxplot_axis(c, axs[ax_i])
                ax_i += 1
            plt.savefig('ecc_all_global{}.png'.format(ch))
            plt.close()

            ch += 1

    @staticmethod
    def draw_boxplot_axis(data, ax):
        coupled_ecc = data[data[:, -1] == 1][:, 5]
        uncoupled_ecc = data[data[:, -1] == 0][:, 5]
        box_data = [coupled_ecc, uncoupled_ecc]
        ax.boxplot(box_data, showmeans=True, widths=0.5, labels=['Coupled', 'Uncoupled'])

    def coupling_dist_boxplots(self):
        """
        Coupling distance based on condition
        """
        data = []
        for cond in self.couples_data:
            data.append(cond[:, 4])
        plt.boxplot(data, showmeans=True, widths=0.5, labels=self.conditions)
        plt.ylabel('Coupling distance (pixels)')
        plt.xlabel('Condition')
        plt.savefig('coupling_dist_boxplot.png')
        plt.close()

    def coupling_prob_boxplots(self):
        """
        Coupling probability for each condition in coupling distance intervals
        """
        dist_bins = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        for b in range(1, len(dist_bins)):
            data = []
            for cond in self.couples_data:
                filt_cond = cond[cond[:, 4] > dist_bins[b-1]]
                filt_cond = filt_cond[filt_cond[:, 4] <= dist_bins[b]]
                data.append(filt_cond[:, 5])
            plt.ylim(0, 1)
            plt.boxplot(data, showmeans=True, widths=0.5, labels=self.conditions)
            plt.ylabel('Coupling probability')
            plt.xlabel('Condition')
            plt.title('{} < Coupling Distance <= {}'.format(dist_bins[b-1], dist_bins[b]))
            plt.savefig('coupling_prob_max_{}_boxplot.png'.format(dist_bins[b]))
            plt.close()

    def coupling_prob_scatter(self):
        """
        Scatterplot of coupling probability by coupling distance
        """
        fig, axs = plt.subplots(1,3)
        fig.set_size_inches(15, 5)
        plt.subplots_adjust(wspace=0.5)
        ax_i = 0
        axs[0].set_title('0Mg + Bicc + Gly')
        axs[1].set_title('KCl')
        axs[2].set_title('Block')
        for cond in self.couples_data:
            data_prob = cond[:, 5]
            data_dist = cond[:, 4]
            # axs[ax_i].hist2d(data_dist, data_prob, bins=(50, 50), cmap='jet', range=[[0, 10], [0, 1]])
            axs[ax_i].set_ylim(0,1.0)
            axs[ax_i].set_xlim(0,10)
            axs[ax_i].scatter(data_dist, data_prob)
            ax_i += 1

        for ax in axs:
            ax.set_ylabel('Coupling probability')
            ax.set_xlabel('Coupling distance (pixels)')
        plt.savefig('coupling_prob_max_density.png')
        plt.close()

    def coupling_index_boxplots(self):
        """
        Boxplots of coupling index for each condition
        """
        data0 = []
        data1 = []
        for cond in self.soda_data:
            data0.append(cond[:, 3])
            data1.append(cond[:, 4])
        data = [data0, data1]
        ch = 0
        for d in data:
            plt.ylim(ymax=0.5)
            plt.boxplot(d, showmeans=True, widths=0.5, labels=self.conditions)
            plt.ylabel('Coupling index')
            plt.xlabel('Condition')
            plt.savefig('coupling_index_boxplot{}.png'.format(ch))
            ch += 1
            plt.close()

    def index_ecc_boxplots(self):
        """
        Boxplots of eccentricity for each condition in different coupling index intervals
        """
        bin_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        ch_i = 0
        for spots_ch in self.spots_data:  # channels
            for b in range(1, len(bin_list)):  # bins
                plot_data = []
                for i in range(len(self.soda_data)):  # conditions
                    soda_data0 = self.soda_data[i][self.soda_data[i][:, 3] >= bin_list[b-1]]
                    soda_data0 = soda_data0[soda_data0[:, 3] < bin_list[b]]
                    soda_data1 = self.soda_data[i][self.soda_data[i][:, 4] >= bin_list[b-1]]
                    soda_data1 = soda_data1[soda_data1[:, 4] < bin_list[b]]
                    if ch_i == 0:
                        data = spots_ch[i][np.isin(spots_ch[i][:, -2], soda_data0[:, -1])][:, 5]
                    else:
                        data = spots_ch[i][np.isin(spots_ch[i][:, -2], soda_data1[:, -1])][:, 5]
                    plot_data.append(data)
                plt.boxplot(plot_data, labels=self.conditions, showmeans=True)
                plt.title('{} <= Coupling Index < {}'.format(bin_list[b-1], bin_list[b]))
                plt.xlabel('Condition')
                plt.ylabel('Eccentricity')
                plt.savefig("index_ecc_max{}_boxplot{}.png".format(bin_list[b], ch_i))
                plt.close()
            ch_i += 1

    def draw_density_ecc(self):
        """
        2D hists of eccentricity by distance to nearest_neighbor for each condition and coupling (coupled/uncoupled/all)
        """
        ch = 0
        for channel in self.spots_data:
            ax_x = 0
            fig, axs = plt.subplots(3, 3)
            fig.set_size_inches(15,10)
            plt.subplots_adjust(hspace=0.5)
            axs[0,0].set_title('0Mg + Bicc + Gly (Coupled)')
            axs[0,1].set_title('KCl (Coupled)')
            axs[0,2].set_title('Block (Coupled)')
            axs[1,0].set_title('0Mg + Bicc + Gly (Uncoupled)')
            axs[1,1].set_title('KCl (Uncoupled)')
            axs[1,2].set_title('Block (Uncoupled)')
            axs[2,0].set_title('0Mg + Bicc + Gly (All)')
            axs[2,1].set_title('KCl (All)')
            axs[2,2].set_title('Block (All)')
            for cond in channel:
                coupled = cond[cond[:, -1] == 1]
                non_coupled = cond[cond[:, -1] == 0]
                x = coupled[:, 3]
                y = coupled[:, 5]
                x2 = non_coupled[:, 3]
                y2 = non_coupled[:, 5]
                x3 = cond[:, 3]
                y3 = cond[:, 5]

                axs[0, ax_x].hist2d(x, y, bins=(50, 50), range=[[0, 50], [0, 1.0]], cmap='jet')
                axs[1, ax_x].hist2d(x2, y2, bins=(50, 50), range=[[0, 50], [0, 1.0]], cmap='jet')
                axs[2, ax_x].hist2d(x3, y3, bins=(50, 50), range=[[0, 50], [0, 1.0]], cmap='jet')

                for y in axs:
                    for ax in y:
                        ax.set_aspect(1. / ax.get_data_ratio())
                        ax.set_xlabel("Distance to nearest neighbor (pixels)")
                        ax.set_ylabel("Eccentricity")
                ax_x += 1

            plt.savefig("density_map_ecc_nn{}".format(ch))
            plt.close()
            ch += 1

    def draw_iso_density(self):
        ch = 0
        for channel in self.spots_data:
            ax_x = 0
            fig, axs = plt.subplots(3, 3)
            fig.set_size_inches(15,10)
            plt.subplots_adjust(hspace=0.5)
            axs[0,0].set_title('0Mg + Bicc + Gly (Coupled)')
            axs[0,1].set_title('KCl (Coupled)')
            axs[0,2].set_title('Block (Coupled)')
            axs[1,0].set_title('0Mg + Bicc + Gly (Uncoupled)')
            axs[1,1].set_title('KCl (Uncoupled)')
            axs[1,2].set_title('Block (Uncoupled)')
            axs[2,0].set_title('0Mg + Bicc + Gly (All)')
            axs[2,1].set_title('KCl (All)')
            axs[2,2].set_title('Block (All)')
            for cond in channel:
                coupled = cond[cond[:, -1] == 1]
                non_coupled = cond[cond[:, -1] == 0]
                x = coupled[:, 3]
                y = (4*np.pi*(coupled[:, 2]))/(coupled[:, 13]**2)
                x2 = non_coupled[:, 3]
                y2 = (4*np.pi*(non_coupled[:, 2]))/(non_coupled[:, 13]**2)
                x3 = cond[:, 3]
                y3 = (4 * np.pi * (cond[:, 2]))/(cond[:, 13] ** 2)

                axs[0, ax_x].hist2d(x,y, bins=(50,50), range=[[0,30], [0,5]], cmap='jet')
                axs[1, ax_x].hist2d(x2, y2, bins=(50, 50), range=[[0,30], [0,5]], cmap='jet')
                axs[2, ax_x].hist2d(x3, y3, bins=(50, 50), range=[[0, 30], [0,5]], cmap='jet')

                for y in axs:
                    for ax in y:
                        ax.set_aspect(1. / ax.get_data_ratio())
                        ax.set_xlabel("Distance to nearest neighbor (pixels)")
                        ax.set_ylabel("Isoperimetric quotient")
                ax_x += 1

            plt.savefig("density_isoperimetric_nn{}".format(ch))
            plt.close()
            ch += 1

    def draw_eq_diameter_density(self):
        ch = 0
        for channel in self.spots_data:
            ax_x = 0
            fig, axs = plt.subplots(3, 3)
            fig.set_size_inches(15,10)
            plt.subplots_adjust(hspace=0.5)
            axs[0,0].set_title('0Mg + Bicc + Gly (Coupled)')
            axs[0,1].set_title('KCl (Coupled)')
            axs[0,2].set_title('Block (Coupled)')
            axs[1,0].set_title('0Mg + Bicc + Gly (Uncoupled)')
            axs[1,1].set_title('KCl (Uncoupled)')
            axs[1,2].set_title('Block (Uncoupled)')
            axs[2,0].set_title('0Mg + Bicc + Gly (All)')
            axs[2,1].set_title('KCl (All)')
            axs[2,2].set_title('Block (All)')
            for cond in channel:
                coupled = cond[cond[:, -1] == 1]
                non_coupled = cond[cond[:, -1] == 0]
                x = coupled[:, 3]
                y = coupled[:, 15]
                x2 = non_coupled[:, 3]
                y2 = non_coupled[:, 15]
                x3 = cond[:, 3]
                y3 = cond[:, 15]

                axs[0, ax_x].hist2d(x,y, bins=(50,50), range=[[0,50], [2,13]], cmap='jet')
                axs[1, ax_x].hist2d(x2, y2, bins=(50, 50), range=[[0,50], [2,13]], cmap='jet')
                axs[2, ax_x].hist2d(x3, y3, bins=(50, 50), range=[[0,50], [2,13]], cmap='jet')

                for y in axs:
                    for ax in y:
                        ax.set_aspect(1. / ax.get_data_ratio())
                        ax.set_xlabel("Distance to nearest neighbor (pixels)")
                        ax.set_ylabel("Equivalent diameter (pixels)")
                ax_x += 1

            plt.savefig("density_diameter_nn{}".format(ch))
            plt.close()
            ch += 1


if __name__ == '__main__':
    start_time = time.clock()
    main(DIRECTORY, SCALE_LIST, SCALE_THRESHOLD, N_RINGS, RING_WIDTH)
    print("--- Running time: %s seconds ---" % (time.clock() - start_time))
