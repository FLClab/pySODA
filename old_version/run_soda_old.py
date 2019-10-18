import wavelet_SODA_old as wv
from skimage import io, filters
from scipy.ndimage import morphology
from scipy.spatial import distance
from matplotlib import patches
from skimage.measure import find_contours, regionprops, label, subdivide_polygon
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import xlsxwriter

"""
This file contains all the code related to running the SODA method using the classes from wavelet_SODA.py
This is the first version of the code used to do most of the analysis. The segmentation takes much longer time
here.
"""

""" Change parameters here! """
DIRECTORY = r"C:\Users\Renaud\Documents\GitHub\pySODA - Copie (2)\figure_image\cropped"  # Path containing TIF files

# For spot detection
SCALE_LIST = [[3,4],  # Channel 0  # Scales to be used for wavelet transform for spot detection
              [3,4],  # Channel 1  # Higher values mean less details.
              [2]]  # Channel 2  # Multiple scales can be used (e.g. [1,2]). Scales must be integers.
SCALE_THRESHOLD = [200,  # Channel 0  # Percent modifier of wavelet transform threshold.
                   200,  # Channel 1  # Higher value = more pixels detected.
                   100]  # Channel 2
MIN_SIZE = [5,  # Channel 0 # Minimum area (pixels) of spots to analyse
            5,  # Channel 1
            5]  # Channel 2
MIN_AXIS_LENGTH = [3,  # Channel 0  # Minimum length of both ellipse axes of spots to analyse
                   3,  # Channel 1
                   1]  # Channel 2

# For SODA analysis
ROI_THRESHOLD = 200  # Percent modifier of ROI threshold. Higher value = more pixels taken.
N_RINGS = 16  # Number of rings around spots (int)
RING_WIDTH = 1  # Width of rings in pixels (15nm)
SELF_SODA = True  # Set to True to compute SODA for couples of spots in the same channel as well
SELF_CHANNELS = [0]  # Enter channels for which to run SODA on themselves
SELF_ONLY = False  # Set to True to only run SODA on each channel with themselves

# Display and graphs
SHOW_ROI = True  # Set to True to display the ROI mask contour and detected spots for every image
SAVE_ROI = True  # Set to True to save TIF images of all spots and filtered spots for each channel
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
    def __init__(self, file, image, directory, min_size=(5, 5), min_axis=(1,1), scale_list=((2,), (2,)), scale_threshold=(100, 100)):
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
        self.min_axis = min_axis

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

        print(roivolume)

        # Calls SODA spatial relations analysis
        prob_write, CIndex, mean_dist, raw_mean_dist, coupling, n_couples = self.spatial_relations(SR, ch0, ch1)
        spots0, spots1, couples_data = SR.get_spots_data(prob_write)

        # Display spot detection, ROI and couples
        if SAVE_ROI or SHOW_ROI:
            self.show_roi(contours, masked_detections, couples_data, SHOW_ROI, SAVE_ROI, ch0, ch1)
            # self.show_roi_shape_factor(contours, masked_detections, ch0, ch1)

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
        marks = [wv.SpatialDistribution(d, i, cs=self.min_size[ch], min_axis=self.min_axis[ch]).mark()
                 for d, i, ch in (detections[ch0], detections[ch1])]
        for m in range(len(marks)):
            print("\tSpots in channel {}:".format(m), len(marks[m]))
            if range(len(marks[m]) < 100):  # Minimum number of spots necessary to compute SODA
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
        print(G)
        print('Done! \n')

        print("Computing variance...")
        var = SR.variance_theo_delta_new()
        print(var)
        print('Done! \n')

        print("Computing A...")
        A = SR.intersection2D_new()
        print(A)
        print('Done! \n')

        print("Computing G0...")
        arg_dict = {'G': G, 'var': var, 'A': A}
        G0 = SR.reduced_Ripley_vector(**arg_dict)
        print('Done! \n')
        print(G0)
        plt.bar(np.arange(0, G0.size)*15, G0, 14)
        plt.xticks(np.arange(-7.5,232.5,15), np.arange(0,240,15))
        plt.axhline(y=np.sqrt(2*np.log(G0.size)))
        plt.savefig(os.path.join(r"figure_image", 'g0_bars.pdf'), transparent=True, bbox_inches='tight')
        plt.show()
        SR.draw_G0(G0)

        print("Computing statistics...")
        prob_write, CIndex, mean_dist, raw_mean_dist, coupling, n_couples = SR.coupling_prob(**arg_dict, G0=G0)
        print("================================"
              "\nCoupling Index 1:", CIndex[0],
              "\nCoupling Index 2:", CIndex[1],
              "\nMean Coupling Distance:", mean_dist, "pixels",
              "\nUnweighted Mean Coupling Distance:", raw_mean_dist, "pixels")

        probs = np.ndarray.tolist(coupling)
        print('Probabilities', probs, '\n\n')

        # Coupling prob. by distance histogram
        if WRITE_HIST:
            dists = [i*RING_WIDTH for i in range(N_RINGS)]
            plt.ylim(0, 1.0)
            plt.bar(dists, probs, align='edge', width=RING_WIDTH, edgecolor='black', linewidth=0.75)
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
            filt += self.image[ch]
        filt = (filters.gaussian(filt, 10))  # Higher sigma for smoother edges

        threshold = np.mean(filt) * (100/ROI_THRESHOLD)
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
    
    def show_roi(self, contour, masked_detections, couples, show, save, ch0, ch1):
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
        # Filtered spots image; checks if spots are too small or too linear for proper analysis and removes rejected
        # spots from image. Image is masked using detected ROI mask.
        filtered_spots = []
        for d, img, ch in masked_detections:
            labels = label(d, connectivity=2)
            label_props = regionprops(labels)
            for i in range(len(label_props)):
                if label_props[i].area < self.min_size[ch] \
                        or label_props[i].minor_axis_length < self.min_axis[ch] \
                        or label_props[i].major_axis_length < self.min_axis[ch]:
                    labels[labels == label_props[i].label] = 0
            labels[labels > 0] = 1
            filtered_spots.append(labels)

        # Save images with all spots and filtered spots for both channels.
        if save:
            for ch in [ch0, ch1]:
                io.imsave("{}_all_spots_ch{}.tif".format(os.path.basename(self.file), ch), self.detections[ch][0].astype('uint16'))
                io.imsave("{}_filtered_spots_ch{}.tif".format(os.path.basename(self.file), ch), filtered_spots[ch].astype('uint16'))

        if show:
            fig, axs = plt.subplots(2, 3, sharex='all', sharey='all')

            # Add patches in second in third images for ROI contour
            for ax in (axs[0,1], axs[0,2], axs[1,0], axs[1,1], axs[1,2]):
                patch = [patches.Polygon(np.fliplr(c), fill=False, color='white', linewidth=1) for c in contour]
                for p in patch:
                    ax.add_patch(p)

            # Original spot detection binary image
            spots_composite = np.zeros((self.detections[ch0][0].shape[0], self.detections[ch0][0].shape[1], 3))
            spots_composite[:,:,0] = self.detections[ch0][0]
            spots_composite[:,:,1] = self.detections[ch1][0]
            spots_composite = spots_composite
            io.imsave('all_spots.tif', (spots_composite).astype('uint8'))
            axs[0,0].imshow(spots_composite)

            # Display filtered spots
            filtered_composite = np.zeros((filtered_spots[ch0].shape[0], filtered_spots[ch0].shape[1], 3))
            filtered_composite[:,:,0] = self.detections[ch0][0]
            filtered_composite[:,:,1] = self.detections[ch1][0]
            filtered_composite = filtered_composite/(2**16-1)*255
            io.imsave('filtered_spots.tif', (filtered_composite*255).astype('uint8'))
            axs[0,1].imshow(filtered_composite)

            # Same as second image with marker on coupled spots.
            if couples.any():
                axs[0,2].plot(couples[:, 0], couples[:, 1], 'g.')
                axs[0,2].plot(couples[:, 2], couples[:, 3], 'm.')
            axs[0,2].imshow(filtered_composite)

            coupled_points_ch0 = np.array((couples[:, 0], couples[:, 1]))
            coupled_points_ch1 = np.array((couples[:, 2], couples[:, 3]))

            coupled_mask_ch0, coupled_centroids_ch0 = self.find_points_to_shapes(filtered_spots[ch0], coupled_points_ch0, ch=ch0)
            #coupled_mask_ch0 = np.array((coupled_mask_ch0, np.zeros(coupled_mask_ch0.shape), np.zeros(coupled_mask_ch0.shape)))
            io.imsave('coupled_ch0.tif', (coupled_mask_ch0).astype('uint8'))
            io.imsave('centers_ch0.tif', (coupled_centroids_ch0).astype('uint8'))

            coupled_mask_ch1, coupled_centroids_ch1 = self.find_points_to_shapes(filtered_spots[ch1], coupled_points_ch1, ch=ch1)
            #coupled_mask_ch1 = np.array((np.zeros(coupled_mask_ch1.shape), coupled_mask_ch1, np.zeros(coupled_mask_ch0.shape)))
            io.imsave('coupled_ch1.tif', (coupled_mask_ch1).astype('uint8'))
            io.imsave('centers_ch1.tif', (coupled_centroids_ch1).astype('uint8'))


            axs[1,0].imshow(coupled_mask_ch0)
            axs[1,1].imshow(coupled_mask_ch1)

            not_coupled_composite = np.zeros((filtered_spots[ch0].shape[0], filtered_spots[ch0].shape[1], 3))
            not_coupled_composite[:,:,0] = filtered_spots[ch0] - coupled_mask_ch0
            not_coupled_composite[:,:,1] = filtered_spots[ch1] - coupled_mask_ch1
            not_coupled_composite = not_coupled_composite/(2**16-1)
            io.imsave('uncoupled.tif', (not_coupled_composite).astype('uint8'))
            axs[1,2].imshow(not_coupled_composite)

            plt.show()
        plt.close()

    def find_points_to_shapes(self, image, points, ch):
        mask_lab, num = label(image, return_num=True)
        mask_props = regionprops(mask_lab, intensity_image=self.image[ch])

        all_spots_positions = np.ndarray((2, num))
        for i in range(num):
            all_spots_positions[:, i] = np.flip(np.array(mask_props[i].weighted_centroid))

        dists = distance.cdist(all_spots_positions.transpose(), points.transpose())
        dists = np.sort(dists, axis=1)
        good_spots = np.argwhere(dists[:,0] < 0.1) + 1

        coupled_mask = np.zeros(mask_lab.shape)
        coupled_mask[np.isin(mask_lab, good_spots)] = 1

        mask_lab, num = label(coupled_mask, return_num=True)
        mask_props = regionprops(mask_lab, intensity_image=self.image[ch])
        coupled_centroids = np.zeros(mask_lab.shape)
        for p in mask_props:
            y, x = p.weighted_centroid
            coupled_centroids[int(y),int(x)] = 255

        return coupled_mask, coupled_centroids

    def show_roi_shape_factor(self, contour, masked_detections, ch0, ch1):
        """
        Displays 3 figures in matplotlib, from left to right:
        Spot detection images, ROI and filtered spots (min size and shape), Marked couples
        First channel is yellow, second channel is cyan, overlap is white
        :param contour: List of polygons delimiting the ROI
        :param masked_detections: spot detection images with ROI mask
        :param ch0: int; first channel used in analysis
        :param ch1: int; second channel used in analysis
        """
        fig, axs = plt.subplots(1, 6, sharex='all', sharey='all')

        # Add patches for ROI contours
        for ax in axs:
            patch = [patches.Polygon(np.fliplr(c), fill=False, color='white', linewidth=1) for c in contour]
            for p in patch:
                ax.add_patch(p)

        # Filtered spots image; checks if spots are too small or too linear for proper analysis and removes rejected
        # spots from image. Image is masked using detected ROI mask.
        bins = [0.0, 0.3, 0.6, 1.0, 1.3, 1.6, 2.0]

        for b in range(1, len(bins)):
            filtered_spots = []
            for d, img, ch in masked_detections:
                labels = label(d, connectivity=2)
                label_props = regionprops(labels)
                for i in range(len(label_props)):
                    area = label_props[i].area
                    perimeter = label_props[i].perimeter
                    if perimeter == 0:
                        labels[labels == i + 1] = 0
                    else:
                        shape_factor = (4*np.pi*area)/(perimeter**2)
                        if label_props[i].area < self.min_size[ch] \
                                or label_props[i].minor_axis_length < self.min_axis[ch] \
                                or label_props[i].major_axis_length < self.min_axis[ch] \
                                or (shape_factor <= bins[b-1] or shape_factor > bins[b]):
                            labels[labels == i + 1] = 0
                labels[labels > 0] = 1
                filtered_spots.append(labels)
            axs[b-1].set_title('{} > SF > {}'.format(bins[b-1], bins[b]))
            axs[b-1].imshow(((filtered_spots[ch0] * 2) + (filtered_spots[ch1])), cmap='nipy_spectral')

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

    img_index = 0
    # Loops through all files in directory and run analysis only on TIF files
    for file in os.listdir(directory):
        if 'filtered' in file or 'all_spots' in file:
            continue
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
            soda = SodaImageAnalysis(file_path, img, directory, MIN_SIZE, MIN_AXIS_LENGTH, scale_list, scale_threshold)
            # Loop through channel combinations; if SELF_SODA is true then run SODA on same-channel combinations as well
            for ch0 in range(img.shape[0]):
                for ch1 in range(img.shape[0]):
                    if ((SELF_SODA is False and ch0 != ch1) or SELF_SODA is True) and {ch0, ch1} not in progress_list:
                        if SELF_ONLY and ch0 != ch1 or SELF_SODA and (ch0 not in SELF_CHANNELS or ch1 not in SELF_CHANNELS):
                            continue
                        try:
                            print("*****Computing SODA for channels {} and {}*****".format(ch0, ch1))
                            # Analysis happens here.
                            prob_write, spots0, spots1, CIndex, mean_dist, \
                                raw_mean_dist, coupling, n_couples, mean_ecc, \
                                spots_data0, spots_data1, couples_data = soda.analysis(ch0, ch1, n_rings,
                                                                                       step, img_index)

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

    try:
        results_workbook.close()
    except PermissionError:
        print("PermissionError: Couldn't write xlsx file. It might be open already!")

if __name__ == '__main__':
    start_time = time.time()
    main(DIRECTORY, SCALE_LIST, SCALE_THRESHOLD, N_RINGS, RING_WIDTH)
    print("--- Running time: %s seconds ---" % (time.time() - start_time))
