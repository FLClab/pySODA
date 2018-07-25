import wavelet_SODA as wv
from skimage import io, filters
from scipy.ndimage import morphology
import os
import time
import matplotlib.pyplot as plt
from matplotlib import patches
from skimage.measure import find_contours, regionprops, label, subdivide_polygon
from skimage.segmentation import slic
import numpy as np
import xlsxwriter

""" Change parameters here! """
# For spot detection
SCALE_LIST = [2]  # Scales to be used for wavelet transform for spot detection
SCALE_THRESHOLD = 100  # Percent modifier of wavelet transform threshold

# For SODA analysis
DIRECTORY = r"C:\Users\Renaud\PycharmProjects\WAVELETS_IMAGES\cocultures\EXP-8-STED"  # Path containing TIF files
N_RINGS = 10  # Number of rings around spots
STEP = 1  # Width of rings in pixels

# Other
SHOW_ROI = False  # Set to True to show the ROI mask contour and detected spots


class SodaImageAnalysis:
    """
    This class contains the steps to run the SODA analysis on an image.
    """
    def __init__(self, file, scale_list=(2,), scale_threshold=100):
        """
        Constructor function.
        :param file: Path of the image file to analyse
        :param scale_list: List of wavelet scales to use
        :param scale_threshold: Percentage of the wavelet threshold to use.
        """
        self.file = file
        self.scale_list = scale_list
        self.scale_threshold = scale_threshold

        self.image = io.imread(file)

        z, y, x = self.image.shape
        print('X:', x, '\nY:', y)
        self.poly = [np.array([[0, 0],
                               [x, 0],
                               [x, y],
                               [0, y]])]

        print("Computing wavelet spot detection...\nScales: {}".format(self.scale_list))
        self.detections = [(wv.Detection_Wavelets((self.image[i]), J_list=self.scale_list,
                                                  scale_threshold=self.scale_threshold).computeDetection(),
                            self.image[i]) for i in range(z)]

    def analysis(self, ch1, ch2, n_rings, step):
        """
        Runs the analysis for chosen channels.
        :param ch1: int; first channel to use in analysis
        :param ch2: int; second channel to use in analysis
        :param max_dist: int; maximum distance in which to detect coupling
        :param step: int; width of rings (number of rings = max_dist/step)
        """
        contours, roivolume, masked_detections = self.find_roi()
        marks = self.spatial_distribution(masked_detections)
        max_dist = n_rings*step
        #contours = self.poly
        #roivolume = self.image.shape[1] * self.image.shape[2]
        SR = wv.Spatial_Relations(*marks, self.image[0].shape, roivolume, contours, self.image, n_rings, step, os.path.basename(self.file))
        prob_write, CIndex, mean_dist, raw_mean_dist, coupling, n_couples = self.spatial_relations(SR)
        SR.write_spots_and_probs(prob_write, 'PySODA_spots_{}_ch{}{}.xlsx'.format(os.path.basename(self.file), ch1, ch2))

        return prob_write, len(marks[0]), len(marks[1]), CIndex, mean_dist, raw_mean_dist, coupling, n_couples

    def spatial_distribution(self, detections):
        """
        Computes the spatial distribution for the chosen list of binary images
        :param detections: list of tuples containing (binary image, base image)
        :return: Marked Point Process of each binary image in detections
        """
        print("Computing spatial distribution...")
        marks = [wv.Spatial_Distribution(d, i, cs=0).mark() for d, i in detections]
        for m in range(len(marks)):
            print("\tSpots in channel {}:".format(m), len(marks[m]))
        return marks

    def spatial_relations(self, SR):
        """
        Computes every step of the spatial relations analysis for the chosen Spatial_Relations object
        :param SR: Spatial Relations object
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
        # print('Probabilities (Javascript version):', SR.main2D_corr(G, var, A))
        probs = np.ndarray.tolist(coupling)
        print('Probabilities', probs, '\n\n')
        #dists = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        #plt.bar(dists, probs, align='edge', width=1, edgecolor='black', linewidth=0.75)
        #plt.savefig('hist_{}.png'.format(os.path.basename(self.file)))
        #plt.close()

        return prob_write, CIndex, mean_dist, raw_mean_dist, coupling, n_couples

    def find_roi_hkmeans(self):
        """
        Attempt to replicate the cell mask from Icy SODA.
        Doesn't work very well!
        :return:
        """
        filt = self.image[0] + self.image[1]
        filt = (filters.gaussian(filt, 1))
        filt = slic(filt, n_segments=2, compactness=0.0001, enforce_connectivity=False)
        plt.imshow(filt)
        plt.show()
        filt[filt > 0] = 1
        filt = morphology.binary_dilation(filt, np.ones((20, 20)))
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

        plt.imshow(mask)
        plt.show()

        roivolume = len(mask[np.nonzero(mask)])

        # Find contours around the mask. Padding is used to properly find contours near maximum edges
        contlabel = np.pad(mask, ((0, 2), (0, 2)), mode='constant', constant_values=0)
        cont = find_contours(contlabel, level=0, fully_connected='high', positive_orientation='high')
        for c in range(len(cont)):
            if len(cont[c]) < 5:
                cont[c] = 0
        cont = [value for value in cont if type(value) != int]

        print('Number of regions in ROI:', len(cont))
        print('Volume of ROI:', roivolume)
        # For smoother contours and more precise points for distance to boundary
        for c in range(len(cont)):
            cont[c] = subdivide_polygon(cont[c], 7)

        masked_detections = [(self.detections[i][0] * mask, self.image[i]) for i in range(len(self.detections))]

        # Display ROI
        patch = [patches.Polygon(np.fliplr(c), fill=False, color='white', linewidth=1) for c in
                 cont]

        fig, ax = plt.subplots()
        for p in patch:
            ax.add_patch(p)

        plt.imshow((masked_detections[0][0] * 0.995) + (masked_detections[1][0] * 0.5), cmap='nipy_spectral')
        plt.show()

        for c in range(len(cont)):
            cont[c] = np.array(cont[c])

        return cont, roivolume, masked_detections

    def find_roi(self):
        """
        Finds the ROI based on a gaussian filter and threshold.
        :return: cont: Numpy array of all the points determining the shape of the ROI
        :return: roivolume: Area of the detected ROI
        :return: masked_detections: list of tuples (binary image, base image) with only the spots inside the ROI
        """

        # Filtered image for masking: gaussian filter + threshold + fill holes
        filt = self.image[0] + self.image[1]
        filt = (filters.gaussian(filt, 10))

        threshold = np.mean(filt)/2
        filt[filt < threshold] = 0
        filt[filt >= threshold] = 1
        # filt = morphology.binary_dilation(filt, np.ones((10, 10)))
        filt = morphology.binary_closing(filt)

        #filt = morphology.binary_fill_holes(filt).astype('uint16')

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
        # io.imsave('MASK_{}.tif'.format(os.path.basename(self.file)), mask.astype('uint16'))

        # TEST MASK
        # mask = io.imread('mask.tif')
        # mask[mask > 0] = 1
        roivolume = len(mask[np.nonzero(mask)])

        # Find contours around the mask. Padding is used to properly find contours near maximum edges
        contlabel = np.pad(mask, ((0, 2), (0, 2)), mode='constant', constant_values=0)
        cont = find_contours(contlabel, level=0, fully_connected='high', positive_orientation='high')

        # Remove any small residual contours that may appear
        for c in range(len(cont)):
            if len(cont[c]) < 5:
                cont[c] = 0
        cont = [value for value in cont if type(value) != int]

        print('Number of regions in ROI:', len(cont))
        print('Volume of ROI:', roivolume)

        # Subdivide polygon for smoother contours and more precise points for distance to boundary
        for c in range(len(cont)):
           cont[c] = subdivide_polygon(cont[c], 7)

        masked_detections = [(self.detections[i][0]*mask, self.image[i]) for i in range(len(self.detections))]

        if SHOW_ROI:
            # Display ROI
            patch = [patches.Polygon(np.fliplr(c), fill=False, color='white', linewidth=1) for c in
                    cont]
            fig, ax = plt.subplots()
            for p in patch:
               ax.add_patch(p)
            plt.imshow((self.detections[0][0] * 0.995) + (self.detections[1][0] * 0.5), cmap='nipy_spectral')
            # plt.imshow(self.image[0], cmap='hot')
            # plt.savefig('ROI_{}.tif'.format(os.path.basename(self.file)))
            #plt.close()
            plt.show()

        for c in range(len(cont)):
            cont[c] = np.array(cont[c])

        return cont, roivolume, masked_detections


def main(directory, scale_list, scale_threshold, n_rings, step):
    """
    Runs the SODA analysis on all images in the chosen directory. Also writes excel files with results.
    :param directory: string; folder from which to analyse every .tif image
    """
    results_workbook = xlsxwriter.Workbook("PySODA_results_{}.xlsx".format(os.path.basename(directory)), {'nan_inf_to_errors': True})
    results01 = results_workbook.add_worksheet(name="SODA")
    results00 = results_workbook.add_worksheet(name="SODAseul 0")
    results11 = results_workbook.add_worksheet(name="SODAseul 1")
    worksheets = [results01, results00, results11]
    titles = ['File', 'Spots in channel 0', 'Spots in channel 1', 'Number of couples',
              'Coupling index 0', 'Coupling index 1', 'Weighted mean coupling distance',
              'Unweighted mean coupling distance']
    row = 1
    for sheet in worksheets:
        for t in range(len(titles)):
            sheet.write(0, t, titles[t])

    for file in os.listdir(directory):
        if file.lower().endswith('.tif') or file.lower().endswith('.tiff'):
            print("*****************Computing SODA*****************")
            print(file)
            file_path = os.path.join(directory, file)

            try:
                soda = SodaImageAnalysis(file_path, scale_list, scale_threshold)

                prob_write, spots0, spots1, CIndex, mean_dist, raw_mean_dist, coupling, n_couples = soda.analysis(0, 1,
                                                                                                                  n_rings,
                                                                                                                  step)

                elemlist = [file, spots0, spots1, n_couples, CIndex[0], CIndex[1], mean_dist, raw_mean_dist]
                for index in range(len(elemlist)):
                    results01.write(row, index, elemlist[index])
            except ValueError:
                print("Skipping invalid TIF file...")

            #print("********Computing SODAseul for channel 0********")
            #prob_write, spots0, spots1, CIndex, mean_dist, raw_mean_dist, coupling, n_couples = soda.analysis(0, 0, 10,
            #                                                                                                  2)
            #elemlist = [file, spots0, spots1, n_couples, CIndex[0], CIndex[1], mean_dist, raw_mean_dist]
            #for index in range(len(elemlist)):
            #    results00.write(row, index, elemlist[index])

            #print("********Computing SODAseul for channel 1********")
            #prob_write, spots0, spots1, CIndex, mean_dist, raw_mean_dist, coupling, n_couples = soda.analysis(1, 1, 10,
            #                                                                                                  2)
            #elemlist = [file, spots0, spots1, n_couples, CIndex[0], CIndex[1], mean_dist, raw_mean_dist]
            #for index in range(len(elemlist)):
            #    results11.write(row, index, elemlist[index])

            row += 1
    results_workbook.close()


if __name__ == '__main__':
    start_time = time.clock()
    try:
        main(DIRECTORY, SCALE_LIST, SCALE_THRESHOLD, N_RINGS, STEP)
    except FileNotFoundError:
        print("Directory is invalid; could not execute SODA.")

    print("--- Running time: %s seconds ---" % (time.clock() - start_time))
