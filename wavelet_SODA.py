import numpy
import math
import sys
import xlsxwriter

from skimage.measure import find_contours, regionprops, label
from skimage.draw import polygon
from matplotlib import pyplot
from scipy import ndimage
import os
import select_ROI
from matplotlib import patches


"""This script is intended to perform a cluster analysis based on the paper

Mapping molecular assemblies with fluorescence microscopy and object-based spatial statistics
Thibault Lagache, Alexandre Grassart, Stéphane Dallongeville, Orestis Faklaris, 
Nathalie Sauvonnet, Alexandre Dufour, Lydia Danglot & Jean-Christophe Olivo-Marin

The first step is to perform the detection of the clusters with wevelet transformation 
of the image and statistical thresholding of wavelets coefficients.
The second sted is to characterize the spatial distribution of the clusters using
a Marked Point Process.
The third and final step is to characterize the spatial relations between the clusters.
The Ripley's K Function will be of great help.

"""


class Detection_Wavelets():
    """
    This is based on the paper
        "Extraction of spots in biological images using multiscale products"
    Some modifications were made to the algorithm:
    .   The threshold in the wavelet function is inverse of what is in the article.
        I had to make these modifications to obtain results that made sense.
    .   The formula for w in the paper is mistaken for W1 = A0 - A1, but is okay for
        Wi = Ai - Ai_1.
    .   Note that the function convolve1d on an array vs on every line on the array
        did not seem to produce the same results.

    :returns : A probability map of clusters
    """
    def __init__(self, img, J_list=(2,), scale_threshold=100):
        """Init function
        :param img: A numpy 2D array
        :param J: The maximum scale
        :param J_list: List of all chosen scales
        """
        self.img = img
        self.J = max(J_list)
        self.J_list = J_list
        self.scale_threshold = scale_threshold

    def computeDetection(self):
        """
        Computes the binary correlation image
        :return: numpy array representing the binary image
        """

        dataIn = numpy.copy(self.img)
        #pyplot.imshow(dataIn)
        #pyplot.title('Base')
        #pyplot.show()
        h, w = dataIn.shape
        prevArray = self.array_to_list(dataIn)

        scales = self.b3WaveletScales2D(prevArray, h, w)

        #for i in scales:
        #    pyplot.imshow(self.list_to_array(i,h,w))
        #    pyplot.title('Convolution')
        #    pyplot.show()
        coefficients = self.b3WaveletCoefficients2D(scales, prevArray, h, w)

        for i in range(len(coefficients)-1):
            coefficients[i] = self.filter_wat(coefficients[i], i, w, h)
        #    pyplot.imshow(self.list_to_array(coefficients[i],h,w))
        #    pyplot.title('Wavelet')
        #    pyplot.show()

        for i in range(len(coefficients[-1])):
            coefficients[-1][i] = 0

        binaryDetectionResult = self.spot_construction(coefficients)
        # pyplot.imshow(self.list_to_array(binaryDetectionResult,h,w))
        # pyplot.title('Correlation')
        # pyplot.show()

        for i in range(len(binaryDetectionResult)):
            if binaryDetectionResult[i] != 0:
                binaryDetectionResult[i] = 255
            else:
                binaryDetectionResult[i] = 0

        # pyplot.imshow(self.list_to_array(binaryDetectionResult,h,w))
        # pyplot.title('Binary')
        # pyplot.show()

        imgout = self.list_to_array(binaryDetectionResult, h, w)

        return imgout

    @staticmethod
    def array_to_list(arrayin):
        """
        Turns 2D numpy array into 1D list
        :param arrayin: 2D numpy array
        :return: 1D list
        """
        listout = []
        datalist = arrayin.tolist()
        for line in datalist:
            for i in line:
                listout.append(i)
        return listout

    @staticmethod
    def list_to_array(listin, h, w):
        """
        Turns 1D list back into 2D numpy array
        :param listin: 1D list
        :param h: Array height
        :param w: Array width
        :return: 2D numpy array
        """
        imgout = numpy.zeros((h,w))
        for y in range(h):
            a = y * w
            for x in range(w):
                if a < len(listin):
                    imgout[y, x] = listin[a]
                a += 1
        return imgout

    def b3WaveletScales2D(self, dataIn, h, w):
        """
        Computes the convolution images for scales J
        :param dataIn: Base image as 1D list
        :param h: image height
        :param w: image width
        :return: List of convoluted images as 1D lists
        """

        prevArray = dataIn.copy()
        resArray = []

        for s in range(1, self.J+1):
            stepS = 2**(s-1)

            currentArray = self.filter_and_swap(prevArray, w, h, stepS)

            if s == 1:
                prevArray = currentArray
                currentArray = []
            else:
                tmp = currentArray
                currentArray = prevArray
                prevArray = tmp

            currentArray = self.filter_and_swap(prevArray, h, w, stepS)
            tmp = currentArray
            currentArray = prevArray
            prevArray = tmp

            resArray.append(prevArray)

        return resArray

    def b3WaveletCoefficients2D(self, scaleCoefficients, originalImage, h, w):
        """
        Computes wavelet images as 1D lists
        :param scaleCoefficients: List of  convoluted images as 1D lists
        :param originalImage: Original image as 1D list
        :param h: Image height
        :param w: Image width
        :return: List of wavelet images as 1D lists
        """

        waveletCoefficients = []
        iterPrev = originalImage.copy()
        for j in range(self.J):
            iterCurrent = scaleCoefficients[j]
            wCoefficients = []
            for i in range(h*w):
                wCoefficients.append(iterPrev[i] - iterCurrent[i])
            waveletCoefficients.append(wCoefficients)
            iterPrev = iterCurrent
        waveletCoefficients.append(scaleCoefficients[self.J-1])
        return waveletCoefficients

    def filter_wat(self, data, depth, width, height):
        """
        Wavelet transform coefficient matrix filter from Icy Spot Detector code
        :param data: image data
        :param depth: number of scale
        :param width: image width
        :param height: image height
        :return: filtered image
        """

        output = data.copy()
        lambdac=[]

        for i in range(self.J+2):
            lambdac.append(numpy.sqrt(2 * numpy.log(width*height / (1 << (2*i)))))

        # mad
        size = width*height
        mean = numpy.mean(data)
        a = 0
        for i in range(len(data)):
            s = data[i] - mean
            a += numpy.abs(s)
        mad = a/size

        # dcoeff = ld?
        dcoeff = (self.scale_threshold/100.0)

        coeffThr = (lambdac[depth+1] * mad)/dcoeff

        for i in range(len(data)):
            if data[i] < coeffThr:
                output[i] = 0

        return output

    def spot_construction(self, inputCoefficients):
        """
        Reconstructs correlation image
        :param inputCoefficients: List of wavelet coefficient images as 1D lists
        :return: Correlation image as 1D list
        """

        output = []
        for i in range(len(inputCoefficients[0])):
            allNotNull = True
            v = 0
            for j in range(self.J):
                if j+1 in self.J_list:
                    if inputCoefficients[j][i] == 0:
                        allNotNull = False
                    v += inputCoefficients[j][i]

            if allNotNull:
                output.append(v)
            else:
                output.append(0)

        return output

    def filter_and_swap(self, arrayIn, w, h, stepS):
        #print(len(arrayIn))

        arrayOut = arrayIn.copy()

        w2 = 1/16
        w1 = 1/4
        w0 = 3/8

        w0idx = 0

        for y in range(h):

            arrayOutIter = 0 + y
            w1idx1 = w0idx + stepS - 1
            w2idx1 = w1idx1 + stepS
            w1idx2 = w0idx + stepS
            w2idx2 = w1idx2 + stepS

            cntX = 0
            while cntX < stepS:
                arrayOut[arrayOutIter] = w2*((arrayIn[w2idx1]) + (arrayIn[w2idx2])) + \
                                         w1*((arrayIn[w1idx1]) + (arrayIn[w1idx2])) + \
                                         w0*(arrayIn[w0idx])
                w1idx1 -= 1
                w2idx1 -= 1
                w1idx2 += 1
                w2idx2 += 1
                w0idx += 1
                arrayOutIter += h
                cntX += 1
            w1idx1 += 1

            while cntX < 2*stepS:
                arrayOut[arrayOutIter] = w2*((arrayIn[w2idx1]) + (arrayIn[w2idx2])) + \
                                         w1*((arrayIn[w1idx1]) + (arrayIn[w1idx2])) + \
                                         w0*(arrayIn[w0idx])
                w1idx1 += 1
                w2idx1 -= 1
                w1idx2 += 1
                w2idx2 += 1
                w0idx += 1
                arrayOutIter += h
                cntX += 1
            w2idx1 += 1

            while cntX < (w - 2*stepS):
                arrayOut[arrayOutIter] = w2*((arrayIn[w2idx1]) + (arrayIn[w2idx2])) + \
                                         w1*((arrayIn[w1idx1]) + (arrayIn[w1idx2])) + \
                                         w0*(arrayIn[w0idx])
                w1idx1 += 1
                w2idx1 += 1
                w1idx2 += 1
                w2idx2 += 1
                w0idx += 1
                arrayOutIter += h
                cntX += 1
            w2idx2 -= 1

            while cntX < (w - stepS):
                arrayOut[arrayOutIter] = w2*((arrayIn[w2idx1]) + (arrayIn[w2idx2])) + \
                                         w1*((arrayIn[w1idx1]) + (arrayIn[w1idx2])) + \
                                         w0*(arrayIn[w0idx])
                w1idx1 += 1
                w2idx1 += 1
                w1idx2 += 1
                w2idx2 -= 1
                w0idx += 1
                arrayOutIter += h
                cntX += 1
            w1idx2 -= 1

            while cntX < w:
                arrayOut[arrayOutIter] = w2*((arrayIn[w2idx1]) + (arrayIn[w2idx2])) + \
                                         w1*((arrayIn[w1idx1]) + (arrayIn[w1idx2])) + \
                                         w0*(arrayIn[w0idx])
                w1idx1 += 1
                w2idx1 += 1
                w1idx2 -= 1
                w2idx2 -= 1
                w0idx += 1
                arrayOutIter += h
                cntX += 1

        return arrayOut


class Spatial_Distribution():
    """ This is based on the Marked Point Process as shown in the cited above paper
    Marked is the attributes of the cluster (shape, size, color)
    Point Process is the position of the clusters (centroid)

    :returns : A list of tuple (polygon, area, centroid)
    NOTE. Centroid is (y, x) coordinates
    """
    def __init__(self, prob_map, img, cs=0):
        """This is the init function
        :param prob_map: A 2D numpy array of boolean detected clusters
        :param cs: Minimum area of a cluster
        """
        self.img = img
        self.P = prob_map
        self.P[self.P > 0] = 1
        self.cs = cs

    def mark(self):
        """ This function creates the Marked Point Processed of every clusters
        """

        # Label each spot for regionprops
        labels = label(self.P, connectivity=2)

        props = regionprops(labels, intensity_image=self.img)

        mark = []
        for p in props:
            nonzero = (p.area != 0)

            if nonzero:  # Make sure that detected contours are not holes in actual contours
                try:
                    s = p.area
                    min_axis = p.minor_axis_length
                    cent_int = p.weighted_centroid

                    # TODO: Put back the shape filter
                    # if s > self.cs and min_axis > 0:  # Make sure that the spot has a good enough shape to be analysed
                    if s > self.cs:
                        mark.append(
                            (p, s, cent_int)
                        )

                except IndexError:
                    print('Index error: spot was ignored.')

        return mark

    # def mark(self):
    #     """ This function creates the Marked Point Processed of every clusters
    #     """
    #
    #     # Pad the image so contours along the maximum edges are well detected
    #     P_pad = numpy.pad(self.P, ((0,2), (0,2)), mode='constant', constant_values=0)
    #     img_pad = numpy.pad(self.img, ((0, 2), (0, 2)), mode='constant', constant_values=0)
    #
    #     contours = []
    #     rois = []
    #
    #     # Label each spot with a different value to isolate them and create accurate contours
    #     labels = ndimage.label(P_pad, structure=[[1,1,1],
    #                                              [1,1,1],
    #                                              [1,1,1]])
    #     for lab in range(1, numpy.max(labels[0])+1):
    #         roipoly = numpy.zeros(P_pad.shape)
    #         roipoly[labels[0] == lab] = 1
    #         roipoly = roipoly.astype('uint8')
    #         cont = find_contours(roipoly, 0, fully_connected='high')
    #         contours.append(cont[0])
    #         rois.append(roipoly)
    #
    #     mark = []
    #     # pat = []
    #     roi_i = 0
    #     for c in contours:
    #         roipoly = rois[roi_i]
    #         nonzeros = numpy.count_nonzero(roipoly)
    #
    #         if nonzeros:  # Make sure that detected contours are not holes in actual contours
    #             try:
    #                 props = regionprops(roipoly, intensity_image=img_pad)
    #                 s = nonzeros
    #                 cent_int = props[0].weighted_centroid
    #                 cent_mass = props[0].centroid
    #                 #pyplot.imshow(roipoly*framedI)
    #                 #pyplot.show()
    #
    #                 if s > self.cs:
    #                     mark.append(
    #                         (c, s, cent_int, cent_mass)
    #                     )
    #                     # pat.append(patches.Polygon(numpy.fliplr(c), color='lime', fill=False, alpha=0.5))
    #
    #             except IndexError:
    #                 print('Index error: spot was ignored.')
    #
    #         roi_i += 1
    #
    #     # fig, ax = pyplot.subplots()
    #     # ax.imshow((self.P*self.img).astype('uint16'))
    #     # pyplot.draw()
    #     # for p in pat:
    #     #     ax.add_patch(p)
    #     # pyplot.show()
    #
    #     return mark

    def PolyArea(self, x, y):
        """ This function computes the area of a cluster
        :param x: A numpy array of x coordinates
        :param y: A numpy array of y coordinates
        """
        return 0.5*numpy.abs(numpy.dot(x,numpy.roll(y,1))-numpy.dot(y,numpy.roll(x,1)))


class Spatial_Relations():
    """ This is based on the the paper cited above
    To characterise the spatial relations between two populations A1 (green) and A2 (red) of
    objects (spots or localisations), we use the Ripley’s K function, a gold standard for analysing
    the second-order properties (i.e., distance to neigh- bours) of point processes.
    """
    def __init__(self, MPP1, MPP2, sROI, roivolume, poly, img, n_rings, step, filename):
        """ The init function
        :param MPP1: The marked point process of every clusters in the first channel
        :param MPP2: The marked point process of every clusters in the second channel
        :param sROI: A tuple (y, x) of the window size
        """

        self.img = img
        self.filename = filename

        self.MPP1 = MPP1
        self.MPP2 = MPP2

        self.ROIarea = roivolume
        self.poly = poly

        # self.MPP1_ROI = self.mpp_in_contours(self.MPP1)
        # self.MPP2_ROI = self.mpp_in_contours(self.MPP2)
        self.MPP1_ROI = self.MPP1
        self.MPP2_ROI = self.MPP2
        print(len(self.MPP1_ROI), len(self.MPP2_ROI))

        self.sROI = sROI
        self.max_dist = n_rings*step
        self.pas = step

        self.rings = numpy.array([r*step for r in range(0, n_rings+1, 1)])
        self.imgw1 = self.image_windows(self.MPP1_ROI)
        self.imgw2 = self.image_windows(self.MPP2_ROI)

        self.neighbors = self.nearest_neighbors()

        self.distance_fit, self.N_fit = self.dist_fit()

    def matrix_dist(self, X, Y):
        return numpy.sqrt((X[:, 0] - Y[:, 0]) ** 2 + (X[:, 1] - Y[:, 1]) ** 2)

    def nearest_neighbors(self):
        """
        Finds the nearest neighbor in both MPPs of each spot and the distance between them
        :return: List of tuples (dist. to neighbor, neighbor index) for each combination of channels
        """

        mpp1_data = numpy.zeros((len(self.MPP1_ROI), 2))
        mpp2_data = numpy.zeros((len(self.MPP2_ROI), 2))

        for i in range(len(self.MPP1_ROI)):
            p, s, (y, x) = self.MPP1_ROI[i]
            mpp1_data[i, 0] = x
            mpp1_data[i, 1] = y

        for i in range(len(self.MPP2_ROI)):
            p, s, (y, x) = self.MPP2_ROI[i]
            mpp2_data[i, 0] = x
            mpp2_data[i, 1] = y

        # MPP1 with MPP1
        distances = []
        min_distance = []
        for x in mpp1_data:
            repmat = numpy.tile(x, (mpp1_data.shape[0], 1))
            distance = self.matrix_dist(repmat, mpp1_data)
            distance_pos = distance[numpy.where((distance > 0))]
            distances.append(distance)
            m = distance_pos.min()
            m_index = numpy.where((distance == m))[0][0]
            min_distance.append((m, m_index))
        min_dist_11 = min_distance

        # MPP1 with MPP2
        distances = []
        min_distance = []
        for x in mpp1_data:
            repmat = numpy.tile(x, (mpp2_data.shape[0], 1))
            distance = self.matrix_dist(repmat, mpp2_data)
            distance_pos = distance[numpy.where((distance > 0))]
            distances.append(distance)
            m = distance_pos.min()
            m_index = numpy.argmin(distance)
            min_distance.append((m, m_index))
        min_dist_12 = min_distance

        # MPP2 with MPP1
        distances = []
        min_distance = []
        for x in mpp2_data:
            repmat = numpy.tile(x, (mpp1_data.shape[0], 1))
            distance = self.matrix_dist(repmat, mpp1_data)
            distance_pos = distance[numpy.where((distance > 0))]
            distances.append(distance)
            m = distance_pos.min()
            m_index = numpy.argmin(distance)
            min_distance.append((m, m_index))
        min_dist_21 = min_distance

        # MPP2 with MPP2
        min_distance = []
        distances = []
        for y in mpp2_data:
            repmat = numpy.tile(y, (mpp2_data.shape[0], 1))
            distance = self.matrix_dist(repmat, mpp2_data)
            distance_pos = distance[numpy.where((distance > 0))]
            distances.append(distance)
            m = distance_pos.min()
            m_index = numpy.where((distance == m))[0][0]
            min_distance.append((m, m_index))
        min_dist_22 = min_distance

        return min_dist_11, min_dist_12, min_dist_21, min_dist_22

        #tree1 = scipy.spatial.KDTree(mpp1_data)
        #tree2 = scipy.spatial.KDTree(mpp2_data)

        # neighbors_MPP11 = []
        # neighbors_MPP12 = []
        # neighbors_MPP22 = []
        #
        # # Spots in channel 1 with spots in both channels
        # for spot in self.MPP1:
        #     neighbor_MPP1 = self.nearest_neighbors(spot, self.MPP1)
        #     neighbor_MPP2 = self.nearest_neighbors(spot, self.MPP2)
        #     neighbors_MPP11.append((spot, neighbor_MPP1))
        #     neighbors_MPP12.append((spot, neighbor_MPP2))
        #
        # # Spots in channel 2 with spots in channel 2
        # for spot in self.MPP1:
        #     neighbor_MPP2 = self.nearest_neighbors(spot, self.MPP2)
        #     neighbors_MPP22.append((spot, neighbor_MPP2))
        #
        # return neighbors_MPP11, neighbors_MPP12, neighbors_MPP22

    def dist_fit(self):
        """ Allows to generate the distance_fit and N_fit as in the paper
        """
        distance_fit = []
        distance_fit.append(0)
        temp = distance_fit[0]
        while temp + self.pas <= self.max_dist:
            temp += self.pas
            distance_fit.append(temp)
        N_fit = len(distance_fit)
        if N_fit == 1:
            distance_fit.append(self.max_dist)
            N_fit = len(distance_fit)

        return distance_fit, N_fit

    def image_windows(self, points):
        """ This function implements the image windows as proposed in the java script
        version of the algorithm.
        """
        h, w = self.sROI
        imagewindows = [[[] for i in range(int(h / self.max_dist) + 1)] for j in range((int(w / self.max_dist) + 1))]
        dist_to_border = self.nearest_contour(points)
        for k in range(len(points)):
            p, s, (y, x) = points[k]
            j, i = int(y / self.max_dist), int(x / self.max_dist)
            imagewindows[i][j].append((y, x, s, dist_to_border[k]))
        return imagewindows

    def correlation_new(self):
        """
        This function computes the Ripley's Correlation as in Ripley2D.java (G)
        """
        result = numpy.zeros((3, self.N_fit - 1))
        delta_K = numpy.zeros(self.N_fit - 1)
        d_mean = numpy.zeros(self.N_fit - 1)
        d_mean_2 = numpy.zeros(self.N_fit - 1)
        count = numpy.zeros(self.N_fit - 1)
        ROIy, ROIx = self.sROI
        n1, n2 = len(self.MPP1_ROI), len(self.MPP2_ROI)
        for i in range(len(self.imgw1)):
            for j in range(len(self.imgw1[i])):
                for y1, x1, s1, db1 in self.imgw1[i][j]:
                    # d = self.distance_pl(self.poly,x1,y1)
                    # d = self.dist_to_contour(x1, y1)
                    d = db1
                    # d = min(x1, ROIx - x1, y1, ROIy - y1) # min distance from ROI
                    for k in range(max(i - 1, 0), min(i + 1, len(self.imgw1) - 1) + 1):
                        for l in range(max(j - 1, 0), min(j + 1, len(self.imgw1[i]) - 1) + 1):
                            for y2, x2, s2, db2 in self.imgw2[k][l]:
                                temp = self.distance(x1,x2,y1,y2) # distance
                                if (y1, x1, s1) != (y2, x2, s2):
                                    weight = 1 # weight
                                    if temp > d:
                                        weight = 1 - (numpy.arccos(d / temp)) / numpy.pi
                                    for m in range(1, self.N_fit):
                                        if (temp < self.distance_fit[m]) & (temp >= self.distance_fit[0]):
                                            delta_K[m - 1] += (1 / weight) * self.ROIarea / (n1 * n2)
                                            count[m - 1] += 1
                                            d_mean[m - 1] += temp
                                            d_mean_2[m - 1] += temp**2
                                            break

        for l in range(self.N_fit - 1):
            result[0, l] = delta_K[l]
            if count[l] > 0:
                result[1, l] = d_mean[l] / count[l]
                result[2, l] = d_mean_2[l] / count[l]
            else:
                result[1, l] = 0
                result[2, l] = 0
        return result

    def nearest_contour(self, mpp):
        """
        Finds the distance between every point and the ROI border
        Because the distance is point to point, the ROI must have points all along its contour for this to be accurate.
        :param mpp: Marked Point Process of a single channel (list of points)
        :return: List of distances
        """
        mpp_data = numpy.zeros((len(mpp), 2))

        poly_sum = numpy.concatenate(self.poly)

        poly_data = numpy.zeros((len(poly_sum), 2))

        for i in range(len(mpp)):
            p, s, (y, x) = mpp[i]
            mpp_data[i, 0] = x
            mpp_data[i, 1] = y

        for i in range(len(poly_sum)):
            y, x = poly_sum[i]
            poly_data[i, 0] = x
            poly_data[i, 1] = y

        distances = []
        min_distance = []
        for x in mpp_data:
            repmat = numpy.tile(x, (poly_data.shape[0], 1))
            distance = self.matrix_dist(repmat, poly_data)
            distances.append(distance)
            m = distance.min()
            min_distance.append(m)

        return min_distance

    def dist_to_contour(self, x, y):
        '''
        Meant to calculate the distance between a point and the edge of a ROI if
        the ROI is composed of multiple contours
        :param x: x coordinate of the point
        :param y: y coordinate of the point
        :return:
        '''
        dist = sys.maxsize
        min_dist = dist

        for c in self.poly:
            min_dist = self.distance_pl(c, x, y)
            min_dist = min(dist, min_dist)

        return min_dist

    def mpp_in_poly(self, MPP, poly):
        """
        This function checks which spots are in the selected polygon
        :param MPP: Marked Point Process; list of marks
        :param poly: A list of (x, y) coordinates corresponding to a polygon's (ROI) vertexes
        :return: List of marks in the ROI
        """
        MPP_ROI = []
        for point in MPP:
            y, x = point[2]

            n = len(poly)
            inside = False

            p1y, p1x = poly[0]
            for i in range(n + 1):
                p2y, p2x = poly[i % n]
                if y > min(p1y, p2y):
                    if y <= max(p1y, p2y):
                        if x <= max(p1x, p2x):
                            if p1y != p2y:
                                xints = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                            else:
                                xints = None
                            if p1x == p2x or x <= xints:
                                inside = not inside
                p1x, p1y = p2x, p2y
            if inside:
                MPP_ROI.append(point)

        return MPP_ROI

    def mpp_in_contours(self, MPP):

        polist = []
        for poly in self.poly:
            polist += self.mpp_in_poly(MPP, poly)

        return polist

    def variance_theo_delta_new(self):
        """ This function implements the computation of the standard deviation of G
        """
        n1, n2 = len(self.MPP1_ROI), len(self.MPP2_ROI)
        mu = self.mean_G()
        result = numpy.zeros(self.N_fit - 1)
        results, N_h = self.betaCorrection(self.max_dist/10, 100)
        # results = numpy.zeros(results.shape)
        ROIy, ROIx = self.sROI

        for k in range(1, self.N_fit):
            distancek_1 = self.distance_fit[k - 1]
            distancek = self.distance_fit[k]

            d2 = distancek_1**2
            d2bis = distancek**2
            e1 = numpy.pi * (distancek_1**2)
            e2 = numpy.pi * (distancek**2)

            temp_A1, temp_A2, temp_A3 = 0, 0, 0

            sum_h_a, sum_h_a_bis = 0, 0

            for i in range(len(self.imgw1)):
                for j in range(len(self.imgw1[i])):
                    for y1, x1, s1, db1 in self.imgw1[i][j]:
                        # dist = self.distance_pl(self.poly, x1, y1)
                        # dist = self.dist_to_contour(x1, y1)
                        # dist = min(x1, ROIx - x1, y1, ROIy - y1)  # min distance from ROI
                        dist = db1
                        if (dist < distancek) & (dist >= self.distance_fit[0]):
                            sum_h_a += results[math.ceil(N_h * dist / distancek)]
                        else:
                            sum_h_a += 1
                        if k > 1:
                            if (dist < distancek_1) & (dist > self.distance_fit[0]):
                                sum_h_a_bis += results[math.ceil(N_h * dist / distancek_1)]
                            else:
                                sum_h_a_bis += 1

                        for m in range(max(i - 1, 0), min(i + 1, len(self.imgw1) - 1) + 1):
                            for l in range(max(j - 1, 0), min(j + 1, len(self.imgw1[i]) - 1) + 1):
                                for y2, x2, s2, db2 in self.imgw1[m][l]:
                                    distance_ij = self.distance(x1, x2, y1, y2)
                                    if distance_ij > 0:
                                        if distance_ij < 2 * distancek_1:
                                            temp1 = 2 * d2 * numpy.arccos(distance_ij / (2 * distancek_1))
                                            temp2 = 0.5 * distance_ij * numpy.sqrt(4 * d2 - distance_ij**2)
                                            temp_A1 += temp1 - temp2
                                        if distance_ij < 2 * distancek:
                                            temp1 = 2 * distancek**2 * numpy.arccos(distance_ij / (2 * distancek))
                                            temp2 = 0.5 * distance_ij * numpy.sqrt(4 * distancek**2 - distance_ij**2)
                                            temp_A2 += temp1 - temp2
                                        if distance_ij < distancek_1 + distancek:
                                            if distance_ij + distancek_1 < distancek:
                                                temp_A3 += 2 * numpy.pi * d2
                                            else:
                                                temp1 = d2 * numpy.arccos((distance_ij**2 + d2 - d2bis) / (2 * distance_ij * distancek_1))
                                                temp2 = d2bis * numpy.arccos((distance_ij**2 + d2bis - d2) / (2 * distance_ij * distancek))
                                                temp3 = 0.5 * numpy.sqrt((- distance_ij + distancek_1 + distancek)
                                                                         * (distance_ij - distancek_1 + distancek)
                                                                         * (distance_ij + distancek_1 - distancek)
                                                                         * (distance_ij + distancek_1 + distancek))
                                                temp_A3 += 2 * (temp1 + temp2 - temp3)
            I2 = (temp_A1 + temp_A2 - temp_A3 - (e1**2 / self.ROIarea + e2**2 / self.ROIarea - 2 * e1 * e2 / self.ROIarea) * (n1 * (n1 - 1))) * n2 / self.ROIarea
            I1 = (e2 * sum_h_a - e1 * sum_h_a_bis - n1 * (e2 - e1)**2 / self.ROIarea) * n2 / self.ROIarea
            result[k - 1] = (self.ROIarea / (n2 * n1))**2 * (I1 + I2)
        return result

    def intersection2D_new(self):
        """
        This function computes the A matrix
        """
        n1, n2 = len(self.MPP1_ROI), len(self.MPP2_ROI)
        A = numpy.zeros((self.N_fit - 1, self.N_fit - 1))
        for r in range(self.N_fit - 1):
            A[r, r] = n1
        S = numpy.zeros((self.N_fit, self.N_fit))

        for k1 in range(0, self.N_fit):
            for k2 in range(0, self.N_fit):
                m = min(self.distance_fit[k1], self.distance_fit[k2])  # create min
                M = max(self.distance_fit[k1], self.distance_fit[k2])  # create max

                for i in range(len(self.imgw1)):
                    for j in range(len(self.imgw1[i])):
                        for y1, x1, s1, db1 in self.imgw1[i][j]:
                            for o in range(max(i - 1, 0), min(i + 1, len(self.imgw1) - 1) + 1):
                                for l in range(max(j - 1, 0), min(j + 1, len(self.imgw1[i]) - 1) + 1):
                                    for y2, x2, s2, db2 in self.imgw1[o][l]:
                                        d = self.distance(x1, x2, y1, y2)
                                        if d > 0:
                                            if m + d < M:
                                                S[k1, k2] = S[k1, k2] + numpy.pi * m ** 2
                                            else:
                                                if d < (m + M):
                                                    temp1 = m ** 2 * numpy.arccos((d ** 2 + m ** 2 - M ** 2) / (2 * d * m))
                                                    temp2 = M ** 2 * numpy.arccos((d ** 2 + M ** 2 - m ** 2) / (2 * d * M))
                                                    temp3 = 0.5 * numpy.sqrt((-d + m + M) * (d + m - M) * (d - m + M) * (d + m + M))
                                                    S[k1, k2] = S[k1, k2] + temp1 + temp2 - temp3

        for i in range(self.N_fit - 1):
            for j in range(self.N_fit - 1):
                vol = numpy.pi * (self.distance_fit[j + 1] ** 2 - self.distance_fit[j] ** 2)
                A[i, j] = A[i, j] + (S[i + 1, j + 1] + S[i, j] - S[i, j + 1] - S[i + 1, j]) / vol
        return A

    def reduced_Ripley_vector(self, **kwargs):
        """
        Vector G0 from paper. This function returns the estimation of the coupling probability
        """
        n1, n2 = len(self.MPP1_ROI), len(self.MPP2_ROI)
        if kwargs:
            G = kwargs["G"][0, :]
            var = kwargs["var"]
            A = kwargs["A"]
        else:
            G = self.correlation_new()[0, :]
            var = self.variance_theo_delta_new()
            A = self.intersection2D_new()
        mean = self.mean_G()

        A_b = A/n1

        G0 = numpy.dot(numpy.linalg.inv(A_b), G - mean) / numpy.sqrt(var)

        return G0

    def draw_G0(self, G0):
        T = numpy.sqrt(2 * numpy.log(len(G0)))
        pyplot.bar(self.rings[1:], G0)
        pyplot.axhline(y=T, color='red', linestyle='dashed')
        pyplot.title("G0")
        pyplot.show()

    def main2D_corr(self, G, var, A):
        """
        Currently unused
        P calculation from the SODA code (non_parametric_object.java)
        Works; very nearly gives the same results as coupling_prob
        """
        n1, n2 = len(self.MPP1_ROI), len(self.MPP2_ROI)
        delta_K = G[0,:]
        sigma_T = numpy.zeros([self.N_fit - 1])
        Num = delta_K * (n1*n2/self.ROIarea)
        mu_tab = self.mean_G() * (n1*n2/self.ROIarea)
        for p in range(self.N_fit-1):
            if p > 0:
                T_p = (numpy.sqrt(2 * numpy.log(p+1)))
            else:
                T_p = numpy.sqrt(2)
            sigma_T[p] = (n1*n2*T_p/self.ROIarea)*numpy.sqrt(var[p])

        A_matb = A * 1/n1
        A_matb_inverse = numpy.linalg.inv(A_matb)
        C = numpy.dot(A_matb_inverse, (Num - mu_tab))

        for p in range(self.N_fit-1):
            if C[p]<sigma_T[p]:
                C[p] = 0

        proba_dist = []
        for i in range(self.N_fit-1):
            if Num[i] > 0:
                proba_dist.append(C[i]/Num[i])
            else:
                proba_dist.append(0)

        return proba_dist

    def coupling_prob(self, **kwargs):
        """
        This function computes the coupling probability between the two channels
        """
        n1, n2 = len(self.MPP1_ROI), len(self.MPP2_ROI)
        if kwargs:
            G = kwargs["G"][0, :]
            var = kwargs["var"]
            A = kwargs["A"]
            G0 = kwargs["G0"]
        else:
            G = self.correlation_new()[0, :]
            var = self.variance_theo_delta_new()
            A = self.intersection2D_new()
            G0 = self.reduced_Ripley_vector(**kwargs)
        mean = self.mean_G()

        # T = [numpy.sqrt(2 * numpy.log(i+1)) if i > 0 else numpy.sqrt(2) for i in range(self.N_fit)]
        T = numpy.sqrt(2 * numpy.log(len(G0)))
        coupling = numpy.array([
            (numpy.sqrt(var[i]) * G0[i]) / G[i] if G0[i] > T else 0 for i in range(len(self.rings)-1)
        ])
        probability = []

        prob_write = []
        for c1, s1, (y1, x1) in self.MPP1_ROI:
            for c2, s2, (y2, x2) in self.MPP2_ROI:
                dist = self.distance(x1, x2, y1, y2)
                p = 0
                if (s1, (y1, x1)) != (s2, (y2, x2)):
                    for i in range(len(self.rings) - 1):
                        if (dist >= self.rings[i]) & (dist < self.rings[i + 1]):
                            p += coupling[i]
                            probability.append([p, dist])

                # Prepare information list for writing in excel file
                if p != 0:
                    problist = [x1, y1, x2, y2, dist, p]
                    prob_write.append(problist)

        # self.data_boxplot(prob_write)

        n_couples = len(probability)
        probability = numpy.array(probability)
        probability = probability[probability[:,0] > 0]
        coupling_index = numpy.sum(probability, axis=0)
        mean_coupling_distance = 0
        if coupling_index[0] > 0:
            mean_coupling_distance = numpy.sum(numpy.prod(probability, axis=1)) / coupling_index[0]
        raw_mean_distance = coupling_index[1]/probability[:,1].shape[0]  # Icy/SODA returns this, unlike the paper
        return prob_write, (coupling_index[0] / n1, coupling_index[0] / n2), mean_coupling_distance, raw_mean_distance, coupling, n_couples

    def data_boxplot(self, prob_write):
        """
        Plots a boxplot of a spot property by channels and coupling
        :param prob_write: List of couples and their properties
        """
        fig, axs = pyplot.subplots(1,2)

        dataC = []
        dataU = []
        for p1, s1, (y1, x1) in self.MPP1_ROI:
            coupled = False
            for xa, ya, xb, yb, dist, p in prob_write:
                if (y1, x1) == (ya, xa):
                    dataC.append(p1.inertia_tensor_eigvals[0])
                    coupled = True
            if not coupled:
                dataU.append(p1.inertia_tensor_eigvals[0])

        data1 = [dataC, dataU]

        dataC = []
        dataU = []
        for p1, s1, (y1, x1) in self.MPP2_ROI:
            coupled = False
            for xa, ya, xb, yb, dist, p in prob_write:
                if (y1, x1) == (yb, xb):
                    dataC.append(p1.inertia_tensor_eigvals[0])
                    coupled = True
            if not coupled:
                dataU.append(p1.inertia_tensor_eigvals[0])

        data2 = [dataC, dataU]

        axs[0].boxplot(data1, showmeans=True)
        axs[1].boxplot(data2, showmeans=True)

        pyplot.title('Eccentricity')
        axs[0].set_title('Channel 0')
        axs[1].set_title('Channel 1')

        pyplot.savefig('boxplot_{}'.format(self.filename))
        pyplot.close()

    def data_scatter(self, mean_coupling_dist, prob_write):
        """
        Plots a scatter plot of two spot proprieties. Saves the resulting plot in a .png
        :param mean_coupling_dist: Mean coupling distance
        :param prob_write: List of couples and their properties
        """
        X = []
        Y = []

        blue = []
        dist_tot = 0

        # for i in range(len(self.MPP1_ROI)):
        #     p1, s1, (y, x) = self.MPP1_ROI[i]
        #     for xa, ya, xb, yb, dist, p in prob_write:
        #         if (y, x) == (ya, xa):
        #             X.append(dist)
        #             Y.append(p1.eccentricity)
        #             blue.append('blue')
        #
        # red = []
        # for i in range(len(self.MPP2_ROI)):
        #     p1, s1, (y, x) = self.MPP2_ROI[i]
        #     for xa, ya, xb, yb, dist, p in prob_write:
        #         if (y, x) == (yb, xb):
        #             X.append(dist)
        #             Y.append(p1.eccentricity)
        #             red.append('red')

        for i in range(len(self.MPP1_ROI)):
            p1, s1, (y,x) = self.MPP1_ROI[i]
            neighbor = self.neighbors[1][i][1]
            #r1, c1 = p1.centroid
            #r2, c2 = p1.weighted_centroid
            #d = self.distance(c1, c2, r1, r2)
            p2, s2, (y2, x2) = self.MPP2_ROI[neighbor]
            Y.append(p1.eccentricity)
            X.append(self.neighbors[0][i][0])
            dist_tot += self.neighbors[0][i][0]
            blue.append('blue')

        red = []
        for i in range(len(self.MPP2_ROI)):
            p1, s1, (y,x) = self.MPP2_ROI[i]
            neighbor = self.neighbors[2][i][1]
            #r1, c1 = p1.centroid
            #r2, c2 = p1.weighted_centroid
            #d = self.distance(c1, c2, r1, r2)
            p2, s2, (y2, x2) = self.MPP1_ROI[neighbor]
            Y.append(p1.eccentricity)
            X.append(self.neighbors[3][i][0])
            dist_tot += self.neighbors[3][i][0]
            red.append('red')

        dist_mean = dist_tot/(len(red)+len(blue))
        pyplot.scatter(X, Y, color=blue+red, marker='.', linewidths=0.5, edgecolors='black')
        pyplot.axvline(x=mean_coupling_dist, color='green', linestyle='dashed')
        pyplot.axvline(x=dist_mean, color='cyan', linestyle='dashed')
        pyplot.xlabel('Distance to nearest neigbor (Pixels)')
        pyplot.ylabel('Eccentricity')
        pyplot.savefig('scatter_{}'.format(self.filename))
        pyplot.close()
        #pyplot.show()

    def write_spots_and_probs(self, prob_write, directory, title):
        """
        Writes informations about couples and single spots
        :param prob_write: list containing lists of information to write about each couple
        :param directory: string containing the path of the output file
        :param title: name of the output excel file as string
        """
        workbook = xlsxwriter.Workbook(os.path.join(directory, title), {'nan_inf_to_errors': True})
        couples = workbook.add_worksheet(name='Couples')
        titles = ['X1', 'Y1', 'X2', 'Y2', 'Distance', 'Coupling probability']
        for t in range(len(titles)):
            couples.write(0, t, titles[t])

        row = 1
        for p_list in prob_write:
            for index in range(len(p_list)):
                couples.write(row, index, p_list[index])
            row += 1

        spots1 = workbook.add_worksheet(name="Spots ch0")
        spots2 = workbook.add_worksheet(name="Spots ch1")
        titles = ['X1', 'Y1', 'Area', 'Distance to Neighbor Ch0', 'Distance to Neighbor Ch1', 'Eccentricity',
                  'Max intensity', 'Min intensity', 'Mean intensity',
                  'Major axis length', 'Minor axis length', 'Orientation',
                  'Perimeter', 'Coupled']
        for t in range(len(titles)):
            spots1.write(0, t, titles[t])
        row = 1
        for (p1, s1, (y1, x1)) in self.MPP1_ROI:
            coupled = 0
            for xa, ya, xb, yb, dist, p in prob_write:
                if (y1, x1) == (ya, xa):
                    coupled = 1
            dnn1, nn1 = self.neighbors[0][row-1]
            dnn2, nn2 = self.neighbors[1][row-1]
            datarow = [x1, y1, s1, dnn1, dnn2, p1.eccentricity,
                       p1.max_intensity, p1.min_intensity, p1.mean_intensity,
                       p1.major_axis_length, p1.minor_axis_length, p1.orientation,
                       p1.perimeter, coupled]
            for i in range(len(datarow)):
                spots1.write(row, i, datarow[i])
            row += 1

        titles = ['X2', 'Y2', 'Area', 'Distance to Neighbor Ch0', 'Distance to Neighbor Ch1', 'Eccentricity',
                  'Max intensity', 'Min intensity', 'Mean intensity',
                  'Major axis length', 'Minor axis length', 'Orientation',
                  'Perimeter', 'Coupled']
        for t in range(len(titles)):
            spots2.write(0, t, titles[t])
        row = 1
        for p2, s2, (y2, x2) in self.MPP2_ROI:
            coupled = 0
            for xc, yc, xd, yd, dist, p in prob_write:
                if (y2, x2) == (yd, xd):
                    coupled = 1
            dnn1, nn1 = self.neighbors[2][row-1]
            dnn2, nn2 = self.neighbors[3][row-1]
            datarow = [x2, y2, s2, dnn1, dnn2, p2.eccentricity,
                       p2.max_intensity, p2.min_intensity, p2.mean_intensity,
                       p2.major_axis_length, p2.minor_axis_length, p2.orientation,
                       p2.perimeter, coupled]
            for i in range(len(datarow)):
                spots2.write(row, i, datarow[i])
            row += 1
        try:
            workbook.close()
        except PermissionError:
            print("Warning: Workbook is open and couldn't be overwritten!")

    def betaCorrection(self, step, nbN):
        """ This function computes the boundary condition
        betaCorrection(maxdist / 10, 100);
        """
        valN = 1/(1/nbN)
        N_h = int(valN) + 1

        alpha = numpy.zeros(N_h + 1)
        results = numpy.zeros(N_h + 1)
        for i in range(1, results.size):
            alpha[i] = (i / N_h)
        for i in range(results.size):
            j = 2
            h = alpha[i] + step
            while h <= 1:
                results[i] = results[i] + h * step / (1 - 1 / numpy.pi * numpy.arccos(alpha[i] / h))
                h = alpha[i] + j * step
                j += 1
            results[i] = results[i] * 2 + alpha[i] * alpha[i]
        return results, N_h

    def distance(self, x1, x2, y1, y2):
        """ This function computes the distance between two points
        :param x1: x coordinates of first point
        :param x2: x coordinates of the second point
        :param y1: y coordinates of the first point
        :param y2: y coordinates of the second point
        """
        return numpy.sqrt((x1 - x2)**2 + (y1 - y2)**2)

    # Not used
    def distance_pl(self, polygon, x0, y0):
        """ This function computes the shortest distance between a point and a line

        :param polygon: A polygon object
        :param x0, y0: x and y coordinate of the points
        """
        min_dist = sys.maxsize
        for i in range(polygon.shape[0]):
            if i < polygon.shape[0] - 1:
                x1, y1 = polygon[i]
                x2, y2 = polygon[i + 1]
            else:
                x1, y1 = polygon[i]
                x2, y2 = polygon[0]
            temp1 = abs((y2 - y1)*x0 - (x2 - x1)*y0 - x2*y1 - y2*x1)
            temp2 = numpy.sqrt((y2 - y1)**2 + (x2 - x1)**2)
            dist = temp1 / temp2

            if dist < min_dist:
                min_dist = dist
        return min_dist

    # def distance_pl(self, polygon, x0, y0):
    #     """ This function computes the shortest distance between a point and a polygon
    #
    #     :param polygon: A polygon object
    #     :param x0, y0: x and y coordinate of the points
    #     """
    #     min_dist = sys.maxsize
    #     for i in range(polygon.shape[0]):
    #         y1, x1 = polygon[i]
    #         disttmp = numpy.sqrt((x0 - x1)**2 + (y0 - y1)**2)
    #
    #         if disttmp < min_dist:
    #             min_dist = disttmp
    #
    #     return min_dist

    def boundary_condition(self, h, x1, x2, y1, y2):
        """ This is to compute the boundary condition (eq 13) in supplementary info
        :param h: The nearest boundary
        :param x1: The x coordinate of point 1
        :param y1: The y coordinate of point 1
        :param X: A numpy 1D array of x random coordinates
        :param Y: A numpy 1D array of y random coordinates

        :returns : The boundary condition
        """
        d = self.distance(x1, x2, y1, y2)
        #minimum = numpy.array([min(h, d) for d in dist])
        if d == 0:
            d = 0.0000001
        minimum = min(h, d)

        k = (1 - 1/numpy.pi * numpy.arccos(minimum / d))
        return 1 / k

    def mean_G(self):
        """ This function computes the mean of G in 2D
        """
        mean = numpy.pi * numpy.diff(numpy.array(self.distance_fit) ** 2)
        return mean

    # Not used
    @staticmethod
    def PolyArea(x, y):
        """ This function computes the area of a cluster
        :param x: A numpy array of x coordinates
        :param y: A numpy array of y coordinates
        """
        return 0.5*numpy.abs(numpy.dot(x,numpy.roll(y,1))-numpy.dot(y,numpy.roll(x,1)))

# Not used
def create_mask(img):
    """ This function creates a mask for the detection

    :param img: A 2D numpy array

    :returns : A 2D numpy array of the masked ROI and coordinates of the polygon
    """
    mask = numpy.zeros(img.shape)
    r = select_ROI.ROI(img)
    poly = r.get()
    rr, cc = polygon(poly[:,1], poly[:,0])
    mask[rr, cc] = 1
    #print(numpy.sum(mask))
    return mask, poly