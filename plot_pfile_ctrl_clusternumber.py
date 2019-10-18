import numpy
import pickle
import os
import glob
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from skimage import morphology
from skimage import filters
from skimage import io

from matplotlib import pyplot


PATH = r"Y:\#PUBLIC\twiesner\SynapticProteins_Analysis\Suppl_clusternumberandratio_BP_diffSTIM\images"
conditions = ["block", "glugly", "0mgglybic"]

labels = ["coupled", "totall_ch0","totall_ch1"]

def histogram(data, _range=(0, 16), bins=16, average=False):
    """
    Computes bins the data according to first dimension

    :param data: A 2D `numpy.ndarray`
    :param range: A `tuple` of the minimal and maximal values to use
    :param bins: The number of bins to use

    :returns: A `tuple` of mean and std of of the binned data if average else a numpy array
              The bin edges used for calculation
    """
    bin_edges = numpy.linspace(_range[0], _range[1], num=bins + 1)
    hist = []
    for i, bin_edge in enumerate(bin_edges[:-1]):
        hist.append(data[(data[:, 0] >= bin_edge) & (data[:, 0] < bin_edges[i + 1]), 1:])
    if average:
        hist = (numpy.array([numpy.mean(h, axis=0) if h.size > 0 else numpy.zeros(data.shape[1] - 1) for h in hist]),
                numpy.array([numpy.std(h, axis=0) if h.size > 0 else numpy.zeros(data.shape[1] - 1) for h in hist]))
    return hist, bin_edges

def find_ROI(image, sigma=10, threshold=200):
    """
    Find the region of interest for the analysis using a gaussian blur and thresholding.
    :param sigma: Sigma of the gaussian blur
    :param threshold: Threshold multiplier
    :return roi_mask: mask of the ROI
    """
    stack = numpy.sum(image, axis=0)
    filt = filters.gaussian(stack, sigma=sigma)
    threshold = numpy.mean(filt) * (100 / threshold)
    filt[filt < threshold] = 0
    filt[filt >= threshold] = 1
    filt = morphology.binary_closing(filt)

    # Keep only areas with a significant volume
    labels = label(filt, connectivity=2)
    label_props = regionprops(labels)
    roi_mask = numpy.copy(labels)
    arealist = []
    for i in range(len(label_props)):
        arealist.append(label_props[i].area)
    for i in range(len(label_props)):
        if label_props[i].area < numpy.mean(arealist):
            roi_mask[roi_mask == i + 1] = 0
    roi_mask[roi_mask > 0] = 1

    return roi_mask


if __name__ == "__main__":

    for condition in conditions:
        files = glob.glob(os.path.join(PATH, "pfile*{}.pkl".format(condition)), recursive=True)
        # print(files)
        clusters = {}

        for file in files:
            clusters[file] = []
            data = pickle.load(open(file, "rb"))
            for xlsx_name, xlsx_file in data.items():
                image_name = str.join('_', os.path.basename(xlsx_name).split('_')[2:-1])
                image_path = os.path.join(PATH, condition, image_name)
                image_data = io.imread(image_path)
                image_mask = find_ROI(image_data)

                data_couple = xlsx_file[0]
                data_ch0 = xlsx_file[1]
                data_ch1 = xlsx_file[2]
                clusternumber_couple = len(data_couple)
                clusternumber_ch0 = len(data_ch0)
                clusternumber_ch1 = len(data_ch1)
                mask_area = numpy.sum(image_mask > 0)

                # print(clusternumber_couple)
                # print(clusternumber_ch0)
                # print(clusternumber_ch1)
                # print(condition)
                # clusters.update({condition: [clusternumber_couple,  clusternumber_ch0, clusternumber_ch1]})
                clusters[file].append([clusternumber_couple,  clusternumber_ch0, clusternumber_ch1])

                #

        print(clusters)
        fig, ax = plt.subplots()
        for file in files:
            values = numpy.array(clusters[file])
            index = numpy.arange(len(values))
            ax.boxplot(values)
            # scatter of data points
            for ind, data_points in zip(index, values.T):
                ax.scatter(numpy.random.normal(loc=ind + 1, scale=0.03, size=data_points.shape), data_points,
                           alpha=0.3)
        # plt.show()
        ax.legend()
        ax.set_xticklabels(labels)
        ax.set_title('{}'.format(condition))
        ax.set_ylabel("clusternumber per image")
        ax.set_ylim(0, 3000)

        for key, val in clusters.items():
            print(key, "=>", val)

        print("-> Sucess! - boxplots  <- \n  | Condition : {} |".format( condition))
        plt.savefig('Clusternumber_{}.png'.format(condition), bbox_inches='tight',dpi=600)
        plt.savefig('Clusternumber_{}.pdf'.format(condition), bbox_inches='tight', dpi=600)

            #     basename = os.path.basename(file)
            #     os.makedirs(os.path.join(os.path.dirname(file), "coupling-distance"), exist_ok=True)
            #     savename = os.path.join(os.path.dirname(file), "coupling-distance", "_".join((str(image_name), *basename.split(".")[0].split("_")[-2:])) + ".png")
            #     pyplot.savefig(savename, bbox_inches="tight", transparent=True)
            #     pyplot.close()
