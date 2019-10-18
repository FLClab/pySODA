from skimage import io
from skimage import measure
from scipy.spatial import distance
from matplotlib import pyplot
import os
import pickle
import numpy


def make_coupled_watershed(pfile_path, images_path, output_path):
    """
    Make a watershed masks of only the coupled spots from the images from pfile
    :param pfile_path: Path of the pfile containing the info from excel files
    :param images_path: Path of the images that are in the pfile
    """
    data = pickle.load(open(pfile_path,'rb'))
    path = images_path
    print(data.shape)

    # Loop through all files
    subspots_dict = dict()
    for file in data:
        # Get image name from excel file name
        image_name = str.join('_', file.split('\\''')[-1].split('_')[2:-1])

        # Get the spots data from the pfile
        couples_data = data[file][0]
        couples_spots_positions = numpy.array((couples_data['X1'], couples_data['Y1']))

        # Get the path to original mask + watershed mask
        mask_path = image_name + '_filtered_spots_ch0.tif'
        watershed_path = '.'.join(image_name.split('.')[:-1]) + '_filtered.tif'

        try:
            base_image = io.imread(os.path.join(path, image_name))
        except FileNotFoundError:
            continue
        try:
            watershed_mask = io.imread(os.path.join(path, watershed_path))
        except FileNotFoundError:
            # Get the name of images with a missing mask
            print(image_name)
            continue

        # Get position of all spots in the original mask
        mask = io.imread(os.path.join(path, mask_path))
        mask_lab, num = measure.label(mask, return_num=True)
        mask_props = measure.regionprops(mask_lab, intensity_image=base_image[0])

        all_spots_positions = numpy.ndarray((2, num))
        for i in range(0, num):
            all_spots_positions[:, i] = numpy.flip(numpy.array(mask_props[i].weighted_centroid))

        # Find corresponding spots in pfile couples info vs mask spots
        dists = distance.cdist(all_spots_positions.transpose(), couples_spots_positions.transpose())
        dists = numpy.sort(dists, axis=1)
        good_spots = numpy.argwhere(dists[:,0] < 0.01) + 1

        # Create new watershed mask with only coupled spots
        coupled_mask = numpy.zeros(mask_lab.shape)
        coupled_mask[numpy.isin(mask_lab, good_spots)] = 1

        coupled_watershed = coupled_mask * watershed_mask
        subspots_per_spot = count_subspots(coupled_mask, coupled_watershed)
        subspots_dict[image_name] = subspots_per_spot
        io.imsave(os.path.join(output_path, image_name + '_filtered_coupled.tif'), coupled_watershed.astype('uint16'))

    pickle.dump(subspots_dict, open(os.path.join(output_path, 'subspots.pkl'), 'wb'))
    return subspots_dict


def count_subspots(spots_image, subspots_image):
    labeled_spots, spots_num = measure.label(spots_image, return_num=True)
    subspots_array = numpy.zeros((spots_num,2))

    spots_props = measure.regionprops(labeled_spots)
    split_subspots = labeled_spots * subspots_image
    for p in spots_props:
        labeled_subspots, subspots_num = measure.label(split_subspots == p.label, return_num=True)
        if subspots_num > 0:
            subspots_array[p.label-1, 0] = subspots_num
            subspots_array[p.label-1, 1] = p.area
    return subspots_array


if __name__ == '__main__':

    for cond in ('0mgglybic', 'block'):
        pfile_path = r"Y:\#PUBLIC\twiesner\SynapticProteins_Analysis\pooled_clean\{}\pfile_data_PSDBassoon_{}.pkl".format(cond, cond)
        images_path = r"Y:\#PUBLIC\twiesner\SynapticProteins_Analysis\pooleddata_PB_selfSodafromCouples\{}".format(cond)
        output_path = r"filtered_coupled\{}".format(cond)
        make_coupled_watershed(pfile_path, images_path, output_path)