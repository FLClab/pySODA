import numpy as np
from scipy.ndimage import measurements
from skimage.filters import threshold_otsu
import random
import tifffile
from scipy.stats import pearsonr
import scipy.stats
import glob

import os

JUL_BASE_DIR = '/home/ulaval.ca/frbea320/projects/ul-val-prj-def-fllac4/julia_Actin_CaMKII/predictions'
ACTIN_COFILIN = "/home/ulaval.ca/frbea320/projects/ul-val-prj-def-fllac4/OptimalTransportDatasets/alexy_Actin-Cofilin/Actin-Cofilin"
ACTIN_COFILIN_PRED = "/home/ulaval.ca/frbea320/projects/ul-val-prj-def-fllac4/OptimalTransportDatasets/alexy_Actin-Cofilin/predictions"


def load_actin_cofilin():
    block = np.load('./results/pele-mele/alexyblock.npz')
    kcl = np.load('./results/pele-mele/alexykcl.npz')
    random = np.load('./results/pele-mele/random.npz')
    coloc = np.load('./results/BassoonFUS/colocalized.npz')
    return block, kcl, random, coloc


def load_pd_data():
    plko = np.load('./results/BassoonPSD_PD-N1/wt.npz')
    pd = np.load('./results/BassoonPSD_PD-N1/inf.npz')

    plko_otc, plko_confidence, distances = plko['otc'], plko['confidence'], plko['distances']
    pd_otc, pd_confidence = pd['otc'], pd['confidence']

    random = np.load('./results/pele-mele/random.npz')
    random_otc, random_confidence = random['otc'], random['confidence']

    coloc = np.load('./results/BassoonFUS/colocalized.npz')
    coloc_otc, coloc_confidence = coloc['otc'], coloc['confidence']
    return (plko_otc, plko_confidence), (pd_otc, pd_confidence), (random_otc, random_confidence), (coloc_otc, coloc_confidence), distances


def load_als_data():
    plko = np.load('./results/BassoonFUS/plko.npz')
    plko_otc, plko_confidence, distances = plko['otc'], plko['confidence'], plko['distances']

    shfus = np.load('./results/BassoonFUS/shfus.npz')
    shfus_otc, shfus_confidence = shfus['otc'], shfus['confidence']

    random = np.load('./results/pele-mele/random.npz')
    random_otc, random_confidence = random['otc'], random['confidence']

    coloc = np.load('./results/BassoonFUS/colocalized.npz')
    coloc_otc, coloc_confidence = coloc['otc'], coloc['confidence']

    return (plko_otc, plko_confidence), (shfus_otc, shfus_confidence), (random_otc, random_confidence), (coloc_otc, coloc_confidence), distances


def load_actin_data():
    gfp = np.load('./results/data_bichette/12-GFP.npz')
    gfp_otc, gfp_confidence, distances = gfp['otc'], gfp['confidence'], gfp['distances']

    nt = np.load('./results/data_bichette/12-Non_Transfected.npz')
    nt_otc, nt_confidence = nt['otc'], nt['confidence']

    rescue = np.load('./results/data_bichette/12-rescue.npz')
    rescue_otc, rescue_confidence = rescue['otc'], rescue['confidence']

    shrna = np.load('./results/data_bichette/12-shRNA-BCamKII.npz')
    shrna_otc, shrna_confidence = shrna['otc'], shrna['confidence']

    shrescue = np.load(
        './results/data_bichette/12-shRNA_rescue.npz')
    shrescue_otc, shrescue_confidence = shrescue['otc'], shrescue['confidence']

    random = np.load('./results/pele-mele/random.npz')
    random_otc, random_confidence, random_distances = random[
        'otc'], random['confidence'], random['distances']
    print(len(random_distances))
    print(len(distances))

    coloc = np.load('./results/BassoonFUS/colocalized.npz')
    coloc_otc, coloc_confidence = coloc['otc'], coloc['confidence']

    return (gfp_otc, gfp_confidence), (nt_otc, nt_confidence), (rescue_otc, rescue_confidence), (shrna_otc, shrna_confidence), (shrescue_otc, shrescue_confidence), (random_otc, random_confidence), (coloc_otc, coloc_confidence), distances


def remove_small_regions(rprops):
    new_rprops = []
    for i, r in enumerate(rprops):
        min_row, min_col, max_row, max_col = r.bbox
        height = max_row - min_row
        width = max_col - min_col
        if height > 1 and width > 1:
            new_rprops.append(r)
    return new_rprops, len(new_rprops)


def random_crop(img_a, img_b, new_w, new_h):
    assert img_a.shape == img_b.shape
    height = img_a.shape[0]
    width = img_a.shape[1]
    x = random.randint(0, width - new_w)
    y = random.randint(0, height - new_h)
    new_a = img_a[y:y + new_h, x:x + new_w]
    new_b = img_b[y:y + new_h, x:x + new_w]
    return new_a, new_b


def img_binary_mask(img):
    thresh = threshold_otsu(img)
    binary = img > thresh
    return binary


def clusterize(img):
    img = img > 0
    dims = len(img.shape)
    structure = np.ones(np.ones(dims, int) * 3)
    labeled, ncomponents = measurements.label(img, structure)
    return labeled, ncomponents


def im2mat(img):
    """Converts an image to a matrix (one pixel per line)"""
    return img.reshape((img.shape[0] * img.shape[1], 1))


def mat2im(X, shape):
    """Converts a matrix back to an image"""
    return X.reshape(shape)


def normalize_mass(transport_plan):
    total_mass = np.sum(transport_plan)
    return transport_plan / total_mass


def compute_OTC(transport_plan, cost_matrix, dist):
    """
    This function does the same thing as `compute_transported_mass` below but in a much slower way
    """
    assert transport_plan.shape == cost_matrix.shape
    transported_mass = 0
    row, col = transport_plan.shape[0], transport_plan.shape[1]
    for i in range(row):
        for j in range(col):
            if transport_plan[i][j] != 0 and cost_matrix[i][j] <= dist:
                transported_mass += transport_plan[i][j]
    return transported_mass


def compute_transported_mass(transport_plan, cost_matrix, dist):
    """
    Computes the amount of transported mass that can be transported for a given maximum distance of transport

    Params:
    --------
    transport plan (WxH matrix): optimal transport plan between two distributions
    cost matrix (WxH matrix): cost matrix of the cost matrix between each pair of elements in the two distributions
    dist (int): maximum distance in pixels allowed for a transport

    Returns:
    ---------
    Fraction of the total mass transported
    """
    gt_dist = cost_matrix <= dist
    return np.sum(transport_plan[gt_dist == True])


def random_distributions(size=128):
    """
    Creates two images both with random spatial distributions

    Params:
    --------
    size (int): size of the images
    """
    img_a = np.zeros((size, size))
    img_b = np.zeros((size, size))
    num_regions_a = np.random.randint(5, 20)
    num_regions_b = np.random.randint(5, 20)
    for _ in range(num_regions_a):
        x_size = np.random.randint(2, 10)
        y_size = np.random.randint(2, 10)
        x = np.random.randint(0, size - x_size)
        y = np.random.randint(0, size - y_size)
        for i in range(x, x + x_size):
            for j in range(y, y + y_size):
                img_a[i, j] = np.random.uniform()
    for _ in range(num_regions_b):
        x_size = np.random.randint(2, 10)
        y_size = np.random.randint(2, 10)
        x = np.random.randint(0, size - x_size)
        y = np.random.randint(0, size - y_size)
        for i in range(x, x + x_size):
            for j in range(y, y + y_size):
                img_b[i, j] = np.random.uniform()
    return img_a, img_b


def highly_colocalized_distributions(size=128):
    """
    Builds two images which have high colocalization in their structures

    Params:
    --------
    size (int): size of the images
    """
    img_a = np.zeros((size, size))
    img_b = np.zeros(img_a.shape)
    num_regions = np.random.randint(5, 20)
    for _ in range(num_regions):
        r_size = np.random.randint(2, 20)
        x_a = np.random.randint(0, size - 2 * r_size)
        y_a = np.random.randint(0, size - 2 * r_size)
        x_b = int(np.floor(np.random.normal(loc=x_a, scale=2.0)))
        y_b = int(np.floor(np.random.normal(loc=y_a, scale=2.0)))
        for i in range(x_a, x_a + r_size):
            for j in range(y_a, y_a + r_size):
                img_a[i, j] = random.random()
        for i in range(x_b, x_b + r_size):
            for j in range(y_b, y_b + r_size):
                img_b[i, j] = random.random()
    pearson, _ = pearsonr(img_a.flatten(), img_b.flatten())
    return img_a, img_b, pearson


def build_colocalized_dataset():
    """
    Builds a dataset of crops in which the regions are highly colocalized
    """
    crops_a, crops_b = [], []
    pearson_coeffs = []
    for i in range(400):
        img_a, img_b, pearson = highly_colocalized_distributions()
        pearson_coeffs.append(pearson)
        crops_a.append(img_a)
        crops_b.append(img_b)

    return crops_a, crops_b, np.array(pearson_coeffs)


def build_random_dist_dataset():
    """
    Builds a dataset of crops in which the regions follow a random spatial distribution
    """
    crops_a, crops_b = [], []
    for i in range(400):
        img_a, img_b = random_distributions()
        crops_a.append(img_a)
        crops_b.append(img_b)
    return crops_a, crops_b


def find_mask_files(file_list):
    mask_files = []
    for f in file_list:
        interdir = f.split('/')[7]
        pred_dir = JUL_BASE_DIR + '/' + interdir
        file_name = f.split('/')[-1].split('.')[0] + '.tif'
        full_name = pred_dir + '/' + file_name
        mask_files.append(full_name)
    return mask_files


def img_to_crops_with_masks(img_file, mask_file, size=128, step=96):
    img_a = tifffile.imread(img_file)[0]  # cofilin
    img_b = tifffile.imread(img_file)[1]  # actin
    mask_rings = tifffile.imread(mask_file)[0]  # rings
    mask_fibres = tifffile.imread(mask_file)[1]  # fibres
    mask_rings = mask_rings >= 0.14
    mask_fibres = mask_fibres >= 0.14
    x, y = img_a.shape
    x_m, y_m = mask_rings.shape
    assert x == img_b.shape[0]
    assert y == img_b.shape[1]
    assert x_m == mask_fibres.shape[0]
    assert y_m == mask_fibres.shape[1]
    img_a_rings = img_a * mask_rings
    img_b_rings = img_b * mask_rings
    img_a_fibres = img_a * mask_fibres
    img_b_fibres = img_b * mask_fibres
    start_xs = np.arange(0, x - size, step)
    start_xs = [int(np.floor(x)) for x in start_xs]
    start_ys = np.arange(0, y - size, step)
    start_ys = [int(np.floor(y)) for y in start_ys]
    crops_a_rings, crops_b_rings, crops_a_fibres, crops_b_fibres = [], [], [], []
    img_a_rings_mean = np.mean(img_a_rings)
    img_b_rings_mean = np.mean(img_b_rings)
    img_a_fibres_mean = np.mean(img_a_fibres)
    img_b_fibres_mean = np.mean(img_b_fibres)
    for s_x in start_xs:
        for s_y in start_ys:
            crop_a_r = img_a_rings[s_x:s_x + size, s_y:s_y + size]
            crop_b_r = img_b_rings[s_x:s_x + size, s_y:s_y + size]
            crop_a_f = img_a_fibres[s_x:s_x + size, s_y:s_y + size]
            crop_b_f = img_b_fibres[s_x:s_x + size, s_y:s_y + size]
            crop_a_r_mean = np.mean(crop_a_r)
            crop_b_r_mean = np.mean(crop_b_r)
            crop_a_f_mean = np.mean(crop_a_f)
            crop_b_f_mean = np.mean(crop_b_f)
            if (crop_a_r_mean >= img_a_rings_mean) and (crop_b_r_mean >= img_b_rings_mean):
                crops_a_rings.append(crop_a_r)
                crops_b_rings.append(crop_b_r)
            if (crop_a_f_mean >= img_a_fibres_mean) and (crop_b_f_mean >= img_b_fibres_mean):
                crops_a_fibres.append(crop_a_f)
                crops_b_fibres.append(crop_b_f)
    return (crops_a_rings, crops_b_rings), (crops_a_fibres, crops_b_fibres)


def img_to_crops(img, size=128, step=96):
    """
    Takes a large image and extracts multiple (can be overlapping) crops.

    Returns:
    ------------
    List of extracted crops
    """
    img_a = tifffile.imread(img)[0]  # Bassoon / VGLUT1 / CaMKII
    img_b = tifffile.imread(img)[1]  # PSD-95 / Actin
    x, y = img_a.shape
    assert x == img_b.shape[0]
    assert y == img_b.shape[1]
    start_xs = np.arange(0, x - size, step)
    start_xs = [int(np.floor(x)) for x in start_xs]
    start_ys = np.arange(0, y - size, step)
    start_ys = [int(np.floor(y)) for y in start_ys]
    crops_a = []
    crops_b = []
    img_a_mean = np.mean(img_a)
    img_b_mean = np.mean(img_b)
    for s_x in start_xs:
        for s_y in start_ys:
            crop_a = img_a[s_x:s_x + size, s_y:s_y + size]
            crop_b = img_b[s_x:s_x + size, s_y:s_y + size]
            crop_a_mean = np.mean(crop_a)
            crop_b_mean = np.mean(crop_b)
            if (crop_a_mean >= img_a_mean) and (crop_b_mean >= img_b_mean):
                crops_a.append(crop_a)
                crops_b.append(crop_b)
    return crops_a, crops_b


def img_to_list(direc):
    """
    In the case we're able to work on whole images and not crops.
    Take a directory of multi-channel tiff files and convert into two list of images
    """
    tiff_files = [fname for fname in os.listdir(
        direc) if fname.endswith('.tif')]
    tiff_files = [os.path.join(direc, p) for p in tiff_files]
    crops_a, crops_b = [], []
    for img in tiff_files:
        img_a = tifffile.imread(img)[2]
        img_b = tifffile.imread(img)[3]
        x, y = img_a.shape
        assert x == img_b.shape[0]
        assert y == img_b.shape[1]
        crops_a.append(img_a)
        crops_b.append(img_b)
    return crops_a, crops_b


def compute_confidence_interval(averages, stds):
    """
    Computes the 95% confidence interval values
    Note that the Student's t distribution is used to compute t_crit.

    Params:
    --------
    averages (array): array of OTC values for different maximum transport distances
    stds (array): array of standard deviations on the OTC values for different maximum transport distances

    Returns:
    ---------
    95% confidence interval values (lower and upper bounds)
    """
    dof = averages.shape[0] - 1
    confidence = 0.95
    t_crit = np.abs(scipy.stats.t.ppf((1 - confidence) / 2, dof))
    confidence = stds * t_crit / np.sqrt(averages.shape[0])
    return confidence


def find_corresponding_mask(img_file, mask_paths=ACTIN_COFILIN_PRED):
    img_name = img_file.split('/')[-1]
    img_name_no_ext = img_name[:-19]
    print(img_name_no_ext)
    for m in mask_paths:
        m_name = m.split('/')[-1].split('.')[0]
        if m_name == img_name_no_ext:
            return m


def load_actin_cofilin_data():
    img_files = glob.glob("{}/*".format(ACTIN_COFILIN))
    mask_files = glob.glob("{}/*".format(ACTIN_COFILIN_PRED))
    sorted_mask_files = []
    for i in img_files:
        m_name = find_corresponding_mask(i, mask_files)
        sorted_mask_files.append(m_name)
    assert len(img_files) == len(sorted_mask_files)
    return img_files, sorted_mask_files
