import matplotlib.pyplot as plt
import utils
import ot
import tifffile
import numpy as np
import time


def calculate_tplans(imgs, crops=True, use_mass_ratio=False):
    """
    Calculates the cost matrix and optimal transport plan in the case where we want to transport
    one image's spatial distribution to another's, and where the mass to be transported is the pixel values
    I.e., we want to transport each pixel

    Params:
    ----------
    imgs (tiff file OR list): the images on which to do optimal transport

    Returns:
    ----------
    transport plan (array): The optimal transport plan
    """
    img_a, img_b = imgs[0], imgs[1]

    # Denoising
    img_a = img_a * (img_a > np.percentile(img_a, 15))
    img_b = img_b * (img_b > np.percentile(img_b, 15))

    # Compute mass_ratio
    mass_a = np.sum(img_a)
    mass_b = np.sum(img_b)
    mass_ratio = mass_a / mass_b

    assert img_a.shape == img_b.shape
    img_a_pos = np.zeros((img_a.shape[0] * img_a.shape[1], 2))
    img_a_prod = np.zeros((img_a.shape[0] * img_a.shape[1], ))
    img_b_prod = np.zeros((img_b.shape[0] * img_b.shape[1], ))

    # Initialize positions and masses
    index = 0
    for i in range(img_a.shape[0]):
        for j in range(img_a.shape[1]):
            img_a_pos[index][0] = i
            img_a_pos[index][1] = j
            img_a_prod[index] = img_a[i, j]
            img_b_prod[index] = img_b[i, j]
            index += 1
    img_b_pos = img_a_pos  # Coordinates of the masses to be transported are the same because imgs have same shape and we're transporting all pixels

    # Normalize mass so that it sums to 1 in both images
    img_a_prod = img_a_prod / img_a_prod.sum()
    img_b_prod = img_b_prod / img_b_prod.sum()
    # Cost matrix
    # default metric is squared euclidean
    cost_matrix = ot.dist(img_a_pos, img_b_pos, metric='euclidean')

    # this will ignore transports over 500nm (25 pixels * 20nm (pixel size))
    # binary_cost_matrix = cost_matrix <= 25
    # cost_matrix = np.multiply(cost_matrix, binary_cost_matrix)
    # Transport plan
    # OT pixel-by-pixel is more accurate but requires many more iterations to converge
    transport_plan = ot.emd(img_a_prod, img_b_prod,
                            cost_matrix, numItermax=500000)
    return transport_plan, cost_matrix, mass_ratio
