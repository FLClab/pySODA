import skimage.filters
import numpy as np
from matplotlib import pyplot as plt

"""
Work in progress of a code to create synthetic images
"""


def synth_images(y=1000, x=1000):
    """
    Creates a 2-channel image of random spots with a noisy background. This was used to test SODA if spots are
    completely random. Result: no coupling is detected.
    :param y: int; number of rows in image
    :param x: int; number of columns in image
    :return image: numpy array with shape (2,y,x) containing the data for the synthetic image
    """
    snr = 0.5
    A = 100
    B = 50

    image = np.ndarray((2,y,x))
    z = image.shape[0]
    for i in range(z):
        sigma = np.sqrt((A ** 2 / snr ** 2) - (A + B))
        G = np.random.random_sample((y,x))
        G[G < 0.9985-i*0.003] = 0
        G[G > 0] = 32000
        G = skimage.filters.gaussian(G,2)
        N = np.random.normal(0, sigma, (y,x))

        image[i] = G+N

    return image

if __name__=='__main__':
    plt.imshow(synth_images())
    plt.show()


