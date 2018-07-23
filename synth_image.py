from matplotlib import pyplot as plt
import skimage.filters
import numpy as np
import wavelet_SODA as wv
"""
Work in progress of a code to create synthetic images
"""

A = 100
snr = 0.6
B = 50

sigma = np.sqrt((A**2/snr**2)-(A+B))

for i in range(500):
    sigma = np.sqrt((A**2/(snr)**2) - (A + 50))
    G = np.random.random_sample((256,256))
    G[G < 0.9985] = 0
    G[G > 0] = 32000
    G = skimage.filters.gaussian(G,2)
    N = np.random.normal(0, sigma, (256,256))

    img = G+N
    det = wv.Detection_Wavelets(img, [2], 100).computeDetection()

    fig, axs = plt.subplots(1,3)
    axs[0].imshow(img)
    axs[1].imshow(det)
    axs[2].imshow(G)
    plt.show()
