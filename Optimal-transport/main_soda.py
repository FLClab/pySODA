import os
import numpy as np
import matplotlib.pyplot as plt
from soda_tplans import soda_tplans, tplans_pixel_by_pixel
from utils import normalize_mass, compute_OTC, compute_OTC_v2
from tqdm import tqdm
from split_crops import img_to_crops
"""
    N.B) For all images in the directories listed below:
        channel 0 --> Bassoon
        channel 1 --> PSD95
"""
BLOCK_DIR = "../../dataset_creator/theresa-dataset/DATA_SYNPROT_BLOCKGLUGLY/Block"
GLUGLY_DIR = "../../dataset_creator/theresa-dataset/DATA_SYNPROT_BLOCKGLUGLY/GluGLY"

FUS_MAP2_VGLUT1_PLKO = '../../../projects/ul-val-prj-def-fllac4/3_color_FUS_MAP2_VGLUT1_unmixed/PLKO'
FUS_MAP2_VGLUT1_318 = '../../../projects/ul-val-prj-def-fllac4/3_color_FUS_MAP2_VGLUT1_unmixed/shFus318'

BassoonFUS_PLKO = './jmdata_ot/composite/bassoon_fus/plko'
BassoonFUS_318 = './jmdata_ot/composite/bassoon_fus/318'

BassoonPSD_PLKO = './jmdata_ot/composite/bassoon_psd/plko'
BassoonPSD_318 = './jmdata_ot/composite/bassoon_psd/318'


def get_control_disease_otc(N, distances, paths=[BassoonFUS_PLKO, BassoonFUS_318]):
    for dir_idx, direc in enumerate(paths):
        tiff_files = [fname for fname in os.listdir(
            direc) if fname.endswith('tif')]
        tiff_files = [os.path.join(direc, p) for p in tiff_files]
        all_crops_a, all_crops_b = [], []
        otc_avgs = np.zeros((2, N))
        otc_stds = np.zeros((2, N))
        # populate our lists of crops
        for img_file in tiff_files:
            crops_a, crops_b = img_to_crops(img_file)
            all_crops_a += crops_a
            all_crops_b += crops_b
        assert len(all_crops_a) == len(all_crops_b)
        # print(len(all_crops_a))
        ot_curve = np.zeros((len(all_crops_a), N))
        for i, (crop_a, crop_b) in enumerate(zip(all_crops_a, all_crops_b)):
            print(f"Processing image {i + 1} of {len(all_crops_a)}")
            transport_plan, cost_matrix = tplans_pixel_by_pixel(
                [crop_a, crop_b])
            otc_values = []
            for dist in tqdm(distances):
                transported_mass = compute_OTC_v2(
                    transport_plan, cost_matrix, dist)
                otc_values.append(transported_mass)
            otc_values = np.array(otc_values)
            ot_curve[i] = otc_values
        ot_avg = np.mean(ot_curve, axis=0)
        ot_std = np.std(ot_curve, axis=0)
        otc_avgs[dir_idx] = ot_avg
        otc_stds[dir_idx] = ot_avg
    return otc_avgs, otc_stds


def main():
    N = 30
    common_dist = np.linspace(0, N, N)
    otc_avgs, otc_stds = get_control_disease_otc(N, common_dist)
    xtick_locs = [1, 5, 10, 15, 20, 25, 30]
    xtick_labels = [str(item * 20) for item in xtick_locs]
    fig = plt.figure()
    plt.plot(common_dist, otc_avgs[0], color='lightblue', label='PLKO')
    plt.fill_between(common_dist, otcs_avgs[0] - otc_stds[0],
                     otc_avgs[0] + otc_stds[0], facecolor='lightblue', alpha=0.5)
    plt.plot(common_dist, otc_avgs[1], color='lightcoral', label='shFUS-318')
    plt.fill_between(common_dist, otcs_avgs[1] - otc_stds[1],
                     otc_avgs[1] + otc_stds[1], facecolor='lightcoral', alpha=0.5)
    plt.xlabel('Distance (nm)', fontsize=14)
    plt.xticks(ticks=xtick_locs, labels=xtick_labels)
    plt.ylabel('OTC', fontsize=14)
    plt.legend()
    plt.title('Bassoon - FUS', fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    fig.savefig('./final_figures/JM_Bassoon_FUS/crops128_pixel_by_pixel_otc.png')


# def main():
#     N = 1500
#     common_dist = np.linspace(0, N, N)
#     ctrl_avg = None
#     ctrl_std = None
#     disease_avg = None
#     disease_std = None
#     for dir_idx, direc in enumerate([BassoonFUS_PLKO, BassoonFUS_318]):
#         tiff_files = [fname for fname in os.listdir(
#             direc) if fname.endswith('.tif')]
#         tiff_files = [os.path.join(direc, p) for p in tiff_files]
#         num_files = len(tiff_files)
#         ot_curve = np.zeros((num_files, N))
#         for i in range(num_files):
#             print("\nProcessing image {} of {}".format(i + 1, num_files))
#             # compute cost matrix and transport plan
#             transport_plan, cost_matrix = soda_tplans(imgs=tiff_files[i])
#             otc_values = []
#             # computes fraction of mass being transported for a range of distances
#             for dist in tqdm(common_dist):
#                 transported_mass = compute_OTC_v2(
#                     transport_plan, cost_matrix, dist)
#                 otc_values.append(transported_mass)
#             otc_values = np.array(otc_values)
#             ot_curve[i] = otc_values
#         ot_avg = np.mean(ot_curve, axis=0)
#         ot_std = np.std(ot_curve, axis=0)
#         if dir_idx == 0:
#             ctrl_avg = ot_avg
#             ctrl_std = ot_std
#         else:
#             disease_avg = ot_avg
#             disease_std = ot_std

#     # plot the OT curves
#     xtick_locs = [200, 600, 1000, 1400]
#     # convert distance from pixels to nm, assuming pixel size = 15
#     xtick_labels = [str(item * 15) for item in xtick_locs]
#     fig = plt.figure()
#     plt.plot(common_dist, ctrl_avg,
#              color='lightblue', label='PLKO')
#     plt.fill_between(common_dist, ctrl_avg - ctrl_std,
#                      ctrl_avg + ctrl_std, facecolor='lightblue', alpha=0.5)
#     plt.plot(common_dist, disease_avg, color='lightcoral', label='shFus318')
#     plt.fill_between(common_dist, disease_avg - disease_std,
#                      disease_avg + disease_std, facecolor='lightcoral', alpha=0.5)
#     plt.xlabel('Distance (nm)', fontsize=14)
#     plt.xticks(ticks=xtick_locs, labels=xtick_labels)
#     plt.ylabel('OTC', fontsize=14)
#     plt.legend()
#     plt.title('Bassoon - FUS OTC', fontsize=18)
#     plt.xticks(fontsize=14)
#     plt.yticks(fontsize=14)
#     fig.savefig('./figures/bassoon_psd_jmichel_OTC_soda_zoomed.png')

if __name__ == "__main__":
    main()
