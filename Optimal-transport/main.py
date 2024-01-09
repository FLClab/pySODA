import os
import numpy as np
import matplotlib.pyplot as plt
import pixel_tplans
import utils
from tqdm import tqdm
import scipy.stats
plt.style.use('dark_background')

"""
 BassoonFUS: Bassoon in channel 0, FUS in channel 1
 BassoonPSD: Bassoon in channel 0, PSD95 in channel 1
 ActinCaMKII: CamKII in channel 0, Actin in channel 1

"""
BassoonFUS_PLKO = './jmdata_ot/composite/bassoon_fus/plko'
BassoonFUS_318 = './jmdata_ot/composite/bassoon_fus/318'
BassoonPSD_PLKO = './jmdata_ot/composite/bassoon_psd/plko'
BassoonPSD_318 = './jmdata_ot/composite/bassoon_psd/318'

GFP = "/home/ulaval.ca/frbea320/projects/ul-val-prj-def-fllac4/julia_Actin_CaMKII/12-GFP"
NT = "/home/ulaval.ca/frbea320/projects/ul-val-prj-def-fllac4/julia_Actin_CaMKII/12-Non_Transfected"
RESCUE = "/home/ulaval.ca/frbea320/projects/ul-val-prj-def-fllac4/julia_Actin_CaMKII/12-rescue"
shRNA_BCaMKII = "/home/ulaval.ca/frbea320/projects/ul-val-prj-def-fllac4/julia_Actin_CaMKII/12-shRNA-BCaMKII"
shRNA_RESCUE = "/home/ulaval.ca/frbea320/projects/ul-val-prj-def-fllac4/julia_Actin_CaMKII/12-shRNA_rescue"

ACTIN_COFILIN = "/home/ulaval.ca/frbea320/projects/ul-val-prj-def-fllac4/alexy_Actin-Cofilin/Actin-Cofilin"
ACTIN_COFILIN_PRED = "/home/ulaval.ca/frbea320/projects/ul-val-prj-def-fllac4/alexy_Actin-Cofilin/predictions"

ACTIN_CAMKII = [GFP, NT, RESCUE, shRNA_BCaMKII, shRNA_RESCUE]

CONDITION = 'Rescue'


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


def extract_crops(directory):
    """
    Takes a directory and splits all the images it contains into crops
    Default size of the crops is 128x128

    Params:
    ---------
    directory (string): path to the directory which contains the images to split into crops

    Returns:
    all_crops_a, all_crops_b (list, list): lists of crops of images
    """
    tiff_files = [fname for fname in os.listdir(
        directory) if fname.endswith('.tif')]
    tiff_files = [os.path.join(directory, p) for p in tiff_files]
    all_crops_a, all_crops_b = [], []
    for img_file in tiff_files:
        crops_a, crops_b = utils.img_to_crops(img_file)
        all_crops_a += crops_a
        all_crops_b += crops_b
    assert len(all_crops_a) == len(all_crops_b)
    return (all_crops_a, all_crops_b)


def extract_masked_crops(directory):
    tiff_files = [fname for fname in os.listdir(
        directory) if fname.endswith('.tif')]
    tiff_files = [os.path.join(directory, p) for p in tiff_files]
    mask_files = utils.find_mask_files(tiff_files)
    rings_a, rings_b, fibres_a, fibres_b = [], [], [], []
    for t_file, m_file in zip(tiff_files, mask_files):
        (ra, rb), (fa, fb) = utils.img_to_crops_with_masks(t_file, m_file)
        rings_a += ra
        rings_b += rb
        fibres_a += fa
        fibres_b += fb
    assert len(rings_a) == len(rings_b)
    assert len(fibres_a) == len(fibres_a)
    return (rings_a, rings_b), (fibres_a, fibres_b)


def extract_masked_crops_v2(img_files, mask_files):
    rings_a, rings_b, fibres_a, fibres_b = [], [], [], []
    for t_file, m_file in zip(img_files, mask_files):
        (ra, rb), (fa, fb) = utils.img_to_crops_with_masks(t_file, m_file)
        rings_a += ra
        rings_b += rb
        fibres_a += fa
        fibres_b += fb
    assert len(rings_a) == len(rings_b)
    assert len(fibres_a) == len(fibres_b)
    return (rings_a, rings_b), (fibres_a, fibres_b)


def compute_OTC(crops, max_dist=30):
    """
    Computes the optimal transport curve considering distances going up to max_dist.
    One point on the transport curve is the fraction of the total mass transported given a maximum distance allowed for transportation
    Considering many such distances gives the full optimal transport curve

    Params:
    ----------
    crops (list): list of image crops for which to do the computation
    max_dist (int): maximum distance to consider when computing the OTC

    Returns:
    ----------
    otc_avg (array): Optimal transport curve average over the number of crops
    otc_std (array): Standard deviations of the OTC at each distance considered
    """
    distances = np.arange(0, 25, 1)
    (crops_a, crops_b) = crops
    num_samples = len(crops_a)
    ot_curve = np.zeros((len(crops_a), distances.shape[0]))
    for i, (c_a, c_b) in enumerate(zip(crops_a, crops_b)):
        print(f"Processing crop {i + 1} of {len(crops_a)}")
        transport_plan, cost_matrix, mass_ratio = pixel_tplans.calculate_tplans([
                                                                                c_a, c_b])
        otc_values = []
        for d in tqdm(distances, desc='Looping through distances'):
            transported_mass = utils.compute_transported_mass(
                transport_plan, cost_matrix, d)
            otc_values.append(transported_mass * mass_ratio)
        otc_values = np.array(otc_values)
        ot_curve[i] = otc_values
    otc_avg = np.mean(ot_curve, axis=0)
    otc_std = np.std(ot_curve, axis=0)
    confidence = compute_confidence_interval(otc_avg, otc_std)
    return otc_avg, otc_std, confidence, distances


def main_rerun():
    plko = np.load('./results/BassoonFUS/plko.npz')
    plko_otc, plko_confidence, distances = plko['otc'], plko['confidence'], plko['distances']

    shfus = np.load('./results/BassoonFUS/shfus.npz')
    shfus_otc, shfus_confidence = shfus['otc'], shfus['confidence']

    random = np.load('./results/pele-mele/random.npz')
    random_otc, random_confidence = random['otc'], random['confidence']

    perfect_otc = np.array([1.0] * len(distances))
    perfect_confidence = np.array([0.0] * len(distances))

    coloc = np.load('./results/BassoonFUS/colocalized.npz')
    coloc_otc, coloc_confidence = coloc['otc'], coloc['confidence']
    xtick_locs = [1, 5, 10, 15, 20]
    xtick_labels = [str(item * 20) for item in xtick_locs]
    fig = plt.figure()
    plt.plot(distances, plko_otc, color='lightblue', label='PLKO')
    plt.fill_between(distances, plko_otc - plko_confidence,
                     plko_otc + plko_confidence, facecolor='lightblue', alpha=0.3)
    plt.plot(distances, shfus_otc, color='lightcoral', label='shFUS')
    plt.fill_between(distances, shfus_otc - shfus_confidence,
                     shfus_otc + shfus_confidence, facecolor='lightcoral', alpha=0.3)
    plt.plot(distances, random_otc, color='magenta', label='Random')
    plt.fill_between(distances, random_otc - random_confidence,
                     random_otc + random_confidence, facecolor='magenta', alpha=0.3)
    plt.plot(distances, coloc_otc, color='gold',
             label='High colocalization')
    plt.fill_between(distances, coloc_otc - coloc_confidence,
                     coloc_otc + coloc_confidence, facecolor='gold', alpha=0.3)
    plt.plot(distances, perfect_otc, color='limegreen', label='Identical')
    plt.fill_between(distances, perfect_otc - perfect_confidence,
                     perfect_otc + perfect_confidence, facecolor='limegreen', alpha=0.3)
    plt.xlabel('Distance (nm)', fontsize=16)
    plt.xticks(ticks=xtick_locs, labels=xtick_labels)
    plt.ylabel('OTC', fontsize=16)
    plt.legend(fontsize=12, loc='lower right')
    plt.title('ALS model: Bassoon - FUS')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    fig.savefig(
        "./results/BassoonFUS/FINS_BassoonFUS.png", bbox_inches='tight')
    fig.savefig(
        "./results/BassoonFUS/FINS_BassoonFUS.pdf", transparent=True, bbox_inches='tight')


def main():
    # img_files, mask_files = utils.load_actin_cofilin_data()
    # rings, fibres = extract_masked_crops_v2(img_files, mask_files)

    # rings_otc, rings_std, rings_confidence, distances = compute_OTC(rings)
    # np.savez("./rings_actin-cofilin", otc=rings_otc,
    #          confidence=rings_confidence, distances=distances)
    # fibres_otc, fibres_std, fibres_confidence, distances = compute_OTC(fibres)
    # np.savez("./fibres_actin-cofilin", otc=fibres_otc,
    #          confidence=fibres_confidence, distances=distances)
    rings = np.load('./rings_actin-cofilin.npz')
    rings_otc, rings_confidence, distances = rings['otc'], rings['confidence'], rings['distances']

    fibres = np.load('./fibres_actin-cofilin.npz')
    fibres_otc, fibres_confidence = fibres['otc'], fibres['confidence']

    random = np.load('./results/pele-mele/random.npz')
    random_otc, random_confidence = random['otc'], random['confidence']

    coloc = np.load('./results/BassoonFUS/colocalized.npz')
    coloc_otc, coloc_confidence = coloc['otc'], coloc['confidence']

    # Plot the results
    xtick_locs = [1, 5, 10]
    xtick_labels = [str(item * 20) for item in xtick_locs]
    fig = plt.figure()
    # Rings
    plt.plot(distances, rings_otc, color='lightcoral', label='Rings')
    plt.fill_between(distances, rings_otc - rings_confidence, rings_otc +
                     rings_confidence, facecolor='lightcoral', alpha=0.3)
    # Fibres
    plt.plot(distances, fibres_otc, color='lightblue', label='Fibers')
    plt.fill_between(distances, fibres_otc - fibres_confidence,
                     fibres_otc + fibres_confidence, facecolor='lightblue', alpha=0.3)

    plt.plot(distances[:20], random_otc, color='magenta', label='Random')
    plt.fill_between(distances[:20], random_otc - random_confidence,
                     random_otc + random_confidence, facecolor='magenta', alpha=0.3)
    plt.plot(distances[:20], coloc_otc, color='gold',
             label='High colocalization')
    plt.fill_between(distances[:20], coloc_otc - coloc_confidence,
                     coloc_otc + coloc_confidence, facecolor='gold', alpha=0.3)

    plt.xlabel('Distance (nm)', fontsize=16)
    plt.xticks(ticks=xtick_locs, labels=xtick_labels)
    plt.ylabel('OTC', fontsize=16)
    plt.legend(fontsize=16)
    plt.title('Actin - Cofilin', fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlim([0, 10])
    fig.savefig(
        "./results/rings-vs-fibres/actin-cofilin.png")
    fig.savefig(
        "./results/rings-vs-fibres/actin-cofilin.pdf", transparent=True, bbox_inches='tight')


if __name__ == "__main__":
    main()
